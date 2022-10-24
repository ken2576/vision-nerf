# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import time
import random
import subprocess
import numpy as np
import shutil
import torch
import torch.utils.data.distributed
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter

from data import eval_dataset_dict
from models.render_ray import render_rays
from models.render_image import render_single_image
from models.model import VisionNerfModel
from models.sample_ray import RaySamplerSingleImage, RaySamplerMultipleImages
from models.criterion import Criterion
from models.projection import Projector
from data.create_training_dataset import create_training_dataset
import opt
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, img2psnr, get_views


# Fix numpy's duplicated RNG issue and make the experiments reproducible
# https://pytorch.org/docs/stable/notes/randomness.html#dataloader
def workder_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)
    torch.set_deterministic(True)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def train(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.ckptdir, args.expname)
    print('checkpoints will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    log_folder = os.path.join(args.logdir, args.expname)
    print('logs will be saved to {}'.format(log_folder))
    os.makedirs(log_folder, exist_ok=True)

    # Save the args and config files
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # Create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               worker_init_fn=workder_init_fn,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               shuffle=True if train_sampler is None else False)

    # Create training visualization dataset
    train_indices = args.train_indices
    if args.debug:
        train_indices = [0]
    train_vis_dataset = eval_dataset_dict[args.data_type](args, 'train')
    train_vis_subset = torch.utils.data.Subset(train_vis_dataset, train_indices)
    train_vis_loader = DataLoader(train_vis_subset, batch_size=1, shuffle=False)

    # Create validation dataset
    val_dataset = eval_dataset_dict[args.data_type](args, 'val')
    val_indices = args.val_indices
    if args.debug:
        val_indices = [0]
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

    # Create model
    model = VisionNerfModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)
    # Create projector
    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    tb_dir = os.path.join(args.logdir, args.expname)
    if args.local_rank == 0:
        writer = SummaryWriter(tb_dir)
        print('saving tensorboard files to {}'.format(tb_dir))
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 0
    while global_step < model.start_step + args.n_iters + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop

            # Load training rays
            ray_sampler = RaySamplerMultipleImages(train_data, device, global_step, bbox_steps=args.bbox_steps)

            ray_batch = ray_sampler.random_sample(args.N_rand,
                                                  sample_mode=args.sample_mode,
                                                  center_ratio=args.center_ratio,
                                                  )
            featmaps = model.encode(ray_batch['src_rgbs']) # (batch, #views, #channels, height', width')

            ret = render_rays(ray_batch=ray_batch,
                              model=model,
                              featmaps=featmaps,
                              projector=projector,
                              N_samples=args.N_samples,
                              inv_uniform=args.inv_uniform,
                              N_importance=args.N_importance,
                              det=args.det,
                              white_bkgd=args.white_bkgd)
           
            # compute loss
            model.optimizer.zero_grad()
            loss, scalars_to_log = criterion(ret['outputs_coarse'], ray_batch, scalars_to_log)

            if ret['outputs_fine'] is not None:
                fine_loss, scalars_to_log = criterion(ret['outputs_fine'], ray_batch, scalars_to_log)
                loss += fine_loss

            loss.backward()
            scalars_to_log['loss'] = loss.item()
            model.optimizer.step()
            if args.use_warmup and global_step < args.warmup_steps:
                model.warmup_scheduler.step()
                model.scheduler.step()
            else:
                model.scheduler.step()

            scalars_to_log['lr'] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(ret['outputs_coarse']['rgb'], ray_batch['rgb']).item() # pylint: disable=unsubscriptable-object
                    scalars_to_log['train/coarse-loss'] = mse_error
                    scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
                    if ret['outputs_fine'] is not None:
                        mse_error = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb']).item() # pylint: disable=unsubscriptable-object
                        scalars_to_log['train/fine-loss'] = mse_error
                        scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)

                    logstr = '{} Epoch: {}  step: {} '.format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    print(logstr)
                    print('each iter time {:.05f} seconds'.format(dt))

                if global_step % args.i_weights == 0:
                    print('Saving checkpoints at {} to {}...'.format(global_step, out_folder))
                    fpath = os.path.join(out_folder, 'model_{:06d}.pth'.format(global_step))
                    model.save_model(fpath)

                if global_step % args.i_img == 0:
                    model.switch_to_eval()
                    
                    print('Logging a random validation view...')
                    output_dicts = []
                    src_imgs = []
                    gt_imgs = []

                    for val_data in val_loader:
                        pairs = get_views(val_data, args.val_src_views, args.val_tgt_views)
                        for idx, pair in enumerate(pairs):
                            tmp_ray_sampler = RaySamplerSingleImage(pair, device, render_stride=args.render_stride)
                            output_dict = render_image(args, model, tmp_ray_sampler, projector, args.render_stride)
                            src_img, gt_img = get_imgs_from_sampler(tmp_ray_sampler, args.render_stride)

                            output_dicts.append(output_dict)
                            src_imgs.append(src_img)
                            gt_imgs.append(gt_img)
                    
                    log_view_to_tb(writer, global_step, src_imgs,
                                gt_imgs, output_dicts, len(args.val_tgt_views), prefix=f'val/')

                    torch.cuda.empty_cache()

                    print('Logging current training view...')
                    output_dicts = []
                    src_imgs = []
                    gt_imgs = []

                    for vis_data in train_vis_loader:
                        pairs = get_views(vis_data, args.train_src_views, args.train_tgt_views)
                        for idx, pair in enumerate(pairs):
                            tmp_ray_sampler = RaySamplerSingleImage(pair, device, render_stride=args.render_stride)
                            output_dict = render_image(args, model, tmp_ray_sampler, projector, args.render_stride)
                            src_img, gt_img = get_imgs_from_sampler(tmp_ray_sampler, args.render_stride)

                            output_dicts.append(output_dict)
                            src_imgs.append(src_img)
                            gt_imgs.append(gt_img)

                    log_view_to_tb(writer, global_step, src_imgs,
                                gt_imgs, output_dicts, len(args.train_tgt_views), prefix=f'train/')

                    torch.cuda.empty_cache()

                    model.switch_to_train()

            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
        epoch += 1

def render_image(args, model, ray_sampler, projector, render_stride=1):
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()

        featmaps = model.encode(ray_batch['src_rgbs']) # (batch, #views, #channels, height', width')

        ret = render_single_image(ray_sampler=ray_sampler,
                                  ray_batch=ray_batch,
                                  model=model,
                                  projector=projector,
                                  chunk_size=args.chunk_size,
                                  N_samples=args.N_samples,
                                  inv_uniform=args.inv_uniform,
                                  N_importance=args.N_importance,
                                  det=True,
                                  white_bkgd=args.white_bkgd,
                                  render_stride=render_stride,
                                  featmaps=featmaps)
    return ret

def get_imgs_from_sampler(ray_sampler, render_stride):
    H, W = ray_sampler.H, ray_sampler.W
    src_img = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))
    gt_img = ray_sampler.rgb.reshape(H, W, 3)
    
    if args.render_stride != 1:
        src_img = src_img[::render_stride, ::render_stride]
        gt_img = gt_img[::render_stride, ::render_stride]

    return src_img, gt_img

def get_rgb_grid(src_img, gt_img, ret):
    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(src_img)

    rgb_pred = img_HWC2CHW(ret['outputs_coarse']['rgb'].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3*w_max)
    rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
    rgb_im[:, :rgb_gt.shape[-2], w_max:w_max+rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], 2*w_max:2*w_max+rgb_pred.shape[-1]] = rgb_pred

    if ret["outputs_fine"] is not None:
        rgb_fine = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, :rgb_fine.shape[-2], :rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)

    # clamping RGB images
    rgb_im = torch.clamp(rgb_im, 0.0, 1.0)

    return rgb_im

def get_depth(ret):

    depth_im = ret['outputs_coarse']['depth'].detach().cpu()
    
    if ret['outputs_fine'] is None:
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
    else:
        depth_im = torch.cat((depth_im, ret['outputs_fine']['depth'].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))

    return depth_im

def get_acc(ret):
    
    acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()

    if ret['outputs_fine'] is None:
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
    else:
        acc_map = torch.cat((acc_map, torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()), dim=-1)
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

    return acc_map

def log_view_to_tb(writer, global_step, src_imgs, gt_imgs, output_dicts, n_views, prefix=''):

    rgb_im = []
    depth_im = []
    acc_map = []
    for src_img, gt_img, output_dict in zip(src_imgs, gt_imgs, output_dicts):
        rgb_im.append(get_rgb_grid(src_img, gt_img, output_dict))
        depth_im.append(get_depth(output_dict))
        acc_map.append(get_acc(output_dict))
    rgb_im = torch.stack(rgb_im, 0)
    depth_im = torch.stack(depth_im, 0)
    acc_map = torch.stack(acc_map, 0)

    rgb_im = make_grid(rgb_im, n_views)
    depth_im = make_grid(depth_im, n_views)
    acc_map = make_grid(acc_map, n_views)

    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)
    writer.add_image(prefix + 'depth_gt-coarse-fine', depth_im, global_step)
    writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)

    # write scalar
    n_total = len(src_imgs)
    n_objs = n_total // n_views

    psnr = []
    for i_obj in range(n_objs):
        for i_view in range(n_views):
            i = i_obj * n_views + i_view
            pred_rgb = output_dicts[i]['outputs_fine']['rgb'] if output_dicts[i]['outputs_fine'] is not None else output_dicts[i]['outputs_coarse']['rgb']
            psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_imgs[i])
            psnr.append(psnr_curr_img)
            writer.add_scalar(prefix + f'psnr_image_{i_obj}_{i_view}', psnr_curr_img, global_step)
    psnr_mean = np.mean(psnr)

    writer.add_scalar(prefix + f'psnr_image', psnr_mean, global_step)

if __name__ == '__main__':
    parser = opt.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)
