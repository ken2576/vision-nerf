import os
import torch
import numpy as np
from network.resnet_mlp import PosEncodeResnet
from network.vit import VIT

def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


class VisionNerfModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device('cuda:{}'.format(args.local_rank))

        self.freq_factor = np.pi

        # create coarse network
        pos_c = 3
        in_c = args.im_feat_dim + 3 + 3
        # create coarse network
        self.net_coarse = PosEncodeResnet(args, pos_c, in_c, args.mlp_feat_dim,
                                          4, args.mlp_block_num).to(device)
        if args.coarse_only:
            self.net_fine = None
        else:
            # create fine network
            self.net_fine = PosEncodeResnet(args, pos_c, in_c, args.mlp_feat_dim,
                                            4, args.mlp_block_num).to(device)


        im_feat = args.im_feat_dim
        # create feature extraction network
        self.feature_net = VIT(im_feat,
                               train_pos_embed=not args.freeze_pos_embed,
                               use_skip_conv=not args.no_skip_conv).cuda()

        # optimizer and learning rate scheduler
        learnable_params = list(self.net_coarse.parameters())
        learnable_params += list(self.feature_net.parameters())
        if self.net_fine is not None:
            learnable_params += list(self.net_fine.parameters())

        params = [
                {'params': self.net_coarse.parameters()},
                {'params': self.feature_net.parameters(), 'lr': args.lrate_feature},
            ]

        if self.net_fine is not None:
            params.append({'params': self.net_fine.parameters()})

        self.optimizer = torch.optim.Adam(params, lr=args.lrate_mlp)

        if args.scheduler == 'steplr':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=args.lrate_decay_steps,
                                                            gamma=args.lrate_decay_factor)
        else:
            raise NotImplementedError

        if args.use_warmup:
            self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                        lr_lambda=lambda step: np.clip((step+1), 0, args.warmup_steps) / args.warmup_steps)
        else:
            self.warmup_scheduler = None

        out_folder = os.path.join(args.ckptdir, args.expname)
        self.start_step = self.load_from_ckpt(out_folder,
                                              load_opt=load_opt,
                                              load_scheduler=load_scheduler)

        if args.distributed:

            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank
                )

    def encode(self, x):
        """
        Args:
            x: input tensor [b, v, h, w, c]
        Returns:
            Extracted feature maps [b, out_c, h, w]
        """
        b, v, h, w, c = x.shape
        x = x*2. - 1. # Normalization for transformer
        feat_maps = self.feature_net(x.reshape([-1, h, w, c]).permute(0, 3, 1, 2))
        _, nc, nh, nw = feat_maps.shape
        feat_maps = feat_maps.reshape([b, v, nc, nh, nw])
        return feat_maps

    def posenc(self, x):
        freq_multiplier = (
            self.freq_factor * 2 ** torch.arange(
                                        self.args.freq_num,
                                        device=f"cuda:{self.args.local_rank}"
                                    )
        ).view(1, 1, 1, -1)
        x_expand = x.unsqueeze(-1)
        sin_val = torch.sin(x_expand * freq_multiplier)
        cos_val = torch.cos(x_expand * freq_multiplier)
        return torch.cat(
            [x_expand, sin_val, cos_val], -1
        ).view(x.shape[:2] + (-1,))

    def switch_to_eval(self):
        self.net_coarse.eval()
        if self.net_fine is not None:
            self.net_fine.eval()
        self.feature_net.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        if self.net_fine is not None:
            self.net_fine.train()
        self.feature_net.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'net_coarse': de_parallel(self.net_coarse).state_dict(),
                   'feature_net': de_parallel(self.feature_net).state_dict(),
                   }

        if self.net_fine is not None:
            to_save['net_fine'] = de_parallel(self.net_fine).state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.net_coarse.load_state_dict(to_load['net_coarse'])
        self.feature_net.load_state_dict(to_load['feature_net'])

        if self.net_fine is not None and 'net_fine' in to_load.keys():
            self.net_fine.load_state_dict(to_load['net_fine'])

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''Load model from existing checkpoints and return the current step
        
        Args:
            out_folder: the directory that stores ckpts
        Returns:
            The current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, training from scratch...')
            step = 0

        return step
