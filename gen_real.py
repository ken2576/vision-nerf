import os
import glob
import shutil
import configargparse
import tqdm
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T

from models.render_image import render_single_image
from models.model import VisionNerfModel
from models.sample_ray import RaySamplerSingleImage
from models.projection import Projector
from utils import img_HWC2CHW

def config_parser():
    parser = configargparse.ArgumentParser()
    # general
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--expname', type=str, help='experiment name')
    parser.add_argument('--ckptdir', type=str, help='checkpoint folder')
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument('--outdir', type=str, help='output video directory')
    parser.add_argument("--local_rank", type=int, default=0, help='rank for distributed training')

    ########## dataset options ##########
    ## render dataset
    parser.add_argument('--data_path', type=str, help='the dataset to train')
    parser.add_argument('--img_hw', type=int, nargs='+', help='image size for the input')
    parser.add_argument("--focal", type=float, default=131.25, help="Focal length")
    parser.add_argument("--radius", type=float, default=1.3, help="Camera distance")
    parser.add_argument('--data_index', type=int,
                        default=[],
                        nargs='+',
                        help='data index to select from the dataset')
    parser.add_argument("--z_near", type=float, default=0.8)
    parser.add_argument("--z_far", type=float, default=1.8)
    parser.add_argument("--fps", type=int, default=12, help="FPS of video")

    parser.add_argument('--no_reload', action='store_true',
                        help='do not reload weights from saved ckpt (not used)')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training (not used)')
    parser.add_argument('--num_frames', type=int, default=40, help='how frames to render')
    parser.add_argument("--elevation", type=float, default=0.0, help="elevation angle (negative is above)")

    ########## model options ##########
    ## ray sampling options
    parser.add_argument('--chunk_size', type=int, default=128,
                        help='number of rays processed in parallel, decrease if running out of memory')
    
    ## model options
    parser.add_argument('--im_feat_dim', type=int, default=128, help='image feature dimension')
    parser.add_argument('--mlp_feat_dim', type=int, default=512, help='mlp hidden dimension')
    parser.add_argument('--freq_num', type=int, default=10, help='how many frequency bases for positional encodings')
    parser.add_argument('--mlp_block_num', type=int, default=2, help='how many resnet blocks for coarse network')
    parser.add_argument('--coarse_only', action='store_true', help='use coarse network only')
    parser.add_argument("--anti_alias_pooling", type=int, default=1, help='if use anti-alias pooling')
    parser.add_argument('--num_source_views', type=int, default=1, help='number of views')
    parser.add_argument('--freeze_pos_embed', action='store_true', help='freeze positional embeddings')
    parser.add_argument('--no_skip_conv', action='store_true', help='disable skip convolution')

    ########### iterations & learning rate options (not used) ##########
    parser.add_argument('--lrate_feature', type=float, default=1e-3, help='learning rate for feature extractor')
    parser.add_argument('--lrate_mlp', type=float, default=5e-4, help='learning rate for mlp')
    parser.add_argument('--lrate_decay_factor', type=float, default=0.5,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument('--lrate_decay_steps', type=int, default=50000,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument('--warmup_steps', type=int, default=10000, help='num of iterations for warm-up')
    parser.add_argument('--scheduler', type=str, default='steplr', help='scheduler type to use [steplr]')
    parser.add_argument('--use_warmup', action='store_true', help='use warm-up scheduler')
    parser.add_argument('--bbox_steps', type=int, default=100000, help='iterations to use bbox sampling')


    ########## rendering options ##########
    parser.add_argument('--N_samples', type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument('--N_importance', type=int, default=128, help='number of important samples per ray')
    parser.add_argument('--inv_uniform', action='store_true',
                        help='if True, will uniformly sample inverse depths')
    parser.add_argument('--det', action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument('--white_bkgd', action='store_true',
                        help='apply the trick to avoid fitting to white background')

    return parser

def parse_intrinsic(focal, cx, cy):
    intrinsic = np.array([[focal, 0, cx, 0],
                          [0, focal, cy, 0],
                          [0,     0,  1, 0],
                          [0,     0,  0, 1]])
    return intrinsic

class RealRenderDataset(Dataset):
    """
    Dataset for rendering
    """
    def __init__(self, args, **kwargs):
        """
        Args:
            args.data_path: path to data directory
            args.img_hw: image size (resize if needed)
        """
        super().__init__()
        self.base_path = args.data_path

        print("Loading real dataset", self.base_path)
        assert os.path.exists(self.base_path)

        self.rgb_paths = sorted(glob.glob(os.path.join(self.base_path, "*_normalize.jpg"))) + \
            sorted(glob.glob(os.path.join(self.base_path, "*_normalize.png")))
        self.poses = []
        self.intrinsics = []
        for i in range(len(self.rgb_paths)):
            intrinsic = parse_intrinsic(args.focal, args.img_hw[0]//2, args.img_hw[1]//2)
            cam_pose = trans_t(args.radius)
            self.poses.append(cam_pose)
            self.intrinsics.append(intrinsic)

        self.rgb_paths = np.array(self.rgb_paths)
        self.poses = np.stack(self.poses, 0)
        self.intrinsics = np.array(self.intrinsics)

        self.define_transforms()
        self.img_hw = args.img_hw

        # default near/far plane depth
        self.z_near = args.z_near
        self.z_far = args.z_far

    def __len__(self):
        return len(self.rgb_paths)

    def define_transforms(self):
        self.img_transforms = T.Compose(
            [T.ToTensor(), T.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
        )
        self.mask_transforms = T.Compose(
            [T.ToTensor(), T.Normalize((0.0,), (1.0,))]
        )

    def __getitem__(self, index):
        # Read source RGB
        src_rgb_path = self.rgb_paths[index]
        src_c2w_mat = self.poses[index]
        src_intrinsics = self.intrinsics[index]

        img = imageio.imread(src_rgb_path)[..., :3]
        mask = (img.sum(axis=-1) != 255*3)[..., None].astype(np.uint8) * 255
        rgb = self.img_transforms(img)
        mask = self.mask_transforms(mask)

        h, w = rgb.shape[-2:]
        if (h != self.img_hw[0]) or (w != self.img_hw[1]):
            scale = self.img_hw[-1] / rgb.shape[-1]
            src_intrinsics[:, :2] *= scale

            rgb = F.interpolate(rgb, size=self.img_hw, mode="area")
            mask = F.interpolate(mask, size=self.img_hw, mode="area")
        
        depth_range = np.array([self.z_near, self.z_far])

        return {
            "rgb_path": src_rgb_path,
            "img_id": index,
            "img_hw": self.img_hw,
            "src_rgbs": rgb[None, ...].permute([0, 2, 3, 1]).float(),
            "src_masks": mask[None, ...].permute([0, 2, 3, 1]).float(),
            "src_c2w_mats": torch.FloatTensor(src_c2w_mat)[None, :],
            "src_intrinsics": torch.FloatTensor(src_intrinsics)[None, :],
            "depth_range": torch.FloatTensor(depth_range)
        }


def trans_t(t):
    return torch.tensor(
        [[-1, 0, 0, 0], [0, 0, -1, t], [0, -1, 0, 0], [0, 0, 0, 1],], dtype=torch.float32,
    )

def rot_theta(angle):
    return torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )

def rot_phi(phi):
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )

def pose_spherical(theta, phi, radius):
    """
    Spherical rendering poses, from NeRF
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w

    return c2w

def gen_video(args):

    device = "cuda"
    print(f"checkpoints reload from {args.ckptdir}")

    dataset = RealRenderDataset(args)

    # Create VisionNeRF model
    model = VisionNerfModel(args, False, False)
    # create projector
    projector = Projector(device=device)
    model.switch_to_eval()

    if not args.data_index:
        args.data_index = [x for x in range(len(dataset))]

    for d_idx in args.data_index:
        out_folder = os.path.join(args.outdir, args.expname, f'{d_idx:06d}')
        print(f'Rendering {dataset[d_idx]["rgb_path"][:-15]}')
        print(f'videos will be saved to {out_folder}')
        os.makedirs(out_folder, exist_ok=True)

        # save the args and config files
        f = os.path.join(out_folder, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))

        if args.config is not None:
            f = os.path.join(out_folder, 'config.txt')
            if not os.path.isfile(f):
                shutil.copy(args.config, f)

        sample = dataset[d_idx]
        pose_index = 0
        data_input = dict(
            rgb_path=sample['rgb_path'],
            img_id=sample['img_id'],
            img_hw=sample['img_hw'],
            tgt_intrinsic=sample['src_intrinsics'][0:1],
            src_masks=sample['src_masks'][pose_index][None, None, :],
            src_rgbs=sample['src_rgbs'][pose_index][None, None, :],
            src_c2w_mats=sample['src_c2w_mats'][pose_index][None, None, :],
            src_intrinsics=sample['src_intrinsics'][pose_index][None, None, :],
            depth_range=sample['depth_range'][None, :]
        )

        input_im = sample['src_rgbs'][pose_index].cpu().numpy()
        filename = os.path.join(out_folder, 'input.png')
        imageio.imwrite(filename, (input_im*255.).astype(np.uint8))

        radius = (dataset.z_near + dataset.z_far) * 0.5
        print("> Using default camera radius", radius)

        # Use 360 pose sequence from NeRF
        render_poses = torch.stack(
            [
                pose_spherical(angle, args.elevation, radius)
                for angle in np.linspace(-180, 180, args.num_frames)[::-1]
            ],
            0,
        )  # (NV, 4, 4)
        # +z is the vertical axis
        
        imgs = []
        with torch.no_grad():

            for idx, pose in enumerate(tqdm.tqdm(render_poses)):
                filename = os.path.join(out_folder, f'{idx:06}.png')
                data_input['tgt_c2w_mat'] = pose[None, :]

                # load training rays
                ray_sampler = RaySamplerSingleImage(data_input, device, render_stride=1)
                ray_batch = ray_sampler.get_all()
                featmaps = model.encode(ray_batch['src_rgbs'])

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
                                        render_stride=1,
                                        featmaps=featmaps)
                
                if ret['outputs_fine']:
                    rgb_im = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
                else:
                    rgb_im = img_HWC2CHW(ret['outputs_coarse']['rgb'].detach().cpu())
                # clamping RGB images
                rgb_im = torch.clamp(rgb_im, 0.0, 1.0).permute([1, 2, 0]).cpu().numpy()
                rgb_im = (rgb_im * 255.).astype(np.uint8)
                imageio.imwrite(filename, rgb_im)
                imgs.append(rgb_im)
                torch.cuda.empty_cache()

            imgs = np.stack(imgs, 0)
            imageio.mimsave(os.path.join(out_folder, f'output.gif'), imgs, fps=12)
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    gen_video(args)