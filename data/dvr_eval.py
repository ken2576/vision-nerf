import os
import glob

import imageio
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import  torchvision.transforms as T
from torch.utils.data import Dataset

def parse_pose(path, num_views):
    cameras = np.load(path)

    intrinsics = []
    c2w_mats = []

    for i in range(num_views):
        # ShapeNet
        wmat_inv_key = "world_mat_inv_" + str(i)
        wmat_key = "world_mat_" + str(i)
        kmat_key = "camera_mat_" + str(i)
        if wmat_inv_key in cameras:
            c2w_mat = cameras[wmat_inv_key]
        else:
            w2c_mat = cameras[wmat_key]
            if w2c_mat.shape[0] == 3:
                w2c_mat = np.vstack((w2c_mat, np.array([0, 0, 0, 1])))
            c2w_mat = np.linalg.inv(w2c_mat)

        intrinsics.append(cameras[kmat_key])
        c2w_mats.append(c2w_mat)

    intrinsics = np.stack(intrinsics, 0)
    c2w_mats = np.stack(c2w_mats, 0)

    return intrinsics, c2w_mats

class DVREvalDataset(Dataset):
    """
    Dataset from DVR (Niemeyer et al. 2020)
    Provides 3D-R2N2 and NMR renderings
    """
    def __init__(self, args, mode, 
                **kwargs):
        """
        Args:
            args.data_path: path to data directory
            args.img_hw: image size (resize if needed)
            mode: train | test | val mode
        """
        super().__init__()
        self.base_path = args.data_path
        self.dataset_name = os.path.basename(args.data_path)
        assert os.path.exists(self.base_path)

        cats = [x for x in glob.glob(os.path.join(args.data_path, "*")) if os.path.isdir(x)]

        list_prefix = "gen_"

        if mode == "train":
            file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        elif mode == "val":
            file_lists = [os.path.join(x, list_prefix + "val.lst") for x in cats]
        elif mode == "test":
            file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]

        print("Loading NMR dataset", self.base_path, "name:", self.dataset_name, "mode:", mode)
        self.mode = mode

        all_objs = []
        for file_list in file_lists:
            if not os.path.exists(file_list):
                continue
            base_dir = os.path.dirname(file_list)
            cat = os.path.basename(base_dir)
            with open(file_list, "r") as f:
                objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)

        self.all_objs = all_objs

        if args.debug:
            self.all_objs = self.all_objs[:1]
    
        if mode == "val" or mode == "test":
            self.all_objs = self.all_objs[:100] # HACK to avoid reading too much things

        self.intrinsics = []
        self.poses = []
        self.rgb_paths = []
        for _, path in tqdm.tqdm(self.all_objs):
            curr_paths = sorted(glob.glob(os.path.join(path, "image", "*")))
            self.rgb_paths.append(curr_paths)

            pose_path = os.path.join(path, 'cameras.npz')
            intrinsics, c2w_mats = parse_pose(pose_path, len(curr_paths))

            self.poses.append(c2w_mats)
            self.intrinsics.append(intrinsics)

        self.rgb_paths = np.array(self.rgb_paths)
        self.poses = np.stack(self.poses, 0)
        self.intrinsics = np.array(self.intrinsics)

        assert(len(self.rgb_paths) == len(self.poses))

        self.define_transforms()
        self.img_hw = args.img_hw

        self.num_views = args.num_source_views
        self.closest_n_views = args.closest_n_views

        # default near/far plane depth
        self.z_near = 1.2
        self.z_far = 4.0

    def __len__(self):
        return len(self.intrinsics)

    def define_transforms(self):
        self.img_transforms = T.Compose(
            [T.ToTensor(), T.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
        )
        self.mask_transforms = T.Compose(
            [T.ToTensor(), T.Normalize((0.0,), (1.0,))]
        )

    def __getitem__(self, index):
        rgb_paths = self.rgb_paths[index]
        c2w_mats = np.array(self.poses[index])
        intrinsics = np.array(self.intrinsics[index])

        rgbs = []
        masks = []
        bboxes = []

        # Read all RGB
        for i in range(len(rgb_paths)):
            img = imageio.imread(rgb_paths[i])[..., :3]
            mask = (img.sum(axis=-1) != 255*3)[..., None].astype(np.uint8) * 255
            rgb = self.img_transforms(img)
            mask = self.mask_transforms(mask)

            intrinsics[i, 0, 0] *= img.shape[1] / 2.0
            intrinsics[i, 1, 1] *= img.shape[0] / 2.0
            intrinsics[i, 0, 2] = img.shape[1] / 2.0
            intrinsics[i, 1, 2] = img.shape[0] / 2.0

            h, w = rgb.shape[-2:]
            if (h != self.img_hw[0]) or (w != self.img_hw[1]):
                scale = self.img_hw[-1] / w
                intrinsics[i, :2] *= scale

                rgb = F.interpolate(rgb[None, :], size=self.img_hw, mode="area")[0]
                mask = F.interpolate(mask[None, :], size=self.img_hw, mode="area")[0]

            rgbs.append(rgb)
            masks.append(mask)

            yy = torch.any(mask, axis=2)
            xx = torch.any(mask, axis=1)
            ynz = torch.nonzero(yy)[:, 1]
            xnz = torch.nonzero(xx)[:, 1]
            ymin, ymax = ynz[[0, -1]]
            xmin, xmax = xnz[[0, -1]]
            bbox = torch.FloatTensor([xmin, ymin, xmax, ymax])

            bboxes.append(bbox)

        depth_range = np.array([self.z_near, self.z_far])

        return {
            "rgb_path": rgb_paths[0],
            "img_id": index,
            "img_hw": self.img_hw,
            "bbox": torch.stack(bboxes, 0),
            "masks": torch.stack(masks).permute([0, 2, 3, 1]).float(),
            "rgbs": torch.stack(rgbs).permute([0, 2, 3, 1]).float(),
            "c2w_mats": torch.FloatTensor(c2w_mats),
            "intrinsics": torch.FloatTensor(intrinsics),
            "depth_range": torch.FloatTensor(depth_range)
        }
