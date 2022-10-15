import os
import glob

import imageio
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import  torchvision.transforms as T
from torch.utils.data import Dataset

def parse_intrinsic(path):
    with open(path, "r") as f:
        lines = f.readlines()
        focal, cx, cy, _ = map(float, lines[0].split())
    intrinsic = np.array([[focal, 0, cx, 0],
                          [0, focal, cy, 0],
                          [0,     0,  1, 0],
                          [0,     0,  0, 1]])
    return intrinsic

def parse_pose(path):
    return np.loadtxt(path, dtype=np.float32).reshape(4, 4)

class SRNEvalDataset(Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """
    def __init__(self, args, mode, **kwargs):
        """
        Args:
            args.data_path: path to data directory
            args.img_hw: image size (resize if needed)
            mode: train | test | val mode
        """
        super().__init__()
        self.base_path = args.data_path + "_" + mode
        self.dataset_name = os.path.basename(args.data_path)

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.mode = mode
        assert os.path.exists(self.base_path)

        is_chair = "chair" in self.dataset_name
        if is_chair and mode == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        intrinsic_paths = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )

        if args.debug:
            intrinsic_paths = intrinsic_paths[:10]

        self.intrinsics = []
        self.poses = []
        self.rgb_paths = []
        for path in tqdm.tqdm(intrinsic_paths):
            dir = os.path.dirname(path)
            curr_paths = sorted(glob.glob(os.path.join(dir, "rgb", "*")))
            self.rgb_paths.append(curr_paths)

            pose_paths = [f.replace('rgb', 'pose').replace('png', 'txt') for f in curr_paths]
            c2w_mats = [parse_pose(x) for x in 
                    pose_paths]
            self.poses.append(c2w_mats)

            self.intrinsics.append(parse_intrinsic(path))

        self.rgb_paths = np.array(self.rgb_paths)
        self.poses = np.stack(self.poses, 0)
        self.intrinsics = np.array(self.intrinsics)

        assert(len(self.rgb_paths) == len(self.poses))

        self.define_transforms()
        self.img_hw = args.img_hw

        self.num_views = args.num_source_views
        self.closest_n_views = args.closest_n_views

        # Default near/far plane depth
        if is_chair:
            self.z_near = 1.25
            self.z_far = 2.75
        else:
            self.z_near = 0.8
            self.z_far = 1.8

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
        intrinsics = np.array([self.intrinsics[index]] * len(rgb_paths))

        rgbs = []
        masks = []
        bboxes = []

        # Read all RGB
        for i in range(len(rgb_paths)):
            
            img = imageio.imread(rgb_paths[i])[..., :3]
            mask = (img.sum(axis=-1) != 255*3)[..., None].astype(np.uint8) * 255
            rgb = self.img_transforms(img)
            mask = self.mask_transforms(mask)

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
