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


import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def divide_safe(num, denom):
    eps = 1e-8
    tmp = denom + eps * torch.le(denom, 1e-20).to(torch.float)
    return num / tmp

def meshgrid_pinhole(h, w,
                    is_homogenous=True, device=None):
    '''Create a meshgrid for image coordinate
    Args:
        h: grid height
        w: grid width
        is_homogenous: return homogenous or not
    Returns:
        Image coordinate meshgrid [height, width, 2 (3 if homogenous)]
    '''
    xs = torch.linspace(0, w-1, steps=w, device=device)
    ys = torch.linspace(0, h-1, steps=h, device=device)
    new_y, new_x = torch.meshgrid(ys, xs)
    grid = (new_x, new_y)

    if is_homogenous:
        ones = torch.ones_like(new_x)
        grid = torch.stack(grid + (ones, ), 2)
    else:
        grid = torch.stack(grid, 2)
    return grid

def normalize(pixel_locations, h, w):
    resize_factor = torch.tensor([w-1., h-1.], device=pixel_locations.device).view([1, 1, 1, 1, 2])
    normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
    return normalized_pixel_locations


class Projector():
    def __init__(self, device):
        self.device = device

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.], device=pixel_locations.device).view([1, 1, 1, 1, 2])
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def normalize_pts(self, xyz, bbox3d):
        near = bbox3d[:, None, None, None, :, 1]
        far = bbox3d[:, None, None, None, :, 0]
        normalized_voxel_locations = 2 * (xyz - near) / (far-near) - 1.
        return normalized_voxel_locations

    def compute_projections(self, xyz, train_ints, train_exts):
        '''Project 3D points into cameras
        Args:
            xyz: [batch, N_rays, N_samples, 3]
            train_ints: intrinsics [batch, num_views, 4, 4] 
            train_exts: extrinsics [batch, num_views, 4, 4]
        Returns:
            Pixel locations [batch, #views, N_rays, N_samples, 2], xyz_c [batch, #views, N_rays*N_samples, 4]
        '''
        batch, N_rays, N_samples, _ = xyz.shape
        xyz = xyz.reshape(batch, -1, 3)  # [batch, n_points, 3]
        num_views = train_ints.shape[1]
        train_intrinsics = train_ints  # [batch, n_views, 4, 4]
        train_poses = train_exts  # [batch, n_views, 4, 4]
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [batch, n_points, 4]
        
        xyz_c = torch.inverse(train_poses) @ (xyz_h.permute([0, 2, 1])[:, None].repeat(1, num_views, 1, 1)) # camera_coodrinates
        projections = train_intrinsics @ xyz_c # [batch, n_views, 4, n_points]
        projections = projections.permute(0, 1, 3, 2)  # [batch, n_views, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [batch, n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        return pixel_locations.reshape((batch, num_views, N_rays, N_samples, 2)), \
               xyz_c.permute(0, 1, 3, 2).reshape((batch, num_views, N_rays, N_samples, 4))

    def compute_directions(self, dir, train_exts):
        '''Transform view directions from world to camera coordinates
        Args:
            dir: [batch, N_rays, N_samples, 3]
            train_exts: extrinsics [batch, num_views, 4, 4]
        Returns:
            Viewing direction in camera coordinates [batch, #views, N_rays*N_samples, 3]
        '''
        _, N_rays, N_samples, _ = dir.shape
        num_views = train_exts.shape[1]
        dir = repeat(dir, 'b nr ns c -> b nv c (nr ns)', nv=num_views)
        train_poses = train_exts[..., :3, :3]  # [batch, n_views, 4, 4]
        dir_c = torch.inverse(train_poses) @ (dir)
        dir_c = rearrange(dir_c, 'b nv c (nr ns) -> b nv nr ns c', nr=N_rays, ns=N_samples)
        return dir_c

    def compute_pixel(self,  xyz, train_imgs, train_ints, train_exts, featmaps):
        '''Original pixelNeRF projection (2D -> samples)
        Args:
            xyz: [batch, n_rays, n_samples, 3]
            train_imgs: [batch, n_views, h, w, 3]
            train_ints: [batch, n_views, 4, 4]
            train_exts: [batch, n_views, 4, 4]
            featmaps: [batch, n_views, c, h, w]
        Returns: rgb_feat_sampled: [batch, n_rays, n_samples, n_views, 3+n_feat],
                 xyz_c: [batch, n_views, n_rays, n_samples, 4]
        '''
        _, views, h, w = train_imgs.shape[:-1]

        train_imgs = train_imgs.permute(0, 1, 4, 2, 3)  # [batch, n_views, 3, h, w]
        train_imgs = train_imgs * 2. - 1. # normalization

        # compute the projection of the query points to each reference image
        pixel_locations, xyz_c = self.compute_projections(xyz, train_ints, train_exts)
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [batch, n_views, n_rays, n_samples, 2]
        N_rays, N_samples = normalized_pixel_locations.shape[2:4]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs.flatten(0, 1), normalized_pixel_locations.flatten(0, 1), align_corners=True)
        rgb_sampled = rearrange(rgbs_sampled, '(b v) c nr ns -> b nr ns v c', v=views)

        # deep feature sampling
        feat_sampled = F.grid_sample(featmaps.flatten(0, 1), normalized_pixel_locations.flatten(0, 1), align_corners=True)
        feat_sampled = rearrange(feat_sampled, '(b v) c nr ns -> b nr ns v c', v=views)
        rgb_feat_sampled = torch.cat([rgb_sampled, feat_sampled], dim=-1)   # [batch, n_rays, n_samples, n_views, c+3]

        return rgb_feat_sampled, xyz_c    
