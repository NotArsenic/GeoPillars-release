import torch
from torch import nn
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


@MODELS.register_module()
class PillarFeatureBuilder(BaseModule):
    vx: torch.Tensor
    vy: torch.Tensor
    x_offset: torch.Tensor
    y_offset: torch.Tensor
    p_range: torch.Tensor

    def __init__(self, voxel_size, point_cloud_range, max_points, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        vx, vy, vz = voxel_size
        self.register_buffer("vx", torch.tensor(vx))
        self.register_buffer("vy", torch.tensor(vy))
        self.register_buffer("x_offset", torch.tensor(vx / 2 + point_cloud_range[0]))
        self.register_buffer("y_offset", torch.tensor(vy / 2 + point_cloud_range[1]))
        self.max_points = max_points

        # Pre-allocate range to avoid re-creation on every LiDAR scan
        self.register_buffer("p_range", torch.arange(max_points))

    def forward(self, voxels, coords, num_points):
        device = voxels.device
        M, P, _ = voxels.shape

        mask = (self.p_range[:P][None, :] < num_points[:, None]).unsqueeze(-1).float()

        # cluster
        points_sum = torch.sum(voxels[:, :, :3] * mask, dim=1, keepdim=True)
        points_mean = points_sum / torch.clamp(num_points.view(-1, 1, 1), min=1)
        f_cluster = voxels[:, :, :3] - points_mean

        # center
        x_center = coords[:, 3].float() * self.vx + self.x_offset
        y_center = coords[:, 2].float() * self.vy + self.y_offset

        f_center = torch.stack(
            [
                voxels[:, :, 0] - x_center.view(-1, 1),
                voxels[:, :, 1] - y_center.view(-1, 1),
            ],
            dim=-1,
        )

        # distance + azimuth
        distance = torch.norm(voxels[:, :, :3], dim=2, keepdim=True)
        azimuth = torch.atan2(voxels[:, :, 1], voxels[:, :, 0]).unsqueeze(-1)

        # density
        density = (num_points.float() / self.max_points).view(-1, 1, 1).expand(-1, P, 1)

        features = torch.cat(
            [voxels, f_cluster, f_center, distance, azimuth, density], dim=-1
        )

        return features * mask, mask
