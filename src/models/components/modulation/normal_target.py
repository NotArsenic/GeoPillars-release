"""
Ground-Truth Normal Preparer & Cosine Similarity Loss

Non-learned geometric utilities for preparing GT surface normals as
supervision targets for NGFM, and the auxiliary loss that trains it.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# GT Normal Downsampler  (no learnable parameters)
class NormalTargetPreparer(nn.Module):
    """
    Geometry-preserving downsampler for ground-truth BEV surface normals.

    Takes full-resolution GT normals + occupancy and produces targets at
    the backbone's spatial bottleneck resolution (H/stride × W/stride).
    """

    def __init__(
        self,
        stride: int = 8,
        min_occupancy: float = 0.0,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.min_occupancy = min_occupancy

    @torch.no_grad()
    def compute_density(
        self,
        occupancy_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the NGFM density gate signal from pillar occupancy.
        """
        density_map = F.avg_pool2d(
            occupancy_mask.float(), kernel_size=self.stride, stride=self.stride
        )  # (B, 1, H/s, W/s)  ∈ [0, 1]
        return density_map

    @torch.no_grad()
    def compute_gt_targets(
        self,
        gt_normals_bev: torch.Tensor,
        occupancy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute GT supervision targets for the NGFM cosine-similarity loss.
        """
        s = self.stride
        patch_area = float(s * s)

        # Pool occupancy
        occupancy_pooled = F.avg_pool2d(
            occupancy_mask.float(), kernel_size=s, stride=s
        )  # (B, 1, H/s, W/s)  ∈ [0, 1]

        density_map = occupancy_pooled
        valid_count = occupancy_pooled * patch_area  # (B, 1, H/s, W/s)

        #  Occupancy-weighted average of normals
        normals_pooled = F.avg_pool2d(
            gt_normals_bev, kernel_size=s, stride=s
        )  # (B, 3, H/s, W/s)
        gt_normals_ds = normals_pooled / occupancy_pooled.clamp(min=1e-6)

        #  Re-normalise to unit length
        gt_normals_ds = F.normalize(gt_normals_ds, p=2, dim=1, eps=1e-6)

        # Valid mask
        valid_mask = valid_count > (self.min_occupancy * patch_area)

        return gt_normals_ds, valid_mask, density_map

    @torch.no_grad()
    def forward(
        self,
        gt_normals_bev: torch.Tensor,
        occupancy_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward-compatible wrapper - delegates to ``compute_gt_targets``."""
        return self.compute_gt_targets(gt_normals_bev, occupancy_mask)

    def extra_repr(self) -> str:
        return f"stride={self.stride}, min_occupancy={self.min_occupancy}"


# Auxiliary Cosine Similarity Loss
def ngfm_cosine_loss(
    pred_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    valid_mask: torch.Tensor,
    density_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Masked cosine similarity loss between predicted and GT surface normals.

    """
    # Cosine similarity per cell: (B, H, W)
    cos_sim = F.cosine_similarity(pred_normals, gt_normals, dim=1, eps=1e-6)

    per_cell_loss = 1.0 - cos_sim  # (B, H, W)

    # Squeeze valid_mask to (B, H, W) for element-wise multiply
    mask = valid_mask.squeeze(1).float()  # (B, H, W)

    if density_weight is not None:
        # Weight by occupancy density: denser patches get more influence
        weight = mask * density_weight.squeeze(1)  # (B, H, W)
    else:
        weight = mask

    # Masked mean - only valid cells contribute
    weighted_loss = (per_cell_loss * weight).sum()
    num_valid = weight.sum().clamp(min=1.0)

    return weighted_loss / num_valid
