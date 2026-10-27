"""
Normal-Guided Feature Modulation (NGFM)

Physics-guided meso-level geometry filter for BEV feature maps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


@MODELS.register_module()
class NormalGuidedFeatureModulation(BaseModule):
    """
    Physics-guided BEV feature modulator gated by predicted surface normals.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_dim: int = 64,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        # Branch 1a: Feature compression
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

        # Branch 1b: Raw 3-vector predictor (pre-normalisation).
        self.normal_proj = nn.Conv2d(hidden_dim, 3, kernel_size=1, bias=False)

        # Branch 1c: Spatial mask expansion  3 -> in_channels.
        self.mask_expand = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

        # Weight init
        self.apply(self._init_weights)

    # Forward
    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features:       (B, in_channels, H, W)

        Returns:
            modulated_features: (B, in_channels, H, W)
            pred_normals:       (B, 3, H, W)  unit vectors
        """
        # Normal-Guided Spatial Mask
        h = self.compress(features)  # (B, hidden, H, W)

        raw_normals = self.normal_proj(h)  # (B, 3, H, W)

        pred_normals = F.normalize(raw_normals, p=2, dim=1, eps=1e-6)  # (B, 3, H, W)

        spatial_mask = self.mask_expand(pred_normals)  # (B, in_ch, H, W) ∈ [0,1]

        modulated_features = features * (1.0 + spatial_mask)  # (B, in_ch, H, W)

        return modulated_features, pred_normals

    # Weight initialisation
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
