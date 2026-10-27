"""
GeoPillar Encoder Adapters for CenterPoint
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

# Module-level BEV cache
_BEV_CACHE: Dict[int, Tensor] = {}


# Voxel encoder adapter
@MODELS.register_module()
class GeoPillarVoxelEncoder(BaseModule):
    """
    Runs the full GeoPillar encoding pipeline (feature building -> vector attention -> BEV scatter), caches the resulting BEV canvas, and returns a dummy per-voxel tensor to satisfy the MVX interface contract.
    """

    def __init__(
        self,
        builder_cfg: Dict,
        pfn_cfg: Dict,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.builder = MODELS.build(builder_cfg)
        self.pfn = MODELS.build(pfn_cfg)

    def forward(
        self,
        voxels: Tensor,
        num_points: Tensor,
        coors: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Run PillarFeatureBuilder -> GeoPillarFeatureNet.
        """
        # PillarFeatureBuilder: (M, P, C_raw) -> (M, P, 12) enriched features
        features_enriched, mask = self.builder(voxels, coors, num_points)

        # GeoPillarFeatureNet: (M, P, 12) -> (B, C_out, H, W) BEV canvas
        bev_canvas = self.pfn(features_enriched, num_points, coors)

        # Cache for the middle encoder
        _BEV_CACHE[id(self)] = bev_canvas

        return voxels.new_zeros(voxels.shape[0], self.pfn.out_channels)


# Middle encoder adapter (passthrough)
@MODELS.register_module()
class GeoPillarMiddleEncoder(BaseModule):
    """
    No-op middle encoder - retrieves the BEV canvas cached by
    `GeoPillarVoxelEncoder` and returns it directly, bypassing the
    standard `PointPillarsScatter` scatter step.
    """

    def __init__(
        self,
        in_channels: int = 64,
        output_shape: Optional[List[int]] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.output_shape = output_shape or [256, 256]

    def forward(
        self,
        voxel_features: Tensor,
        coors: Tensor,
        batch_size: int = None,
    ) -> Tensor:
        """
        Retrieve the cached BEV canvas from `GeoPillarVoxelEncoder`.

        """
        if not _BEV_CACHE:
            raise RuntimeError(
                "GeoPillarMiddleEncoder: BEV cache is empty. "
                "Ensure GeoPillarVoxelEncoder is used as pts_voxel_encoder."
            )

        # Pop the single cached entry
        key = next(iter(_BEV_CACHE))
        bev_canvas = _BEV_CACHE.pop(key)
        return bev_canvas
