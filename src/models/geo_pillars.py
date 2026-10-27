"""
Geo-Pillars Full Model
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

from src.models.components.modulation.normal_target import (
    NormalTargetPreparer,
    ngfm_cosine_loss,
)


@MODELS.register_module()
class GeoPillarsModel(BaseModule):
    """
    Full Geo-Pillars perception model.
    """

    def __init__(
        self,
        voxel_encoder: Optional[Dict] = None,  # PillarFeatureBuilder
        middle_encoder: Optional[Dict] = None,  # GeoPillarFeatureNet
        backbone: Optional[Dict] = None,  # RepLKBEVBackbone
        neck: Optional[Dict] = None,  # NormalGuidedFeatureModulation
        upsampler: Optional[Dict] = None,  # BEVUpsampler
        bbox_head: Optional[Dict] = None,  # mmdet3d CenterHead
        normal_target: Optional[Dict] = None,  # NormalTargetPreparer params
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if cfg is not None:
            self._init_from_cfg(cfg)
        else:

            self._init_from_kwargs(
                voxel_encoder,
                middle_encoder,
                backbone,
                neck,
                upsampler,
                bbox_head,
                normal_target,
            )

    def _init_from_kwargs(
        self,
        voxel_encoder,
        middle_encoder,
        backbone,
        neck,
        upsampler,
        bbox_head,
        normal_target,
    ) -> None:
        """Build sub-modules from per-component config dicts (mmdet3d path)."""
        self.pillar_builder = MODELS.build(voxel_encoder)
        self.pillar_feature_net = MODELS.build(middle_encoder)
        self.backbone = MODELS.build(backbone)
        self.ngfm = MODELS.build(neck)
        self.upsampler = MODELS.build(upsampler)
        self.detection_head = MODELS.build(bbox_head)

        if normal_target is not None:
            self.normal_target_preparer = NormalTargetPreparer(
                stride=normal_target["stride"],
                min_occupancy=normal_target["min_occupancy"],
            )
        else:
            self.normal_target_preparer = NormalTargetPreparer(
                stride=8, min_occupancy=0.1
            )

    def _init_from_cfg(self, cfg: Dict[str, Any]) -> None:
        """Build sub-modules from a nested YAML config dict (Lightning path)."""
        from src.models.components.lidar_processor import PillarFeatureBuilder
        from src.models.components.encoders.pillar_feature_net import (
            GeoPillarFeatureNet,
        )
        from src.models.components.backbones.replk_bev_backbone import RepLKBEVBackbone
        from src.models.components.modulation.NGFM import NormalGuidedFeatureModulation
        from src.models.components.upsample.upsampling import BEVUpsampler

        model_cfg = cfg["model"]
        data_cfg = cfg["data"]

        # Pillar Feature Builder
        self.pillar_builder = PillarFeatureBuilder(
            voxel_size=data_cfg["voxel_size"],
            point_cloud_range=data_cfg["point_cloud_range"],
            max_points=data_cfg["max_points_per_voxel"],
        )

        # Pillar Feature Net
        pfn_cfg = model_cfg["pillar_feature_net"]
        self.pillar_feature_net = GeoPillarFeatureNet(
            in_channels=pfn_cfg["in_channels"],
            out_channels=pfn_cfg["out_channels"],
            max_points=pfn_cfg["max_points"],
            grid_size_x=data_cfg["grid_size_x"],
            grid_size_y=data_cfg["grid_size_y"],
        )

        # RepLK-BEV Backbone
        bb_cfg = model_cfg["backbone"]
        self.backbone = RepLKBEVBackbone(
            in_channels=bb_cfg["in_channels"],
            stem_channels=bb_cfg["stem_channels"],
            stage_channels=bb_cfg["stage_channels"],
            num_blocks=bb_cfg["num_blocks"],
            kernel_sizes=bb_cfg["kernel_sizes"],
            small_kernels=bb_cfg["small_kernels"],
            expand=bb_cfg["expand"],
        )

        # NGFM
        ngfm_cfg = model_cfg["ngfm"]
        self.ngfm = NormalGuidedFeatureModulation(
            in_channels=ngfm_cfg["in_channels"],
            hidden_dim=ngfm_cfg["hidden_dim"],
        )

        # Normal Target Preparer (no learnable params
        nt_cfg = model_cfg["normal_target"]
        self.normal_target_preparer = NormalTargetPreparer(
            stride=nt_cfg["stride"],
            min_occupancy=nt_cfg["min_occupancy"],
        )

        # BEV Upsampler
        up_cfg = model_cfg["upsampler"]
        self.upsampler = BEVUpsampler(channels=up_cfg["channels"])

        # CenterPoint Stage-1 Head (mmdet3d CenterHead)
        head_cfg = model_cfg["detection_head"]
        raise NotImplementedError(
            "Legacy Lightning cfg path no longer supports the custom CenterPointHead. "
            "Use the mmdet3d registry path (bbox_head=dict(type='CenterHead', ...)) instead."
        )

    # forward
    def forward(
        self,
        voxels: Tensor,
        coords: Tensor,
        num_points: Tensor,
        gt_normals_bev: Optional[Tensor] = None,
        occupancy_mask: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Geo-Pillars forward pass (backbone + stage-1 CenterHead).
        """
        output: Dict[str, Any] = {}

        # Pillar encoding
        features_11d, _ = self.pillar_builder(voxels, coords, num_points)
        bev_map = self.pillar_feature_net(features_11d, num_points, coords)

        # Backbone
        backbone_out = self.backbone(bev_map)  # (B, C, H/s, W/s)

        with torch.no_grad():
            occ_from_bev = (
                bev_map.detach().abs().sum(dim=1, keepdim=True) > 0
            ).float()  # (B, 1, H, W)

        density_map = self.normal_target_preparer.compute_density(occ_from_bev)
        output["density_map"] = density_map

        if gt_normals_bev is not None:
            gt_occ = occupancy_mask if occupancy_mask is not None else occ_from_bev
            gt_normals_ds, valid_mask, _ = (
                self.normal_target_preparer.compute_gt_targets(gt_normals_bev, gt_occ)
            )
            output["gt_normals_ds"] = gt_normals_ds
            output["valid_mask"] = valid_mask

        # NGFM
        modulated, pred_normals = self.ngfm(backbone_out)
        output["pred_normals"] = pred_normals

        # Upsample to full BEV resolution
        upsampled = self.upsampler(modulated)  # (B, C, H, W)
        output["bev_feat_bchw"] = upsampled
        return output
