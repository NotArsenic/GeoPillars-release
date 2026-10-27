"""
GeoPillars mmengine/mmdet3d Detector Wrapper
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmengine.structures import InstanceData


# Detector
@MODELS.register_module()
class GeoPillarsDetector(Base3DDetector):
    """
    mmengine `Base3DDetector` wrapper around `GeoPillarsModel`.

    """

    def __init__(
        self,
        model: Dict,
        data_preprocessor: Dict,
        train_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:

        if isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)

        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )

        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

        inner_cfg = dict(**model)
        inner_cfg.setdefault("train_cfg", train_cfg)
        inner_cfg.setdefault("test_cfg", test_cfg)
        if "bbox_head" in inner_cfg:
            inner_cfg["bbox_head"] = dict(**inner_cfg["bbox_head"])
            inner_cfg["bbox_head"].setdefault("train_cfg", (train_cfg or {}).get("pts"))
            inner_cfg["bbox_head"].setdefault("test_cfg", (test_cfg or {}).get("pts"))
        self.inner = MODELS.build(inner_cfg)

        # NGFM loss weight
        self.ngfm_loss_weight = float(self.train_cfg.get("ngfm_normal_weight", 0.1))

    # Base3DDetector abstract methods

    def extract_feat(self, batch_inputs_dict, batch_input_metas=None):
        """Not called directly; full pipeline runs inside loss() / predict()."""
        raise NotImplementedError("Use loss() or predict() - not extract_feat()")

    def _forward(self, batch_inputs_dict, batch_data_samples=None, **kwargs):
        """Tensor-mode forward (debugging). Returns the inner model output dict."""
        voxels, coords, num_points = self._unpack_voxels(batch_inputs_dict)
        return self.inner(voxels, coords, num_points)

    # Helpers
    @staticmethod
    def _unpack_voxels(batch_inputs_dict: Dict) -> Tuple[Tensor, Tensor, Tensor]:
        """Extract (voxels, coors, num_points) from the preprocessed batch."""
        v = batch_inputs_dict["voxels"]
        return v["voxels"], v["coors"], v["num_points"]

    # Training
    def loss(
        self,
        batch_inputs_dict: Dict[str, Any],
        batch_data_samples: List[Det3DDataSample],
    ) -> Dict[str, Tensor]:
        """
        Compute the Geo-Pillars training loss.
        """
        voxels, coords, num_points = self._unpack_voxels(batch_inputs_dict)
        device = voxels.device

        # Backbone + neck forward
        gt_normals_bev = batch_inputs_dict.get("gt_normals_bev")
        occupancy_mask = batch_inputs_dict.get("occupancy_mask")
        if gt_normals_bev is not None:
            gt_normals_bev = gt_normals_bev.to(device)
        if occupancy_mask is not None:
            occupancy_mask = occupancy_mask.to(device)

        output = self.inner(
            voxels,
            coords,
            num_points,
            gt_normals_bev=gt_normals_bev,
            occupancy_mask=occupancy_mask,
        )
        bev_feat_bchw = output["bev_feat_bchw"]  # (B, C, H, W)

        # Stage-1 losses via CenterHead.loss()
        losses = self.inner.detection_head.loss(
            [bev_feat_bchw],
            batch_data_samples,
        )

        # NGFM auxiliary loss
        ngfm_loss = torch.zeros(1, device=device).squeeze()
        if "gt_normals_ds" in output:
            from src.models.components.modulation.normal_target import ngfm_cosine_loss

            ngfm_loss = ngfm_cosine_loss(
                pred_normals=output["pred_normals"],
                gt_normals=output["gt_normals_ds"],
                valid_mask=output["valid_mask"],
                density_weight=output["density_map"],
            )

        losses["loss_ngfm"] = self.ngfm_loss_weight * ngfm_loss
        return losses

    # Inference
    def predict(
        self,
        batch_inputs_dict: Dict[str, Any],
        batch_data_samples: List[Det3DDataSample],
    ) -> List[Det3DDataSample]:
        """Run inference using CenterHead.predict() for stage-1 decoding."""
        voxels, coords, num_points = self._unpack_voxels(batch_inputs_dict)

        output = self.inner(voxels, coords, num_points)
        bev_feat_bchw = output["bev_feat_bchw"]

        # Stage-1 decode via CenterHead.predict()
        predictions = self.inner.detection_head.predict(
            [bev_feat_bchw],
            batch_data_samples,
        )  # list[InstanceData]

        # Package into Det3DDataSample
        results: List[Det3DDataSample] = []
        for sample, pred in zip(batch_data_samples, predictions):
            sample = sample.clone()
            sample.pred_instances_3d = pred
            sample.pred_instances = InstanceData()  # required by NuScenesMetric
            results.append(sample)

        return results
