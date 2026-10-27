"""
Geo-Pillars BEV Normal Scatter Transforms
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Sequence, Union

from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS, MODELS
from mmdet3d.datasets.transforms import Pack3DDetInputs
from mmdet3d.datasets.transforms.loading import LoadPointsFromFile
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor


@TRANSFORMS.register_module()
class LoadPointsFromNpy(LoadPointsFromFile):
    """Drop-in replacement for `LoadPointsFromFile` that handles `.npy` files correctly."""

    def _load_points(self, pts_filename: str) -> np.ndarray:
        if pts_filename.endswith(".npy"):
            return np.load(pts_filename).astype(np.float32)
        return super()._load_points(pts_filename)


# 1.  Scatter per-point normals -> full-resolution BEV grid
@TRANSFORMS.register_module()
class ScatterNormalsToBEV(BaseTransform):
    """Scatter per-point surface normals onto a full-resolution BEV grid."""

    def __init__(
        self,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        normal_dims: Sequence[int] = (4, 5, 6),
        min_points_per_pillar: int = 3,
    ) -> None:
        super().__init__()
        self.vx = float(voxel_size[0])
        self.vy = float(voxel_size[1])
        self.x_min = float(point_cloud_range[0])
        self.y_min = float(point_cloud_range[1])
        self.x_max = float(point_cloud_range[3])
        self.y_max = float(point_cloud_range[4])
        self.W = int(round((self.x_max - self.x_min) / self.vx))
        self.H = int(round((self.y_max - self.y_min) / self.vy))
        self.normal_dims = tuple(normal_dims)
        self.min_points_per_pillar = min_points_per_pillar
        self._grid_cells = self.H * self.W

    # core
    def transform(self, results: Dict) -> Dict:
        pts = results["points"].tensor.numpy()  # (N, D)
        N, D = pts.shape

        max_dim_needed = max(self.normal_dims)

        if D <= max_dim_needed:
            results["gt_normals_bev"] = np.zeros((3, self.H, self.W), dtype=np.float32)
            results["occupancy_mask"] = np.zeros((1, self.H, self.W), dtype=np.float32)
            return results

        # Extract normal components
        nx = pts[:, self.normal_dims[0]].copy()
        ny = pts[:, self.normal_dims[1]].copy()
        nz = pts[:, self.normal_dims[2]].copy()

        # Augmentation consistency: rotation
        # GlobalRotScaleTrans stores the 3×3 rotation in results['pcd_rotation'].
        pcd_rotation = results.get("pcd_rotation")
        if pcd_rotation is not None:
            rot = np.asarray(pcd_rotation, dtype=np.float64)
            if rot.shape == (3, 3):
                normals_mat = np.stack([nx, ny, nz], axis=1)  # (N, 3)
                # mmdet3d applies:  points = points @ rot_mat_T
                normals_mat = (normals_mat @ rot).astype(np.float32)
                nx, ny, nz = normals_mat[:, 0], normals_mat[:, 1], normals_mat[:, 2]

        # Augmentation consistency: flips
        # RandomFlip3D negates Y for horizontal flip, X for vertical flip.
        if results.get("pcd_horizontal_flip", False):
            ny = -ny
        if results.get("pcd_vertical_flip", False):
            nx = -nx

        # Compute BEV grid indices
        x, y = pts[:, 0], pts[:, 1]
        col = np.floor((x - self.x_min) / self.vx).astype(np.intp)
        row = np.floor((y - self.y_min) / self.vy).astype(np.intp)

        valid = (col >= 0) & (col < self.W) & (row >= 0) & (row < self.H)
        if not valid.all():
            col, row = col[valid], row[valid]
            nx, ny, nz = nx[valid], ny[valid], nz[valid]

        # Bincount scatter (optimised)
        flat = row * self.W + col  # 1-D pillar index

        sum_nx = np.bincount(flat, weights=nx, minlength=self._grid_cells)
        sum_ny = np.bincount(flat, weights=ny, minlength=self._grid_cells)
        sum_nz = np.bincount(flat, weights=nz, minlength=self._grid_cells)
        count = np.bincount(flat, minlength=self._grid_cells)

        sum_nx = sum_nx[: self._grid_cells].reshape(self.H, self.W).astype(np.float32)
        sum_ny = sum_ny[: self._grid_cells].reshape(self.H, self.W).astype(np.float32)
        sum_nz = sum_nz[: self._grid_cells].reshape(self.H, self.W).astype(np.float32)
        count = count[: self._grid_cells].reshape(self.H, self.W).astype(np.float32)

        # Average per pillar + L2 renormalise
        occupied = count >= self.min_points_per_pillar
        inv_count = np.where(occupied, 1.0 / count.clip(min=1.0), 0.0).astype(
            np.float32
        )

        avg_nx = sum_nx * inv_count
        avg_ny = sum_ny * inv_count
        avg_nz = sum_nz * inv_count

        # L2 norm per cell
        norm = np.sqrt(avg_nx * avg_nx + avg_ny * avg_ny + avg_nz * avg_nz)
        norm = norm.clip(min=1e-6)

        gt_normals_bev = np.stack(
            [avg_nx / norm, avg_ny / norm, avg_nz / norm],
            axis=0,
        )  # (3, H, W), unit-length where occupied, 0 elsewhere

        occupancy_mask = occupied.astype(np.float32)[np.newaxis]  # (1, H, W)

        results["gt_normals_bev"] = gt_normals_bev
        results["occupancy_mask"] = occupancy_mask

        # Drop the normal columns to prevent data leakage into the voxel encoder
        keep_dims = [i for i in range(D) if i not in self.normal_dims]
        results["points"].tensor = results["points"].tensor[:, keep_dims]

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"grid=({self.H}, {self.W}), "
            f"voxel=({self.vx}, {self.vy}), "
            f"normal_dims={self.normal_dims})"
        )


# 2.  Extended packer - carries BEV normals through to the batch
@TRANSFORMS.register_module()
class PackGeoPillarsInputs(Pack3DDetInputs):
    """Drop-in replacement for `Pack3DDetInputs` that also packs BEV normals."""

    _EXTRA_INPUT_KEYS = ("gt_normals_bev", "occupancy_mask")

    def transform(self, results: Dict) -> Dict:
        # Stash our extra fields before super() drops them
        extras = {}
        for key in self._EXTRA_INPUT_KEYS:
            if key in results:
                extras[key] = results.pop(key)

        # Standard mmdet3d packing
        packed = super().transform(results)

        # Inject normal BEV tensors into inputs
        if "inputs" not in packed:
            packed["inputs"] = {}
        for key, val in extras.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            packed["inputs"][key] = val

        return packed


# 3.  Custom data preprocessor - passes through BEV normals
@MODELS.register_module()
class GeoPillarsPreprocessor(Det3DDataPreprocessor):
    """Extends `Det3DDataPreprocessor` to carry extra input tensors through."""

    _EXTRA_KEYS = ("gt_normals_bev", "occupancy_mask")

    def simple_process(self, data: dict, training: bool = False) -> dict:
        """Standard preprocessing + passthrough of NGFM supervision tensors."""
        result = super().simple_process(data, training)

        collated_inputs = data.get("inputs", {})
        batch_inputs = result["inputs"]

        for key in self._EXTRA_KEYS:
            if key in collated_inputs:
                val = collated_inputs[key]
                # pseudo_collate returns a list of per-sample tensors;
                # stack them into a batch tensor.
                if isinstance(val, (list, tuple)):
                    batch_inputs[key] = torch.stack(
                        [
                            v if isinstance(v, torch.Tensor) else torch.from_numpy(v)
                            for v in val
                        ]
                    )
                elif isinstance(val, torch.Tensor):
                    batch_inputs[key] = val

        return result
