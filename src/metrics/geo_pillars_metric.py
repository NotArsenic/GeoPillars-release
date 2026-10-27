"""
GeoPillars Custom 3D Detection Evaluator
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.registry import METRICS

try:
    from shapely.geometry import Polygon as ShapelyPolygon

    try:
        from shapely.validation import make_valid as _make_valid
    except ImportError:
        try:
            from shapely import make_valid as _make_valid  # shapely ≥ 2.0
        except ImportError:
            _make_valid = None  # shapely < 1.8
    import shapely as _shapely_mod

    _SHAPELY_VERSION = getattr(_shapely_mod, "__version__", "unknown")
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False
    _SHAPELY_VERSION = None


IOU_BACKEND_ROTATED = "shapely_rotated"
IOU_BACKEND_AABB = "aabb_axis_aligned"


# 0.  BEV IoU helpers
def _box_to_corners_2d(
    cx: float, cy: float, dx: float, dy: float, yaw: float
) -> np.ndarray:
    """Convert a single oriented BEV box -> 4 corner points (CCW)."""
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    hdx, hdy = dx / 2.0, dy / 2.0
    # corners in the box-local frame (CCW from bottom-left)
    local = np.array(
        [[-hdx, -hdy], [hdx, -hdy], [hdx, hdy], [-hdx, hdy]], dtype=np.float64
    )
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cy], dtype=np.float64)


def _make_polygon(box7: np.ndarray) -> "ShapelyPolygon":
    """Build a Shapely polygon from a 7-element box vector."""
    corners = _box_to_corners_2d(box7[0], box7[1], box7[3], box7[4], box7[6])
    poly = ShapelyPolygon(corners)
    if not poly.is_valid and _make_valid is not None:
        poly = _make_valid(poly)
    return poly


def resolve_iou_backend(requested: str = "auto") -> str:
    """Resolve the IoU backend name actually used for a `iou_backend` request."""
    if requested == "aabb":
        return IOU_BACKEND_AABB
    if requested == "shapely":
        if not _HAS_SHAPELY:
            raise ImportError(
                "iou_backend='shapely' requested but Shapely is not importable"
            )
        return IOU_BACKEND_ROTATED
    if requested != "auto":
        raise ValueError(
            f"iou_backend must be 'auto', 'shapely' or 'aabb', got {requested!r}"
        )
    return IOU_BACKEND_ROTATED if _HAS_SHAPELY else IOU_BACKEND_AABB


def compute_bev_iou_matrix(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, backend: Optional[str] = None
) -> np.ndarray:
    """Pairwise rotated BEV IoU between *pred* and *gt* boxes."""
    N, M = len(pred_boxes), len(gt_boxes)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float64)

    if backend is None:
        backend = resolve_iou_backend("auto")
    if backend == IOU_BACKEND_ROTATED:
        return _bev_iou_shapely(pred_boxes, gt_boxes, N, M)
    return _bev_iou_aabb(pred_boxes, gt_boxes, N, M)


def _bev_iou_shapely(pred_boxes, gt_boxes, N, M):
    iou = np.zeros((N, M), dtype=np.float64)

    pred_polys = [_make_polygon(b) for b in pred_boxes]
    gt_polys = [_make_polygon(b) for b in gt_boxes]

    for i, pp in enumerate(pred_polys):
        pa = pp.area
        if pa < 1e-8:
            continue
        for j, gp in enumerate(gt_polys):
            ga = gp.area
            if ga < 1e-8:
                continue
            if not pp.intersects(gp):
                continue
            inter = pp.intersection(gp).area
            union = pa + ga - inter
            if union > 1e-8:
                iou[i, j] = inter / union
    return iou


def _bev_iou_aabb(pred_boxes, gt_boxes, N, M):
    """Axis-aligned BEV IoU (rotation ignored) - used when Shapely is absent."""

    def _to_aabb(boxes):
        cx, cy = boxes[:, 0], boxes[:, 1]
        hdx, hdy = boxes[:, 3] / 2, boxes[:, 4] / 2
        return np.stack([cx - hdx, cy - hdy, cx + hdx, cy + hdy], axis=1)

    pa = _to_aabb(pred_boxes)
    ga = _to_aabb(gt_boxes)
    iou = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        x1 = np.maximum(pa[i, 0], ga[:, 0])
        y1 = np.maximum(pa[i, 1], ga[:, 1])
        x2 = np.minimum(pa[i, 2], ga[:, 2])
        y2 = np.minimum(pa[i, 3], ga[:, 3])
        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        a_p = (pa[i, 2] - pa[i, 0]) * (pa[i, 3] - pa[i, 1])
        a_g = (ga[:, 2] - ga[:, 0]) * (ga[:, 3] - ga[:, 1])
        union = a_p + a_g - inter
        valid = union > 1e-8
        iou[i, valid] = inter[valid] / union[valid]
    return iou


# 1.  Orientation error helper
def _angle_diff(yaw_pred: float, yaw_gt: float) -> float:
    """Compute the smallest unsigned yaw angle difference in [0, π].
    Args:
        yaw_pred: Predicted yaw angle in radians.
        yaw_gt:   Ground-truth yaw angle in radians.

    Returns:
        Angle difference in [0, π].
    """
    diff = abs(yaw_pred - yaw_gt) % (2.0 * np.pi)
    return float(min(diff, 2.0 * np.pi - diff))


def _fold_heading(diff: float) -> float:
    """Fold a `[0, π]` yaw error into `[0, π/2]` (heading-flip invariant)."""
    return float(min(diff, np.pi - diff))


# 2.  AP computation (with proper interpolation)
def compute_ap(precision: np.ndarray, recall: np.ndarray, method: str = "r40") -> float:
    """Compute Average Precision from cumulative precision / recall arrays."""
    if len(precision) == 0 or len(recall) == 0:
        return 0.0

    if method == "r40":

        ap = 0.0
        for t in np.linspace(1.0 / 40, 1.0, 40):
            mask = recall >= (t - 1e-9)
            ap += precision[mask].max() if mask.any() else 0.0
        return ap / 40.0

    elif method == "r11":

        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            mask = recall >= (t - 1e-9)
            ap += precision[mask].max() if mask.any() else 0.0
        return ap / 11.0

    elif method == "all_point":

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    raise ValueError(f"Unknown AP method: {method!r}  (use r40 / r11 / all_point)")


# 3.  PR curve plotting utility
def plot_pr_curves(
    pr_data_path: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Plot Precision-Recall curves from the JSON exported by the metric."""
    import matplotlib

    matplotlib.use("Agg" if save_path else "TkAgg")
    import matplotlib.pyplot as plt

    with open(pr_data_path, "r") as f:
        pr_data = json.load(f)

    thresholds = sorted({d["iou_threshold"] for d in pr_data})
    n_thr = len(thresholds)
    fig, axes = plt.subplots(1, n_thr, figsize=figsize, squeeze=False)

    colors = plt.cm.tab10.colors

    for ti, thr in enumerate(thresholds):
        ax = axes[0, ti]
        curves = [d for d in pr_data if d["iou_threshold"] == thr]
        curves.sort(key=lambda c: c["ap"], reverse=True)

        for ci, curve in enumerate(curves):
            color = colors[ci % len(colors)]
            rec, pre = np.array(curve["recall"]), np.array(curve["precision"])
            # plot the monotone envelope for a cleaner curve
            env_rec = np.concatenate(([0.0], rec))
            env_pre = np.concatenate(([1.0], pre))
            for i in range(len(env_pre) - 2, -1, -1):
                env_pre[i] = max(env_pre[i], env_pre[i + 1])
            ax.step(
                env_rec,
                env_pre,
                where="post",
                color=color,
                linewidth=1.5,
                label=f"{curve['class_name']} "
                f"(AP={curve['ap']:.3f}, "
                f"GT={curve['n_gt']}, Pred={curve['n_pred']})",
            )

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"PR Curve - BEV IoU ≥ {thr}", fontsize=13, fontweight="bold")
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Geo-Pillars Detection - Precision-Recall Curves",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"PR curves saved to {save_path}")
    else:
        plt.show()
    plt.close()


# 4.  Distance-range label helper
def _range_label(lo: float, hi: float) -> str:
    """Return a compact human-readable label for a distance range."""
    lo_str = f"{int(lo)}" if lo == int(lo) else f"{lo}"
    if hi == float("inf"):
        return f"{lo_str}m+"
    hi_str = f"{int(hi)}" if hi == int(hi) else f"{hi}"
    return f"{lo_str}-{hi_str}m"


# 5.  GeoPillarsMetric
@METRICS.register_module()
class GeoPillarsMetric(BaseMetric):
    """Custom 3D detection evaluator for Geo-Pillars."""

    default_prefix = "GeoPillars"

    def __init__(
        self,
        ann_file: str,
        iou_thresholds: Union[List[float], Tuple[float, ...]] = (0.7, 0.5),
        ap_method: str = "r40",
        pr_save_dir: Optional[str] = None,
        min_score: float = 0.0,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        compute_orientation: bool = True,
        distance_ranges: Optional[List[Tuple[float, float]]] = None,
        mbeviou_mode: str = "single_threshold",
        iou_backend: str = "auto",
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ann_file = ann_file
        self.iou_thresholds = sorted(iou_thresholds)
        self.ap_method = ap_method
        self.pr_save_dir = pr_save_dir
        self.min_score = min_score
        self.compute_orientation = compute_orientation

        if mbeviou_mode not in ("single_threshold", "legacy_pooled", "off"):
            raise ValueError(
                "mbeviou_mode must be 'single_threshold', 'legacy_pooled' or "
                f"'off', got {mbeviou_mode!r}"
            )
        self.mbeviou_mode = mbeviou_mode

        self.iou_backend_requested = iou_backend
        self.iou_backend = resolve_iou_backend(iou_backend)
        self.shapely_version = _SHAPELY_VERSION
        self.iou_backend_info = dict(
            requested=iou_backend,
            resolved=self.iou_backend,
            rotated=self.iou_backend == IOU_BACKEND_ROTATED,
            shapely_importable=_HAS_SHAPELY,
            shapely_version=_SHAPELY_VERSION,
        )

        # mAOE is computed at the first (lowest) IoU threshold
        self._aoe_iou_thr = self.iou_thresholds[0]
        # mBEVIoU (single_threshold mode) uses the same operating point
        self._mbeviou_thr = self.iou_thresholds[0]

        # Distance-stratified evaluation
        if distance_ranges is None:
            distance_ranges = [(0.0, 20.0), (20.0, 40.0), (40.0, float("inf"))]
        self.distance_ranges: List[Tuple[float, float]] = list(distance_ranges)

        assert ap_method in (
            "r40",
            "r11",
            "all_point",
        ), f"ap_method must be 'r40', 'r11', or 'all_point', got {ap_method!r}"

        if not _HAS_SHAPELY:
            print(
                "[GeoPillarsMetric] WARNING: Shapely not found - using "
                "axis-aligned BEV IoU (less accurate).  "
                "Install: pip install shapely"
            )
        elif self.iou_backend == IOU_BACKEND_AABB:
            print(
                "[GeoPillarsMetric] WARNING: axis-aligned BEV IoU forced via "
                "iou_backend='aabb' - rotation is ignored in every IoU."
            )

    # process(): called once per batch during validation / test
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Store per-sample predictions for later aggregation."""
        for data_sample in data_samples:
            pred_3d = data_sample["pred_instances_3d"]

            bboxes = pred_3d["bboxes_3d"]
            if hasattr(bboxes, "tensor"):  # LiDARInstance3DBoxes
                bboxes = bboxes.tensor
            bboxes_np = bboxes.detach().cpu().numpy()

            self.results.append(
                dict(
                    sample_idx=data_sample["sample_idx"],
                    pred_bboxes=bboxes_np,  # (N, 7+)
                    pred_scores=pred_3d["scores_3d"].detach().cpu().numpy(),  # (N,)
                    pred_labels=pred_3d["labels_3d"].detach().cpu().numpy(),  # (N,)
                )
            )

    # compute_metrics(): called once after all batches are processed
    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute mAP, BEV IoU, mAOE, distance-stratified mAP/mAOE.

        Returns:
            Dict mapping metric names to float values.
        """
        logger = MMLogger.get_current_instance()

        # Load GT annotations
        ann_data = load(self.ann_file)
        data_list = ann_data["data_list"]
        classes: List[str] = list(self.dataset_meta["classes"])
        num_classes = len(classes)

        pkl_categories = ann_data.get("metainfo", {}).get("categories", {})
        if pkl_categories:
            pkl_id_to_name = {int(v): k for k, v in pkl_categories.items()}
            label_remap: Dict[int, int] = {}
            for pkl_id, name in pkl_id_to_name.items():
                if name in classes:
                    label_remap[pkl_id] = classes.index(name)
                else:
                    label_remap[pkl_id] = -1
        else:
            # No categories dict - assume pkl labels already match config
            label_remap = {i: i for i in range(num_classes)}

        gt_by_class: Dict[int, Dict[int, List[np.ndarray]]] = {
            c: defaultdict(list) for c in range(num_classes)
        }
        for info in data_list:
            sample_idx = info["sample_idx"]
            for inst in info.get("instances", []):
                if not inst.get("bbox_3d_isvalid", True):
                    continue
                raw_label = int(inst["bbox_label_3d"])
                cls_id = label_remap.get(raw_label, -1)
                if 0 <= cls_id < num_classes:
                    gt_by_class[cls_id][sample_idx].append(
                        np.array(inst["bbox_3d"][:7], dtype=np.float64)
                    )

        pred_by_class: Dict[int, List[Tuple[int, np.ndarray, float]]] = defaultdict(
            list
        )

        for res in results:
            sidx = res["sample_idx"]
            for i in range(len(res["pred_scores"])):
                score = float(res["pred_scores"][i])
                if score < self.min_score:
                    continue
                cls_id = int(res["pred_labels"][i])
                if 0 <= cls_id < num_classes:
                    pred_by_class[cls_id].append(
                        (
                            sidx,
                            res["pred_bboxes"][i][:7].astype(np.float64),
                            score,
                        )
                    )

        metric_dict: Dict[str, float] = {}
        all_aps: List[float] = []

        tp_ious_by_thr: Dict[float, List[float]] = defaultdict(list)
        all_tp_angle_errors: List[float] = []  # for mAOE (first IoU thr only)
        per_cls_angle_errors: Dict[int, List[float]] = defaultdict(list)
        pr_curve_data: List[dict] = []

        for thr in self.iou_thresholds:
            class_aps = []
            for cls_id in range(num_classes):
                cls_name = classes[cls_id]
                n_gt = sum(len(v) for v in gt_by_class[cls_id].values())

                ap, tp_ious, tp_angle_errors, precision, recall = self._eval_class(
                    pred_by_class[cls_id],
                    gt_by_class[cls_id],
                    iou_threshold=thr,
                )

                key = f"{cls_name}_AP@{thr}"
                metric_dict[key] = round(ap, 4)
                class_aps.append(ap)
                all_aps.append(ap)
                tp_ious_by_thr[thr].extend(tp_ious)

                # Collect orientation errors at the designated IoU threshold
                if self.compute_orientation and thr == self._aoe_iou_thr:
                    all_tp_angle_errors.extend(tp_angle_errors)
                    per_cls_angle_errors[cls_id].extend(tp_angle_errors)

                # Store raw PR data for export
                if len(precision) > 0:
                    pr_curve_data.append(
                        dict(
                            class_name=cls_name,
                            class_id=cls_id,
                            iou_threshold=thr,
                            ap=round(ap, 4),
                            precision=precision.tolist(),
                            recall=recall.tolist(),
                            n_gt=n_gt,
                            n_pred=len(pred_by_class[cls_id]),
                        )
                    )

            mean_ap_thr = float(np.mean(class_aps)) if class_aps else 0.0
            metric_dict[f"AP@{thr}"] = round(mean_ap_thr, 4)

        # Aggregate headline numbers
        mAP = float(np.mean(all_aps)) if all_aps else 0.0
        metric_dict["mAP"] = round(mAP, 4)

        if self.mbeviou_mode == "legacy_pooled":
            pooled = [v for thr in self.iou_thresholds for v in tp_ious_by_thr[thr]]
            mBEVIoU = float(np.mean(pooled)) if pooled else 0.0
            metric_dict["mBEVIoU"] = round(mBEVIoU, 4)
        elif self.mbeviou_mode == "single_threshold":
            single = tp_ious_by_thr[self._mbeviou_thr]
            mBEVIoU = float(np.mean(single)) if single else 0.0
            metric_dict[f"mBEVIoU@{self._mbeviou_thr}"] = round(mBEVIoU, 4)
        else:
            mBEVIoU = float("nan")

        # mAOE and per-class AOE
        cls_aoe_vals: List[float] = []
        cls_aoe_folded_vals: List[float] = []
        if self.compute_orientation:
            for cls_id in range(num_classes):
                cls_name = classes[cls_id]
                errs = per_cls_angle_errors[cls_id]
                cls_aoe = float(np.mean(errs)) if errs else float("nan")
                metric_dict[f"{cls_name}_AOE"] = (
                    round(cls_aoe, 4) if not np.isnan(cls_aoe) else float("nan")
                )
                if not np.isnan(cls_aoe):
                    cls_aoe_vals.append(cls_aoe)

                errs_f = [_fold_heading(e) for e in errs]
                cls_aoe_f = float(np.mean(errs_f)) if errs_f else float("nan")
                metric_dict[f"{cls_name}_AOE_folded"] = (
                    round(cls_aoe_f, 4) if not np.isnan(cls_aoe_f) else float("nan")
                )
                if not np.isnan(cls_aoe_f):
                    cls_aoe_folded_vals.append(cls_aoe_f)

            mAOE = float(np.mean(cls_aoe_vals)) if cls_aoe_vals else float("nan")
            metric_dict["mAOE"] = round(mAOE, 4) if not np.isnan(mAOE) else float("nan")
            mAOE_folded = (
                float(np.mean(cls_aoe_folded_vals))
                if cls_aoe_folded_vals
                else float("nan")
            )
            metric_dict["mAOE_folded"] = (
                round(mAOE_folded, 4) if not np.isnan(mAOE_folded) else float("nan")
            )

            if all_tp_angle_errors:
                flipped = float(
                    np.mean(
                        [1.0 if e > np.pi / 2 else 0.0 for e in all_tp_angle_errors]
                    )
                )
                metric_dict["heading_flip_rate"] = round(flipped, 4)
        else:
            mAOE = float("nan")
            mAOE_folded = float("nan")

        # Distance-stratified evaluation
        dist_results: List[dict] = []  # structured data for summary table
        for lo, hi in self.distance_ranges:
            lbl = _range_label(lo, hi)
            range_aps: List[float] = []
            range_angle_errors: List[float] = []
            range_tp_ious: List[float] = []

            for thr in self.iou_thresholds:
                range_class_aps = []
                for cls_id in range(num_classes):
                    cls_name = classes[cls_id]

                    # Filter GT by distance
                    gt_range = self._filter_gt_by_distance(gt_by_class[cls_id], lo, hi)
                    # Filter predictions by distance
                    pred_range = self._filter_pred_by_distance(
                        pred_by_class[cls_id], lo, hi
                    )

                    n_gt_range = sum(len(v) for v in gt_range.values())

                    ap_r, tp_ious_r, tp_aoe_r, prec_r, rec_r = self._eval_class(
                        pred_range, gt_range, iou_threshold=thr
                    )

                    rkey = f"{cls_name}_AP@{thr}@{lbl}"
                    metric_dict[rkey] = round(ap_r, 4)
                    range_class_aps.append(ap_r)

                    # Collect orientation errors for the range at AOE threshold
                    if self.compute_orientation and thr == self._aoe_iou_thr:
                        range_angle_errors.extend(tp_aoe_r)
                    if thr == self._mbeviou_thr:
                        range_tp_ious.extend(tp_ious_r)

                    if len(prec_r) > 0:
                        pr_curve_data.append(
                            dict(
                                class_name=cls_name,
                                class_id=cls_id,
                                iou_threshold=thr,
                                ap=round(ap_r, 4),
                                precision=prec_r.tolist(),
                                recall=rec_r.tolist(),
                                n_gt=n_gt_range,
                                n_pred=len(pred_range),
                                distance_range=[lo, hi if hi != float("inf") else None],
                            )
                        )

                range_aps.extend(range_class_aps)
                mean_ap_range_thr = (
                    float(np.mean(range_class_aps)) if range_class_aps else 0.0
                )
                metric_dict[f"AP@{thr}@{lbl}"] = round(mean_ap_range_thr, 4)

            mAP_range = float(np.mean(range_aps)) if range_aps else 0.0
            metric_dict[f"mAP@{lbl}"] = round(mAP_range, 4)

            if self.mbeviou_mode == "single_threshold":
                miou_range = float(np.mean(range_tp_ious)) if range_tp_ious else 0.0
                metric_dict[f"mBEVIoU@{self._mbeviou_thr}@{lbl}"] = round(miou_range, 4)

            mAOE_range = float("nan")
            if self.compute_orientation:
                mAOE_range = (
                    float(np.mean(range_angle_errors))
                    if range_angle_errors
                    else float("nan")
                )
                metric_dict[f"mAOE@{lbl}"] = (
                    round(mAOE_range, 4) if not np.isnan(mAOE_range) else float("nan")
                )
                range_errors_folded = [_fold_heading(e) for e in range_angle_errors]
                mAOE_range_f = (
                    float(np.mean(range_errors_folded))
                    if range_errors_folded
                    else float("nan")
                )
                metric_dict[f"mAOE_folded@{lbl}"] = (
                    round(mAOE_range_f, 4)
                    if not np.isnan(mAOE_range_f)
                    else float("nan")
                )

            dist_results.append(
                dict(
                    label=lbl,
                    lo=lo,
                    hi=hi,
                    mAP=mAP_range,
                    mAOE=mAOE_range,
                    per_thr={
                        thr: metric_dict[f"AP@{thr}@{lbl}"]
                        for thr in self.iou_thresholds
                    },
                )
            )

        metric_dict["iou_rotated"] = (
            1.0 if self.iou_backend == IOU_BACKEND_ROTATED else 0.0
        )

        # Save PR curves to disk
        if self.pr_save_dir and pr_curve_data:
            os.makedirs(self.pr_save_dir, exist_ok=True)
            pr_path = os.path.join(self.pr_save_dir, "pr_curves.json")
            with open(pr_path, "w") as f:
                json.dump(pr_curve_data, f, indent=2)
            logger.info(f"PR curve data saved -> {pr_path}")
            logger.info(f'  Plot: plot_pr_curves("{pr_path}", "pr_curves.pdf")')

        self._log_summary(
            logger,
            classes,
            mAP,
            mBEVIoU,
            mAOE,
            gt_by_class,
            num_classes,
            metric_dict,
            dist_results,
        )

        return metric_dict

    # Distance filter helpers
    @staticmethod
    def _bev_distance(box7: np.ndarray) -> float:
        """BEV distance from sensor origin: sqrt(x² + y²)."""
        return float(np.sqrt(box7[0] ** 2 + box7[1] ** 2))

    def _filter_gt_by_distance(
        self,
        gt_per_sample: Dict[int, List[np.ndarray]],
        lo: float,
        hi: float,
    ) -> Dict[int, List[np.ndarray]]:
        """Return a copy of *gt_per_sample* keeping only boxes in [lo, hi)."""
        filtered: Dict[int, List[np.ndarray]] = defaultdict(list)
        for sidx, boxes in gt_per_sample.items():
            for box in boxes:
                d = self._bev_distance(box)
                if lo <= d < hi:
                    filtered[sidx].append(box)
        return filtered

    def _filter_pred_by_distance(
        self,
        predictions: List[Tuple[int, np.ndarray, float]],
        lo: float,
        hi: float,
    ) -> List[Tuple[int, np.ndarray, float]]:
        """Return predictions whose box centre falls within [lo, hi) metres."""
        return [
            (sidx, box, score)
            for sidx, box, score in predictions
            if lo <= self._bev_distance(box) < hi
        ]

    # per-class evaluation
    def _eval_class(
        self,
        predictions: List[Tuple[int, np.ndarray, float]],
        gt_per_sample: Dict[int, List[np.ndarray]],
        iou_threshold: float,
    ) -> Tuple[float, List[float], List[float], np.ndarray, np.ndarray]:
        """Evaluate AP (and orientation errors) for one class at one IoU threshold."""
        n_gt = sum(len(bboxes) for bboxes in gt_per_sample.values())

        if n_gt == 0 or len(predictions) == 0:
            return 0.0, [], [], np.array([]), np.array([])

        # sort all predictions by score (descending)
        predictions = sorted(predictions, key=lambda x: x[2], reverse=True)

        tp = np.zeros(len(predictions), dtype=np.float64)
        fp = np.zeros(len(predictions), dtype=np.float64)
        tp_ious: List[float] = []
        tp_angle_errors: List[float] = []

        # per-sample set of already-matched GT indices
        matched: Dict[int, set] = defaultdict(set)

        # pre-build GT arrays per sample (avoid repeated stacking)
        gt_arrays: Dict[int, np.ndarray] = {}
        for sidx, boxes in gt_per_sample.items():
            if boxes:
                gt_arrays[sidx] = np.stack(boxes, axis=0)

        for det_i, (sidx, pred_box, score) in enumerate(predictions):
            gt_arr = gt_arrays.get(sidx)

            if gt_arr is None:
                fp[det_i] = 1.0
                continue

            # IoU between this single prediction and all GT in the sample
            iou_vec = compute_bev_iou_matrix(
                pred_box[np.newaxis, :], gt_arr, backend=self.iou_backend
            )[
                0
            ]  # (M,)

            best_j = int(np.argmax(iou_vec))
            best_iou = iou_vec[best_j]

            if best_iou >= iou_threshold and best_j not in matched[sidx]:
                tp[det_i] = 1.0
                matched[sidx].add(best_j)
                tp_ious.append(float(best_iou))
                # Orientation error for this TP match
                tp_angle_errors.append(
                    _angle_diff(float(pred_box[6]), float(gt_arr[best_j, 6]))
                )
            else:
                fp[det_i] = 1.0

        # cumulative precision / recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / float(n_gt)
        precision = cum_tp / (cum_tp + cum_fp)

        ap = compute_ap(precision, recall, method=self.ap_method)

        return ap, tp_ious, tp_angle_errors, precision, recall

    def _log_summary(
        self,
        logger,
        classes: List[str],
        mAP: float,
        mBEVIoU: float,
        mAOE: float,
        gt_by_class: Dict[int, Dict[int, List]],
        num_classes: int,
        metric_dict: Dict[str, float],
        dist_results: List[dict],
    ) -> None:
        """Log the formatted evaluation summary table."""
        sep = "=" * 72
        thin = "-" * 72

        logger.info(f"\n{sep}")
        logger.info(
            f"  GeoPillars Evaluation   " f"(AP interpolation: {self.ap_method})"
        )
        logger.info(f"{sep}")
        logger.info(f"  mAP      : {mAP:.4f}")
        logger.info(
            f"  BEV IoU  : {self.iou_backend}"
            + (f" (shapely {self.shapely_version})" if self.shapely_version else "")
        )
        if not np.isnan(mBEVIoU):
            label = (
                "mBEVIoU (pooled)"
                if self.mbeviou_mode == "legacy_pooled"
                else f"mBEVIoU@{self._mbeviou_thr}"
            )
            logger.info(f"  {label:<9}: {mBEVIoU:.4f}")
        if self.compute_orientation and not np.isnan(mAOE):
            logger.info(
                f"  mAOE     : {mAOE:.4f} rad  " f"(IoU thr={self._aoe_iou_thr})"
            )
            folded = metric_dict.get("mAOE_folded", float("nan"))
            if isinstance(folded, float) and not np.isnan(folded):
                logger.info(
                    f"  mAOE_fold: {folded:.4f} rad  "
                    f"(heading-flip folded into [0, π/2]; "
                    f"flip rate "
                    f'{metric_dict.get("heading_flip_rate", float("nan")):.4f})'
                )
        for thr in self.iou_thresholds:
            logger.info(f'  AP@{thr:<5} : {metric_dict[f"AP@{thr}"]:.4f}')
        logger.info(thin)

        # Per-class table header
        aoe_col = self.compute_orientation
        header = f'  {"Class":<20s}'
        for thr in self.iou_thresholds:
            header += f"  AP@{thr:<6}"
        if aoe_col:
            header += "     AOE(rad)"
        header += "  #GT"
        logger.info(header)
        logger.info(thin)

        for cls_id in range(num_classes):
            cls_name = classes[cls_id]
            n_gt = sum(len(v) for v in gt_by_class[cls_id].values())
            row = f"  {cls_name:<20s}"
            for thr in self.iou_thresholds:
                val = metric_dict.get(f"{cls_name}_AP@{thr}", 0.0)
                row += f"  {val:<10.4f}"
            if aoe_col:
                aoe_val = metric_dict.get(f"{cls_name}_AOE", float("nan"))
                if isinstance(aoe_val, float) and np.isnan(aoe_val):
                    row += f'       {"N/A":>8}'
                else:
                    row += f"  {aoe_val:>10.4f}"
            row += f"  {n_gt:>5d}"
            logger.info(row)

        # Distance-stratified sub-table
        if dist_results:
            logger.info(thin)
            logger.info("  Distance-Stratified Results")
            logger.info(thin)
            dist_header = f'  {"Range":<12s}'
            for thr in self.iou_thresholds:
                dist_header += f"  AP@{thr:<6}"
            dist_header += f'  {"mAP":<10}'
            if self.compute_orientation:
                dist_header += f'  {"mAOE(rad)":<12}'
            logger.info(dist_header)
            logger.info("  " + "-" * 68)

            for dr in dist_results:
                row = f'  {dr["label"]:<12s}'
                for thr in self.iou_thresholds:
                    row += f'  {dr["per_thr"][thr]:<10.4f}'
                row += f'  {dr["mAP"]:<10.4f}'
                if self.compute_orientation:
                    if np.isnan(dr["mAOE"]):
                        row += f'  {"N/A":<12}'
                    else:
                        row += f'  {dr["mAOE"]:<12.4f}'
                logger.info(row)

        logger.info(f"{sep}\n")
