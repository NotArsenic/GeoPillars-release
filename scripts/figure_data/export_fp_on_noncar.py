"""How often do false positives land on unlabelled foreground?

``ObjectNameFilter(['car'])`` removes truck / bus / pedestrian *boxes* from the
detection target, but their points stay in the input cloud, and
``GeoPillarsMetric`` remaps every non-car pkl label to −1 and discards it
(``geo_pillars_metric.py:487-498``). A detection on a bus is therefore scored as
a false positive with no way to tell it apart from a hallucination.

This reproduces the metric's own greedy matching (by subclassing it, so the
TP/FP split is exact by construction), then tests every false positive at
BEV IoU 0.5 against the frame's non-car GT boxes read straight from the pkl.

Writes ``figures/data/fp_on_noncar.json``.
"""

import argparse
import json
import os.path as osp
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from score_predictions import load_dump  # noqa: E402

SCORE_BANDS = [(0.05, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]


def build_recording_metric(ann_file, iou_thresholds, match_thr):
    """GeoPillarsMetric that records the per-detection TP/FP decision."""
    from src.metrics.geo_pillars_metric import GeoPillarsMetric

    class Recording(GeoPillarsMetric):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.record_thr = match_thr
            self.decisions = None

        def _eval_class(self, predictions, gt_per_sample, iou_threshold):
            out = super()._eval_class(predictions, gt_per_sample, iou_threshold)
            # only the global (non-distance-filtered) pass at the match threshold,
            # identified by being the first call at that threshold
            if iou_threshold == self.record_thr and self.decisions is None:
                self.decisions = self._replay(predictions, gt_per_sample, iou_threshold)
            return out

        @staticmethod
        def _replay(predictions, gt_per_sample, iou_threshold):
            """Same loop as _eval_class, returning the decision per detection."""
            from src.metrics.geo_pillars_metric import compute_bev_iou_matrix

            preds = sorted(predictions, key=lambda x: x[2], reverse=True)
            gt_arrays = {s: np.stack(b, 0) for s, b in gt_per_sample.items() if b}
            matched = defaultdict(set)
            rec = []
            for sidx, box, score in preds:
                gt = gt_arrays.get(sidx)
                if gt is None:
                    rec.append((sidx, box, score, False, 0.0))
                    continue
                iou = compute_bev_iou_matrix(box[None, :], gt)[0]
                j = int(np.argmax(iou))
                best = float(iou[j])
                tp = best >= iou_threshold and j not in matched[sidx]
                if tp:
                    matched[sidx].add(j)
                rec.append((sidx, box, score, tp, best))
            return rec

    m = Recording(
        ann_file=ann_file,
        iou_thresholds=iou_thresholds,
        ap_method="r40",
        pr_save_dir=None,
        distance_ranges=[],
    )
    m.dataset_meta = dict(classes=["car"])
    return m


def noncar_gt_by_sample(ann_file):
    with open(ann_file, "rb") as f:
        data = pickle.load(f)
    cats = data["metainfo"]["categories"]
    inv = {v: k for k, v in cats.items()}
    car_id = cats["car"]
    out, counts = {}, Counter()
    for info in data["data_list"]:
        boxes, labels = [], []
        for inst in info.get("instances", []):
            lab = int(inst["bbox_label_3d"])
            if lab == car_id or not inst.get("bbox_3d_isvalid", True):
                continue
            boxes.append(np.asarray(inst["bbox_3d"][:7], dtype=np.float64))
            labels.append(inv.get(lab, str(lab)))
            counts[inv.get(lab, str(lab))] += 1
        if boxes:
            out[int(info["sample_idx"])] = (np.stack(boxes, 0), labels)
    return out, dict(counts)


def analyse(dump, ann_file, match_thr=0.5, overlap_thr=0.3, verbose=True):
    from src.metrics.geo_pillars_metric import compute_bev_iou_matrix

    results, meta, stats = load_dump(dump)
    stats.pop("lidar_path", None)
    metric = build_recording_metric(ann_file, [0.7, match_thr], match_thr)
    t0 = time.perf_counter()
    metrics = metric.compute_metrics(results)
    rec = metric.decisions
    assert rec is not None, "decision recording did not fire"

    noncar, noncar_counts = noncar_gt_by_sample(ann_file)
    fps = [(s, b, sc) for s, b, sc, tp, _ in rec if not tp]
    n_tp = len(rec) - len(fps)

    hit = Counter()
    hit_band = defaultdict(Counter)
    band_total = Counter()
    best_iou_all = []
    for sidx, box, score in fps:
        band = next(f"{lo}-{hi}" for lo, hi in SCORE_BANDS if lo <= score < hi)
        band_total[band] += 1
        g = noncar.get(sidx)
        if g is None:
            best_iou_all.append(0.0)
            continue
        boxes, labels = g
        # cheap centre-distance gate: IoU > 0 needs overlapping footprints
        d = np.hypot(boxes[:, 0] - box[0], boxes[:, 1] - box[1])
        reach = (np.hypot(boxes[:, 3], boxes[:, 4]) + np.hypot(box[3], box[4])) / 2
        cand = np.flatnonzero(d <= reach)
        if not len(cand):
            best_iou_all.append(0.0)
            continue
        iou = compute_bev_iou_matrix(
            box[None, :], boxes[cand], backend=metric.iou_backend
        )[0]
        j = int(np.argmax(iou))
        best = float(iou[j])
        best_iou_all.append(best)
        if best >= overlap_thr:
            hit[labels[cand[j]]] += 1
            hit_band[band][labels[cand[j]]] += 1

    best_iou_all = np.asarray(best_iou_all)
    n_fp = len(fps)
    n_hit = int(sum(hit.values()))
    out = dict(
        run=meta["letter"],
        run_dir=meta["run"],
        variant=meta["variant"],
        dump=osp.relpath(osp.abspath(dump), REPO_ROOT),
        prediction_stats=stats,
        settings=dict(
            match_iou_threshold=match_thr,
            noncar_overlap_iou_threshold=overlap_thr,
            iou_backend=metric.iou_backend_info,
            score_bands=[f"{lo}-{hi}" for lo, hi in SCORE_BANDS],
        ),
        ap_check={
            k: v
            for k, v in metrics.items()
            if k.startswith(("AP@", "mAP", "mBEVIoU", "mAOE", "iou_"))
        },
        detections=dict(
            total=len(rec),
            true_positives=n_tp,
            false_positives=n_fp,
            fp_fraction=round(n_fp / max(len(rec), 1), 6),
        ),
        noncar_gt=dict(boxes_by_class=noncar_counts, frames_with_noncar_gt=len(noncar)),
        fp_on_noncar=dict(
            count=n_hit,
            fraction_of_false_positives=round(n_hit / max(n_fp, 1), 6),
            percent_of_false_positives=round(100 * n_hit / max(n_fp, 1), 3),
            by_class=dict(hit),
            by_class_fraction_of_fp={
                k: round(v / max(n_fp, 1), 6) for k, v in hit.items()
            },
        ),
        fp_by_score_band=dict(
            totals=dict(band_total),
            on_noncar={b: int(sum(c.values())) for b, c in hit_band.items()},
            on_noncar_fraction={
                b: round(sum(hit_band[b].values()) / max(band_total[b], 1), 6)
                for b in band_total
            },
            on_noncar_by_class={b: dict(c) for b, c in hit_band.items()},
        ),
        fp_best_noncar_iou=dict(
            mean=round(float(best_iou_all.mean()), 6),
            fraction_above_0=round(float((best_iou_all > 0).mean()), 6),
            fraction_above_0p1=round(float((best_iou_all >= 0.1).mean()), 6),
            fraction_above_0p3=round(float((best_iou_all >= 0.3).mean()), 6),
            fraction_above_0p5=round(float((best_iou_all >= 0.5).mean()), 6),
            fraction_above_0p7=round(float((best_iou_all >= 0.7).mean()), 6),
        ),
        wall_time_s=round(time.perf_counter() - t0, 1),
    )
    if verbose:
        print(
            f'[fp] {meta["letter"]}: {n_fp} FP of {len(rec)} detections; '
            f"{n_hit} ({100 * n_hit / max(n_fp, 1):.2f}%) overlap a non-car GT "
            f"at IoU >= {overlap_thr}  {dict(hit)}",
            flush=True,
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--overlap-iou", type=float, default=0.3)
    ap.add_argument(
        "--out", default=osp.join(REPO_ROOT, "figures", "data", "fp_on_noncar.json")
    )
    args = ap.parse_args()

    _, meta, _ = load_dump(args.dumps[0])
    out = dict(
        description=(
            "Fraction of false positives (at BEV IoU 0.5 against car GT) that overlap "
            "a non-car GT box at BEV IoU >= 0.3. Non-car boxes come from the pkl "
            "instance list, i.e. before ObjectNameFilter removes them; the evaluator "
            "never sees them, so these detections are scored as hallucinations."
        ),
        ann_file=meta["ann_file"],
        runs={},
    )
    for d in args.dumps:
        r = analyse(d, meta["ann_file"], args.match_iou, args.overlap_iou)
        out["runs"][r["run"]] = r
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[fp] -> {args.out}")


if __name__ == "__main__":
    main()
