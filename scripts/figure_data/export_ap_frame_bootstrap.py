#!/usr/bin/env python3
"""How much of an AP difference is finite-validation-set noise? (Section VI variance.)

There are no seed repeats, so *training* variance cannot be measured from this tree.
What can be measured exactly is the other half of the question: how much AP@0.7 moves
when the 30,842-scan validation split is resampled. This bounds the evaluation noise
a reported difference has to clear before training variance is even the question.

Method. ``GeoPillarsMetric._eval_class`` picks, for every detection, the highest-IoU
GT box *in its own frame*; the IoU threshold only decides whether that match is
accepted. So the expensive part - one rotated-IoU argmax per detection - is computed
once per run and cached as ``(sample_idx, score, best_gt, best_iou)``. Greedy matching,
the PR curve and R40 AP are then exact for any subset of frames, and a frame-level
bootstrap costs nothing extra. Frames are the resampling unit because detections
within a frame are not independent.

Paired differences (A-G for NGFM, A-B and D-E for the intensity column) are
bootstrapped on the *same* resampled frames, which cancels the common part and is the
interval the ablation claims actually need.

Writes ``figures/data/ap_frame_bootstrap.json`` (+ a per-run ``.npz`` cache under
``work_dirs/preds/``).

Usage (repo root)::

    PYTHONPATH=$(pwd) python scripts/figure_data/export_ap_frame_bootstrap.py
"""

import argparse
import json
import os.path as osp
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from _carla_infer import CARLA_ROOT, CARLA_TEST_PKL  # noqa: E402
from score_predictions import load_dump  # noqa: E402

PAIRS = [
    ("A", "G", "NGFM (A) vs no NGFM (G)"),
    ("A", "B", "intensity column, GeoPillars"),
    ("D", "E", "intensity column, CenterPoint"),
]


def gt_by_sample(ann_file, classes=("car",)):
    with open(ann_file, "rb") as f:
        data = pickle.load(f)
    cats = data["metainfo"]["categories"]
    keep = {cats[c] for c in classes}
    out = {}
    for info in data["data_list"]:
        boxes = [
            np.asarray(i["bbox_3d"][:7], dtype=np.float64)
            for i in info.get("instances", [])
            if int(i["bbox_label_3d"]) in keep and i.get("bbox_3d_isvalid", True)
        ]
        out[int(info["sample_idx"])] = np.stack(boxes, 0) if boxes else np.zeros((0, 7))
    return out


def build_cache(dump, gt, cache_path):
    """One rotated-IoU argmax per detection, in the metric's global score order."""
    from src.metrics.geo_pillars_metric import (
        compute_bev_iou_matrix,
        resolve_iou_backend,
    )

    backend = resolve_iou_backend("auto")
    results, meta, stats = load_dump(dump)
    preds = []
    for r in results:
        s = int(r["sample_idx"])
        for box, score in zip(r["pred_bboxes"], r["pred_scores"]):
            preds.append((s, box, float(score)))
    preds = sorted(preds, key=lambda x: x[2], reverse=True)  # same order as _eval_class
    n = len(preds)
    sidx = np.empty(n, np.int64)
    score = np.empty(n, np.float64)
    bestj = np.full(n, -1, np.int64)
    bestiou = np.zeros(n, np.float64)
    t0 = time.perf_counter()
    for i, (s, box, sc) in enumerate(preds):
        sidx[i], score[i] = s, sc
        g = gt.get(s)
        if g is None or len(g) == 0:
            continue
        iou = compute_bev_iou_matrix(box[None, :], g, backend=backend)[0]
        j = int(np.argmax(iou))
        bestj[i], bestiou[i] = j, float(iou[j])
        if (i + 1) % 200000 == 0:
            print(f"    {i + 1}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)
    np.savez_compressed(
        cache_path,
        sample_idx=sidx,
        score=score,
        best_gt=bestj,
        best_iou=bestiou,
        backend=np.array(str(backend)),
    )
    print(
        f"  cached {n} detections in {time.perf_counter() - t0:.0f}s -> {cache_path}",
        flush=True,
    )
    return dict(sample_idx=sidx, score=score, best_gt=bestj, best_iou=bestiou)


def tp_flags(cache, thr):
    """Greedy TP/FP decision per detection, computed once.

    The decision for a detection depends only on the higher-scoring detections in
    its *own* frame (matching is per-frame, the score order is global but the
    ``matched`` set is per-sample), so it is invariant to which other frames are in
    the evaluation set. That is what makes a frame-level bootstrap exact without
    re-matching.
    """
    s, bj, bi = cache["sample_idx"], cache["best_gt"], cache["best_iou"]
    matched = defaultdict(set)
    tp = np.zeros(len(s), np.float64)
    ok = (bj >= 0) & (bi >= thr)
    for i in np.nonzero(ok)[0]:
        key = s[i]
        if bj[i] not in matched[key]:
            matched[key].add(bj[i])
            tp[i] = 1.0
    return tp


def ap_weighted(tp, sidx, n_gt_per_frame, frame_weight=None):
    """R40 AP over frames weighted by multiplicity (1 = the full split)."""
    from src.metrics.geo_pillars_metric import compute_ap as _ap

    if frame_weight is None:
        w = np.ones(len(tp), np.float64)
        n_gt = float(n_gt_per_frame.sum())
    else:
        w = frame_weight[sidx]
        keep = w > 0
        tp, w = tp[keep], w[keep]
        n_gt = float((n_gt_per_frame * frame_weight).sum())
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    cum_tp = np.cumsum(tp * w)
    cum_fp = np.cumsum((1.0 - tp) * w)
    return float(_ap(cum_tp / (cum_tp + cum_fp), cum_tp / n_gt, method="r40"))


def main():
    ap_ = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap_.add_argument("--runs", nargs="*", default=["A", "B", "D", "E", "G"])
    ap_.add_argument("--n_boot", type=int, default=500)
    ap_.add_argument("--thresholds", type=float, nargs="+", default=[0.7, 0.5])
    ap_.add_argument("--seed", type=int, default=0)
    args = ap_.parse_args()

    ann = osp.join(CARLA_ROOT, CARLA_TEST_PKL)
    gt = gt_by_sample(ann)
    frames = np.array(sorted(gt))
    n_frames = len(frames)
    max_idx = int(frames.max()) + 1
    n_gt_per_frame = np.zeros(max_idx, np.float64)
    for s, g in gt.items():
        n_gt_per_frame[s] = len(g)
    print(f"{n_frames} frames, {int(n_gt_per_frame.sum())} car GT boxes", flush=True)

    caches = {}
    for L in args.runs:
        cp = osp.join(REPO_ROOT, "work_dirs", "preds", f"{L}_baseline_match.npz")
        if osp.exists(cp):
            z = np.load(cp)
            caches[L] = {
                k: z[k] for k in ("sample_idx", "score", "best_gt", "best_iou")
            }
            print(f'{L}: cache hit ({len(caches[L]["score"])} detections)', flush=True)
        else:
            print(f"{L}: matching...", flush=True)
            caches[L] = build_cache(
                osp.join(REPO_ROOT, "work_dirs", "preds", f"{L}_baseline.npz"), gt, cp
            )

    rng = np.random.default_rng(args.seed)
    boot_frames = [
        rng.choice(frames, size=n_frames, replace=True) for _ in range(args.n_boot)
    ]

    out = dict(
        description="Frame-level bootstrap of AP@0.7 (and AP@0.5) on the CARLA validation split, "
        "from cached per-detection rotated-IoU matches. Bounds the evaluation noise "
        "of an AP difference; it does NOT bound training / seed variance, which this "
        "tree cannot measure (no seed repeats).",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/figure_data/export_ap_frame_bootstrap.py",
        settings=dict(
            ann_file=ann,
            n_frames=n_frames,
            n_gt=int(n_gt_per_frame.sum()),
            n_bootstrap=args.n_boot,
            seed=args.seed,
            ap_method="r40",
            resampling_unit="frame (with replacement)",
        ),
        runs={},
        pairs={},
    )
    weights = [
        np.bincount(fr, minlength=max_idx).astype(np.float64) for fr in boot_frames
    ]
    per_run_boot = {}
    for L in args.runs:
        out["runs"][L] = {}
        sidx = caches[L]["sample_idx"]
        tps = {}
        for thr in args.thresholds:
            tps[thr] = tp_flags(caches[L], thr)
            full = ap_weighted(tps[thr], sidx, n_gt_per_frame)
            out["runs"][L][f"AP@{thr:g}_full_split"] = round(full, 5)
            out["runs"][L][f"TP@{thr:g}"] = int(tps[thr].sum())
            out["runs"][L][f"FP@{thr:g}"] = int(len(tps[thr]) - tps[thr].sum())
            print(
                f"{L} AP@{thr:g} full split = {full:.5f} "
                f"(TP {int(tps[thr].sum())}, FP {int(len(tps[thr]) - tps[thr].sum())})",
                flush=True,
            )
        thr = args.thresholds[0]
        out["runs"][L]["n_detections"] = int(len(sidx))
        t0 = time.perf_counter()
        vals = np.asarray(
            [ap_weighted(tps[thr], sidx, n_gt_per_frame, w) for w in weights]
        )
        per_run_boot[L] = vals
        out["runs"][L][f"AP@{thr:g}_bootstrap"] = dict(
            mean=round(float(vals.mean()), 5),
            std=round(float(vals.std(ddof=1)), 5),
            ci95=[
                round(float(np.percentile(vals, 2.5)), 5),
                round(float(np.percentile(vals, 97.5)), 5),
            ],
            width_pp=round(
                float(np.percentile(vals, 97.5) - np.percentile(vals, 2.5)) * 100, 3
            ),
        )
        print(
            f"{L} bootstrap AP@{thr:g}: std {vals.std(ddof=1) * 100:.4f} pp "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    thr = args.thresholds[0]
    for a, b, label in PAIRS:
        if a not in per_run_boot or b not in per_run_boot:
            continue
        d = (per_run_boot[a] - per_run_boot[b]) * 100
        obs = (
            out["runs"][a][f"AP@{thr:g}_full_split"]
            - out["runs"][b][f"AP@{thr:g}_full_split"]
        ) * 100
        out["pairs"][f"{a}-{b}"] = dict(
            label=label,
            observed_delta_pp=round(obs, 3),
            paired_bootstrap=dict(
                mean_pp=round(float(d.mean()), 3),
                std_pp=round(float(d.std(ddof=1)), 3),
                ci95_pp=[
                    round(float(np.percentile(d, 2.5)), 3),
                    round(float(np.percentile(d, 97.5)), 3),
                ],
                frac_same_sign=round(float((np.sign(d) == np.sign(obs)).mean()), 4),
            ),
        )
        print(
            f"{a}-{b}: {obs:+.3f} pp, paired 95% CI "
            f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] pp",
            flush=True,
        )

    path = osp.join(REPO_ROOT, "figures", "data", "ap_frame_bootstrap.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
