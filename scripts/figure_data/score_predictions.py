"""Score a prediction dump with one or more GeoPillarsMetric variants, on the CPU.

``GeoPillarsMetric.compute_metrics`` depends only on the per-sample records that
``process`` collects and on the annotation pkl, so a dump written by
``dump_predictions.py`` can be re-scored under any metric setting without
touching the GPU. Used to put the mBEVIoU / mAOE definitions and the
rotated-vs-axis-aligned IoU difference on the record from one inference pass.

Variants (``--variants``)
------------------------
``current``  the metric as it now stands: ``mBEVIoU@0.5``, ``mAOE`` +
             ``mAOE_folded``, rotated Shapely IoU.
``legacy``   ``mbeviou_mode='legacy_pooled'`` - reproduces the pooled
             ``mBEVIoU`` key of the pre-2026-09-20 evaluation JSONs.
``aabb``     ``iou_backend='aabb'`` - the Shapely-missing fallback path, to
             quantify what silently substituting axis-aligned IoU would cost.

Usage::

    PYTHONPATH=$(pwd) python scripts/figure_data/score_predictions.py \
        --dump work_dirs/preds/A_baseline.npz --out work_dirs/preds/A_baseline.json
"""

import argparse
import ast
import json
import math
import os
import os.path as osp
import pickle
import sys
import time

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

VARIANT_KWARGS = {
    "current": dict(mbeviou_mode="single_threshold", iou_backend="auto"),
    "legacy": dict(mbeviou_mode="legacy_pooled", iou_backend="auto"),
    "aabb": dict(mbeviou_mode="single_threshold", iou_backend="aabb"),
}


def load_dump(path):
    """Return (results in GeoPillarsMetric.process format, meta dict)."""
    z = np.load(path, allow_pickle=False)
    meta = ast.literal_eval(str(z["meta"].item()))
    n_det, off = z["n_det"], np.concatenate([[0], np.cumsum(z["n_det"])])
    boxes, scores, labels = z["boxes"], z["scores"], z["labels"]
    results = []
    for i, sidx in enumerate(z["sample_idx"]):
        a, b = off[i], off[i + 1]
        results.append(
            dict(
                sample_idx=int(sidx),
                pred_bboxes=boxes[a:b],
                pred_scores=scores[a:b],
                pred_labels=labels[a:b].astype(np.int64),
            )
        )
    assert off[-1] == len(scores) == len(boxes), (off[-1], len(scores), len(boxes))
    return (
        results,
        meta,
        dict(
            n_frames=len(results),
            n_det=int(off[-1]),
            n_det_per_frame_mean=float(n_det.mean()),
            n_det_per_frame_max=int(n_det.max()) if len(n_det) else 0,
            lidar_path=z["lidar_path"],
        ),
    )


def remap_to_pkl_sample_idx(results, lidar_path, ann_file):
    """Map subset-renumbered sample_idx back onto the annotation pkl's numbering.

    ``Det3DDataset`` renumbers ``sample_idx`` when ``dataset.indices`` selects a
    subset, so a dump made with ``--limit`` must be keyed by lidar file instead.
    Returns the number of records whose sample_idx was corrected.
    """
    with open(ann_file, "rb") as f:
        data_list = pickle.load(f)["data_list"]
    by_path = {i["lidar_points"]["lidar_path"]: i["sample_idx"] for i in data_list}
    fixed = 0
    for res, lp in zip(results, lidar_path):
        key = osp.basename(str(lp)) if str(lp) not in by_path else str(lp)
        if key not in by_path:
            # dumped paths are absolute; the pkl stores them relative to data_root
            cand = [p for p in by_path if str(lp).endswith(p)]
            if len(cand) != 1:
                raise KeyError(f"cannot map lidar_path {lp!r} onto the pkl")
            key = cand[0]
        want = by_path[key]
        if res["sample_idx"] != want:
            res["sample_idx"] = want
            fixed += 1
    return fixed


def restrict_ann_file(ann_file, sample_ids, out_path):
    """Write a copy of ``ann_file`` holding only ``sample_ids``.

    ``GeoPillarsMetric`` takes ``n_gt`` from the whole pkl, so scoring a subset
    dump against the full split makes recall unreachable and r40 AP collapse to
    ~0 (the same flaw affects ``eval_test_checkpoints.py --smoke``). Restricting
    the GT to the frames that were actually inferred is what makes a subset
    AP comparable to a full-split AP.
    """
    with open(ann_file, "rb") as f:
        data = pickle.load(f)
    want = set(int(s) for s in sample_ids)
    kept = [i for i in data["data_list"] if int(i["sample_idx"]) in want]
    if len(kept) != len(want):
        raise RuntimeError(f"{len(kept)} pkl entries for {len(want)} dumped frames")
    data["data_list"] = kept
    os.makedirs(osp.dirname(osp.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    return out_path, len(kept)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dump", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--variants", nargs="+", default=["current"], choices=sorted(VARIANT_KWARGS)
    )
    p.add_argument(
        "--ann-file",
        default=None,
        help="override the annotation pkl recorded in the dump",
    )
    p.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.7, 0.5])
    p.add_argument(
        "--first-n-frames",
        type=int,
        default=None,
        help="score only the first N frames of the dump (implies "
        "--restrict-gt-to-dump)",
    )
    p.add_argument(
        "--restrict-gt-to-dump",
        action="store_true",
        help="score against only the frames present in the dump "
        "(required for subset dumps, else recall cannot reach "
        "the r40 grid and AP collapses to ~0)",
    )
    p.add_argument("--classes", nargs="+", default=["car"])
    return p.parse_args()


def main():
    args = parse_args()
    from src.metrics.geo_pillars_metric import GeoPillarsMetric

    results, meta, stats = load_dump(args.dump)
    ann_file = args.ann_file or meta["ann_file"]
    lidar_path = stats.pop("lidar_path")
    if meta.get("limit"):
        fixed = remap_to_pkl_sample_idx(results, lidar_path, ann_file)
        print(
            f"[score] subset dump: remapped {fixed}/{len(results)} sample_idx "
            "onto the pkl numbering",
            flush=True,
        )

    if args.first_n_frames:
        results = results[: args.first_n_frames]
        stats["n_frames_scored"] = len(results)
        stats["n_det_scored"] = int(sum(len(r["pred_scores"]) for r in results))
        args.restrict_gt_to_dump = True
        print(
            f"[score] sliced to the first {len(results)} frames of the dump", flush=True
        )

    restricted = None
    if args.restrict_gt_to_dump:
        sub = osp.splitext(osp.abspath(args.out))[0] + "_gt_subset.pkl"
        ann_file, n_kept = restrict_ann_file(
            ann_file, [r["sample_idx"] for r in results], sub
        )
        restricted = dict(subset_pkl=ann_file, n_frames=n_kept)
        print(f"[score] GT restricted to the {n_kept} dumped frames", flush=True)

    out = dict(
        dump=osp.relpath(osp.abspath(args.dump), REPO_ROOT),
        gt_restricted_to_dump=restricted,
        dump_meta=meta,
        prediction_stats=stats,
        ann_file=ann_file,
        iou_thresholds=args.iou_thresholds,
        classes=args.classes,
        variants={},
    )
    for name in args.variants:
        kw = VARIANT_KWARGS[name]
        metric = GeoPillarsMetric(
            ann_file=ann_file,
            ap_method="r40",
            iou_thresholds=args.iou_thresholds,
            pr_save_dir=None,
            **kw,
        )
        metric.dataset_meta = dict(classes=list(args.classes))
        t0 = time.perf_counter()
        m = metric.compute_metrics(results)
        out["variants"][name] = dict(
            kwargs=kw,
            iou_backend=metric.iou_backend_info,
            mbeviou_mode=metric.mbeviou_mode,
            wall_time_s=round(time.perf_counter() - t0, 1),
            metrics={f"GeoPillars/{k}": v for k, v in sorted(m.items())},
        )
        print(
            f"[score] {name}: "
            + "  ".join(
                f"{k}={m[k]}"
                for k in sorted(m)
                if k.startswith(("AP@", "mAOE", "mBEVIoU", "iou_", "heading"))
                and "@0-" not in k
                and "@20-" not in k
                and "@40" not in k
            ),
            flush=True,
        )

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(json_safe(out), f, indent=2, allow_nan=False)
    print(f"[score] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
