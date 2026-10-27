"""Build ``eval_results_carla_test_v2.json`` from the per-run prediction dumps.

The original ``eval_results_carla_test.json`` records what the *pre-2026-09-20*
metric produced and is referenced by SHA-256 from ``paper_metrics.json``, so it
is left untouched. This writes a second aggregate in the same schema, carrying
the corrected metrics (``mBEVIoU@0.5`` instead of the threshold-pooled
``mBEVIoU``, ``mAOE_folded`` and ``heading_flip_rate`` alongside ``mAOE``) plus
the ``iou_rotated`` field that records which IoU implementation ran.

Inputs are the JSONs written by ``score_predictions.py`` over the full-split
dumps from ``dump_predictions.py``.

Usage (repo root)::

    PYTHONPATH=$(pwd) python scripts/figure_data/build_carla_eval_aggregate.py \
        --scored work_dirs/preds/*_baseline.json
"""

import argparse
import glob
import json
import os.path as osp
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.join(REPO_ROOT, "tools"))

from eval_test_checkpoints import (
    RESULTS_DIR,
    RUNS,
    _git,
    _sha256,  # noqa: E402
    _write_json,
    checkpoint_path,
    find_config,
    training_log_val_metrics,
)

LETTERS = {
    "geo_pillars_h100_intensity": "A",
    "geo_pillars_h100_nointensity_final": "B",
    "centerpoint_h100_intensity_original": "C",
    "centerpoint_h100_intensity": "D",
    "centerpoint_h100_no_intensity": "E",
    "centerpoint_osf1_ablation": "F",
    "centerpoint_geopillar_encoder_replk_distributed": "G",
}
HEADLINE = [
    "GeoPillars/AP@0.7",
    "GeoPillars/AP@0.5",
    "GeoPillars/mAP",
    "GeoPillars/mBEVIoU@0.5",
    "GeoPillars/mAOE",
    "GeoPillars/mAOE_folded",
    "GeoPillars/heading_flip_rate",
    "GeoPillars/AP@0.7@0-20m",
    "GeoPillars/AP@0.7@20-40m",
    "GeoPillars/AP@0.7@40m+",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scored",
        nargs="+",
        required=True,
        help="score_predictions.py JSONs, one per run",
    )
    ap.add_argument(
        "--variant",
        default="current",
        help="which metric variant in each scored JSON to publish",
    )
    ap.add_argument("--epoch", type=int, default=8)
    ap.add_argument(
        "--out", default=osp.join(RESULTS_DIR, "eval_results_carla_test_v2.json")
    )
    args = ap.parse_args()

    metric_file = osp.join(REPO_ROOT, "src", "metrics", "geo_pillars_metric.py")
    import torch
    import platform

    env = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        git_commit=_git("rev-parse", "HEAD"),
        git_describe=_git("log", "-1", "--format=%h %s"),
        git_dirty=bool(_git("status", "--porcelain")),
        metric_file="src/metrics/geo_pillars_metric.py",
        metric_file_sha256=_sha256(metric_file),
        metric_file_last_commit=_git(
            "log", "-1", "--format=%H", "--", "src/metrics/geo_pillars_metric.py"
        ),
    )

    runs = {}
    for path in sorted(args.scored):
        rec = json.load(open(path))
        v = rec["variants"][args.variant]
        run = rec["dump_meta"]["run"]
        letter = LETTERS[run]
        ckpt = checkpoint_path(run, args.epoch)
        tl = training_log_val_metrics(run, args.epoch)
        m = v["metrics"]
        delta = None
        if tl:
            delta = {
                k: round(100 * (m[k] - tl["metrics"][k]), 4)
                for k in tl["metrics"]
                if k in m
            }
        runs[run] = dict(
            paper_role=RUNS[run]["paper_role"],
            setup=RUNS[run]["setup"],
            letter=letter,
            epochs={
                str(args.epoch): dict(
                    run=run,
                    letter=letter,
                    epoch=args.epoch,
                    checkpoint=osp.relpath(ckpt, REPO_ROOT),
                    checkpoint_sha256=_sha256(ckpt),
                    config=osp.relpath(find_config(run), REPO_ROOT),
                    data=dict(
                        dataset="carla_test",
                        data_root=rec["dump_meta"]["data_root"],
                        ann_file=osp.basename(rec["ann_file"]),
                        evaluator_ann_file=rec["ann_file"],
                        num_samples_evaluated=rec["prediction_stats"]["n_frames"],
                        use_dim=rec["dump_meta"]["use_dim"],
                        test_max_voxels=rec["dump_meta"]["test_max_voxels"],
                        max_num_points=rec["dump_meta"]["max_num_points"],
                        voxel_size=rec["dump_meta"]["voxel_size"],
                    ),
                    metric_config=dict(
                        type="GeoPillarsMetric",
                        ap_method="r40",
                        iou_thresholds=rec["iou_thresholds"],
                        aoe_iou_threshold=min(rec["iou_thresholds"]),
                        mbeviou_mode=v["mbeviou_mode"],
                        mbeviou_iou_threshold=min(rec["iou_thresholds"]),
                        iou_backend=v["iou_backend"],
                        note="AP in [0,1]; AOE in radians; values rounded to 4 dp",
                    ),
                    metrics=dict(sorted(m.items())),
                    predictions=dict(
                        dump=rec["dump"],
                        dump_sha256=_sha256(osp.join(REPO_ROOT, rec["dump"])),
                        variant=rec["dump_meta"]["variant"],
                        batch_size=rec["dump_meta"]["batch_size"],
                        inference_wall_time_s=rec["dump_meta"]["wall_time_s"],
                        detections=rec["prediction_stats"]["n_det"],
                        scoring_wall_time_s=v["wall_time_s"],
                    ),
                    train_log_val=tl,
                    delta_pp=delta,
                    status="ok",
                )
            },
        )

    summary = {}
    for run, r in runs.items():
        e = r["epochs"][str(args.epoch)]
        summary[r["letter"]] = dict(
            run=run, **{k.split("/")[-1]: e["metrics"].get(k) for k in HEADLINE}
        )

    doc = dict(
        description=(
            "CARLA test split (nuscenes_infos_val.pkl, 30,842 scans) "
            "re-evaluation of the released epoch-8 checkpoints with the "
            "corrected GeoPillarsMetric. NOTE: this split is the same one "
            "used for validation during training; it is not held out."
        ),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/figure_data/build_carla_eval_aggregate.py",
        supersedes="results/eval_results_carla_test.json",
        metric_changes=[
            "mBEVIoU is now reported at a single IoU threshold as mBEVIoU@0.5; the "
            "old threshold-pooled mBEVIoU key double-counted strict matches and is "
            "still reachable with mbeviou_mode='legacy_pooled'.",
            "mAOE_folded added: the yaw error folded into [0, pi/2] by min(d, pi-d), "
            "so a heading-reversed box costs 0 instead of pi. heading_flip_rate "
            "reports the share of TP matches beyond pi/2.",
            "iou_rotated records whether exact rotated-rectangle BEV IoU (Shapely) "
            "or the axis-aligned fallback produced the numbers.",
        ],
        environment=env,
        notes=[
            "predictions were dumped once per run and scored on the CPU, so the "
            "metric variants below are all computed from identical detections",
            "the nuScenes zero-shot half of paper_metrics.json still comes from "
            "eval_results_nusc_zeroshot.json, computed with the pre-correction "
            "metric; the source data has since been deleted",
        ],
        runs=runs,
        summary=summary,
    )
    _write_json(args.out, doc)
    print(json.dumps(summary, indent=1))
    print(f"[aggregate] -> {args.out}")


if __name__ == "__main__":
    main()
