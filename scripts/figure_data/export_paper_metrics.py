#!/usr/bin/env python3
"""Collect the scalar results behind Figs. 2, 3, 5 and 6 into ``figures/data/paper_metrics.json``.

Sources (all already on disk, no data or GPU needed):

* ``results/eval_results_carla_test_v2.json`` when present, else
  ``eval_results_carla_test.json``: epoch-8 re-evaluation on the CARLA validation
  split (AP, distance-binned AP, mAOE). The v2 file carries the corrected metrics
  (``mBEVIoU@0.5`` in place of the threshold-pooled ``mBEVIoU``, plus
  ``mAOE_folded``, ``heading_flip_rate`` and ``iou_rotated``).
* ``results/eval_results_nusc_zeroshot.json``: zero-shot nuScenes.
  Still produced by the pre-correction metric - the source data has been deleted,
  so it cannot be regenerated. ``metric_version`` records this per dataset.
* ``<GEOPILLARS_CKPT_ROOT>/<run>/<timestamp>/<timestamp>.log``: validation printed
  during training at epochs 2/4/6/8. Only the ``GeoPillars/*`` keys are read; the
  ``GeoPillars_*`` keys on the same line echo the *previous* evaluation.

Usage (repo root)::

    python scripts/figure_data/export_paper_metrics.py
"""

import glob
import os
import hashlib
import json
import os.path as osp
import re
from datetime import datetime, timezone

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
CKPT = os.environ.get("GEOPILLARS_CKPT_ROOT", osp.join(REPO_ROOT, "checkpoints"))
RESULTS = osp.join(REPO_ROOT, "results")
RUNS = {
    "A": "geo_pillars_h100_intensity",
    "B": "geo_pillars_h100_nointensity_final",
    "C": "centerpoint_h100_intensity_original",
    "D": "centerpoint_h100_intensity",
    "E": "centerpoint_h100_no_intensity",
    "F": "centerpoint_osf1_ablation",
    "G": "centerpoint_geopillar_encoder_replk_distributed",
}
VAL_LINE = re.compile(r"Epoch\(val\) \[(\d+)\]\[(\d+)/(\d+)\]\s+(.*)$")
METRIC = re.compile(r"(GeoPillars/[\w@.\-+]+): ([0-9.]+)")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def training_val(run):
    """Epoch -> GeoPillars/* metrics from the log that holds the completed validations."""
    found = {}
    for log in sorted(glob.glob(osp.join(CKPT, run, "2026*", "2026*.log"))):
        per_epoch = {}
        for line in open(log, errors="replace"):
            m = VAL_LINE.search(line)
            if m and m.group(2) == m.group(3):
                metrics = {k: float(v) for k, v in METRIC.findall(m.group(4))}
                if metrics:
                    per_epoch[int(m.group(1))] = metrics
        if per_epoch:
            found[log] = per_epoch
    assert (
        len(found) == 1
    ), f"{run}: expected exactly one log with validations, got {list(found)}"
    log, per_epoch = next(iter(found.items()))
    return osp.relpath(log, REPO_ROOT), per_epoch


COMMON_KEYS = [
    "AP@0.7",
    "AP@0.5",
    "mAOE",
    "AP@0.7@0-20m",
    "AP@0.7@20-40m",
    "AP@0.7@40m+",
    "mAOE@0-20m",
    "mAOE@20-40m",
    "mAOE@40m+",
]
V2_ONLY_KEYS = [
    "mBEVIoU@0.5",
    "mAOE_folded",
    "heading_flip_rate",
    "iou_rotated",
    "mAOE_folded@0-20m",
    "mAOE_folded@20-40m",
    "mAOE_folded@40m+",
]
LEGACY_ONLY_KEYS = ["mBEVIoU"]


def main():
    carla_v2_path = osp.join(RESULTS, "eval_results_carla_test_v2.json")
    carla_path = (
        carla_v2_path
        if osp.exists(carla_v2_path)
        else osp.join(RESULTS, "eval_results_carla_test.json")
    )
    carla = json.load(open(carla_path))
    carla_is_v2 = carla_path == carla_v2_path
    nusc = json.load(open(osp.join(RESULTS, "eval_results_nusc_zeroshot.json")))
    carla_keys = COMMON_KEYS + (V2_ONLY_KEYS if carla_is_v2 else LEGACY_ONLY_KEYS)
    nusc_keys = COMMON_KEYS + LEGACY_ONLY_KEYS
    runs = {}
    for letter, run in RUNS.items():
        cm = carla["runs"][run]["epochs"]["8"]["metrics"]
        nm = nusc["runs"][run]["epochs"]["8"]["metrics"]
        log, per_epoch = training_val(run)
        runs[letter] = dict(
            run=run,
            paper_role=carla["runs"][run].get("paper_role"),
            setup=carla["runs"][run].get("setup"),
            carla_test={
                k: cm[f"GeoPillars/{k}"] for k in carla_keys if f"GeoPillars/{k}" in cm
            },
            nusc_zeroshot={
                k: nm[f"GeoPillars/{k}"] for k in nusc_keys if f"GeoPillars/{k}" in nm
            },
            training_val=dict(
                log=log,
                epochs=sorted(per_epoch),
                **{
                    k: [per_epoch[e][f"GeoPillars/{k}"] for e in sorted(per_epoch)]
                    for k in ("AP@0.7", "AP@0.5")
                },
            ),
        )
        # the epoch-8 training-log value must agree with the re-evaluation (metric changed after training)
        assert (
            abs(runs[letter]["training_val"]["AP@0.7"][-1] - cm["GeoPillars/AP@0.7"])
            <= 0.0003
        ), letter

    ap = {k: runs[k]["carla_test"]["AP@0.7"] * 100 for k in RUNS}
    steps = [
        ("D", "CenterPoint at GeoPillars budget (0.4 m, 8 pts, 16K pillars, OSF=4)"),
        ("F", "+ head resolution (OSF=4 -> 1)"),
        ("G", "+ GeoPillars encoder, RepLK backbone, BEV upsampler"),
        ("A", "+ NGFM (full GeoPillars)"),
    ]
    decomposition = dict(
        metric="AP@0.7 (%), CARLA test re-evaluation, epoch 8",
        start=dict(run="D", ap=round(ap["D"], 2)),
        increments=[
            dict(
                to=k,
                label=lab,
                from_ap=round(ap[p], 2),
                to_ap=round(ap[k], 2),
                delta=round(ap[k] - ap[p], 2),
            )
            for (p, _), (k, lab) in zip(steps[:-1], steps[1:])
        ],
        end=dict(run="A", ap=round(ap["A"], 2)),
        total_gap=round(ap["A"] - ap["D"], 2),
    )
    doc = dict(
        description="Scalar results for the GeoPillars paper figures (car class). AP in [0, 1] unless noted; "
        "mAOE in radians over true positives at IoU 0.5; distance = BEV range of the box centre.",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/figure_data/export_paper_metrics.py",
        sources=dict(
            carla_test=osp.relpath(carla_path, REPO_ROOT),
            carla_test_sha256=sha256(carla_path),
            nusc_zeroshot="results/eval_results_nusc_zeroshot.json",
            nusc_zeroshot_sha256=sha256(
                osp.join(RESULTS, "eval_results_nusc_zeroshot.json")
            ),
            training_val="per-run training logs; the validation split IS this "
            '"CARLA test split" (nuscenes_infos_val.pkl, 30,842 scans) '
            "- it is not held out; metric code before b800b57",
        ),
        metric_version=dict(
            carla_test=(
                "corrected 2026-09-20: mBEVIoU@0.5 single-threshold, "
                "mAOE_folded and heading_flip_rate added, iou_rotated "
                "recorded"
                if carla_is_v2
                else "pre-correction"
            ),
            nusc_zeroshot=(
                "pre-correction: mBEVIoU is the threshold-pooled key and "
                "no mAOE_folded is available; the zero-shot source data "
                "was deleted, so it cannot be regenerated"
            ),
            comparable=(
                "AP@0.5, AP@0.7, mAP, mAOE and the distance-binned AP are "
                "identically defined in both and are directly comparable; "
                "mBEVIoU is not"
            ),
        ),
        runs=runs,
        decomposition=decomposition,
    )
    path = osp.join(REPO_ROOT, "figures", "data", "paper_metrics.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps(decomposition, indent=1))
    for k, r in runs.items():
        print(
            k,
            r["run"],
            "val",
            r["training_val"]["epochs"],
            r["training_val"]["AP@0.7"],
            "| mAOE bins",
            [r["carla_test"][f"mAOE@{b}"] for b in ("0-20m", "20-40m", "40m+")],
            "| carla",
            r["carla_test"]["AP@0.7"],
            "nusc",
            r["nusc_zeroshot"]["AP@0.7"],
        )


if __name__ == "__main__":
    main()
