#!/usr/bin/env python3
"""Compact the evaluation PR curves into one small JSON for plotting (Fig. 7).

Reads ``work_dirs/eval_{carla_test,nusc_zeroshot}/<run>/epoch_8/pr_curves/pr_curves.json``
(100-230 MB each: one precision/recall point per detection) and writes
``figures/data/pr_curves.json`` with, for every run, dataset, IoU threshold and
distance bin:

* ``recall``: grid 0, 0.001, ... up to the curve's maximum recall (the exact
  maximum is appended as the last point),
* ``precision``: the interpolated precision envelope at those recalls,
  ``max{p : r' >= r}``, the same quantity the R40 AP integrates,
* ``max_recall``, ``ap_r40`` (recomputed from the raw arrays and checked against
  the stored metric), ``n_gt``, ``n_pred``.

Usage (repo root)::

    python scripts/figure_data/export_pr_curves.py
"""

import json
import os.path as osp
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
RUNS = {
    "A": "geo_pillars_h100_intensity",
    "B": "geo_pillars_h100_nointensity_final",
    "C": "centerpoint_h100_intensity_original",
    "D": "centerpoint_h100_intensity",
    "E": "centerpoint_h100_no_intensity",
    "F": "centerpoint_osf1_ablation",
    "G": "centerpoint_geopillar_encoder_replk_distributed",
}
DATASETS = ("carla_test", "nusc_zeroshot")
STEP = 1e-3


def r40(p, r):
    return (
        sum(
            (p[r >= t - 1e-9].max() if (r >= t - 1e-9).any() else 0.0)
            for t in np.linspace(1 / 40, 1, 40)
        )
        / 40
    )


def envelope(p, r):
    """Interpolated precision max{p : r' >= r} on a fixed recall grid."""
    env = np.maximum.accumulate(p[::-1])[::-1]  # suffix max; r is non-decreasing
    rmax = float(r[-1])
    grid = np.arange(0.0, rmax, STEP)
    grid = np.append(grid, rmax)
    idx = np.searchsorted(r, grid - 1e-12, side="left")  # first index with r >= grid
    return grid, env[np.minimum(idx, len(env) - 1)]


def key_for(e):
    dr = e.get("distance_range")
    rng = "all" if not dr else ("40m+" if dr[1] is None else f"{dr[0]:g}-{dr[1]:g}m")
    return f"{e['iou_threshold']}", rng


def main():
    out = dict(
        description="PR curves (car) of the released runs at epoch 8, compacted from the "
        "per-detection arrays written by GeoPillarsMetric. precision = interpolated envelope.",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/figure_data/export_pr_curves.py",
        recall_grid_step=STEP,
        run_letters=RUNS,
        curves={},
    )
    for ds in DATASETS:
        for letter, run in RUNS.items():
            root = osp.join(REPO_ROOT, "work_dirs", f"eval_{ds}", run, "epoch_8")
            with open(osp.join(root, "result.json")) as f:
                metrics = json.load(f)["metrics"]
            with open(osp.join(root, "pr_curves", "pr_curves.json")) as f:
                entries = json.load(f)
            for e in entries:
                thr, rng = key_for(e)
                p = np.asarray(e["precision"], dtype=np.float64)
                r = np.asarray(e["recall"], dtype=np.float64)
                ap = round(r40(p, r), 4)
                mkey = f"GeoPillars/AP@{thr}" + ("" if rng == "all" else f"@{rng}")
                assert ap == metrics[mkey], (ds, run, mkey, ap, metrics[mkey])
                grid, prec = envelope(p, r)
                out["curves"].setdefault(ds, {}).setdefault(letter, {}).setdefault(
                    thr, {}
                )[rng] = dict(
                    run=run,
                    ap_r40=ap,
                    max_recall=round(float(r[-1]), 5),
                    precision_at_max_recall=round(float(p[-1]), 5),
                    n_gt=e["n_gt"],
                    n_pred=e["n_pred"],
                    recall=[round(float(x), 5) for x in grid],
                    precision=[round(float(x), 5) for x in prec],
                )
            print(
                f'{ds} {letter} {run}: AP@0.7 {metrics["GeoPillars/AP@0.7"]} ok',
                flush=True,
            )
    path = osp.join(REPO_ROOT, "figures", "data", "pr_curves.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {path} ({osp.getsize(path) / 2**20:.2f} MiB)")


if __name__ == "__main__":
    main()
