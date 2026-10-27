#!/usr/bin/env python3
"""Measure pillar occupancy of every CARLA scan under the voxelisation budgets (Fig. 9).

For each scan (train and test split, all of them) the in-range points are
binned exactly as mmcv hard voxelisation does (checked against
``mmcv.ops.Voxelization`` on 40 random scans x both budgets x both caps: identical
pillar and point counts):

* cell index ``c = floor((p - pcr_min) / voxel_size)`` in float32, points kept
  only when ``0 <= c < grid`` on x, y and z (one 8 m z cell = a pillar);
* pillars are admitted in order of their first point until ``max_voxels`` is
  reached, later pillars (and their points) are dropped;
* each admitted pillar keeps its first ``max_num_points`` points.

Budgets measured:
  ``geopillars``: 0.4 m, 8 points, 16,000 pillars (train) / 40,000 (test), used by A, B, D, E, F, G
  ``cp_native``: 0.2 m, 20 points, 30,000 / 40,000, used by C (reference only)

Writes ``figures/data/pillar_stats.json`` (histograms, cap-hit rates) and
``figures/data/pillar_stats_scans.npz`` (per-scan numbers).

Usage (repo root)::

    python scripts/figure_data/export_pillar_stats.py
"""

import argparse
import json
import os.path as osp
import sys
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(REPO_ROOT, "tools"))
from eval_test_checkpoints import CARLA_ROOT as DATA_ROOT  # noqa: E402
PCR = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
BUDGETS = {
    "geopillars": dict(
        voxel=0.4,
        max_points=8,
        max_voxels_train=16000,
        max_voxels_test=40000,
        runs="A, B, D, E, F, G",
    ),
    "cp_native": dict(
        voxel=0.2,
        max_points=20,
        max_voxels_train=30000,
        max_voxels_test=40000,
        runs="C",
    ),
}
HIST_MAX = 512  # per-pillar point counts above this land in the last bin


def budget_stats(xyz, voxel, max_points, max_voxels_train, max_voxels_test, **_):
    # mmcv hard_voxelize: float32 cell index c = floor((p - min) / size), keep 0 <= c < grid on x, y, z
    size = np.array([voxel, voxel, PCR[5] - PCR[2]], np.float32)
    grid = np.round((PCR[3:] - PCR[:3]) / size).astype(np.int64)
    c = np.floor((xyz - PCR[:3].astype(np.float32)) / size).astype(np.int64)
    c = c[np.all((c >= 0) & (c < grid), axis=1)]
    flat = c[:, 0] * grid[1] + c[:, 1]
    uniq, first, counts = np.unique(flat, return_index=True, return_counts=True)
    order = np.argsort(first, kind="stable")  # admission order = first appearance
    counts_in_order = counts[order]
    capped = np.minimum(counts_in_order, max_points)
    out = dict(
        occupied=len(uniq),
        pillar_hist=np.bincount(np.minimum(counts, HIST_MAX), minlength=HIST_MAX + 1),
        pillars_at_point_cap=int((counts >= max_points).sum()),
        pillars_over_point_cap=int((counts > max_points).sum()),
        points_after_point_cap=int(capped.sum()),
    )
    for tag, cap in (("train", max_voxels_train), ("test", max_voxels_test)):
        out[f"pillars_kept_{tag}"] = min(len(uniq), cap)
        out[f"points_kept_{tag}"] = int(capped[:cap].sum())
    return out


def scan_stats(path):
    pts = np.load(path, mmap_mode="r")[:, :3]
    xyz = np.asarray(pts, dtype=np.float32)
    in_range = int(np.all((xyz >= PCR[:3]) & (xyz < PCR[3:]), axis=1).sum())
    return dict(
        n_points=len(pts),
        n_in_range=in_range,
        **{name: budget_stats(xyz, **b) for name, b in BUDGETS.items()},
    )


def summarise(values):
    v = np.asarray(values, dtype=np.float64)
    return dict(
        mean=float(v.mean()),
        std=float(v.std()),
        min=float(v.min()),
        max=float(v.max()),
        **{f"p{q}": float(np.percentile(v, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--limit", type=int, default=None, help="first N scans per split (smoke test)"
    )
    args = ap.parse_args()

    doc = dict(
        description="Pillar occupancy of CARLA scans under the hard-voxelisation budgets, computed on every scan.",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/figure_data/export_pillar_stats.py",
        point_cloud_range=PCR.tolist(),
        budgets=BUDGETS,
        semantics="mmcv hard voxelisation: pillars admitted in first-point order up to max_voxels; first "
        "max_num_points points per pillar kept",
        splits={},
    )
    per_scan = {}
    for split in ("train", "val"):
        with open(osp.join(DATA_ROOT, f"nuscenes_infos_{split}.pkl"), "rb") as f:
            infos = pickle.load(f)["data_list"][: args.limit]
        paths = [osp.join(DATA_ROOT, i["lidar_points"]["lidar_path"]) for i in infos]
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(scan_stats, paths, chunksize=64))
        s = dict(
            split_file=f"nuscenes_infos_{split}.pkl",
            role="train" if split == "train" else "test",
            scans=len(res),
            points_per_scan=summarise([r["n_points"] for r in res]),
            in_range_points_per_scan=summarise([r["n_in_range"] for r in res]),
            budgets={},
        )
        arrays = dict(
            n_points=np.array([r["n_points"] for r in res], np.int32),
            n_in_range=np.array([r["n_in_range"] for r in res], np.int32),
        )
        for name, b in BUDGETS.items():
            rr = [r[name] for r in res]
            hist = np.sum([r["pillar_hist"] for r in rr], axis=0)
            occ = np.array([r["occupied"] for r in rr])
            n_in = arrays["n_in_range"].astype(np.float64)
            bs = dict(
                pillar_point_count_hist=dict(
                    counts=hist.tolist(),
                    note=f"index k = number of occupied pillars holding k points before the cap "
                    f"(index {HIST_MAX} = {HIST_MAX} or more); summed over all scans",
                ),
                occupied_pillars_per_scan=summarise(occ),
                occupied_pillars_hist_1000=dict(
                    bin_edges=list(range(0, int(occ.max()) + 1001, 1000)),
                    counts=np.histogram(
                        occ, bins=range(0, int(occ.max()) + 1001, 1000)
                    )[0].tolist(),
                ),
                pillars_at_or_over_point_cap_frac=float(
                    sum(r["pillars_at_point_cap"] for r in rr) / hist[1:].sum()
                ),
                pillars_over_point_cap_frac=float(
                    sum(r["pillars_over_point_cap"] for r in rr) / hist[1:].sum()
                ),
                in_range_points_dropped_by_point_cap_frac=float(
                    1 - sum(r["points_after_point_cap"] for r in rr) / n_in.sum()
                ),
            )
            for tag in ("train", "test"):
                cap = b[f"max_voxels_{tag}"]
                kept = np.array([r[f"points_kept_{tag}"] for r in rr], np.float64)
                bs[f"max_voxels_{tag}"] = dict(
                    cap=cap,
                    scans_exceeding_cap_frac=float((occ > cap).mean()),
                    scans_exceeding_cap=int((occ > cap).sum()),
                    in_range_points_kept_frac=float(kept.sum() / n_in.sum()),
                    in_range_points_kept_frac_per_scan=summarise(
                        kept / np.maximum(n_in, 1)
                    ),
                )
                arrays[f"{name}_points_kept_{tag}cap"] = kept.astype(np.int32)
            s["budgets"][name] = bs
            arrays[f"{name}_occupied"] = occ.astype(np.int32)
        doc["splits"][split] = s
        per_scan[split] = arrays
        g = s["budgets"]["geopillars"]
        print(
            f"{split}: {len(res)} scans | occupied pillars median {g['occupied_pillars_per_scan']['p50']:.0f} "
            f"p95 {g['occupied_pillars_per_scan']['p95']:.0f} | >16K {100 * g['max_voxels_train']['scans_exceeding_cap_frac']:.2f}% "
            f"| pillars >8 pts {100 * g['pillars_over_point_cap_frac']:.1f}% | points kept (train caps) "
            f"{100 * g['max_voxels_train']['in_range_points_kept_frac']:.1f}%",
            flush=True,
        )

    tag = "" if args.limit is None else f"_smoke{args.limit}"
    with open(
        osp.join(REPO_ROOT, "figures", "data", f"pillar_stats{tag}.json"), "w"
    ) as f:
        json.dump(doc, f, indent=1)
    np.savez_compressed(
        osp.join(REPO_ROOT, "figures", "data", f"pillar_stats_scans{tag}.npz"),
        **{
            f"{split}_{k}": v
            for split, arrs in per_scan.items()
            for k, v in arrs.items()
        },
    )


if __name__ == "__main__":
    main()
