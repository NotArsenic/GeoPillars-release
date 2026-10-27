#!/usr/bin/env python3
"""Export the zero-shot nuScenes GT-filter statistics and the intensity audit.

Writes ``figures/data/nusc_gt_stats.json`` (methodology numbers, histograms,
intensity handling) and ``figures/data/nusc_gt_cars.npz`` (one row per car
annotation, so histograms can be re-binned without nuScenes on disk).

Sources:
  * ``nuscenes_infos_zeroshot.pkl`` (raw ``.bin`` paths) and
    ``nuscenes_infos_zeroshot_npy.pkl`` (the pkl the eval actually used),
    both built by ``scripts/offline_preprocessing/nuscenes_partial_trainval_infos.py``.
  * ``v1.0-trainval/sample_annotation.json`` as an independent cross-check of
    the car counts and the ``num_lidar_pts`` cut.
  * every raw LIDAR_TOP ``.bin`` and every eval ``.npy`` for the intensity audit.

Usage (repo root)::

    python scripts/figure_data/export_nusc_gt_stats.py
"""

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
MIN_PTS = 5
MAX_DIST = 50.0
EVAL_RUNS = [
    "geo_pillars_h100_intensity",
    "geo_pillars_h100_nointensity_final",
    "centerpoint_h100_intensity_original",
    "centerpoint_h100_intensity",
    "centerpoint_h100_no_intensity",
    "centerpoint_osf1_ablation",
    "centerpoint_geopillar_encoder_replk_distributed",
]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def hist(values, edges):
    counts, _ = np.histogram(values, bins=edges)
    return dict(
        bin_edges=[float(e) if np.isfinite(e) else None for e in edges],
        counts=counts.astype(int).tolist(),
        note="bins are [lo, hi); the last bin includes hi",
    )


def audit_scan(args):
    """Per-scan intensity stats of the raw .bin and exact comparison with the eval .npy."""
    bin_path, npy_path = args
    raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    npy = np.load(npy_path)
    inten = raw[:, 3]
    return dict(
        n_points=len(raw),
        raw_min=float(inten.min()),
        raw_max=float(inten.max()),
        raw_non_integer=int(np.count_nonzero(inten != np.round(inten))),
        raw_hist=np.bincount(np.clip(inten, 0, 255).astype(np.int64), minlength=256),
        npy_shape_ok=npy.shape == (len(raw), 4) and npy.dtype == np.float32,
        npy_intensity_all_one=bool(np.all(npy[:, 3] == 1.0)),
        npy_intensity_min=float(npy[:, 3].min()),
        npy_intensity_max=float(npy[:, 3].max()),
        npy_xyz_identical=bool(np.array_equal(npy[:, :3], raw[:, :3])),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sys.path.insert(0, osp.join(REPO_ROOT, "tools"))
    from eval_test_checkpoints import NUSC_ZEROSHOT_ROOT

    ap.add_argument("--data_root", default=NUSC_ZEROSHOT_ROOT)
    ap.add_argument("--out_dir", default=osp.join(REPO_ROOT, "figures", "data"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw_pkl = osp.join(args.data_root, "nuscenes_infos_zeroshot.pkl")
    npy_pkl = osp.join(args.data_root, "nuscenes_infos_zeroshot_npy.pkl")
    with open(raw_pkl, "rb") as f:
        raw_infos = pickle.load(f)
    with open(npy_pkl, "rb") as f:
        npy_infos = pickle.load(f)
    raw_list, npy_list = raw_infos["data_list"], npy_infos["data_list"]
    assert len(raw_list) == len(npy_list)
    car_id = npy_infos["metainfo"]["categories"]["car"]

    # -- per-car table from the eval pkl ------------------------------------
    rows = dict(
        sample_idx=[],
        sample_token=[],
        scene_name=[],
        nusc_split=[],
        box=[],
        num_lidar_pts=[],
        range_m=[],
        isvalid=[],
    )
    scans_by_split = Counter()
    scenes_by_split = {}
    for r, n in zip(raw_list, npy_list):
        assert r["token"] == n["token"] and r["instances"] == n["instances"]
        scans_by_split[n["nusc_split"]] += 1
        scenes_by_split.setdefault(n["nusc_split"], set()).add(n["scene_name"])
        for inst in n["instances"]:
            if inst["bbox_label_3d"] != car_id:
                continue
            b = np.asarray(inst["bbox_3d"], dtype=np.float64)
            rows["sample_idx"].append(n["sample_idx"])
            rows["sample_token"].append(n["token"])
            rows["scene_name"].append(n["scene_name"])
            rows["nusc_split"].append(n["nusc_split"])
            rows["box"].append(b)
            rows["num_lidar_pts"].append(inst["num_lidar_pts"])
            rows["range_m"].append(float(np.hypot(b[0], b[1])))
            rows["isvalid"].append(bool(inst["bbox_3d_isvalid"]))

    pts = np.asarray(rows["num_lidar_pts"], dtype=np.int64)
    rng = np.asarray(rows["range_m"])
    isvalid = np.asarray(rows["isvalid"])
    fail_pts = pts < MIN_PTS
    fail_rng = rng > MAX_DIST
    passing = ~fail_pts & ~fail_rng
    # the pkl's stored flag must equal the rule recomputed here
    assert np.array_equal(
        passing, isvalid
    ), "bbox_3d_isvalid disagrees with the recomputed rule"

    # -- independent cross-check against the raw nuScenes annotations -------
    meta = osp.join(args.data_root, "v1.0-trainval_meta", "v1.0-trainval")
    with open(osp.join(meta, "category.json")) as f:
        cat_token = next(c["token"] for c in json.load(f) if c["name"] == "vehicle.car")
    with open(osp.join(meta, "instance.json")) as f:
        car_instances = {
            i["token"] for i in json.load(f) if i["category_token"] == cat_token
        }
    tokens = {n["token"] for n in npy_list}
    with open(osp.join(meta, "sample_annotation.json")) as f:
        anns = [
            a
            for a in json.load(f)
            if a["sample_token"] in tokens and a["instance_token"] in car_instances
        ]
    raw_json_cars = len(anns)
    raw_json_lt5 = sum(a["num_lidar_pts"] < MIN_PTS for a in anns)
    assert raw_json_cars == len(pts), (raw_json_cars, len(pts))
    assert raw_json_lt5 == int(fail_pts.sum()), (raw_json_lt5, int(fail_pts.sum()))

    # n_gt used by GeoPillarsMetric, read back from the saved PR curves
    pr_n_gt = {}
    for run in EVAL_RUNS:
        p = osp.join(
            REPO_ROOT,
            "work_dirs",
            "eval_nusc_zeroshot",
            run,
            "epoch_8",
            "pr_curves",
            "pr_curves.json",
        )
        if osp.isfile(p):
            with open(p) as f:
                entries = json.load(f)
            pr_n_gt[run] = sorted(
                {
                    (
                        e["n_gt"],
                        tuple(e["distance_range"]) if e.get("distance_range") else None,
                    )
                    for e in entries
                },
                key=str,
            )
    overall_n_gt = {run: [n for n, d in v if d is None] for run, v in pr_n_gt.items()}
    assert all(v == [int(passing.sum())] for v in overall_n_gt.values()), overall_n_gt

    # -- intensity audit over every scan ------------------------------------
    jobs = [
        (
            osp.join(args.data_root, r["lidar_points"]["lidar_path"]),
            osp.join(args.data_root, n["lidar_points"]["lidar_path"]),
        )
        for r, n in zip(raw_list, npy_list)
    ]
    raw_hist = np.zeros(256, dtype=np.int64)
    n_points, audits = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for a in ex.map(audit_scan, jobs, chunksize=32):
            raw_hist += a.pop("raw_hist")
            n_points.append(a["n_points"])
            audits.append(a)
    n_points = np.asarray(n_points)

    eval_inputs = {}
    for run in EVAL_RUNS:
        p = osp.join(
            REPO_ROOT, "work_dirs", "eval_nusc_zeroshot", run, "epoch_8", "result.json"
        )
        if osp.isfile(p):
            with open(p) as f:
                d = json.load(f)["data"]
            use_dim = d["use_dim"]
            use_dim = list(range(use_dim)) if isinstance(use_dim, int) else use_dim
            eval_inputs[run] = dict(
                ann_file=d["ann_file"],
                load_dim=d["load_dim"],
                use_dim=use_dim,
                reads_intensity_column=3 in use_dim,
                num_samples_evaluated=d["num_samples_evaluated"],
            )

    passing_pts, passing_rng = pts[passing], rng[passing]
    pct = lambda v, q: {f"p{k}": float(np.percentile(v, k)) for k in q}
    stats = dict(
        description="Zero-shot nuScenes GT protocol and intensity handling for the GeoPillars paper. "
        "Car class only; boxes in the nuScenes LIDAR_TOP frame.",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/figure_data/export_nusc_gt_stats.py",
        sources=dict(
            data="nuScenes v1.0-trainval, lidar blobs 01+02 (keyframes only) + trainval meta",
            info_pkl_eval=osp.basename(npy_pkl),
            info_pkl_eval_sha256=sha256(npy_pkl),
            info_pkl_raw_bin=osp.basename(raw_pkl),
            info_pkl_raw_bin_sha256=sha256(raw_pkl),
            pkl_builder="scripts/offline_preprocessing/nuscenes_partial_trainval_infos.py",
            npy_builder="scripts/offline_preprocessing/nuscenes_to_npy_single_sweep.py --constant_intensity 1.0",
        ),
        keyframes_evaluated=len(npy_list),
        scenes=len({n["scene_name"] for n in npy_list}),
        split_breakdown={
            s: dict(keyframes=scans_by_split[s], scenes=len(scenes_by_split[s]))
            for s in sorted(scans_by_split)
        },
        split_note="official nuScenes train/val scene lists (nuscenes.utils.splits); both are used, "
        "since no model saw nuScenes during training",
        gt_rule=dict(
            text=f"a car counts as GT iff num_lidar_pts >= {MIN_PTS} and hypot(x, y) <= {MAX_DIST:g} m "
            "(box centre, LIDAR_TOP frame)",
            min_lidar_pts=MIN_PTS,
            max_range_m=MAX_DIST,
            range_definition="BEV distance of box centre from LIDAR_TOP origin",
            category_mapping="only 'vehicle.car' maps to car (mmdet3d NuScenesNameMapping); "
            "vehicle.emergency.* and others are not car",
            mirrors="CARLA test pkl from offline_processor.py (>= 5 lidar points, max GT distance exactly 50.0 m)",
            ignored_gt_handling="cars failing the rule are dropped from GT; detections on them count as false positives",
        ),
        car_annotations=dict(
            total_before_filtering=int(len(pts)),
            failed_num_lidar_pts_ge_5=int(fail_pts.sum()),
            failed_range_le_50m=int(fail_rng.sum()),
            failed_both=int((fail_pts & fail_rng).sum()),
            failed_any=int((fail_pts | fail_rng).sum()),
            failed_only_num_lidar_pts=int((fail_pts & ~fail_rng).sum()),
            failed_only_range=int((fail_rng & ~fail_pts).sum()),
            zero_lidar_pts=int((pts == 0).sum()),
            final_gt_cars=int(passing.sum()),
            final_gt_cars_by_split={
                s: int((passing & (np.asarray(rows["nusc_split"]) == s)).sum())
                for s in sorted(scans_by_split)
            },
            keyframes_with_at_least_one_gt_car=int(
                len(set(np.asarray(rows["sample_idx"])[passing].tolist()))
            ),
            gt_cars_per_keyframe_mean=float(passing.sum() / len(npy_list)),
            gt_cars_by_metric_distance_bin={
                "0-20m": int(((passing_rng >= 0) & (passing_rng < 20)).sum()),
                "20-40m": int(((passing_rng >= 20) & (passing_rng < 40)).sum()),
                "40m+": int((passing_rng >= 40).sum()),
            },
            cross_checks=dict(
                raw_sample_annotation_json_vehicle_car=raw_json_cars,
                raw_sample_annotation_json_num_lidar_pts_lt_5=raw_json_lt5,
                pkl_isvalid_flag_equals_recomputed_rule=True,
                geopillars_metric_n_gt_from_pr_curves=overall_n_gt,
            ),
        ),
        passing_car_num_lidar_pts=dict(
            histogram_log_bins=hist(
                passing_pts,
                np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]),
            ),
            histogram_linear_0_200=hist(
                np.clip(passing_pts, 0, 200), np.arange(0, 205, 5)
            ),
            linear_note="values above 200 are clipped into the last bin [195, 200]",
            min=int(passing_pts.min()),
            max=int(passing_pts.max()),
            mean=float(passing_pts.mean()),
            median=float(np.median(passing_pts)),
            **pct(passing_pts, [10, 25, 75, 90]),
        ),
        passing_car_range_m=dict(
            histogram_2m=hist(passing_rng, np.arange(0, 52, 2.0)),
            histogram_5m=hist(passing_rng, np.arange(0, 55, 5.0)),
            min=float(passing_rng.min()),
            max=float(passing_rng.max()),
            mean=float(passing_rng.mean()),
            median=float(np.median(passing_rng)),
            **pct(passing_rng, [10, 25, 75, 90]),
        ),
        points_per_keyframe=dict(
            note="raw LIDAR_TOP keyframe, single sweep, all points (no range crop)",
            min=int(n_points.min()),
            max=int(n_points.max()),
            mean=float(n_points.mean()),
            median=float(np.median(n_points)),
            total=int(n_points.sum()),
        ),
        intensity_handling=dict(
            raw_nuscenes_intensity=dict(
                observed_min=min(a["raw_min"] for a in audits),
                observed_max=max(a["raw_max"] for a in audits),
                non_integer_values=int(sum(a["raw_non_integer"] for a in audits)),
                points_audited=int(n_points.sum()),
                keyframes_audited=len(audits),
                histogram_0_255=raw_hist.tolist(),
                source="column 3 of every raw LIDAR_TOP .pcd.bin (float32, 5 columns)",
            ),
            fed_to_model=dict(
                value=1.0,
                npy_files_with_intensity_exactly_1=int(
                    sum(a["npy_intensity_all_one"] for a in audits)
                ),
                npy_intensity_min=min(a["npy_intensity_min"] for a in audits),
                npy_intensity_max=max(a["npy_intensity_max"] for a in audits),
                npy_xyz_bitwise_identical_to_raw=int(
                    sum(a["npy_xyz_identical"] for a in audits)
                ),
                npy_shape_and_dtype_ok=int(sum(a["npy_shape_ok"] for a in audits)),
                eval_pkl_point_features=npy_infos["metainfo"].get("point_features"),
                npy_dir=osp.dirname(npy_list[0]["lidar_points"]["lidar_path"]),
            ),
            eval_pipeline=dict(
                loader="LoadPointsFromNpy (np.load, no value changes) -> Pack3DDetInputs; no intensity normalisation",
                runs=eval_inputs,
            ),
            rationale="every CARLA training scan has intensity exactly 1.0 (atmosphere_attenuation_rate=0 in the "
            "collector), so intensity checkpoints get the same constant; xyz-only runs never read it",
            confirmed=bool(
                all(
                    a["npy_intensity_all_one"]
                    and a["npy_xyz_identical"]
                    and a["npy_shape_ok"]
                    for a in audits
                )
                and all(
                    v["ann_file"] == osp.basename(npy_pkl) for v in eval_inputs.values()
                )
            ),
        ),
        companion_file="nusc_gt_cars.npz",
    )
    with open(osp.join(args.out_dir, "nusc_gt_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    np.savez_compressed(
        osp.join(args.out_dir, "nusc_gt_cars.npz"),
        sample_idx=np.asarray(rows["sample_idx"], dtype=np.int32),
        sample_token=np.asarray(rows["sample_token"]),
        scene_name=np.asarray(rows["scene_name"]),
        nusc_split=np.asarray(rows["nusc_split"]),
        box=np.stack(rows["box"]),
        box_format=np.asarray(
            "x y z l w h yaw; LIDAR_TOP frame; z = box gravity centre"
        ),
        num_lidar_pts=pts.astype(np.int32),
        range_m=rng,
        is_gt=passing,
        failed_num_lidar_pts=fail_pts,
        failed_range=fail_rng,
        keyframe_n_points=n_points.astype(np.int32),
        keyframe_token=np.asarray([n["token"] for n in npy_list]),
    )
    print(
        json.dumps(
            {k: stats[k] for k in ("keyframes_evaluated", "scenes", "split_breakdown")},
            indent=1,
        )
    )
    print(json.dumps({k: v for k, v in stats["car_annotations"].items()}, indent=1))
    ih = stats["intensity_handling"]
    print(
        "raw intensity",
        ih["raw_nuscenes_intensity"]["observed_min"],
        ih["raw_nuscenes_intensity"]["observed_max"],
        "non-int",
        ih["raw_nuscenes_intensity"]["non_integer_values"],
    )
    print({k: v for k, v in ih["fed_to_model"].items()}, "confirmed", ih["confirmed"])


if __name__ == "__main__":
    main()
