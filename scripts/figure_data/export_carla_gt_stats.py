"""CARLA GT accounting for the 30,842-scan evaluation split.

The CARLA counterpart of ``figures/data/nusc_gt_stats.json``. ``GeoPillarsMetric``
filters GT only on ``bbox_3d_isvalid``, which ``offline_processor.py:556`` writes
as literal ``True`` for every instance, so the effective filter is whatever the
collection and preprocessing stages already applied. This script states what
that is, in numbers, and checks it against what the training dataloader would
have kept (``num_lidar_pts > 0`` plus ``ObjectRangeFilter``).

Writes ``figures/data/carla_gt_stats.json``.
"""

import argparse
import hashlib
import json
import os.path as osp
import pickle
import sys
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from _carla_infer import CARLA_ROOT, CARLA_TEST_PKL  # noqa: E402

PCR = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
DIST_BINS = [(0, 20), (20, 40), (40, float("inf"))]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def pct(a, q):
    return [round(float(v), 4) for v in np.percentile(a, q)]


def hist(a, edges):
    c, _ = np.histogram(np.clip(a, edges[0], edges[-1]), bins=edges)
    return dict(
        bin_edges=[float(e) for e in edges],
        counts=[int(v) for v in c],
        note="bins are [lo, hi); the last bin includes hi",
    )


def dim_stats(v, name):
    return {
        name: dict(
            n=int(len(v)),
            mean=round(float(v.mean()), 4),
            median=round(float(np.median(v)), 4),
            q1=round(float(np.percentile(v, 25)), 4),
            q3=round(float(np.percentile(v, 75)), 4),
            iqr=round(float(np.percentile(v, 75) - np.percentile(v, 25)), 4),
            std=round(float(v.std()), 4),
            min=round(float(v.min()), 4),
            max=round(float(v.max()), 4),
            n_unique_rounded_2dp=int(len(np.unique(np.round(v, 2)))),
        )
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pkl", default=osp.join(CARLA_ROOT, CARLA_TEST_PKL))
    ap.add_argument(
        "--train-pkl", default=osp.join(CARLA_ROOT, "nuscenes_infos_train.pkl")
    )
    ap.add_argument(
        "--out", default=osp.join(REPO_ROOT, "figures", "data", "carla_gt_stats.json")
    )
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    data_list = data["data_list"]
    cats = data["metainfo"]["categories"]
    car_id = cats["car"]
    inv = {v: k for k, v in cats.items()}

    n_by_label, isvalid_true, isvalid_false = {}, 0, 0
    car_box, car_pts, car_town, car_scan = [], [], [], []
    per_scan_cars = np.zeros(len(data_list), dtype=np.int32)
    out_of_pcr = zero_pts = lt5_pts = 0

    for si, info in enumerate(data_list):
        town = info["lidar_points"]["lidar_path"].split("/")[0]
        for inst in info.get("instances", []):
            lab = int(inst["bbox_label_3d"])
            n_by_label[inv.get(lab, lab)] = n_by_label.get(inv.get(lab, lab), 0) + 1
            if inst.get("bbox_3d_isvalid", True):
                isvalid_true += 1
            else:
                isvalid_false += 1
            if lab != car_id:
                continue
            b = inst["bbox_3d"]
            npts = int(inst.get("num_lidar_pts", -1))
            car_box.append(b[:7])
            car_pts.append(npts)
            car_town.append(town)
            car_scan.append(si)
            per_scan_cars[si] += 1
            if not (PCR[0] <= b[0] <= PCR[3] and PCR[1] <= b[1] <= PCR[4]):
                out_of_pcr += 1
            if npts == 0:
                zero_pts += 1
            if 0 <= npts < 5:
                lt5_pts += 1

    car_box = np.asarray(car_box, dtype=np.float64)
    car_pts = np.asarray(car_pts)
    rng_m = np.hypot(car_box[:, 0], car_box[:, 1])
    dx, dy, dz = car_box[:, 3], car_box[:, 4], car_box[:, 5]
    yaw = car_box[:, 6]
    towns, town_counts = np.unique(car_town, return_counts=True)

    bin_counts = {}
    for lo, hi in DIST_BINS:
        lbl = f"{int(lo)}-{int(hi)}m" if np.isfinite(hi) else f"{int(lo)}m+"
        bin_counts[lbl] = int(((rng_m >= lo) & (rng_m < hi)).sum())

    out = dict(
        description=(
            "CARLA GT accounting for the 30,842-scan split that "
            "GeoPillarsMetric scores (nuscenes_infos_val.pkl). This is "
            "the same split used for validation during training and for "
            "checkpoint selection - see naming_check below."
        ),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/figure_data/export_carla_gt_stats.py",
        sources=dict(
            data_root=CARLA_ROOT,
            pkl=osp.basename(args.pkl),
            pkl_sha256=sha256(args.pkl),
            builder="scripts/offline_preprocessing/offline_processor.py",
            scans=len(data_list),
        ),
        gt_rule=dict(
            text=(
                "GeoPillarsMetric keeps an instance iff "
                "inst['bbox_3d_isvalid'] is truthy (geo_pillars_metric.py:506-513) "
                "and its label maps to a configured class. offline_processor.py:556 "
                "writes bbox_3d_isvalid as literal True for every instance, so the "
                "metric applies no filter of its own: it scores every car box in "
                "the pkl."
            ),
            effective_filter=(
                "inherited from upstream, not applied at evaluation time: "
                "(1) collection-time class-dependent 2-D range cut, 50 m for car "
                "(S2R_Carla_dataset_collection_NC.py:209-218); "
                "(2) num_lidar_pts >= 5 (offline_processor.py:295-297); "
                "(3) |roll|,|pitch| <= 15 deg and grounded checks; "
                "(4) per-scan height-sanity rejection. "
                "No range or point-count cut is re-applied by the evaluator."
            ),
            metric_code="src/metrics/geo_pillars_metric.py:506-513",
            isvalid_written_at="scripts/offline_preprocessing/offline_processor.py:556",
        ),
        instances=dict(
            by_class=n_by_label,
            total=int(sum(n_by_label.values())),
            bbox_3d_isvalid_true=isvalid_true,
            bbox_3d_isvalid_false=isvalid_false,
            note="every instance of every class carries bbox_3d_isvalid = True",
        ),
        car_boxes=dict(
            total_in_pkl=int(len(car_box)),
            scored_by_evaluator=int(len(car_box)),
            excluded_by_evaluator=0,
            by_town={str(t): int(c) for t, c in zip(towns, town_counts)},
            outside_point_cloud_range_xy=int(out_of_pcr),
            num_lidar_pts_zero=int(zero_pts),
            num_lidar_pts_below_5=int(lt5_pts),
            max_range_m=round(float(rng_m.max()), 4),
            by_metric_distance_bin=bin_counts,
            cars_per_scan_mean=round(float(per_scan_cars.mean()), 4),
            cars_per_scan_median=float(np.median(per_scan_cars)),
            cars_per_scan_max=int(per_scan_cars.max()),
            scans_with_zero_scored_cars=int((per_scan_cars == 0).sum()),
            fraction_scans_with_zero_scored_cars=round(
                float((per_scan_cars == 0).mean()), 6
            ),
            note=(
                "the evaluator scores all of these; a scan with zero cars still "
                "contributes false positives (geo_pillars_metric.py:779-782)"
            ),
        ),
        training_dataloader_comparison=dict(
            text=(
                "the training/val dataloader additionally applies "
                "num_lidar_pts > 0 (use_valid_flag defaults False, "
                "nuscenes_dataset.py:112-131) and ObjectRangeFilter on the "
                "+-51.2 m box, then ObjectNameFilter([car])"
            ),
            car_boxes_dropped_by_num_lidar_pts_gt_0=int(zero_pts),
            car_boxes_dropped_by_object_range_filter=int(out_of_pcr),
            conclusion=(
                "identical GT set"
                if (zero_pts == 0 and out_of_pcr == 0)
                else "the two GT sets differ - see counts"
            ),
        ),
        num_lidar_pts=dict(
            **{
                k: (round(float(v), 4) if isinstance(v, float) else int(v))
                for k, v in dict(
                    min=car_pts.min(),
                    max=car_pts.max(),
                    mean=float(car_pts.mean()),
                    median=float(np.median(car_pts)),
                ).items()
            },
            percentiles=dict(
                zip(
                    ["p10", "p25", "p75", "p90", "p99"],
                    pct(car_pts, [10, 25, 75, 90, 99]),
                )
            ),
            histogram_log_bins=hist(
                car_pts,
                [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, float(car_pts.max())],
            ),
            histogram_linear_0_200=hist(car_pts, list(np.arange(0, 205, 5))),
            linear_note="values above 200 are clipped into the last bin",
        ),
        range_m=dict(
            min=round(float(rng_m.min()), 4),
            max=round(float(rng_m.max()), 4),
            mean=round(float(rng_m.mean()), 4),
            median=round(float(np.median(rng_m)), 4),
            percentiles=dict(
                zip(
                    ["p10", "p25", "p75", "p90", "p99"],
                    pct(rng_m, [10, 25, 75, 90, 99]),
                )
            ),
            histogram_2m=hist(rng_m, list(np.arange(0, 54, 2))),
            definition="BEV distance of the box centre from the sensor origin",
        ),
        car_dimensions_m=dict(
            note=(
                "bbox_3d = [cx, cy, cz, dx, dy, dz, yaw]; dx is length, dy width, "
                "dz height. All car boxes in the split."
            ),
            **dim_stats(dx, "dx_length"),
            **dim_stats(dy, "dy_width"),
            **dim_stats(dz, "dz_height"),
            median_dx_by_dy=f"{np.median(dx):.2f} x {np.median(dy):.2f} m",
            yaw_exactly_zero=int((yaw == 0.0).sum()),
            yaw_exactly_zero_fraction=round(float((yaw == 0.0).mean()), 6),
            yaw_zero_note=(
                "static level_bb_vehicle boxes fall back to yaw 0.0 "
                "(S2R_Carla_dataset_collection_NC.py:1600-1605), but the "
                "collector stores atan2(cos rel, sin rel) = pi/2 - rel, "
                "so a zero relative yaw is written as +pi/2, not 0"
            ),
        ),
        car_yaw_rad=dict(
            note=(
                "yaw is pi/2 - radians(actor_yaw - lidar_yaw), so a target exactly "
                "parallel to the ego heading is stored as +pi/2 and an exactly "
                "antiparallel one as -pi/2"
            ),
            at_plus_half_pi_1e6=int((np.abs(yaw - np.pi / 2) < 1e-6).sum()),
            at_minus_half_pi_1e6=int((np.abs(yaw + np.pi / 2) < 1e-6).sum()),
            axis_aligned_count=int((np.abs(np.abs(yaw) - np.pi / 2) < 1e-6).sum()),
            axis_aligned_fraction=round(
                float((np.abs(np.abs(yaw) - np.pi / 2) < 1e-6).mean()), 6
            ),
            axis_aligned_fraction_1e3=round(
                float((np.abs(np.abs(yaw) - np.pi / 2) < 1e-3).mean()), 6
            ),
            perpendicular_count=int(
                ((np.abs(yaw) < 1e-6) | (np.abs(np.abs(yaw) - np.pi) < 1e-6)).sum()
            ),
            histogram_pi_over_36=hist(yaw, list(np.linspace(-np.pi, np.pi, 37))),
            interpretation=(
                "a large exactly-axis-aligned share means most CARLA "
                "targets share the ego lane heading bit-for-bit, which "
                "makes yaw regression far easier than on real data and "
                "inflates the CARLA-vs-nuScenes mAOE gap"
            ),
        ),
        user_reported_cross_check=dict(
            claim="median 4.61 x 1.97 m, n = 41,149",
            measured_median=f"{np.median(dx):.2f} x {np.median(dy):.2f} m",
            measured_n=int(len(car_box)),
            n_matches=bool(len(car_box) == 41149),
            equivalent_scan_count_for_n_41149=round(
                41149 / (len(car_box) / len(data_list)), 1
            ),
            comment=(
                "n = 41,149 is not the number of car boxes in this split; it "
                "corresponds to roughly the number in a "
                f"{round(41149 / (len(car_box) / len(data_list)))}-scan subsample "
                "at this split density"
            ),
        ),
    )

    # cross-check: train split totals, for the paper's dataset table
    if osp.exists(args.train_pkl):
        with open(args.train_pkl, "rb") as f:
            tr = pickle.load(f)["data_list"]
        tr_by_label = {}
        for info in tr:
            for inst in info.get("instances", []):
                nm = inv.get(int(inst["bbox_label_3d"]), inst["bbox_label_3d"])
                tr_by_label[nm] = tr_by_label.get(nm, 0) + 1
        out["train_split_cross_check"] = dict(
            scans=len(tr),
            by_class=tr_by_label,
            total=int(sum(tr_by_label.values())),
            train_plus_val_car=int(tr_by_label.get("car", 0) + len(car_box)),
        )

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    keys = (
        "instances",
        "car_boxes",
        "training_dataloader_comparison",
        "car_dimensions_m",
        "user_reported_cross_check",
        "train_split_cross_check",
    )
    print(json.dumps({k: out[k] for k in keys if k in out}, indent=2)[:4200])
    print(
        f'\n[carla-gt] range median {out["range_m"]["median"]} m, '
        f'num_lidar_pts median {out["num_lidar_pts"]["median"]}'
    )
    print(f"[carla-gt] -> {args.out}")


if __name__ == "__main__":
    main()
