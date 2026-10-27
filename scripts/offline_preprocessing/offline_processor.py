#!/usr/bin/env python3
"""
offline_processor.py - Geo-Pillars Offline Data Preprocessor
=============================================================

Converts raw CARLA synthetic LiDAR data into mmdet3d-compatible nuScenes format with precomputed KD-Tree surface normals.

Pipeline:
  1. Proportional Sampling  - sample --target_scans across towns without geographic bias.
  2. Open3D Multiprocessing  - compute surface normals (radius=1.0m, max_nn=30) and produce 7D point clouds [X, Y, Z, I, Nx, Ny, Nz].
  3. Occlusion Filter  - discard objects with num_lidar_pts < 5.
  4. Geometric Sanity Filter - poison and drop entire scans containing
     bounding boxes with physically impossible dimensions.
  5. PKL Builder  - emit nuscenes_infos_{train,val}.pkl structurally
     identical to official nuScenes for mmdet3d NuScenesDataset.

Usage:
  python offline_processor.py \
      --data_root /mnt/data/carla_dataset \
      --out_dir   /mnt/data/carla_processed \
      --target_scans 100000 \
      --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np

#  Constants
NORMAL_SEARCH_RADIUS: float = 1.0  # metres
NORMAL_MAX_NN: int = 30
MIN_LIDAR_PTS: int = 5  # occlusion visibility threshold

# Dimension sanity thresholds (metres)
# "Too Large" - blueprint fallback / trucks with inflated bounding volumes
DIM_MAX_H: float = 3.5
DIM_MAX_W: float = 3.0
DIM_MAX_L: float = 7.0
# "Too Small" - collapsed incline boxes / degenerate geometry
DIM_MIN_H: float = 0.7
DIM_MIN_W: float = 1.0
DIM_MIN_L: float = 1.5

# Map CARLA nuScenes-style class strings -> mmdet3d short names.
# Objects whose class is not in this dict are silently dropped.
NUSCENES_TO_MMDET3D: dict[str, str] = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}


# Proportional Sampling (The Splicer)
def discover_towns(data_root: Path) -> dict[str, list[str]]:
    """Return {town_name: sorted list of frame stems} for every town that
    has both a ``lidar/`` and ``labels/`` subfolder."""

    towns: dict[str, list[str]] = {}
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue

        lidar_dir = entry / "lidar"
        label_dir = entry / "labels"

        if lidar_dir.is_dir() and label_dir.is_dir():
            # Use label JSON stems as the canonical frame list
            # every frame that has a label must also have a point cloud.
            stems = sorted(
                p.stem
                for p in label_dir.glob("*.json")
                if (lidar_dir / f"{p.stem}.npy").exists()
            )

            if stems:
                towns[entry.name] = stems
    return towns


def proportional_sample(
    towns: dict[str, list[str]], target: int, seed: int = 42
) -> dict[str, list[str]]:
    """Proportionally sample ``target`` frames across all towns.

    Uses largest-remainder allocation to guarantee the sum equals target while keeping the per-town ratio as close as possible to the original distribution.
    """
    rng = random.Random(seed)
    total = sum(len(v) for v in towns.values())

    if target >= total:
        print(
            f"[Splicer] Target ({target:,}) >= available ({total:,}). "
            f"Using all available scans."
        )

        return {t: list(s) for t, s in towns.items()}

    # Largest-remainder method (Hamilton / Hare-Niemeyer)
    quotas: dict[str, float] = {t: (len(s) / total) * target for t, s in towns.items()}
    floors: dict[str, int] = {t: int(math.floor(q)) for t, q in quotas.items()}
    remainders = {t: quotas[t] - floors[t] for t in quotas}
    leftover = target - sum(floors.values())

    # Award one extra scan to the towns with the largest remainders.
    for t in sorted(remainders, key=remainders.get, reverse=True)[:leftover]:
        floors[t] += 1

    sampled: dict[str, list[str]] = {}
    for town, stems in towns.items():
        n = floors[town]

        if n >= len(stems):
            sampled[town] = list(stems)

        else:
            sampled[town] = sorted(rng.sample(stems, n))

    return sampled


# Open3D Geometry Engine (Normal Estimation Worker)
def _estimate_normals_for_scan(
    args: tuple[str, str, str, str],
) -> str | None:
    """Worker function executed in a child process.

    Parameters
    ----------
    args : tuple (town, stem, src_npy_path, dst_npy_path)

    Returns
    -------
    dst_npy_path on success, None on failure.
    """
    town, stem, src_path, dst_path = args
    try:
        # Lazy import - Open3D is NOT fork-safe.  Each spawned worker
        # must initialise its own Open3D runtime from scratch.
        import open3d as o3d

        raw = np.load(src_path)  # (N, 4) - X Y Z I
        if raw.ndim != 2 or raw.shape[1] < 4:
            print(f"[WARN] Skipping {src_path}: unexpected shape {raw.shape}")
            return None

        xyz = raw[:, :3].astype(np.float64)
        intensity = raw[:, 3:4]  # keep as column

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # KD-Tree hybrid search - radius + max neighbours
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=NORMAL_SEARCH_RADIUS,
                max_nn=NORMAL_MAX_NN,
            )
        )
        # Orient normals towards sensor origin (ego vehicle position)
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

        normals = np.asarray(pcd.normals).astype(np.float32)  # (N, 3)

        # Concatenate -> (N, 7): X Y Z I Nx Ny Nz
        cloud_7d = np.hstack(
            [
                raw[:, :4].astype(np.float32),
                normals,
            ]
        )

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        np.save(dst_path, cloud_7d)
        return dst_path

    except Exception as exc:
        print(f"[ERROR] {town}/{stem}: {exc}")
        return None


def process_normals_parallel(
    data_root: Path,
    out_dir: Path,
    sampled: dict[str, list[str]],
    workers: int,
) -> dict[str, list[str]]:
    """Launch a multiprocessing pool to compute normals for all sampled scans.

    Returns
    -------
    {town: [stems that succeeded]}.
    """

    tasks: list[tuple[str, str, str, str]] = []
    for town, stems in sampled.items():
        for stem in stems:
            src = str(data_root / town / "lidar" / f"{stem}.npy")
            dst = str(out_dir / town / "lidar_7d" / f"{stem}.npy")
            tasks.append((town, stem, src, dst))

    total = len(tasks)
    print(f"[Geometry] Dispatching {total:,} scans across {workers} workers ...")

    succeeded: dict[str, list[str]] = {t: [] for t in sampled}
    done = 0

    # Use 'spawn' context - Open3D is NOT fork-safe.  'spawn' starts
    # fresh interpreter processes instead of forking, so each child
    # initialises Open3D cleanly without inheriting corrupted state.
    ctx = __import__('multiprocessing').get_context('spawn')
    with ctx.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(
            _estimate_normals_for_scan, tasks, chunksize=64
        ):
            done += 1
            if done % 5000 == 0 or done == total:
                print(f"  [{done:>7,} / {total:,}]")
            if result is not None:
                # Recover town/stem from path
                p = Path(result)
                town = p.parent.parent.name
                stem = p.stem
                succeeded[town].append(stem)

    # Sort for deterministic PKL ordering
    for t in succeeded:
        succeeded[t].sort()

    ok = sum(len(v) for v in succeeded.values())
    print(f"[Geometry] Completed - {ok:,} / {total:,} succeeded.")
    return succeeded


#  3. Occlusion Filter + Label Parser
def parse_label(
    label_path: str | Path,
) -> tuple[list[dict[str, Any]], float | int, dict[str, int], dict[str, int]]:
    """Parse a single CARLA JSON label file.

    Returns
    -------
    objects : list[dict]
        Each dict has keys: center, size, yaw_rad, class, velocity,
        num_lidar_pts - **only** objects passing the visibility filter
        and class-name mapping.

    timestamp : float | int
        Frame timestamp from the JSON root.

    passed_counts : dict[str, int]
        Per-class (mmdet3d name) count of objects that passed the
        occlusion filter.

    filtered_counts : dict[str, int]
        Per-class (mmdet3d name) count of objects that were removed
        by the occlusion filter or class-name mapping.
    """
    with open(label_path, "r") as fh:
        data = json.load(fh)

    ego_rot = data.get("ego_transform", {}).get("rotation", [0.0, 0.0, 0.0])
    if abs(ego_rot[0]) > 2.0 or abs(ego_rot[1]) > 2.0:
        return [], 0.0, {}, {} 

    timestamp = data.get("timestamp", 0.0)
    raw_objects: list[dict] = data.get("objects", [])

    filtered: list[dict[str, Any]] = []
    passed_counts: dict[str, int] = {}
    filtered_counts: dict[str, int] = {}

    for obj in raw_objects:
        raw_class = str(obj.get("class", ""))
        mmdet_class = NUSCENES_TO_MMDET3D.get(raw_class)

        if mmdet_class is None:
            # Unmapped class - skip silently
            continue

        if obj.get("num_lidar_pts", 0) < MIN_LIDAR_PTS:
            filtered_counts[mmdet_class] = filtered_counts.get(mmdet_class, 0) + 1
            continue  # occluded / phantom - hard skip

        # Re-write class to mmdet3d short name
        obj = dict(obj)  # shallow copy to avoid mutating the original
        obj["class"] = mmdet_class
        filtered.append(obj)
        passed_counts[mmdet_class] = passed_counts.get(mmdet_class, 0) + 1

    return filtered, timestamp, passed_counts, filtered_counts


#  4. Blueprint Profiler
def compute_blueprint_medians(data_root: Path) -> dict[str, float]:
    """Calculate the median height of each dynamic blueprint across the entire dataset."""
    import statistics
    from tqdm import tqdm

    all_paths = []
    for town_dir in data_root.iterdir():
        if not town_dir.is_dir():
            continue
        label_dir = town_dir / "labels"
        if label_dir.exists():
            all_paths.extend(list(label_dir.glob("*.json")))

    bp_heights: dict[str, list[float]] = {}
    
    print(f"\n[Profiler] Scanning {len(all_paths):,} frames for blueprint height medians...")
    
    for path in tqdm(all_paths, desc="Profiling Blueprints"):
        if not path.exists():
            continue
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue
            
        for obj in data.get("objects", []):
            bp = str(obj.get("blueprint", ""))
            # Skip static bounds which have massive variance
            if not bp or bp == "level_bb_vehicle":
                continue
                
            # Collector exports full dimensions (no half-extents)
            h = float(obj.get("size", [0, 0, 0])[2])
            if h <= 0:
                continue
                
            if bp not in bp_heights:
                bp_heights[bp] = []
            bp_heights[bp].append(h)
            
    bp_medians = {bp: statistics.median(heights) for bp, heights in bp_heights.items()}
    print(f"[Profiler] Computed medians for {len(bp_medians)} dynamic blueprints.")
    return bp_medians


#  5. MMDET3D PKL Builder

# Official nuScenes category ordering (from reference nuscenes_infos_train.pkl)
# This MUST match the config's class_names list and the NuScenesDataset expectations.
CLASS_NAMES: list[str] = [
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
]
CATEGORIES: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Dummy 4x4 identity SE(3) transform - CARLA LiDAR is already in ego frame
# and we don't track a global pose, so identity is the correct placeholder.
IDENTITY_4x4: list[list[float]] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

# Empty camera instance dict - 6 cameras, all empty (LiDAR-only pipeline)
EMPTY_CAM_INSTANCES: dict[str, list] = {
    "CAM_FRONT": [], "CAM_FRONT_RIGHT": [], "CAM_FRONT_LEFT": [],
    "CAM_BACK": [], "CAM_BACK_LEFT": [], "CAM_BACK_RIGHT": [],
}


def build_dataset_list(
    data_root: Path,
    out_dir: Path,
    succeeded: dict[str, list[str]],
    bp_medians: dict[str, float],
    cleanup: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Compile all valid frames into a single mmdet3d v2-format info dict
    and serialise it as a pickle file.

    The output is *structurally identical* to the official nuScenes
    ``nuscenes_infos_{train,val}.pkl`` so that ``NuScenesDataset`` can
    load it without modification.

    Layout::

        {
            "metainfo": {
                "categories": {"car": 0, ...},
                "dataset":    "nuscenes",
                "version":    "v1.0-trainval",
                "info_version": "1.1",
            },
            "data_list": [
                {
                    "sample_idx": int,
                    "token":      str,
                    "timestamp":  float,
                    "ego2global": [[4x4 floats]],
                    "images":     {},
                    "lidar_points": {
                        "lidar_path":    str,
                        "num_pts_feats": 7,
                        "lidar2ego":     [[4x4 floats]],
                    },
                    "lidar_sweeps": [],
                    "instances": [
                        {
                            "bbox_label":     int,
                            "bbox_3d":        [7 floats],
                            "bbox_3d_isvalid": True,
                            "bbox_label_3d":  int,
                            "num_lidar_pts":  int,
                            "num_radar_pts":  0,
                            "velocity":       [vx, vy],
                        }, ...
                    ],
                    "cam_instances": {"CAM_FRONT": [], ...},
                }, ...
            ],
        }

    Geometric sanity filters
    ~~~~~~~~~~~~~~~~~~~~~~~~
    After the occlusion filter (``parse_label``), each surviving bounding
    box is checked against dimension thresholds.  If *any* box in a scan
    has impossible dimensions, the **entire scan is poisoned** and dropped.

    Returns
    -------
    data_list : list[dict]
        The raw dataset list containing pristine frames.
    stats : dict[str, int]
    stats : dict[str, int]
        Telemetry counters: ``total_processed``, ``nuked_large``,
        ``nuked_small``, ``pristine``.
    """
    class_to_label = CATEGORIES

    data_list: list[dict[str, Any]] = []
    skipped = 0
    empty_label_scans = 0  # frames where all objects were filtered out

    # Telemetry counters (Task 3)
    total_processed = 0
    nuked_large = 0
    nuked_small = 0

    # Aggregate class-wise occlusion stats across all frames
    total_passed: dict[str, int] = {}
    total_filtered: dict[str, int] = {}

    sample_idx = 0
    for town in sorted(succeeded):
        label_dir = data_root / town / "labels"
        for stem in succeeded[town]:
            label_path = label_dir / f"{stem}.json"
            if not label_path.exists():
                skipped += 1
                continue

            total_processed += 1

            # Step 1: Occlusion filter (inside parse_label)
            objects, timestamp, passed_counts, filtered_counts = parse_label(label_path)

            # Accumulate class-wise occlusion stats
            for cls, cnt in passed_counts.items():
                total_passed[cls] = total_passed.get(cls, 0) + cnt
            for cls, cnt in filtered_counts.items():
                total_filtered[cls] = total_filtered.get(cls, 0) + cnt

            # Step 2: Geometric dimension filter (poisoned-scan logic)
            poisoned = False
            poison_reason = ""
            scan_classes = {}
            scan_blueprints = {}
            
            for obj in objects:
                w, l, h = obj["size"]  # width, length, height

                obj["size"] = [w, l, h]
                
                class_name = str(obj.get("class", ""))
                blueprint = str(obj.get("blueprint", ""))
                
                if class_name:
                    scan_classes[class_name] = scan_classes.get(class_name, 0) + 1
                if blueprint:
                    scan_blueprints[blueprint] = scan_blueprints.get(blueprint, 0) + 1
                    
                HEIGHT_TOLERANCE = 0.4

                # Task 2: Class-Isolated Filtering with Blueprint Profiler
                if blueprint == "level_bb_vehicle":
                    # Static Vehicle Bypass
                    if h < 0.7 or h > 3.5:
                        poisoned = True
                        poison_reason = "large" if h > 3.5 else "small"
                        break
                elif class_name in ["car", "truck", "bus"]:
                    # Dynamic Vehicle Median Check
                    expected_median = bp_medians.get(blueprint)
                    if expected_median is not None:
                        if abs(h - expected_median) > HEIGHT_TOLERANCE:
                            poisoned = True
                            poison_reason = "large" if h > expected_median else "small"
                            break
                    else:
                        # Fallback for unknown dynamic blueprints
                        if h < 0.7 or h > 3.5:
                            poisoned = True
                            poison_reason = "large" if h > 3.5 else "small"
                            break

            if poisoned:
                if poison_reason == "large":
                    nuked_large += 1
                else:
                    nuked_small += 1
                if cleanup:
                    try:
                        os.remove(out_dir / town / "lidar_7d" / f"{stem}.npy")
                    except FileNotFoundError:
                        pass
                continue  # skip this entire scan

            # Step 3: Build per-object instance list (nuScenes-identical format)
            # Relative path from out_dir so the dataloader can join it
            lidar_rel = str(Path(town) / "lidar_7d" / f"{stem}.npy")
            token = f"{town}__{stem}"

            instances: list[dict[str, Any]] = []
            for obj in objects:
                cx, cy, cz = obj["center"]
                w, l, h = obj["size"]
                yaw = float(obj["yaw_rad"])
                vel = obj.get("velocity", [0.0, 0.0, 0.0])
                vx = float(vel[0]) if len(vel) > 0 else 0.0
                vy = float(vel[1]) if len(vel) > 1 else 0.0

                label_idx = class_to_label.get(str(obj["class"]), -1)
                instances.append({
                    "bbox_label": label_idx,
                    "bbox_3d": [cx, cy, cz, l, w, h, yaw],
                    "bbox_3d_isvalid": True,
                    "bbox_label_3d": label_idx,
                    "num_lidar_pts": int(obj["num_lidar_pts"]),
                    "num_radar_pts": 0,
                    "velocity": [vx, vy],
                })

            data_list.append({
                "token": token,
                "timestamp": timestamp,
                "ego2global": IDENTITY_4x4,
                "images": {},
                "lidar_points": {
                    "lidar_path": lidar_rel,
                    "num_pts_feats": 7,
                    "lidar2ego": IDENTITY_4x4,
                },
                "lidar_sweeps": [],
                "instances": instances,
                "cam_instances": EMPTY_CAM_INSTANCES,
                "__stats__": {
                    "classes": scan_classes,
                    "blueprints": scan_blueprints
                }
            })

            if len(instances) == 0:
                empty_label_scans += 1
                if cleanup:
                    try:
                        os.remove(out_dir / town / "lidar_7d" / f"{stem}.npy")
                    except FileNotFoundError:
                        pass

    # -- Class-wise occlusion filter summary ----------------------------------
    all_classes = sorted(set(total_passed) | set(total_filtered))
    if all_classes:
        print(f"\n{'-' * 64}")
        print(f"  Occlusion Filter Summary  (min_lidar_pts >= 5)")
        print(f"{'-' * 64}")
        print(f"  {'Class':<25s} {'Passed':>9s} {'Filtered':>10s} {'Filter %':>9s}")
        print(f"  {'-' * 55}")
        grand_passed = 0
        grand_filtered = 0
        for cls in all_classes:
            p = total_passed.get(cls, 0)
            f = total_filtered.get(cls, 0)
            grand_passed += p
            grand_filtered += f
            pct = 100.0 * f / (p + f) if (p + f) > 0 else 0.0
            print(f"  {cls:<25s} {p:>9,d} {f:>10,d} {pct:>8.1f}%")
        grand_total = grand_passed + grand_filtered
        grand_pct = 100.0 * grand_filtered / grand_total if grand_total > 0 else 0.0
        print(f"  {'-' * 55}")
        print(f"  {'TOTAL':<25s} {grand_passed:>9,d} {grand_filtered:>10,d} {grand_pct:>8.1f}%")
        print(f"{'-' * 64}")

    stats = {
        "total_processed": total_processed,
        "nuked_large": nuked_large,
        "nuked_small": nuked_small,
        "pristine": len(data_list),
        "skipped": skipped,
        "empty_label_scans": empty_label_scans
    }
    return data_list, stats


def write_pkl(
    data_list: list[dict[str, Any]], 
    out_path: Path, 
    split_name: str
) -> dict[str, Any]:
    """Assigns sequential sample_idx, builds the metadata wrapper, and writes the PKL."""
    class_counts = {}
    bp_counts = {}
    
    new_data_list = []
    for i, item in enumerate(data_list):
        # Pop out the temporary telemetry stats
        stats = item.pop("__stats__", {})
        for c, count in stats.get("classes", {}).items():
            class_counts[c] = class_counts.get(c, 0) + count
        for bp, count in stats.get("blueprints", {}).items():
            bp_counts[bp] = bp_counts.get(bp, 0) + count
            
        # Rebuild dict so sample_idx is explicitly the first key (preserves official order)
        new_item = {"sample_idx": i}
        new_item.update(item)
        new_data_list.append(new_item)

    pkl_data = {
        "metainfo": {
            "categories": CATEGORIES,
            "dataset": "nuscenes",
            "version": "v1.0-trainval",
            "info_version": "1.1",
        },
        "data_list": new_data_list,
    }

    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(pkl_data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    total_boxes = sum(len(entry["instances"]) for entry in new_data_list)
    print(f"[PKL] Saved {len(new_data_list):,} frames ({total_boxes:,} GT boxes) -> {out_path}")
    
    return {
        "scans": len(new_data_list),
        "boxes": total_boxes,
        "classes": class_counts,
        "blueprints": bp_counts
    }


#  CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Geo-Pillars offline preprocessor: CARLA -> mmdet3d nuScenes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing Town*/ subfolders with lidar/ and labels/.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for 7D point clouds and the info PKL.",
    )
    p.add_argument(
        "--target_scans",
        type=int,
        default=100_000,
        help="Total number of scans to sample across all towns.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of multiprocessing workers. 0 = cpu_count().",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible proportional sampling.",
    )
    p.add_argument(
        "--pkl_name",
        type=str,
        default="nuscenes_infos_train.pkl",
        help="Filename of the output info pickle.",
    )
    p.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data for training (remainder is validation). E.g. 0.8 = 80/20.",
    )
    p.add_argument(
        "--cleanup_poisoned",
        action="store_true",
        help="Delete 7D .npy files for scans that are poisoned by filters or empty.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    workers = args.workers if args.workers > 0 else cpu_count()

    print("=" * 72)
    print("  Geo-Pillars Offline Preprocessor")
    print("=" * 72)
    print(f"  data_root    : {data_root}")
    print(f"  out_dir      : {out_dir}")
    print(f"  target_scans : {args.target_scans:,}")
    print(f"  workers      : {workers}")
    print(f"  seed         : {args.seed}")
    print("=" * 72)

    # Discover all
    towns = discover_towns(data_root)
    if not towns:
        raise FileNotFoundError(
            f"No valid town directories found under {data_root}. "
            "Expected Town*/lidar/ and Town*/labels/ subfolders."
        )
    total_available = sum(len(v) for v in towns.values())
    print(
        f"\n[Splicer] Discovered {len(towns)} towns, "
        f"{total_available:,} total frames:"
    )
    for t in sorted(towns):
        print(f"  {t:>12s}  {len(towns[t]):>8,} frames")

    # Pass 1: Blueprint Profiler
    bp_medians = compute_blueprint_medians(data_root)

    # Pass 2: Filter pristine scans before sampling
    # We run build_dataset_list on ALL available frames to drop poisoned/empty ones.
    print(f"\n[Filter] Scanning all {total_available:,} frames to find pristine subset...")
    data_list, global_stats = build_dataset_list(
        data_root, out_dir, towns, bp_medians, cleanup=False
    )

    # Reconstruct towns dict containing only pristine frames
    pristine_towns: dict[str, list[str]] = {}
    for item in data_list:
        token_parts = item["token"].split("__")
        t = token_parts[0]
        s = token_parts[1]
        if t not in pristine_towns:
            pristine_towns[t] = []
        pristine_towns[t].append(s)

    pristine_available = sum(len(v) for v in pristine_towns.values())
    print(f"\n[Filter] {pristine_available:,} pristine frames survived filtering.")

    # Pass 3: Proportional sample from the pristine subset
    sampled = proportional_sample(pristine_towns, args.target_scans, seed=args.seed)
    total_sampled = sum(len(v) for v in sampled.values())
    print(f"\n[Splicer] Sampled {total_sampled:,} pristine frames for geometry processing:")
    for t in sorted(sampled):
        pct = 100 * len(sampled[t]) / total_sampled if total_sampled else 0
        print(f"  {t:>12s}  {len(sampled[t]):>8,} ({pct:5.1f}%)")

    # Pass 4: Compute normals (Geometry Engine) ONLY for the sampled pristine frames
    succeeded = process_normals_parallel(data_root, out_dir, sampled, workers)

    # Filter the data_list to strictly those that succeeded in normal computation
    valid_tokens = {f"{t}__{s}" for t, stems in succeeded.items() for s in stems}
    final_data_list = [item for item in data_list if item["token"] in valid_tokens]

    # -- 80 / 20 Train / Val split -----------------------------------------
    split_rng = random.Random(args.seed)
    split_rng.shuffle(final_data_list)
    split_idx = int(len(final_data_list) * args.train_split)
    train_list = final_data_list[:split_idx]
    val_list = final_data_list[split_idx:]

    print(
        f"\n[Split] {len(final_data_list):,} pristine and processed frames -> "
        f"{len(train_list):,} train / {len(val_list):,} val"
    )

    # Pass 5: Write PKLs and telemetry
    train_stats = write_pkl(train_list, out_dir / "nuscenes_infos_train.pkl", "train")
    val_stats = write_pkl(val_list, out_dir / "nuscenes_infos_val.pkl", "val")
    
    # Save statistics JSON
    final_stats = {
        "train": train_stats,
        "val": val_stats,
        "total": {
            "scans": train_stats["scans"] + val_stats["scans"],
            "boxes": train_stats["boxes"] + val_stats["boxes"],
            "classes": {k: train_stats["classes"].get(k, 0) + val_stats["classes"].get(k, 0) for k in set(train_stats["classes"]) | set(val_stats["classes"])},
            "blueprints": {k: train_stats["blueprints"].get(k, 0) + val_stats["blueprints"].get(k, 0) for k in set(train_stats["blueprints"]) | set(val_stats["blueprints"])}
        }
    }
    stats_json_path = out_dir / "dataset_statistics.json"
    with open(stats_json_path, "w") as f:
        json.dump(final_stats, f, indent=4)
    print(f"\n[Telemetry] Full dataset statistics written to {stats_json_path}")

    # -- Dataset Yield Report (Task 3) ----------------------------------------
    t_proc = global_stats["total_processed"]
    t_large = global_stats["nuked_large"]
    t_small = global_stats["nuked_small"]
    t_prist = len(final_data_list)

    print("\n" + "=" * 72)
    print("  Geo-Pillars Dataset Yield Report")
    print("=" * 72)
    print(f"  {'Scans processed:':<30s} {t_proc:>10,d}")
    print(f"  {'Nuked (large/mislabeled):':<30s} {t_large:>10,d}")
    print(f"  {'Nuked (collapsed/incline):':<30s} {t_small:>10,d}")
    print(f"  {'Pristine (saved):':<30s} {t_prist:>10,d}")
    print("=" * 72)
    print(f"  7D clouds  : {out_dir}/*/lidar_7d/")
    print(f"  Train PKL  : {out_dir / 'nuscenes_infos_train.pkl'}")
    print(f"  Val PKL    : {out_dir / 'nuscenes_infos_val.pkl'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
