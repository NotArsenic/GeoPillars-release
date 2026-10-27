"""What the NGFM auxiliary task is actually supervised on.

Normals are estimated on the whole scene before any class filtering
(``offline_processor.py:137-193``), and ``ScatterNormalsToBEV`` scatters every
point's normal, while ``ObjectNameFilter(['car'])`` has already removed the
truck / bus / pedestrian *boxes* from the detection target. The auxiliary loss
therefore supervises geometry the detector is told to ignore.

This runs run A's real train pipeline (augmentation included) with
``ObjectNameFilter`` removed so all four annotated classes survive, rebuilds the
bottleneck ``valid_mask`` exactly as ``NormalTargetPreparer`` does, and
classifies every supervised 4x4 patch by what its *occupied* full-resolution
cells contain: car points, non-car object points, or neither.

Because ``density_map`` is identically 1 in this configuration (the BEV prior
makes ``occ_from_bev`` all-ones), the NGFM loss is an unweighted mean over
valid patches - so the fraction of valid patches in a category is exactly the
fraction of the auxiliary loss spent on it.

Writes ``figures/data/ngfm_target_composition.json``.
"""

import argparse
import json
import os.path as osp
import sys
from collections import Counter

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.join(REPO_ROOT, "tools"))

from _carla_infer import CARLA_ROOT  # noqa: E402

CLASSES = ["car", "truck", "bus", "pedestrian"]
STRIDE = 4  # NormalTargetPreparer(stride=4) == backbone stride
MIN_OCCUPANCY = 0.1  # NormalTargetPreparer(min_occupancy=0.1)
SCATTER_MIN_POINTS = 3  # ScatterNormalsToBEV.min_points_per_pillar default


def build_dataset(ann_file="nuscenes_infos_train.pkl"):
    """Run A's train pipeline, with ObjectNameFilter dropped and all 4 classes kept."""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet3d.registry import DATASETS
    import src.data.transforms  # noqa: F401  (registers LoadPointsFromNpy etc.)
    from eval_test_checkpoints import find_config

    init_default_scope("mmdet3d")

    cfg = Config.fromfile(find_config("geo_pillars_h100_intensity"))
    ds = cfg.train_dataloader.dataset
    pipeline = [t for t in ds.pipeline if t["type"] != "ObjectNameFilter"]
    kept = [t["type"] for t in pipeline]
    assert "ScatterNormalsToBEV" in kept and "GlobalRotScaleTrans" in kept
    # stop after the scatter: Pack* would drop the extra classes' boxes
    pipeline = pipeline[: kept.index("ScatterNormalsToBEV") + 1]
    scatter = [t for t in pipeline if t["type"] == "ScatterNormalsToBEV"][0]

    dataset = DATASETS.build(
        dict(
            type=ds.type,
            data_root=CARLA_ROOT,
            ann_file=ann_file,
            data_prefix=ds.data_prefix,
            modality=ds.modality,
            box_type_3d=ds.box_type_3d,
            test_mode=False,
            metainfo=dict(classes=CLASSES),
            pipeline=pipeline,
        )
    )
    return dataset, scatter, [t["type"] for t in pipeline]


def valid_mask_from_occupancy(occ):
    """``NormalTargetPreparer`` bottleneck validity: avg-pool then threshold."""
    H, W = occ.shape
    p = occ.reshape(H // STRIDE, STRIDE, W // STRIDE, STRIDE)
    occ_pooled = p.mean(axis=(1, 3))  # avg_pool2d(k=s, s=s)
    valid_count = occ_pooled * STRIDE**2
    return valid_count > MIN_OCCUPANCY * STRIDE**2, valid_count


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n-scans", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="nuscenes_infos_train.pkl")
    ap.add_argument(
        "--out",
        default=osp.join(REPO_ROOT, "figures", "data", "ngfm_target_composition.json"),
    )
    args = ap.parse_args()

    from mmdet3d.structures.ops import box_np_ops

    dataset, scatter_cfg, pipeline = build_dataset(args.split)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(dataset), size=args.n_scans, replace=False)
    pick.sort()
    print(
        f"[ngfm] {len(dataset)} scans in {args.split}; sampling {args.n_scans}\n"
        f'[ngfm] pipeline: {">".join(pipeline)}',
        flush=True,
    )

    tot = Counter()
    cell_hist = Counter()  # occupied cells per patch, by category
    patch_cellmix = Counter()  # (n_car_cells, n_noncar_cells, n_bg_cells) coarse
    per_class_patch = Counter()
    n_scans_used = 0
    occ_frac, valid_frac = [], []

    for n, di in enumerate(pick):
        # bypass Det3DDataset.prepare_data: without a Pack transform it would
        # look for example['data_samples'] to run its empty-GT filter
        res = dataset.pipeline(dataset.get_data_info(int(di)))
        pts = res["points"].tensor.numpy()[:, :3].astype(np.float32)
        occ = res["occupancy_mask"][0].astype(np.float32)
        boxes = res["gt_bboxes_3d"].tensor.numpy()[:, :7].astype(np.float32)
        labels = np.asarray(res["gt_labels_3d"])
        H, W = occ.shape

        valid, _ = valid_mask_from_occupancy(occ)
        occ_frac.append(float(occ.mean()))
        valid_frac.append(float(valid.mean()))

        # point -> full-resolution cell (same binning as ScatterNormalsToBEV)
        col = np.floor(
            (pts[:, 0] - scatter_cfg["point_cloud_range"][0])
            / scatter_cfg["voxel_size"][0]
        ).astype(np.int64)
        row = np.floor(
            (pts[:, 1] - scatter_cfg["point_cloud_range"][1])
            / scatter_cfg["voxel_size"][1]
        ).astype(np.int64)
        ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)

        # which points sit inside which boxes (mmdet3d's own convention)
        if len(boxes):
            inb = box_np_ops.points_in_rbbox(
                pts.astype(np.float64), boxes.astype(np.float64), origin=(0.5, 0.5, 0)
            )
        else:
            inb = np.zeros((len(pts), 0), dtype=bool)
        is_car = (
            inb[:, labels == 0].any(axis=1)
            if (labels == 0).any()
            else np.zeros(len(pts), bool)
        )
        noncar_cols = labels > 0
        is_noncar = (
            inb[:, noncar_cols].any(axis=1)
            if noncar_cols.any()
            else np.zeros(len(pts), bool)
        )

        # per full-resolution cell: does it hold any car / non-car point
        flat = (row * W + col)[ok]
        car_cell = np.zeros(H * W, bool)
        noncar_cell = np.zeros(H * W, bool)
        np.logical_or.at(car_cell, flat, is_car[ok])
        np.logical_or.at(noncar_cell, flat, is_noncar[ok])
        car_cell = car_cell.reshape(H, W) & (occ > 0)
        noncar_cell = noncar_cell.reshape(H, W) & (occ > 0)
        occ_b = occ > 0

        def pool_any(m):
            return m.reshape(H // STRIDE, STRIDE, W // STRIDE, STRIDE).any(axis=(1, 3))

        def pool_sum(m):
            return m.reshape(H // STRIDE, STRIDE, W // STRIDE, STRIDE).sum(axis=(1, 3))

        p_car, p_noncar = pool_any(car_cell), pool_any(noncar_cell)
        n_car_c, n_noncar_c = pool_sum(car_cell), pool_sum(noncar_cell)
        n_occ_c = pool_sum(occ_b)

        v = valid
        tot["valid_patches"] += int(v.sum())
        tot["patches_total"] += v.size
        tot["with_car"] += int((v & p_car).sum())
        tot["no_car_some_noncar"] += int((v & ~p_car & p_noncar).sum())
        tot["background_only"] += int((v & ~p_car & ~p_noncar).sum())
        tot["with_car_and_noncar"] += int((v & p_car & p_noncar).sum())
        tot["occupied_cells_in_valid_patches"] += int(n_occ_c[v].sum())
        tot["car_cells_in_valid_patches"] += int(n_car_c[v].sum())
        tot["noncar_cells_in_valid_patches"] += int(n_noncar_c[v].sum())

        # histogram of how many of the (up to 16) occupied cells are car / non-car
        for c, nc, o in zip(n_car_c[v], n_noncar_c[v], n_occ_c[v]):
            cell_hist[f"car_cells={int(c)}"] += 1
            cell_hist[f"noncar_cells={int(nc)}"] += 1
            cell_hist[f"occupied_cells={int(o)}"] += 1
            patch_cellmix[f"{int(c)}car_{int(nc)}noncar"] += 1

        # which non-car classes drive it
        for cls_id, name in enumerate(CLASSES):
            if cls_id == 0 or not (labels == cls_id).any():
                continue
            cols = labels == cls_id
            hit = np.zeros(len(pts), bool)
            hit |= inb[:, cols].any(axis=1)
            cc = np.zeros(H * W, bool)
            np.logical_or.at(cc, flat, hit[ok])
            cc = cc.reshape(H, W) & occ_b
            per_class_patch[name] += int((v & pool_any(cc)).sum())
        n_scans_used += 1
        if (n + 1) % 50 == 0:
            vp = tot["valid_patches"]
            print(
                f"[ngfm] {n + 1}/{args.n_scans} scans  valid patches={vp}  "
                f'no-car={tot["no_car_some_noncar"] + tot["background_only"]} '
                f'({100 * (tot["no_car_some_noncar"] + tot["background_only"]) / max(vp, 1):.1f}%)',
                flush=True,
            )

    vp = tot["valid_patches"]
    no_car = tot["no_car_some_noncar"] + tot["background_only"]
    out = dict(
        description=(
            "Composition of the NGFM auxiliary target over supervised 4x4 "
            "bottleneck patches (valid_mask == True). Run A train pipeline with "
            "ObjectNameFilter removed so truck/bus/pedestrian boxes survive; "
            "cells are classified only where occupied (count >= "
            f"{SCATTER_MIN_POINTS}), since only occupied cells contribute normals."
        ),
        settings=dict(
            n_scans=n_scans_used,
            seed=args.seed,
            split=args.split,
            pipeline=pipeline,
            classes=CLASSES,
            stride=STRIDE,
            min_occupancy=MIN_OCCUPANCY,
            min_points_per_pillar=SCATTER_MIN_POINTS,
            grid="256x256 full resolution, 64x64 bottleneck",
            loss_weighting="unweighted mean over valid patches "
            "(density_map is identically 1)",
        ),
        counts=dict(tot),
        fractions_of_supervised_patches=dict(
            any_car_points=round(tot["with_car"] / max(vp, 1), 6),
            no_car_but_noncar_object_points=round(
                tot["no_car_some_noncar"] / max(vp, 1), 6
            ),
            background_only=round(tot["background_only"] / max(vp, 1), 6),
            car_and_noncar_mixed=round(tot["with_car_and_noncar"] / max(vp, 1), 6),
            no_car_geometry_at_all=round(no_car / max(vp, 1), 6),
        ),
        headline=dict(
            percent_of_auxiliary_loss_with_no_car_geometry=round(
                100.0 * no_car / max(vp, 1), 2
            ),
            percent_of_auxiliary_loss_on_patches_touching_a_car=round(
                100.0 * tot["with_car"] / max(vp, 1), 2
            ),
        ),
        occupied_cell_composition=dict(
            car_cells_share=round(
                tot["car_cells_in_valid_patches"]
                / max(tot["occupied_cells_in_valid_patches"], 1),
                6,
            ),
            noncar_cells_share=round(
                tot["noncar_cells_in_valid_patches"]
                / max(tot["occupied_cells_in_valid_patches"], 1),
                6,
            ),
        ),
        noncar_class_patch_counts=dict(per_class_patch),
        noncar_class_patch_fractions={
            k: round(v / max(vp, 1), 6) for k, v in per_class_patch.items()
        },
        histograms=dict(
            cells_per_patch=dict(sorted(cell_hist.items())),
            car_noncar_cell_mix=dict(
                sorted(patch_cellmix.items(), key=lambda kv: -kv[1])[:40]
            ),
        ),
        grid_occupancy=dict(
            mean_fraction_cells_occupied=round(float(np.mean(occ_frac)), 6),
            mean_fraction_patches_supervised=round(float(np.mean(valid_frac)), 6),
        ),
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "counts",
                    "fractions_of_supervised_patches",
                    "headline",
                    "occupied_cell_composition",
                    "noncar_class_patch_fractions",
                    "grid_occupancy",
                )
            },
            indent=2,
        )
    )
    print(f"[ngfm] -> {args.out}")


if __name__ == "__main__":
    main()
