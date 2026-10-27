#!/usr/bin/env python3
"""Pick CARLA test scans that show a heading difference and export them as NPZ (Fig. 8, row 1).

Stages:
  1. Pool: CARLA test scans with >= ``--min_far`` GT cars at 20-50 m, a seeded
     random sample of ``--pool_size`` of them.
  2. Inference with the released epoch_8 checkpoints (A, C, D, F, G) on the pool
     only, same dumped test configs as the evaluation.
  3. Per scan, match detections (score >= ``--score_thr``) to GT with the
     GeoPillarsMetric rule at IoU 0.5 (the mAOE threshold) and take the yaw
     error of every matched car, folded to [0, pi] as in the metric.
  4. Rank scans by the number of mid/far cars (>= 20 m) that A and D both
     detect where A's heading error is < ``--good_deg`` and D's is
     > ``--bad_deg``, minus the reverse case; one scan per town/sequence
     window. Export the top ``--n_scenes`` and ``--n_alternates`` more.

Writes ``figures/data/carla_qualitative/scene_<token>.npz`` (same keys as the
nuScenes scene files) and ``candidates.json``.

Usage (repo root)::

    PYTHONPATH=$(pwd) python scripts/figure_data/export_carla_qualitative.py
"""

import argparse
import json
import os
import os.path as osp
import pickle
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from _carla_infer import (
    CARLA_ROOT,
    CARLA_TEST_PKL,
    REPO_ROOT,
    RUNS,
    build,
)  # noqa: E402

MODELS = {  # NPZ key -> run letter (keys match the nuScenes scene files)
    "geopillars": "A",
    "centerpoint_budget_osf4": "D",
    "centerpoint_osf1": "F",
    "centerpoint_native": "C",
    "cp_encoder_replk": "G",
}


def child_infer(run, indices_path, out_path):
    import torch

    indices = np.load(indices_path)
    runner, model, loader, cfg = build(run, indices=indices)
    sample_idx, frames, boxes, scores, labels = [], [], [], [], []
    with open(osp.join(CARLA_ROOT, CARLA_TEST_PKL), "rb") as f:
        rel_to_idx = {
            i["lidar_points"]["lidar_path"]: i["sample_idx"]
            for i in pickle.load(f)["data_list"]
        }
    with torch.no_grad():
        for data_batch in loader:
            for ds in model.test_step(data_batch):
                idx = rel_to_idx[osp.relpath(ds.lidar_path, CARLA_ROOT)]
                p = ds.pred_instances_3d
                t = p.bboxes_3d.tensor[:, :7].cpu().numpy().copy()
                t[:, 2] = (
                    p.bboxes_3d.gravity_center[:, 2].cpu().numpy()
                )  # bottom -> gravity centre, as in the pkl
                sample_idx.append(np.full(len(t), idx, np.int32))
                frames.append(idx)
                boxes.append(t.astype(np.float32))
                scores.append(p.scores_3d.cpu().numpy().astype(np.float32))
                labels.append(p.labels_3d.cpu().numpy().astype(np.int16))
    assert sorted(frames) == sorted(int(i) for i in indices)
    np.savez_compressed(
        out_path,
        sample_idx=np.concatenate(sample_idx),
        boxes=np.concatenate(boxes),
        scores=np.concatenate(scores),
        labels=np.concatenate(labels),
        frames=np.asarray(frames, np.int64),
    )
    print(f"[infer] {run}: {len(frames)} frames", flush=True)


def match(pred_boxes, pred_scores, gt_boxes, thr):
    """GeoPillarsMetric._eval_class matching for one scan -> (pred_tp, gt_matched, gt_heading_err_rad)."""
    from src.metrics.geo_pillars_metric import _angle_diff, compute_bev_iou_matrix

    pred_tp = np.zeros(len(pred_boxes), bool)
    gt_matched = np.zeros(len(gt_boxes), bool)
    err = np.full(len(gt_boxes), np.nan)
    if len(pred_boxes) and len(gt_boxes):
        iou = compute_bev_iou_matrix(
            pred_boxes.astype(np.float64), gt_boxes.astype(np.float64)
        )
        for i in np.argsort(-pred_scores, kind="stable"):
            j = int(np.argmax(iou[i]))
            if iou[i, j] >= thr and not gt_matched[j]:
                pred_tp[i] = gt_matched[j] = True
                err[j] = _angle_diff(float(pred_boxes[i, 6]), float(gt_boxes[j, 6]))
    return pred_tp, gt_matched, err


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--infer", nargs=3, help=argparse.SUPPRESS)
    ap.add_argument(
        "--out_dir", default=osp.join(REPO_ROOT, "figures", "data", "carla_qualitative")
    )
    ap.add_argument(
        "--cache_dir",
        default=osp.join(REPO_ROOT, "work_dirs", "figure_data", "carla_qualitative"),
    )
    ap.add_argument("--pool_size", type=int, default=2000)
    ap.add_argument("--min_far", type=int, default=3)
    ap.add_argument("--score_thr", type=float, default=0.25)
    ap.add_argument("--good_deg", type=float, default=15.0)
    ap.add_argument("--bad_deg", type=float, default=90.0)
    ap.add_argument("--min_frame_gap", type=int, default=1000)
    ap.add_argument("--n_scenes", type=int, default=3)
    ap.add_argument("--n_alternates", type=int, default=3)
    args = ap.parse_args()
    if args.infer:
        return child_infer(*args.infer)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    with open(osp.join(CARLA_ROOT, CARLA_TEST_PKL), "rb") as f:
        infos = pickle.load(f)
    car_id = infos["metainfo"]["categories"]["car"]
    data_list = infos["data_list"]

    def cars(info):
        inst = [
            i
            for i in info["instances"]
            if i["bbox_label_3d"] == car_id and i["bbox_3d_isvalid"]
        ]
        return (
            np.asarray([i["bbox_3d"][:7] for i in inst], np.float32).reshape(-1, 7),
            np.asarray([i["num_lidar_pts"] for i in inst], np.int32),
        )

    # 1. pool
    eligible = []
    for info in data_list:
        gt, _ = cars(info)
        r = np.hypot(gt[:, 0], gt[:, 1])
        if ((r >= 20) & (r <= 50)).sum() >= args.min_far:
            eligible.append(info["sample_idx"])
    rng = np.random.default_rng(0)
    pool = np.sort(
        rng.choice(eligible, size=min(args.pool_size, len(eligible)), replace=False)
    )
    idx_path = osp.join(args.cache_dir, "pool_indices.npy")
    np.save(idx_path, pool)
    print(f"eligible {len(eligible)} scans, pool {len(pool)}", flush=True)

    # 2. inference
    preds = {}
    for key, letter in MODELS.items():
        run = RUNS[letter]
        out = osp.join(args.cache_dir, f"preds_{run}.npz")
        if not (
            osp.isfile(out) and np.array_equal(np.sort(np.load(out)["frames"]), pool)
        ):
            env = dict(os.environ, PYTHONPATH=REPO_ROOT, WANDB_MODE="disabled")
            subprocess.run(
                [sys.executable, osp.abspath(__file__), "--infer", run, idx_path, out],
                check=True,
                cwd=REPO_ROOT,
                env=env,
            )
        d = np.load(out)
        keep = (d["labels"] == 0) & (d["scores"] >= args.score_thr)
        order = np.argsort(d["sample_idx"][keep], kind="stable")
        sidx, bx, sc = (
            d["sample_idx"][keep][order],
            d["boxes"][keep][order],
            d["scores"][keep][order],
        )
        starts = np.searchsorted(sidx, pool)
        ends = np.searchsorted(sidx, pool, side="right")
        preds[key] = {int(s): (bx[a:b], sc[a:b]) for s, a, b in zip(pool, starts, ends)}

    # 3. per-scan heading comparison
    good, bad = np.deg2rad(args.good_deg), np.deg2rad(args.bad_deg)
    table = []
    for s in pool:
        info = data_list[s]
        gt, _ = cars(info)
        far = np.hypot(gt[:, 0], gt[:, 1]) >= 20
        row = dict(
            sample_idx=int(s),
            token=info["token"],
            n_gt=len(gt),
            n_gt_ge20m=int(far.sum()),
        )
        errs = {}
        for key in MODELS:
            pb, ps = preds[key][int(s)]
            _, gm, err = match(pb, ps, gt, 0.5)
            errs[key] = err
            row[f"{key}_n_det"] = len(pb)
            row[f"{key}_recall@0.5"] = round(float(gm.mean()), 4) if len(gt) else 0.0
            row[f"{key}_mean_err_deg_ge20m"] = (
                round(float(np.rad2deg(np.nanmean(err[far]))), 2)
                if np.isfinite(err[far]).any()
                else None
            )
        a, dd = errs["geopillars"], errs["centerpoint_budget_osf4"]
        both = far & np.isfinite(a) & np.isfinite(dd)
        row["n_both_matched_ge20m"] = int(both.sum())
        row["n_A_good_D_bad_ge20m"] = int((both & (a < good) & (dd > bad)).sum())
        row["n_A_bad_D_good_ge20m"] = int((both & (a > bad) & (dd < good)).sum())
        row["score"] = row["n_A_good_D_bad_ge20m"] - row["n_A_bad_D_good_ge20m"]
        table.append(row)

    # 4. rank; keep scans far apart in the same town (consecutive frames are near-duplicates)
    table.sort(
        key=lambda r: (
            -r["score"],
            -r["n_A_good_D_bad_ge20m"],
            -r["n_both_matched_ge20m"],
            r["sample_idx"],
        )
    )
    chosen = []
    for r in table:
        town, frame = r["token"].split("__")
        if all(
            not (
                c["token"].split("__")[0] == town
                and abs(int(c["token"].split("__")[1]) - int(frame))
                < args.min_frame_gap
            )
            for c in chosen
        ):
            chosen.append(r)
        if len(chosen) == args.n_scenes + args.n_alternates:
            break

    for rank, r in enumerate(chosen):
        info = data_list[r["sample_idx"]]
        raw = np.load(osp.join(CARLA_ROOT, info["lidar_points"]["lidar_path"]))
        gt, gt_pts = cars(info)
        out = dict(
            points=raw[:, :4].astype(np.float32),
            points_note=np.asarray(
                "x y z intensity in the CARLA lidar frame (nuScenes-matched); intensity is 1.0 "
                "for every CARLA point"
            ),
            gt_boxes=gt,
            gt_num_lidar_pts=gt_pts,
            gt_ignored_boxes=np.zeros((0, 7), np.float32),
            gt_ignored_num_lidar_pts=np.zeros(0, np.int32),
            box_format=np.asarray(
                "x y z l w h yaw; lidar frame; z = box gravity centre; yaw about +z"
            ),
            gt_rule=np.asarray(
                "CARLA test GT: cars with >= 5 lidar points within 50 m (offline_processor.py)"
            ),
            score_threshold=np.float32(args.score_thr),
            sample_token=np.asarray(info["token"]),
            scene_name=np.asarray(info["token"].split("__")[0]),
            scene_description=np.asarray(
                f"CARLA {info['token'].split('__')[0]}, test split"
            ),
            nusc_split=np.asarray("carla_test"),
        )
        for key, letter in MODELS.items():
            pb, ps = preds[key][r["sample_idx"]]
            order = np.argsort(-ps, kind="stable")
            pb, ps = pb[order], ps[order]
            out[f"pred_boxes_{key}"], out[f"pred_scores_{key}"] = pb, ps
            for thr, tag in ((0.5, "05"), (0.7, "07")):
                tp, gm, err = match(pb, ps, gt, thr)
                out[f"pred_tp_iou{tag}_{key}"] = tp
                out[f"gt_matched_iou{tag}_{key}"] = gm
                if tag == "05":
                    out[f"gt_heading_err_rad_iou05_{key}"] = err.astype(np.float32)
            out[f"run_{key}"] = np.asarray(RUNS[letter])
        sub = (
            args.out_dir
            if rank < args.n_scenes
            else osp.join(args.out_dir, "alternates")
        )
        os.makedirs(sub, exist_ok=True)
        name = f"scene_{info['token']}.npz"
        np.savez_compressed(osp.join(sub, name), **out)
        r.update(
            rank=rank,
            role="primary" if rank < args.n_scenes else "alternate",
            n_points=len(raw),
            file=osp.relpath(osp.join(sub, name), args.out_dir),
        )
        print(json.dumps(r), flush=True)

    with open(osp.join(args.out_dir, "candidates.json"), "w") as f:
        json.dump(
            dict(
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                generated_by="scripts/figure_data/export_carla_qualitative.py",
                selection=dict(
                    pool_rule=f"CARLA test scans with >= {args.min_far} GT cars at 20-50 m; seeded random sample "
                    f"of {len(pool)} of {len(eligible)}",
                    matching="GeoPillarsMetric greedy rotated-BEV-IoU matching at IoU 0.5 (the mAOE threshold); "
                    "heading error folded to [0, pi]",
                    ranking=f"#cars >= 20 m matched by A and D with A error < {args.good_deg:g} deg and D error > "
                    f"{args.bad_deg:g} deg, minus the reverse; >= {args.min_frame_gap} frames apart within a town",
                    score_threshold=args.score_thr,
                    models={k: RUNS[v] for k, v in MODELS.items()},
                ),
                pool_size=len(pool),
                exported=chosen,
                pool_table=table,
            ),
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
