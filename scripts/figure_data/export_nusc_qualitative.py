#!/usr/bin/env python3
"""Pick nuScenes keyframes that show recall collapse and export them as NPZ.

Stages (parent process):
  1. Candidate pool: keyframes with >= ``--min_gt`` GT cars (the zero-shot GT
     rule: num_lidar_pts >= 5, <= 50 m), at least 2 of them nearer than 20 m and
     at least 2 at or beyond 30 m.
  2. Inference on the pool only (not a full eval), one subprocess per model,
     with the run's dumped config and the same nuScenes dataset/pipeline as
     ``tools/eval_test_checkpoints.py --dataset nusc_zeroshot`` (intensity 1.0).
  3. Per-keyframe recall at ``--score_thr`` with the GeoPillarsMetric matching
     rule (greedy by score, rotated BEV IoU).
  4. ``--selection worst`` (default): rank by GeoPillars recall@0.5 (lowest
     first), then by GT count, one keyframe per nuScenes scene; export the top
     ``--n_scenes`` and ``--n_alternates`` more to ``alternates/``.
     ``--selection representative``: keyframes whose GeoPillars recall@0.5 is
     within 0.10-0.25 (around the pool median) with >= 4 GT cars both nearer
     than 20 m and at or beyond 30 m, ranked by GT count; exported to
     ``representative/`` (and ``representative/alternates/``).

Each ``scene_<sample_token>.npz`` is self-contained; see ``README`` keys inside
and ``candidates.json`` for the selection table.

Usage (repo root)::

    PYTHONPATH=$(pwd) python scripts/figure_data/export_nusc_qualitative.py
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

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.join(REPO_ROOT, "tools"))

# short key -> run name (config/<run>.py, checkpoints/<run>/)
MODELS = {
    "geopillars": "geo_pillars_h100_intensity",
    "centerpoint_osf1": "centerpoint_osf1_ablation",
    "centerpoint_native": "centerpoint_h100_intensity_original",
    "cp_encoder_replk": "centerpoint_geopillar_encoder_replk_distributed",
    "centerpoint_budget_osf4": "centerpoint_h100_intensity",
}


# -----------------------------------------------------------------------------
# child: inference on a subset
# -----------------------------------------------------------------------------


def child_infer(run, indices_path, out_path):
    os.environ.setdefault("WANDB_MODE", "disabled")
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner, load_checkpoint
    from eval_test_checkpoints import DATASETS, checkpoint_path, find_config

    cfg = Config.fromfile(find_config(run))
    cfg.launcher = "none"
    for key in (
        "train_dataloader",
        "train_cfg",
        "optim_wrapper",
        "param_scheduler",
        "val_dataloader",
        "val_cfg",
        "val_evaluator",
    ):
        cfg[key] = None
    cfg.work_dir = osp.join(REPO_ROOT, "work_dirs", "figure_data_nusc_qualitative", run)
    cfg.test_evaluator.pr_save_dir = None
    cfg.visualizer = dict(
        type="Visualizer", vis_backends=[dict(type="LocalVisBackend")]
    )

    dataset = DATASETS["nusc_zeroshot"]
    ds_cfg = cfg.test_dataloader.dataset
    assert ds_cfg.pipeline[0].type == "LoadPointsFromNpy"
    ds_cfg.pipeline[0].load_dim = dataset["load_dim"]
    ds_cfg.data_root = dataset["data_root"]
    ds_cfg.ann_file = dataset["ann_file"]
    cfg.test_evaluator.ann_file = osp.join(dataset["data_root"], dataset["ann_file"])
    ds_cfg.indices = [int(i) for i in np.load(indices_path)]

    runner = Runner.from_cfg(cfg)
    model = runner.model
    load_checkpoint(model, checkpoint_path(run, 8), map_location="cpu")
    model.eval()
    loader = Runner.build_dataloader(cfg.test_dataloader)

    # Det3DDataset renumbers sample_idx to the position inside the subset, so map
    # each output back to the pkl entry through its lidar file name.
    with open(cfg.test_evaluator.ann_file, "rb") as f:
        path_to_idx = {
            osp.basename(i["lidar_points"]["lidar_path"]): i["sample_idx"]
            for i in pickle.load(f)["data_list"]
        }
    sample_idx, frames, boxes, scores, labels = [], [], [], [], []
    with torch.no_grad():
        for data_batch in loader:
            for ds in model.test_step(data_batch):
                p = ds.pred_instances_3d
                b = p.bboxes_3d
                grav = b.gravity_center.cpu().numpy()
                t = b.tensor[:, :7].cpu().numpy().copy()
                t[:, 2] = grav[
                    :, 2
                ]  # bottom centre -> gravity centre, as in the info pkl
                idx = path_to_idx[osp.basename(ds.lidar_path)]
                sample_idx.append(np.full(len(t), idx, dtype=np.int32))
                frames.append(idx)
                boxes.append(t.astype(np.float32))
                scores.append(p.scores_3d.cpu().numpy().astype(np.float32))
                labels.append(p.labels_3d.cpu().numpy().astype(np.int16))
    assert sorted(frames) == sorted(ds_cfg.indices)
    np.savez_compressed(
        out_path,
        sample_idx=np.concatenate(sample_idx),
        boxes=np.concatenate(boxes),
        scores=np.concatenate(scores),
        labels=np.concatenate(labels),
        frames=np.asarray(frames, dtype=np.int64),
    )
    print(
        f"[infer] {run}: {len(frames)} frames, {sum(len(s) for s in scores)} boxes -> {out_path}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# parent
# -----------------------------------------------------------------------------


def match(pred_boxes, pred_scores, gt_boxes, thr):
    """GeoPillarsMetric._eval_class matching, for one frame. Returns (pred_tp, gt_matched)."""
    from src.metrics.geo_pillars_metric import compute_bev_iou_matrix

    order = np.argsort(-pred_scores, kind="stable")
    pred_tp = np.zeros(len(pred_boxes), dtype=bool)
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return pred_tp, gt_matched
    iou = compute_bev_iou_matrix(
        pred_boxes.astype(np.float64), gt_boxes.astype(np.float64)
    )
    for i in order:
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= thr and not gt_matched[j]:
            pred_tp[i] = True
            gt_matched[j] = True
    return pred_tp, gt_matched


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--infer",
        nargs=3,
        metavar=("RUN", "INDICES_NPY", "OUT_NPZ"),
        help=argparse.SUPPRESS,
    )
    from eval_test_checkpoints import NUSC_ZEROSHOT_ROOT

    ap.add_argument("--data_root", default=NUSC_ZEROSHOT_ROOT)
    ap.add_argument(
        "--out_dir", default=osp.join(REPO_ROOT, "figures", "data", "nusc_qualitative")
    )
    ap.add_argument(
        "--cache_dir",
        default=osp.join(REPO_ROOT, "work_dirs", "figure_data_nusc_qualitative"),
    )
    ap.add_argument("--score_thr", type=float, default=0.25)
    ap.add_argument("--min_gt", type=int, default=8)
    ap.add_argument("--n_scenes", type=int, default=3)
    ap.add_argument("--n_alternates", type=int, default=3)
    ap.add_argument("--selection", choices=["worst", "representative"], default="worst")
    args = ap.parse_args()

    if args.infer:
        return child_infer(*args.infer)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    with open(osp.join(args.data_root, "nuscenes_infos_zeroshot.pkl"), "rb") as f:
        infos = pickle.load(f)
    car_id = infos["metainfo"]["categories"]["car"]
    data_list = infos["data_list"]

    def cars(info, valid):
        inst = [
            i
            for i in info["instances"]
            if i["bbox_label_3d"] == car_id and i["bbox_3d_isvalid"] == valid
        ]
        return (
            np.asarray([i["bbox_3d"][:7] for i in inst], dtype=np.float32).reshape(
                -1, 7
            ),
            np.asarray([i["num_lidar_pts"] for i in inst], dtype=np.int32),
        )

    # 1. candidate pool
    pool = []
    for info in data_list:
        gt, _ = cars(info, True)
        r = np.hypot(gt[:, 0], gt[:, 1])
        if len(gt) >= args.min_gt and (r < 20).sum() >= 2 and (r >= 30).sum() >= 2:
            pool.append(info["sample_idx"])
    pool = np.asarray(pool, dtype=np.int64)
    idx_path = osp.join(args.cache_dir, "pool_indices.npy")
    np.save(idx_path, pool)
    print(f"candidate pool: {len(pool)} keyframes")

    # 2. inference on the pool
    preds = {}
    for key, run in MODELS.items():
        out = osp.join(args.cache_dir, f"preds_{run}.npz")
        cached = osp.isfile(out) and np.array_equal(
            np.sort(np.load(out)["frames"]), pool
        )
        if not cached:
            env = dict(os.environ, PYTHONPATH=REPO_ROOT, WANDB_MODE="disabled")
            subprocess.run(
                [sys.executable, osp.abspath(__file__), "--infer", run, idx_path, out],
                check=True,
                cwd=REPO_ROOT,
                env=env,
            )
        d = np.load(out)
        keep = (d["labels"] == 0) & (d["scores"] >= args.score_thr)
        by_frame = defaultdict(
            lambda: (np.zeros((0, 7), np.float32), np.zeros(0, np.float32))
        )
        for s in np.unique(d["sample_idx"][keep]):
            m = keep & (d["sample_idx"] == s)
            by_frame[int(s)] = (d["boxes"][m], d["scores"][m])
        preds[key] = by_frame

    # 3. per-keyframe recall
    table = []
    for s in pool:
        info = data_list[s]
        gt, gt_pts = cars(info, True)
        row = dict(
            sample_idx=int(s),
            sample_token=info["token"],
            scene_name=info["scene_name"],
            n_gt=len(gt),
            n_gt_near_lt20=int((np.hypot(gt[:, 0], gt[:, 1]) < 20).sum()),
            n_gt_far_ge30=int((np.hypot(gt[:, 0], gt[:, 1]) >= 30).sum()),
        )
        for key in MODELS:
            pb, ps = preds[key][int(s)]
            row[f"{key}_n_det"] = len(pb)
            for thr in (0.5, 0.7):
                tp, gm = match(pb, ps, gt, thr)
                row[f"{key}_recall@{thr}"] = round(float(gm.mean()), 4)
        table.append(row)

    # 4. rank, one keyframe per scene
    if args.selection == "worst":
        ranking = (
            "GeoPillars recall@0.5 ascending, then CP OSF=1 recall@0.5, then GT count descending; "
            "one keyframe per nuScenes scene"
        )
        table.sort(
            key=lambda r: (
                r["geopillars_recall@0.5"],
                r["centerpoint_osf1_recall@0.5"],
                -r["n_gt"],
            )
        )
        eligible = table
    else:
        ranking = (
            "GeoPillars recall@0.5 in [0.10, 0.25] (pool median ~0.15), >= 4 GT nearer than 20 m and "
            ">= 4 at or beyond 30 m, GT count descending; one keyframe per nuScenes scene"
        )
        table.sort(key=lambda r: (-r["n_gt"], r["sample_idx"]))
        eligible = [
            r
            for r in table
            if 0.10 <= r["geopillars_recall@0.5"] <= 0.25
            and r["n_gt_near_lt20"] >= 4
            and r["n_gt_far_ge30"] >= 4
        ]
    out_root = (
        args.out_dir
        if args.selection == "worst"
        else osp.join(args.out_dir, "representative")
    )
    chosen, seen = [], set()
    for r in eligible:
        if r["scene_name"] not in seen:
            chosen.append(r)
            seen.add(r["scene_name"])
        if len(chosen) == args.n_scenes + args.n_alternates:
            break

    # scene descriptions + CAM_FRONT, from the raw meta
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot=osp.join(args.data_root, "v1.0-trainval_meta"),
        verbose=False,
    )

    def se3(rec):
        m = np.eye(4)
        m[:3, :3] = Quaternion(rec["rotation"]).rotation_matrix
        m[:3, 3] = rec["translation"]
        return m

    for rank, r in enumerate(chosen):
        info = data_list[r["sample_idx"]]
        sample = nusc.get("sample", info["token"])
        scene = nusc.get("scene", sample["scene_token"])
        raw = np.fromfile(
            osp.join(args.data_root, info["lidar_points"]["lidar_path"]),
            dtype=np.float32,
        ).reshape(-1, 5)
        gt, gt_pts = cars(info, True)
        ign, ign_pts = cars(info, False)
        out = dict(
            points=raw[:, :4],
            points_note=np.asarray(
                "x y z intensity in LIDAR_TOP frame; raw nuScenes intensity (0-255) kept "
                "for the record. Inference used intensity = 1.0 for every point."
            ),
            gt_boxes=gt,
            gt_num_lidar_pts=gt_pts,
            gt_ignored_boxes=ign,
            gt_ignored_num_lidar_pts=ign_pts,
            box_format=np.asarray(
                "x y z l w h yaw; LIDAR_TOP frame; z = box gravity centre; yaw about +z"
            ),
            gt_rule=np.asarray(
                "GT car iff num_lidar_pts >= 5 and hypot(x, y) <= 50 m; gt_ignored_* are cars "
                "failing the rule (not GT, detections on them are FPs)"
            ),
            score_threshold=np.float32(args.score_thr),
            sample_token=np.asarray(info["token"]),
            scene_name=np.asarray(scene["name"]),
            scene_description=np.asarray(scene["description"]),
            nusc_split=np.asarray(info["nusc_split"]),
            timestamp=np.float64(info["timestamp"]),
            lidar2ego=np.asarray(info["lidar_points"]["lidar2ego"], dtype=np.float64),
            ego2global=np.asarray(info["ego2global"], dtype=np.float64),
        )
        for key, run in MODELS.items():
            pb, ps = preds[key][r["sample_idx"]]
            order = np.argsort(-ps, kind="stable")
            pb, ps = pb[order], ps[order]
            out[f"pred_boxes_{key}"] = pb
            out[f"pred_scores_{key}"] = ps
            for thr, tag in ((0.5, "05"), (0.7, "07")):
                tp, gm = match(pb, ps, gt, thr)
                out[f"pred_tp_iou{tag}_{key}"] = tp
                out[f"gt_matched_iou{tag}_{key}"] = gm
            out[f"run_{key}"] = np.asarray(run)

        cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        lid = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cam_cs = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
        lid_cs = nusc.get("calibrated_sensor", lid["calibrated_sensor_token"])
        # lidar -> ego(t_lidar) -> global -> ego(t_cam) -> cam
        l2c = (
            np.linalg.inv(se3(cam_cs))
            @ np.linalg.inv(se3(nusc.get("ego_pose", cam["ego_pose_token"])))
            @ se3(nusc.get("ego_pose", lid["ego_pose_token"]))
            @ se3(lid_cs)
        )
        jpg_path = None
        for d in sorted(os.listdir(args.data_root)):
            p = osp.join(args.data_root, d, cam["filename"])
            if osp.isfile(p):
                jpg_path = p
                break
        if jpg_path:
            with open(jpg_path, "rb") as f:
                out["cam_front_jpeg"] = np.frombuffer(f.read(), dtype=np.uint8)
            out["cam_front_intrinsic"] = np.asarray(
                cam_cs["camera_intrinsic"], dtype=np.float64
            )
            out["lidar_to_cam_front"] = l2c
            out["cam_front_note"] = np.asarray(
                "JPEG bytes: PIL.Image.open(io.BytesIO(arr.tobytes())). "
                "Project p_lidar with lidar_to_cam_front then cam_front_intrinsic."
            )

        sub = out_root if rank < args.n_scenes else osp.join(out_root, "alternates")
        os.makedirs(sub, exist_ok=True)
        np.savez_compressed(osp.join(sub, f"scene_{info['token']}.npz"), **out)
        r.update(
            rank=rank,
            role="primary" if rank < args.n_scenes else "alternate",
            scene_description=scene["description"],
            n_points=len(raw),
            file=osp.relpath(osp.join(sub, f"scene_{info['token']}.npz"), out_root),
            has_cam_front=bool(jpg_path),
        )
        print(json.dumps(r))

    with open(osp.join(out_root, "candidates.json"), "w") as f:
        json.dump(
            dict(
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                generated_by="scripts/figure_data/export_nusc_qualitative.py",
                selection=dict(
                    pool_rule=f">= {args.min_gt} GT cars, >= 2 nearer than 20 m, >= 2 at or beyond 30 m",
                    selection=args.selection,
                    ranking=ranking,
                    score_threshold=args.score_thr,
                    matching="GeoPillarsMetric greedy rotated-BEV-IoU matching",
                    inference="intensity overwritten to 1.0; nuscenes_infos_zeroshot_npy.pkl; epoch_8.pth",
                    models=MODELS,
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
