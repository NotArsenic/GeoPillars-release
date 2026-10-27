"""Run one released checkpoint over the CARLA test split and dump raw predictions.

``GeoPillarsMetric.compute_metrics`` is a pure function of the per-sample
prediction records produced by ``GeoPillarsMetric.process`` plus the annotation
pkl. Dumping those records once therefore lets every metric variant (gated
mBEVIoU, folded mAOE, axis-aligned vs rotated IoU) and every downstream analysis
(false positives on unlabelled foreground) be recomputed on the CPU without
another GPU pass.

Variants
--------
``baseline``            the run's own test pipeline, unchanged.
``point_shuffle_test``  ``PointShuffle`` inserted immediately before
                        ``Pack3DDetInputs``, i.e. the train pipeline's point
                        ordering at test time. Hard voxelisation keeps the first
                        ``max_num_points`` points of each pillar in arrival
                        order, so this changes which points survive truncation.
``zero_bev_pos``        ``bev_pos_encoding`` (found by parameter name) zeroed after
                        loading the checkpoint, nothing else changed.

Usage (from the repo root)::

    PYTHONPATH=$(pwd) python scripts/figure_data/dump_predictions.py \
        --run A --out work_dirs/preds/A_baseline.npz
    PYTHONPATH=$(pwd) python scripts/figure_data/dump_predictions.py \
        --run A --variant point_shuffle_test --out work_dirs/preds/A_shuffle.npz
"""

import argparse
import os
import os.path as osp
import sys
import time

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from _carla_infer import RUNS, build  # noqa: E402

VARIANTS = ("baseline", "point_shuffle_test", "zero_bev_pos")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", required=True, help="run letter A-G or full run name")
    p.add_argument("--out", required=True, help="output .npz path")
    p.add_argument("--variant", default="baseline", choices=VARIANTS)
    p.add_argument("--epoch", type=int, default=8)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N frames of the split",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="torch/numpy seed (point_shuffle_test is stochastic)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    run = RUNS.get(args.run, args.run)
    if run not in RUNS.values():
        raise SystemExit(f"unknown run {args.run!r}; choose from {sorted(RUNS)}")

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    indices = list(range(args.limit)) if args.limit else None
    runner, model, loader, cfg = build(
        run,
        indices=indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epoch=args.epoch,
    )
    pipeline = [t["type"] for t in cfg.test_dataloader.dataset.pipeline]

    # -- variant: PointShuffle at test time --------------------------------
    if args.variant == "point_shuffle_test":
        from mmengine.registry import TRANSFORMS

        ds_pipe = list(cfg.test_dataloader.dataset.pipeline)
        pack = [i for i, t in enumerate(ds_pipe) if t["type"].startswith("Pack")]
        assert len(pack) == 1, f"expected one Pack* transform, got {ds_pipe}"
        ds_pipe.insert(pack[0], dict(type="PointShuffle"))
        cfg.test_dataloader.dataset.pipeline = ds_pipe
        from mmengine.runner import Runner

        loader = Runner.build_dataloader(cfg.test_dataloader)
        pipeline = [t["type"] for t in ds_pipe]
        assert "PointShuffle" in pipeline

    # -- variant: zero the learned BEV positional prior --------------------
    zeroed = None
    if args.variant == "zero_bev_pos":
        # the parameter lives at inner.pillar_feature_net.bev_pos_encoding in
        # GeoPillarsDetector and pts_voxel_encoder.pfn.bev_pos_encoding in the
        # CenterPoint adapter, so find it by name instead of by path
        hits = [
            (n, p)
            for n, p in model.named_parameters()
            if n.endswith("bev_pos_encoding")
        ]
        if len(hits) != 1:
            raise SystemExit(
                f"{run}: found {len(hits)} bev_pos_encoding parameters "
                f"{[n for n, _ in hits]} - expected exactly one"
            )
        name, param = hits[0]
        zeroed = dict(
            parameter=name,
            shape=list(param.shape),
            numel=int(param.numel()),
            abs_mean_before=float(param.detach().abs().mean()),
            std_before=float(param.detach().std()),
        )
        with torch.no_grad():
            param.zero_()
        assert float(param.detach().abs().sum()) == 0.0
        print(f"[dump] zeroed {name}: {zeroed}", flush=True)

    n_frames = len(loader.dataset)
    print(
        f"[dump] run={run} variant={args.variant} frames={n_frames} "
        f'batch={loader.batch_size} pipeline={">".join(pipeline)}',
        flush=True,
    )

    sample_idx, lidar_path, n_det = [], [], []
    boxes, scores, labels = [], [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for bi, data_batch in enumerate(loader):
            for ds in model.test_step(data_batch):
                pred = ds.pred_instances_3d
                b = pred["bboxes_3d"]
                b = b.tensor if hasattr(b, "tensor") else b
                b = b.detach().cpu().numpy().astype(np.float32)
                sample_idx.append(int(ds.sample_idx))
                lidar_path.append(str(ds.metainfo.get("lidar_path", "")))
                n_det.append(len(b))
                if len(b):
                    boxes.append(b)
                    scores.append(
                        pred["scores_3d"].detach().cpu().numpy().astype(np.float32)
                    )
                    labels.append(
                        pred["labels_3d"].detach().cpu().numpy().astype(np.int16)
                    )
            if bi % 50 == 0:
                done = len(sample_idx)
                el = time.perf_counter() - t0
                eta = el / max(done, 1) * (n_frames - done)
                print(
                    f"[dump] {done}/{n_frames} frames  {el:.0f}s elapsed  "
                    f"ETA {eta / 60:.1f} min",
                    flush=True,
                )

    wall = time.perf_counter() - t0
    box_dim = boxes[0].shape[1] if boxes else 9
    out = dict(
        sample_idx=np.asarray(sample_idx, dtype=np.int64),
        lidar_path=np.asarray(lidar_path),
        n_det=np.asarray(n_det, dtype=np.int32),
        boxes=np.concatenate(boxes) if boxes else np.zeros((0, box_dim), np.float32),
        scores=np.concatenate(scores) if scores else np.zeros((0,), np.float32),
        labels=np.concatenate(labels) if labels else np.zeros((0,), np.int16),
        meta=np.array(
            repr(
                dict(
                    run=run,
                    letter=args.run,
                    variant=args.variant,
                    epoch=args.epoch,
                    n_frames=n_frames,
                    limit=args.limit,
                    seed=args.seed,
                    batch_size=loader.batch_size,
                    pipeline=pipeline,
                    ann_file=cfg.test_evaluator.ann_file,
                    data_root=cfg.test_dataloader.dataset.data_root,
                    use_dim=cfg.test_dataloader.dataset.pipeline[0].get("use_dim"),
                    test_max_voxels=cfg.model.data_preprocessor.voxel_layer.max_voxels[
                        1
                    ],
                    max_num_points=cfg.model.data_preprocessor.voxel_layer.max_num_points,
                    voxel_size=cfg.model.data_preprocessor.voxel_layer.voxel_size,
                    zero_bev_pos=zeroed,
                    wall_time_s=round(wall, 1),
                    torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(0),
                )
            )
        ),
    )
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(
        f'[dump] {len(sample_idx)} frames, {len(out["scores"])} detections, '
        f"{wall / 60:.1f} min -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
