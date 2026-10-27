#!/usr/bin/env python3
"""Measure per-sample inference latency and VRAM of every released checkpoint (Fig. 4).

The eval JSONs time whole batches at batch sizes that differ per run (16/24/64),
so their ms/sample are not comparable. Here every run gets the same protocol:

* batch size 1, fp32, the run's dumped test config, CARLA test scans;
* the same ``--n_frames`` scans (evenly spaced over the split), the first
  ``--warmup`` of them untimed;
* latency = ``model.test_step`` wall time between ``torch.cuda.synchronize()``
  calls: GPU voxelisation + network + box decoding/NMS (data loading excluded),
  as in ``InferenceProfilerHook``;
* VRAM = ``torch.cuda.max_memory_allocated`` over the timed loop (weights +
  activations + workspace), plus the reserved-pool peak and the weights alone.

Each run is measured in a fresh subprocess, sequentially, with nothing else on
the GPU. Writes ``figures/data/inference_cost.json``.

Usage (repo root)::

    PYTHONPATH=$(pwd) python scripts/figure_data/measure_inference_cost.py
"""

import argparse
import json
import os
import os.path as osp
import platform
import subprocess
import sys
import time
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

N_TEST = 30842


def child(run, n_frames, warmup, out):
    import torch

    indices = np.linspace(0, N_TEST - 1, n_frames).round().astype(int)
    runner, model, loader, cfg = build(
        run, indices=indices, batch_size=1, num_workers=4
    )
    torch.cuda.synchronize()
    weights_mb = torch.cuda.memory_allocated() / 2**20
    lat = []
    with torch.no_grad():
        for i, data_batch in enumerate(loader):
            if i == warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.test_step(data_batch)
            torch.cuda.synchronize()
            if i >= warmup:
                lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.asarray(lat)
    vl = cfg.model.data_preprocessor.voxel_layer
    res = dict(
        run=run,
        n_timed=int(len(lat)),
        warmup=warmup,
        batch_size=1,
        precision="fp32",
        latency_ms_mean=round(float(lat.mean()), 3),
        latency_ms_median=round(float(np.median(lat)), 3),
        latency_ms_p90=round(float(np.percentile(lat, 90)), 3),
        latency_ms_std=round(float(lat.std()), 3),
        peak_vram_allocated_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
        peak_vram_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20, 1),
        weights_vram_mb=round(weights_mb, 1),
        params=int(sum(p.numel() for p in model.parameters())),
        voxel_size=list(vl.voxel_size),
        max_num_points=vl.max_num_points,
        max_voxels_test=vl.max_voxels[1],
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
    )
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(
        f'[cost] {run}: {res["latency_ms_mean"]} ms, {res["peak_vram_allocated_mb"]} MB',
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--child", nargs=2, metavar=("RUN", "OUT"), help=argparse.SUPPRESS)
    ap.add_argument("--n_frames", type=int, default=550)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", nargs="*", default=list(RUNS))
    args = ap.parse_args()
    if args.child:
        return child(args.child[0], args.n_frames, args.warmup, args.child[1])

    import torch

    tmp = osp.join(REPO_ROOT, "work_dirs", "figure_data", "inference_cost")
    os.makedirs(tmp, exist_ok=True)
    results = {}
    for letter in args.runs:
        run = RUNS[letter]
        out = osp.join(tmp, f"{run}.json")
        env = dict(os.environ, PYTHONPATH=REPO_ROOT, WANDB_MODE="disabled")
        subprocess.run(
            [
                sys.executable,
                osp.abspath(__file__),
                "--child",
                run,
                out,
                "--n_frames",
                str(args.n_frames),
                "--warmup",
                str(args.warmup),
            ],
            check=True,
            cwd=REPO_ROOT,
            env=env,
        )
        with open(out) as f:
            results[letter] = json.load(f)

    agg_path = osp.join(REPO_ROOT, "results", "eval_results_carla_test.json")
    with open(agg_path) as f:
        agg = json.load(f)
    for letter, r in results.items():
        e = agg["runs"][r["run"]]["epochs"]["8"]
        r["ap_0.7_carla_test"] = e["metrics"]["GeoPillars/AP@0.7"]
        rt = e["eval_runtime"]
        r["full_eval_batched_reference"] = dict(
            batch_size=rt["batch_size"],
            ms_per_batch=rt["model_forward_ms_per_batch_mean"],
            peak_vram_allocated_mb=rt["peak_gpu_mem_allocated_mb"],
            note="from eval_results_carla_test.json; batch sizes differ per run, not comparable",
        )

    doc = dict(
        description="Per-sample inference cost of the released epoch_8 checkpoints at batch size 1 on CARLA test "
        "scans. latency = model.test_step (GPU voxelisation + network + decode/NMS), data loading "
        "excluded; VRAM = peak torch allocation during the timed loop.",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/figure_data/measure_inference_cost.py",
        environment=dict(
            host=platform.node(),
            gpu=torch.cuda.get_device_name(0),
            torch=torch.__version__,
            python=platform.python_version(),
        ),
        data=dict(
            data_root=CARLA_ROOT,
            ann_file=CARLA_TEST_PKL,
            n_frames=args.n_frames,
            warmup=args.warmup,
            frame_selection=f"np.linspace(0, {N_TEST - 1}, {args.n_frames}).round()",
        ),
        runs=results,
    )
    path = osp.join(REPO_ROOT, "figures", "data", "inference_cost.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    for letter, r in results.items():
        print(
            letter,
            r["run"],
            r["latency_ms_mean"],
            "ms",
            r["peak_vram_allocated_mb"],
            "MB",
            "AP",
            r["ap_0.7_carla_test"],
        )


if __name__ == "__main__":
    main()
