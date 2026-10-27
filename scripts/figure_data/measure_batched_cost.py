#!/usr/bin/env python3
"""Peak VRAM and throughput of every released checkpoint at one *fixed* batch size.

``measure_inference_cost.py`` covers batch 1; the eval JSONs carry a batched
number but at a batch size that differs per run (16 / 24 / 64), so their VRAM
figures are not comparable. Here every run gets the same batch size, the same
frames and the same protocol:

* fixed ``--batch_size`` (default 24), fp32, the run's dumped test config,
  CARLA test scans evenly spaced over the split;
* the first ``--warmup`` *batches* are untimed and the peak-memory counter is
  reset after them, so weights + activations + workspace of the steady state
  are what is reported;
* latency = ``model.test_step`` wall time between ``torch.cuda.synchronize()``
  calls (data loading excluded), as in ``measure_inference_cost.py``.

Each run runs in a fresh subprocess, sequentially. A run that does not fit is
recorded with ``status='oom'`` rather than aborting the sweep.

Writes ``figures/data/batched_inference_cost.json``.

Usage (repo root)::

    PYTHONPATH=$(pwd) python scripts/figure_data/measure_batched_cost.py --batch_size 24
"""

import argparse
import json
import os.path as osp
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


def pick_frames(n_frames, select):
    """Frame indices into the CARLA val split: evenly spaced, or the densest scans."""
    if select == "even":
        return np.linspace(0, N_TEST - 1, n_frames).round().astype(int)
    # 'densest': the scans with the most in-range points, the worst case for any
    # encoder whose activations scale with occupied pillars. Budget-independent,
    # so every run is measured on the same frames.
    npz = np.load(osp.join(REPO_ROOT, "figures", "data", "pillar_stats_scans.npz"))
    n_in_range = npz["val_n_in_range"]
    assert len(n_in_range) == N_TEST
    return np.sort(np.argsort(n_in_range)[::-1][:n_frames])


def child(run, batch_size, n_batches, warmup, out, select="even"):
    import torch

    n_frames = batch_size * (n_batches + warmup)
    indices = pick_frames(n_frames, select)
    res = dict(run=run, batch_size=batch_size, precision="fp32", frame_selection=select)
    try:
        runner, model, loader, cfg = build(
            run, indices=indices, batch_size=batch_size, num_workers=6
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
        res.update(
            status="ok",
            n_batches_timed=int(len(lat)),
            warmup_batches=warmup,
            ms_per_batch_mean=round(float(lat.mean()), 2),
            ms_per_batch_median=round(float(np.median(lat)), 2),
            ms_per_sample_mean=round(float(lat.mean()) / batch_size, 3),
            peak_vram_allocated_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
            peak_vram_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20, 1),
            weights_vram_mb=round(weights_mb, 1),
            params=int(sum(p.numel() for p in model.parameters())),
            voxel_size=list(vl.voxel_size),
            max_num_points=vl.max_num_points,
            max_voxels_test=vl.max_voxels[1],
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        )
        print(
            f'[batched] {run}: {res["ms_per_batch_mean"]} ms/batch, '
            f'{res["peak_vram_allocated_mb"]} MB',
            flush=True,
        )
    except torch.cuda.OutOfMemoryError as e:
        res.update(status="oom", error=str(e)[:300])
        print(f"[batched] {run}: OOM at batch {batch_size}", flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--child", nargs=2, metavar=("RUN", "OUT"), help=argparse.SUPPRESS)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--n_batches", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", nargs="*", default=list(RUNS))
    ap.add_argument(
        "--select",
        choices=["even", "densest"],
        default="even",
        help="'densest' measures the worst case: the scans with the most in-range "
        "points, the same frames for every run",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.child:
        return child(
            args.child[0],
            args.batch_size,
            args.n_batches,
            args.warmup,
            args.child[1],
            args.select,
        )

    import torch

    out = dict(
        description=f"Inference cost of the released epoch_8 checkpoints at a FIXED batch size "
        f"({args.batch_size}), fp32, on CARLA test scans. Companion to "
        f"inference_cost.json (batch 1). VRAM = peak torch allocation over the timed "
        f"batches, counter reset after warm-up.",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/figure_data/measure_batched_cost.py",
        environment=dict(
            host=__import__("platform").node(),
            gpu=torch.cuda.get_device_name(0),
            gpu_total_mb=round(
                torch.cuda.get_device_properties(0).total_memory / 2**20
            ),
            torch=torch.__version__,
            python=__import__("platform").python_version(),
        ),
        settings=dict(
            batch_size=args.batch_size,
            n_batches_timed=args.n_batches,
            warmup_batches=args.warmup,
            precision="fp32",
            frame_selection=args.select,
            data_root=CARLA_ROOT,
            ann_file=CARLA_TEST_PKL,
        ),
        runs={},
    )
    tmp = osp.join(REPO_ROOT, "work_dirs", "figure_data", "_batched_cost.json")
    for letter in args.runs:
        run = RUNS[letter]
        cmd = [
            sys.executable,
            osp.abspath(__file__),
            "--child",
            run,
            tmp,
            "--batch_size",
            str(args.batch_size),
            "--n_batches",
            str(args.n_batches),
            "--warmup",
            str(args.warmup),
            "--select",
            args.select,
        ]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        with open(tmp) as f:
            out["runs"][letter] = json.load(f)
    path = args.out or osp.join(
        REPO_ROOT, "figures", "data", "batched_inference_cost.json"
    )
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
