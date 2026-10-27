#!/usr/bin/env python3
"""Distribution of the NGFM spatial gate (Fig. 12), from the trained checkpoints.

``GeoPillarsWandbHook`` logged ``ngfm/spatial_mask_hist`` as a ``wandb.Histogram``
during training; that never reached disk, and only the scalar mean/std survive in
``<run>/metrics.csv``. This reconstructs the distribution from the checkpoint by
re-registering the *same* forward hook the training hook used
(``model.inner.ngfm.mask_expand``, i.e. the sigmoid output) over a batch of CARLA
validation scans.

Two things the pooled histogram alone cannot show, both reported here:

* the gate is ``sigmoid(BN(1x1 conv(pred_normals)))`` - 256 channels expanded from
  a 3-vector per cell - so its spread can come from *between-channel* offsets
  rather than from spatial selectivity. The variance is decomposed accordingly.
* the channel-average of the gate at each BEV cell is reported separately from the
  pooled spread, because a wide pooled histogram is also consistent with a gate
  that varies only *across channels* and applies a near-constant gain everywhere.

Not split by occupancy: after ``bev_pos_encoding`` is added the BEV canvas is
non-zero in every cell (see export_encoder_internals.py), so an occupancy mask derived
from the feature map would be identically true; a real one would have to come from
the voxel coordinates and is out of scope here.

``--mode train`` puts BatchNorm in batch-statistics mode to match the training-time
logged value; the default ``eval`` is the deployed behaviour.

Writes ``figures/data/ngfm_mask_hist.json``.

Usage (repo root)::

    PYTHONPATH=$(pwd) python scripts/figure_data/export_ngfm_mask_hist.py
"""

import argparse
import json
import os.path as osp
import sys
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
NGFM_RUNS = ("A", "B")  # only the two runs that have an NGFM
BINS = np.linspace(0.0, 1.0, 101)
ZOOM_BINS = np.linspace(0.40, 0.60, 201)


def stats(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    q = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return dict(
        n=int(x.size),
        mean=round(float(x.mean()), 6),
        std=round(float(x.std()), 6),
        percentiles={
            f"p{p}": round(float(v), 6)
            for p, v in zip([0, 1, 5, 25, 50, 75, 95, 99, 100], q)
        },
        frac_below_0p1=round(float((x < 0.1).mean()), 6),
        frac_above_0p9=round(float((x > 0.9).mean()), 6),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n_frames", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--mode", choices=["eval", "train", "both"], default="both")
    ap.add_argument("--runs", nargs="*", default=list(NGFM_RUNS))
    args = ap.parse_args()

    import torch

    out = dict(
        description="Distribution of the NGFM spatial gate (NGFM.mask_expand output, the sigmoid) "
        "over CARLA validation scans, reconstructed from the epoch_8 checkpoints with "
        "the same forward hook GeoPillarsWandbHook used during training.",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        generated_by="scripts/figure_data/export_ngfm_mask_hist.py",
        environment=dict(
            host=__import__("platform").node(),
            gpu=torch.cuda.get_device_name(0),
            torch=torch.__version__,
        ),
        settings=dict(
            n_frames=args.n_frames,
            batch_size=args.batch_size,
            hook="model.inner.ngfm.mask_expand (nn.Sequential ending in Sigmoid)",
            data_root=CARLA_ROOT,
            ann_file=CARLA_TEST_PKL,
            bin_edges="np.linspace(0, 1, 101)",
        ),
        note="training-time reference values come from <run>/metrics.csv columns "
        "train/ngfm/spatial_mask_{mean,std}, logged in train mode on training scans",
        runs={},
    )
    modes = ["eval", "train"] if args.mode == "both" else [args.mode]

    for letter in args.runs:
        run = RUNS[letter]
        indices = np.linspace(0, N_TEST - 1, args.n_frames).round().astype(int)
        runner, model, loader, cfg = build(
            run, indices=indices, batch_size=args.batch_size, num_workers=4
        )
        core = model.module if hasattr(model, "module") else model
        ngfm = core.inner.ngfm
        grabbed = []
        h = ngfm.mask_expand.register_forward_hook(
            lambda m, i, o: grabbed.append(o.detach().float().cpu().numpy())
        )
        entry = dict(run=run, ngfm_in_channels=int(ngfm.mask_expand[0].out_channels))
        for mode in modes:
            grabbed.clear()
            model.train(mode == "train")
            with torch.no_grad():
                for data_batch in loader:
                    model.test_step(data_batch)
            m = np.concatenate(
                [g.reshape(g.shape[0], g.shape[1], -1) for g in grabbed], axis=0
            )
            # m: (N, C, HW)
            ch_mean = m.mean(axis=(0, 2))  # per-channel mean
            within = m - ch_mean[None, :, None]  # channel offsets removed
            cellwise = m.mean(axis=1)  # (N, HW) mean gate per BEV cell
            entry[mode] = dict(
                shape=list(m.shape),
                pooled=stats(m),
                histogram=dict(
                    bin_edges=[round(float(x), 3) for x in BINS],
                    counts=[int(c) for c in np.histogram(m.ravel(), bins=BINS)[0]],
                ),
                cellwise_histogram=dict(
                    note="channel-average of the gate at each (sample, BEV cell), same bins as `histogram`",
                    bin_edges=[round(float(x), 3) for x in BINS],
                    counts=[
                        int(c) for c in np.histogram(cellwise.ravel(), bins=BINS)[0]
                    ],
                ),
                cellwise_histogram_zoom=dict(
                    note="same quantity, 0.001-wide bins over [0.40, 0.60]",
                    bin_edges=[round(float(x), 4) for x in ZOOM_BINS],
                    counts=[
                        int(c)
                        for c in np.histogram(cellwise.ravel(), bins=ZOOM_BINS)[0]
                    ],
                ),
                variance_decomposition=dict(
                    total_std=round(float(m.std()), 6),
                    between_channel_std=round(float(ch_mean.std()), 6),
                    within_channel_std=round(float(within.std()), 6),
                    between_channel_share_of_variance=round(
                        float(ch_mean.var() / m.var()), 6
                    ),
                    note="total var = var(per-channel means) + mean over channels of "
                    "the spatial/sample variance within a channel",
                ),
                per_channel_mean=dict(
                    min=round(float(ch_mean.min()), 6),
                    max=round(float(ch_mean.max()), 6),
                    mean=round(float(ch_mean.mean()), 6),
                    std=round(float(ch_mean.std()), 6),
                ),
                cellwise_channel_mean=stats(cellwise),
            )
            print(
                f"[ngfm] {letter} {mode}: mean {m.mean():.4f} std {m.std():.4f} "
                f"(between-ch {ch_mean.std():.4f}, within-ch {within.std():.4f})",
                flush=True,
            )
        h.remove()
        out["runs"][letter] = entry
        del model, runner, loader
        torch.cuda.empty_cache()

    # training-time reference from metrics.csv
    import csv as _csv

    for letter in args.runs:
        from eval_test_checkpoints import CKPT_ROOT

        p = osp.join(CKPT_ROOT, RUNS[letter], "metrics.csv")
        rows = [
            r for r in _csv.DictReader(open(p)) if r.get("train/ngfm/spatial_mask_mean")
        ]
        mm = [float(r["train/ngfm/spatial_mask_mean"]) for r in rows]
        ss = [float(r["train/ngfm/spatial_mask_std"]) for r in rows]
        out["runs"][letter]["training_log_reference"] = dict(
            source=osp.relpath(p, REPO_ROOT),
            n_logged_iters=len(mm),
            mask_mean_final=round(mm[-1], 6),
            mask_std_final=round(ss[-1], 6),
            mask_mean_over_run=round(float(np.mean(mm)), 6),
            mask_std_over_run=round(float(np.mean(ss)), 6),
            mask_mean_range=[round(min(mm), 6), round(max(mm), 6)],
            mask_std_range=[round(min(ss), 6), round(max(ss), 6)],
        )

    path = osp.join(REPO_ROOT, "figures", "data", "ngfm_mask_hist.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
