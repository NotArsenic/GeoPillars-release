"""Measure what the GeoPillars encoder actually learned: GVA gate and BEV prior.

Two questions the paper leaves bracketed:

1. ``GroupedVectorAttention.gamma`` initialises to 0, so the attention block
   starts as the identity. Reading ``gamma`` and ``relation_scale`` off
   ``epoch_8.pth``, and measuring ‖γ·out‖ / ‖x‖ on a real batch, says whether
   the block ended up doing anything.
2. ``bev_pos_encoding`` is 4.19 M of the model's 7.28 M parameters, added to the
   BEV canvas after the scatter. Comparing its magnitude against the scattered
   pillar features - separately over occupied and empty cells - says whether it
   is the dominant term in the BEV representation.

Both use forward hooks and exact subtraction: the GVA block returns
``x + γ·out`` so ``γ·out = y − x``, and the encoder returns
``canvas + bev_pos_encoding`` so the pre-prior canvas is ``out − prior``.

Writes ``figures/data/encoder_internals.json``.
"""

import argparse
import json
import os.path as osp
import sys

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from _carla_infer import RUNS, build  # noqa: E402

from eval_test_checkpoints import checkpoint_path  # noqa: E402


def tensor_stats(t):
    t = t.detach().float()
    return dict(
        numel=int(t.numel()),
        abs_mean=round(float(t.abs().mean()), 6),
        mean=round(float(t.mean()), 6),
        std=round(float(t.std()), 6),
        min=round(float(t.min()), 6),
        max=round(float(t.max()), 6),
        l2=round(float(t.norm()), 4),
    )


def read_checkpoint_params(run, epoch=8):
    """gamma / relation_scale / bev_pos_encoding straight out of the checkpoint."""
    import torch

    ckpt = checkpoint_path(run, epoch)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    out = {}
    for key in ("gamma", "relation_scale", "bev_pos_encoding"):
        hits = [k for k in sd if k.endswith(key)]
        if len(hits) != 1:
            out[key] = dict(error=f"{len(hits)} matches: {hits}")
            continue
        t = sd[hits[0]]
        rec = dict(parameter=hits[0], shape=list(t.shape), **tensor_stats(t))
        if key == "gamma":
            rec["value"] = round(float(t.reshape(-1)[0]), 8)
        if key == "relation_scale":
            rec["values"] = [round(float(v), 6) for v in t.reshape(-1)]
        out[key] = rec
    return out


def measure_forward(run, batch_size=8, epoch=8, device="cuda"):
    """One real batch through the trained model, with hooks on the encoder."""
    import torch
    from src.models.components.encoders.pillar_feature_net import (
        GeoPillarFeatureNet,
        GroupedVectorAttention,
    )

    runner, model, loader, cfg = build(
        run,
        indices=list(range(batch_size)),
        batch_size=batch_size,
        num_workers=2,
        epoch=epoch,
    )
    gva = [m for m in model.modules() if isinstance(m, GroupedVectorAttention)]
    pfn = [m for m in model.modules() if isinstance(m, GeoPillarFeatureNet)]
    assert len(gva) == 1 and len(pfn) == 1, (len(gva), len(pfn))
    gva, pfn = gva[0], pfn[0]
    cap = {}

    def gva_hook(mod, args, kwargs, out):
        x = args[0]
        mask = kwargs.get("mask", args[1] if len(args) > 1 else None)
        resid = (out - x).detach()  # exactly γ · attention output
        x = x.detach()
        if mask is not None:
            m = mask.detach().unsqueeze(-1)
            xv, rv = x[m.expand_as(x)], resid[m.expand_as(resid)]
        else:
            xv, rv = x.reshape(-1), resid.reshape(-1)
        # per-point ratio, over valid points only
        if mask is not None:
            xn = (x * mask.unsqueeze(-1)).norm(dim=-1)
            rn = (resid * mask.unsqueeze(-1)).norm(dim=-1)
            sel = mask.detach()
            ratio = rn[sel] / xn[sel].clamp_min(1e-12)
        else:
            ratio = resid.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-12)
        cap["gva"] = dict(
            x=tensor_stats(xv),
            gamma_times_out=tensor_stats(rv),
            n_valid_points=int(sel.sum()) if mask is not None else int(x.numel()),
            n_padded_slots=int((~sel).sum()) if mask is not None else 0,
            global_l2_ratio=round(float(rv.norm() / xv.norm().clamp_min(1e-12)), 8),
            per_point_l2_ratio=dict(
                mean=round(float(ratio.mean()), 8),
                median=round(float(ratio.median()), 8),
                p99=round(float(torch.quantile(ratio.float(), 0.99)), 8),
                max=round(float(ratio.max()), 8),
            ),
        )

    def pfn_hook(mod, args, out):
        prior = mod.bev_pos_encoding.detach().float()
        after = out.detach().float()
        before = after - prior  # the scattered canvas
        coors = args[2].detach()
        occ = torch.zeros(
            after.shape[0],
            after.shape[2],
            after.shape[3],
            dtype=torch.bool,
            device=after.device,
        )
        occ[coors[:, 0].long(), coors[:, 2].long(), coors[:, 3].long()] = True
        occ_c = occ.unsqueeze(1).expand_as(before)
        p_b = prior.expand_as(before)

        def grp(name, sel):
            n = int(sel.sum())
            if n == 0:
                return {name: dict(n_elements=0)}
            cb, pb = before[sel], p_b[sel]
            return {
                name: dict(
                    n_cells=int(sel[:, 0].sum()) if sel.dim() == 4 else None,
                    n_elements=n,
                    canvas_before_prior=tensor_stats(cb),
                    prior=tensor_stats(pb),
                    abs_mean_ratio_prior_over_canvas=(
                        round(float(pb.abs().mean() / cb.abs().mean()), 6)
                        if float(cb.abs().mean()) > 0
                        else None
                    ),
                    l2_ratio_prior_over_canvas=(
                        round(float(pb.norm() / cb.norm()), 6)
                        if float(cb.norm()) > 0
                        else None
                    ),
                )
            }

        cap["prior"] = dict(
            grid=list(after.shape),
            occupied_cells=int(occ.sum()),
            empty_cells=int((~occ).sum()),
            occupancy_fraction=round(float(occ.float().mean()), 6),
            prior_whole_tensor=tensor_stats(prior),
            canvas_before_prior_whole_tensor=tensor_stats(before),
            canvas_after_prior_whole_tensor=tensor_stats(after),
            **grp("occupied", occ_c),
            **grp("empty", ~occ_c),
            empty_cells_are_exactly_zero_before_prior=bool(
                float(before[~occ_c].abs().max()) == 0.0
            ),
            fraction_of_canvas_energy_from_prior=round(
                float(p_b.pow(2).sum() / after.pow(2).sum()), 6
            ),
        )

    h1 = gva.register_forward_hook(gva_hook, with_kwargs=True)
    h2 = pfn.register_forward_hook(pfn_hook)
    try:
        with torch.no_grad():
            for data_batch in loader:
                model.test_step(data_batch)
                break
    finally:
        h1.remove()
        h2.remove()
    cap["batch"] = dict(
        batch_size=batch_size,
        frames=batch_size,
        device=str(next(model.parameters()).device),
    )
    return cap


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="+", default=["A", "B", "G"])
    ap.add_argument(
        "--forward-runs",
        nargs="+",
        default=["A"],
        help="runs to also measure with a real forward pass",
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epoch", type=int, default=8)
    ap.add_argument(
        "--out",
        default=osp.join(REPO_ROOT, "figures", "data", "encoder_internals.json"),
    )
    args = ap.parse_args()

    out = dict(
        description=(
            "Trained values of the GroupedVectorAttention gate (gamma, "
            "relation_scale) and of bev_pos_encoding, plus what each contributes on "
            "a real batch. gamma initialises to 0 and bev_pos_encoding to "
            "randn*0.01."
        ),
        epoch=args.epoch,
        runs={},
    )
    for L in args.runs:
        run = RUNS[L]
        rec = dict(run=run, checkpoint_params=read_checkpoint_params(run, args.epoch))
        g = rec["checkpoint_params"].get("gamma", {})
        print(
            f'[enc] {L} {run}: gamma={g.get("value")} '
            f'relation_scale={rec["checkpoint_params"]["relation_scale"].get("values")}',
            flush=True,
        )
        if L in args.forward_runs:
            rec["forward"] = measure_forward(run, args.batch_size, args.epoch)
            gv = rec["forward"]["gva"]
            pr = rec["forward"]["prior"]
            print(
                f"[enc] {L} forward: ||gamma*out||/||x|| global="
                f'{gv["global_l2_ratio"]} per-point mean={gv["per_point_l2_ratio"]["mean"]}',
                flush=True,
            )
            print(
                f"[enc] {L} prior/canvas abs-mean ratio: occupied="
                f'{pr["occupied"]["abs_mean_ratio_prior_over_canvas"]} '
                f'empty_cells_zero_before_prior={pr["empty_cells_are_exactly_zero_before_prior"]} '
                f'occupancy={pr["occupancy_fraction"]}',
                flush=True,
            )
        out["runs"][L] = rec

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[enc] -> {args.out}")


if __name__ == "__main__":
    main()
