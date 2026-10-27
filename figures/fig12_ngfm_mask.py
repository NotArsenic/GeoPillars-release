#!/usr/bin/env python3
"""Fig. 12: what the normal-guided gate actually does, run A, epoch 8."""

import json
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps

RUN = "A"
MODE = "eval"
FILL = ps.RUN_STYLE["A"]["color"]


def bars(ax, hist, color, alpha):
    edges = np.asarray(hist["bin_edges"], float)
    counts = np.asarray(hist["counts"], float)
    frac = counts / counts.sum()
    ax.bar(
        edges[:-1],
        frac,
        width=np.diff(edges),
        align="edge",
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=3,
    )
    return frac


def main():
    ps.apply()
    doc = json.load(open(osp.join(HERE, "data", "ngfm_mask_hist.json")))
    r = doc["runs"][RUN]
    e = r[MODE]
    ref = r["training_log_reference"]
    pooled, cell = e["pooled"], e["cellwise_channel_mean"]

    fig, (ax, bx) = plt.subplots(
        2,
        1,
        figsize=(ps.COLUMN_W, 3.15),
        gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=0.60),
    )
    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.115, top=0.90)

    # (a) pooled over channels and cells
    frac = bars(ax, e["histogram"], FILL, 0.80)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, frac.max() * 1.42)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("Gate value", labelpad=2)
    ax.set_ylabel("Fraction", labelpad=3)
    ax.grid(axis="x", visible=False)
    ps.despine(ax)
    ax.axvline(pooled["mean"], color=ps.INK, lw=0.7, ls=(0, (3, 1.6)), zorder=4)
    ax.text(
        0.5,
        1.045,
        f"all channels × cells:  μ {pooled['mean']:.4f}   σ {pooled['std']:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=ps.INK,
    )
    ax.text(
        0.985,
        0.90,
        f"training log\n{ref['mask_mean_final']:.4f} / {ref['mask_std_final']:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        color=ps.INK_MUTED,
        linespacing=1.15,
    )
    ax.text(
        -0.145,
        1.10,
        "(a)",
        transform=ax.transAxes,
        fontsize=7.5,
        color=ps.INK,
        va="bottom",
    )

    # (b) averaged over the channel axis, same x-axis
    bars(bx, e["cellwise_histogram"], ps.INK_MUTED, 0.85)
    bx.set_xlim(0, 1)
    bx.set_xticks(np.arange(0, 1.01, 0.25))
    bx.set_xlabel("Channel-mean gate at a BEV cell", labelpad=2)
    bx.set_ylabel("Fraction", labelpad=3)
    bx.grid(axis="x", visible=False)
    ps.despine(bx)
    bx.text(
        0.5,
        1.045,
        f"per cell:  μ {cell['mean']:.4f}   σ {cell['std']:.4f}",
        transform=bx.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=ps.INK,
    )
    bx.text(
        -0.145,
        1.10,
        "(b)",
        transform=bx.transAxes,
        fontsize=7.5,
        color=ps.INK,
        va="bottom",
    )

    ins = bx.inset_axes([0.605, 0.34, 0.375, 0.58])
    z = e["cellwise_histogram_zoom"]
    ze = np.asarray(z["bin_edges"], float)
    zc = np.asarray(z["counts"], float)
    ins.bar(
        ze[:-1],
        zc / zc.sum(),
        width=np.diff(ze),
        align="edge",
        color=ps.INK_MUTED,
        alpha=0.85,
        linewidth=0,
        zorder=3,
    )
    ins.set_xlim(0.44, 0.56)
    ins.set_xticks([0.45, 0.50, 0.55])
    ins.set_yticks([])
    ins.tick_params(labelsize=6.3, pad=1.5)
    ins.grid(False)
    for side in ("top", "right", "left"):
        ins.spines[side].set_visible(False)
    ins.set_title("0.44–0.56, ×0.001 bins", fontsize=6.3, color=ps.INK_MUTED, pad=1.5)

    ps.save(fig, "fig12_ngfm_mask")


if __name__ == "__main__":
    main()
