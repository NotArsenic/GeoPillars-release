#!/usr/bin/env python3
"""Fig. 9: how full the pillars are under the GeoPillars voxelisation budget (CARLA train split)."""

import json
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps

KEPT = "#5E6E7E"  # pillars within the point cap
TRUNC = "#B7C0C9"  # pillars the cap truncates
CAP_LINE = ps.RUN_STYLE["A"]["color"]
SPLIT, BUDGET = "train", "geopillars"
MAX_K = 24


def thousands(x, _):
    return f"{x / 1000:g}K" if x else "0"


def main():
    ps.apply()
    doc = json.load(open(osp.join(HERE, "data", "pillar_stats.json")))
    s = doc["splits"][SPLIT]
    b = s["budgets"][BUDGET]
    cfg = doc["budgets"][BUDGET]
    cap_pts, cap_pillars = cfg["max_points"], cfg["max_voxels_train"]
    hist = np.asarray(b["pillar_point_count_hist"]["counts"], dtype=np.float64)
    total = hist[1:].sum()
    pct = 100 * np.append(hist[1:MAX_K], hist[MAX_K:].sum()) / total
    occ = np.load(osp.join(HERE, "data", "pillar_stats_scans.npz"))[
        f"{SPLIT}_{BUDGET}_occupied"
    ]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(ps.COLUMN_W, 3.05), gridspec_kw=dict(height_ratios=[1.25, 1])
    )
    fig.subplots_adjust(left=0.125, right=0.975, bottom=0.115, top=0.94, hspace=0.66)

    # (a) points per occupied pillar
    k = np.arange(1, MAX_K + 1)
    ax1.bar(
        k,
        pct,
        width=0.72,
        color=np.where(k <= cap_pts, KEPT, TRUNC),
        linewidth=0,
        zorder=3,
    )
    ax1.axvline(cap_pts + 0.5, color=CAP_LINE, lw=0.9, zorder=4)
    ax1.text(
        cap_pts + 0.75,
        21.5,
        f"{cap_pts}-point cap",
        color=ps.INK,
        fontsize=6.5,
        ha="left",
        va="top",
    )
    over = 100 * b["pillars_over_point_cap_frac"]
    dropped = 100 * b["in_range_points_dropped_by_point_cap_frac"]
    single = 100 * hist[1] / total
    ax1.text(
        MAX_K + 0.4,
        15.5,
        f"{over:.1f}% of pillars exceed the cap,\nremoving {dropped:.1f}% of in-range points",
        ha="right",
        va="top",
        fontsize=6.5,
        color=ps.INK,
        linespacing=1.15,
    )
    ax1.text(
        1.55,
        single + 0.2,
        f"{single:.1f}%",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=ps.INK,
    )
    ax1.set_xlim(0.3, MAX_K + 0.7)
    ax1.set_ylim(0, 24)
    ax1.set_xticks([1, 4, 8, 12, 16, 20, MAX_K])
    ax1.set_xticklabels(["1", "4", "8", "12", "16", "20", f"≥{MAX_K}"])
    ax1.set_xlabel("Points in an occupied pillar (before the cap)", labelpad=2)
    ax1.set_ylabel("Pillars (%)", labelpad=3)
    ax1.grid(axis="x", visible=False)
    ax1.tick_params(axis="x", length=0, pad=2)
    ps.despine(ax1)
    ax1.text(
        -0.1, 1.02, "(a)", transform=ax1.transAxes, ha="left", va="bottom", fontsize=8
    )

    # (b) occupied pillars per scan vs the pillar cap
    bins = np.arange(0, cap_pillars + 1001, 200)
    counts, edges = np.histogram(occ, bins=bins)
    share = 100 * counts / len(occ)
    ytop = float(np.ceil(share.max() * 1.15 / 2) * 2)
    ax2.bar(edges[:-1] + 100, share, width=160, color=KEPT, linewidth=0, zorder=3)
    ax2.axvline(cap_pillars, color=CAP_LINE, lw=0.9, zorder=4)
    ax2.text(
        cap_pillars - 250,
        0.95 * ytop,
        f"{cap_pillars // 1000}K training cap\n(test cap 40K)",
        ha="right",
        va="top",
        fontsize=6.5,
        color=ps.INK,
        linespacing=1.15,
    )
    ax2.text(
        occ.max() + 400,
        0.5 * ytop,
        f"max {occ.max():,} pillars; {int((occ > cap_pillars).sum())} of {len(occ):,} scans\n"
        f"reach the cap (median {int(np.median(occ)):,})",
        ha="left",
        va="center",
        fontsize=6.5,
        color=ps.INK,
        linespacing=1.15,
    )
    ax2.set_xlim(0, cap_pillars + 1000)
    ax2.set_ylim(0, ytop)
    ax2.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax2.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000])
    ax2.set_xlabel("Occupied pillars per scan", labelpad=2)
    ax2.set_ylabel("Scans (%)", labelpad=3)
    ax2.grid(axis="x", visible=False)
    ps.despine(ax2)
    ax2.text(
        -0.1, 1.02, "(b)", transform=ax2.transAxes, ha="left", va="bottom", fontsize=8
    )
    ps.save(fig, "fig9_pillar_occupancy")
    print(
        f'{s["scans"]} {SPLIT} scans: single-point pillars {single:.1f}%, over cap {over:.1f}%, '
        f"points dropped {dropped:.1f}%, occupied median {np.median(occ):.0f} max {occ.max()}"
    )


if __name__ == "__main__":
    main()
