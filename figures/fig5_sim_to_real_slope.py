#!/usr/bin/env python3
"""Fig. 5: sim-to-real collapse, AP@0.7 on CARLA test vs zero-shot nuScenes."""

import json
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps

ORDER = ["D", "F", "G", "C", "A"]
YLO, YHI = 2.0, 100.0


def main():
    ps.apply()
    runs = json.load(open(osp.join(HERE, "data", "paper_metrics.json")))["runs"]
    carla = {k: 100 * runs[k]["carla_test"]["AP@0.7"] for k in ORDER}
    nusc = {k: 100 * runs[k]["nusc_zeroshot"]["AP@0.7"] for k in ORDER}

    fig, ax = plt.subplots(figsize=(ps.COLUMN_W, 2.75))
    fig.subplots_adjust(left=0.115, right=0.5, bottom=0.26, top=0.9)
    for z, k in enumerate(ORDER):
        ps.run_line(ax, [0, 1], [carla[k], nusc[k]], k, lw=1.1, ms=3.8, z=3 + z)

    ax.set_yscale("log")
    ax.set_ylim(YLO, YHI)
    ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
    ax.set_ylabel("AP@0.7 (%), log scale", labelpad=2)
    ax.set_xlim(-0.1, 1.06)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CARLA\nval", "nuScenes\nzero-shot"])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=3)
    ax.grid(axis="x", visible=False)
    for side in ("bottom", "right", "top"):
        ax.spines[side].set_visible(False)

    right = sorted(ORDER, key=nusc.get)
    rpos = 10 ** ps.spread(
        [np.log10(nusc[k]) for k in right], min_gap=0.078, lo=np.log10(YLO) + 0.03
    )
    for k, yl in zip(right, rpos):
        ax.annotate(
            f"{k}  {carla[k]:.1f} → {nusc[k]:.1f}  ({100 * nusc[k] / carla[k]:.1f}%)",
            xy=(1, nusc[k]),
            xytext=(1.14, yl),
            ha="left",
            va="center",
            fontsize=6.5,
            color=ps.INK,
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-", color=ps.INK_MUTED, lw=0.4, shrinkA=0, shrinkB=3
            ),
        )
    ax.text(
        1.14,
        10 ** (np.log10(YHI) - 0.02),
        "AP CARLA → nuScenes\n(share retained)",
        ha="left",
        va="top",
        fontsize=6.5,
        color=ps.INK_MUTED,
        transform=ax.transData,
        clip_on=False,
    )

    handles = [
        plt.Line2D(
            [],
            [],
            color=ps.RUN_STYLE[k]["color"],
            ls=ps.RUN_STYLE[k]["ls"],
            lw=1.1,
            marker=ps.RUN_STYLE[k]["marker"],
            ms=3.6,
            mec="white",
            mew=0.5,
        )
        for k in reversed(ORDER)
    ]
    fig.legend(
        handles,
        [f"{k}  {ps.short(k)}" for k in reversed(ORDER)],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=2.4,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.3,
        fontsize=6.5,
    )
    ps.save(fig, "fig5_sim_to_real_slope")


if __name__ == "__main__":
    main()
