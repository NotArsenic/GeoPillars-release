#!/usr/bin/env python3
"""Fig. 6: validation AP@0.7 at epochs 2/4/6/8 for all seven runs."""

import json
import os.path as osp
import sys

import matplotlib.pyplot as plt

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps

ORDER = ["E", "B", "D", "C", "F", "G", "A"]
LEGEND = ["A", "B", "G", "C", "F", "D", "E"]


def main():
    ps.apply()
    runs = json.load(open(osp.join(HERE, "data", "paper_metrics.json")))["runs"]

    fig, ax = plt.subplots(figsize=(ps.COLUMN_W, 2.75))
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.36, top=0.98)
    handles, ends = {}, {}
    for z, k in enumerate(ORDER):
        tv = runs[k]["training_val"]
        y = [100 * v for v in tv["AP@0.7"]]
        handles[k] = ps.run_line(
            ax, tv["epochs"], y, k, lw=1.0 if k in "BE" else 1.15, ms=3.4, z=3 + z
        )
        ends[k] = y[-1]

    ax.set_xlim(1.7, 8.3)
    ax.set_ylim(40, 85)
    ax.set_xticks([2, 4, 6, 8])
    ax.set_xlabel("Epoch", labelpad=2)
    ax.set_ylabel("Validation AP@0.7 (%)", labelpad=3)
    ax.grid(axis="x", visible=False)
    ps.despine(ax)

    letters = sorted(ends, key=ends.get)
    pos = ps.spread([ends[k] for k in letters], min_gap=2.45, hi=85.5)
    for k, yl in zip(letters, pos):
        ax.annotate(
            f"{k}  {ends[k]:.1f}",
            xy=(8, ends[k]),
            xytext=(8.55, yl),
            ha="left",
            va="center",
            fontsize=6.5,
            color=ps.INK,
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-", color=ps.INK_MUTED, lw=0.4, shrinkA=0, shrinkB=3
            ),
        )

    fig.legend(
        [handles[k] for k in LEGEND],
        [f'{k}  {ps.RUN_STYLE[k]["label"]}' for k in LEGEND],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=2.4,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.3,
        fontsize=6.5,
    )
    ps.save(fig, "fig6_val_ap_across_epochs")


if __name__ == "__main__":
    main()
