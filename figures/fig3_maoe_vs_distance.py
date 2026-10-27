#!/usr/bin/env python3
"""Fig. 3: orientation error vs distance, folded against unfolded, plus heading-flip rate."""

import json
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps

ORDER = ["C", "F", "G", "A"]
BARS = ["A", "G", "D", "B", "E", "F", "C"]
BINS = ["0-20m", "20-40m", "40m+"]
BIN_LABELS = ["0–20", "20–40", "≥40"]


def main():
    ps.apply()
    runs = json.load(open(osp.join(HERE, "data", "paper_metrics.json")))["runs"]
    x = np.arange(len(BINS))

    fig, (ax, bx) = plt.subplots(
        2,
        1,
        figsize=(ps.COLUMN_W, 3.55),
        gridspec_kw=dict(height_ratios=[1.62, 1.0], hspace=0.62),
    )
    fig.subplots_adjust(left=0.145, right=0.865, bottom=0.115, top=0.965)

    # (a) folded vs unfolded
    ends, handles = {}, {}
    for z, letter in enumerate(ORDER):
        st = ps.RUN_STYLE[letter]
        unf = [runs[letter]["carla_test"][f"mAOE@{b}"] for b in BINS]
        ax.plot(
            x,
            unf,
            color=st["color"],
            ls=st["ls"],
            lw=0.7,
            alpha=0.42,
            zorder=2 + z,
            clip_on=False,
        )
        fold = [runs[letter]["carla_test"][f"mAOE_folded@{b}"] for b in BINS]
        handles[letter] = ps.run_line(ax, x, fold, letter, z=6 + z)
        ends[letter] = fold[-1]

    ymax = 0.47
    ax.set_xlim(-0.2, 2.72)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Distance to sensor (m)", labelpad=2)
    ax.set_ylabel("Orientation error (rad)", labelpad=3)
    ax.set_yticks(np.arange(0, 0.46, 0.1))
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)

    sec = ax.secondary_yaxis("right", functions=(np.rad2deg, np.deg2rad))
    sec.set_ylabel("(°)", labelpad=2)
    sec.set_yticks([0, 5, 10, 15, 20, 25])
    sec.spines["right"].set_color(ps.INK_MUTED)
    sec.spines["right"].set_linewidth(0.5)
    sec.tick_params(colors=ps.INK_MUTED)

    ax.text(
        1.96,
        runs["C"]["carla_test"]["mAOE@40m+"] + 0.012,
        "unfolded",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=ps.INK_MUTED,
    )

    letters = sorted(ends, key=ends.get)
    pos = ps.spread([ends[k] for k in letters], min_gap=0.032, lo=0.02, hi=ymax - 0.06)
    for letter, yl in zip(letters, pos):
        ax.annotate(
            f"{letter}  {np.rad2deg(ends[letter]):.1f}°",
            xy=(2, ends[letter]),
            xytext=(2.13, yl),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=6.5,
            color=ps.INK,
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-", color=ps.INK_MUTED, lw=0.4, shrinkA=0, shrinkB=3
            ),
        )

    legend_order = ["A", "G", "F", "C"]
    ax.legend(
        [handles[k] for k in legend_order],
        [f"{k}  {ps.short(k)}" for k in legend_order],
        loc="upper left",
        handlelength=2.6,
        handletextpad=0.5,
        labelspacing=0.3,
        borderaxespad=0.15,
        title="folded",
        title_fontsize=6.5,
    )
    ax.get_legend().get_title().set_color(ps.INK_MUTED)
    ax.text(
        -0.132,
        1.03,
        "(a)",
        transform=ax.transAxes,
        fontsize=7.5,
        color=ps.INK,
        va="bottom",
    )

    # (b) heading-flip rate
    rates = [100 * runs[k]["carla_test"]["heading_flip_rate"] for k in BARS]
    ypos = np.arange(len(BARS))[::-1]
    for y, letter, r in zip(ypos, BARS, rates):
        st = ps.RUN_STYLE[letter]
        bx.barh(
            y,
            r,
            height=0.62,
            color="white" if st.get("hollow") else st["color"],
            edgecolor=st["color"],
            linewidth=0.8,
            zorder=3,
        )
        bx.text(
            r + 0.09,
            y,
            f"{r:.2f}",
            va="center",
            ha="left",
            fontsize=6.5,
            color=ps.INK,
            zorder=4,
        )
    bx.set_yticks(ypos)
    bx.set_yticklabels([f"{k}" for k in BARS], fontsize=7)
    bx.set_ylim(-0.6, len(BARS) - 0.4)
    bx.set_xlim(0, 4.35)
    bx.set_xticks([0, 1, 2, 3, 4])
    bx.set_xlabel("Heading-flip rate (% of matches)", labelpad=2)
    bx.grid(axis="y", visible=False)
    ps.despine(bx)
    bx.text(
        -0.132,
        1.06,
        "(b)",
        transform=bx.transAxes,
        fontsize=7.5,
        color=ps.INK,
        va="bottom",
    )

    ps.save(fig, "fig3_maoe_vs_distance")


if __name__ == "__main__":
    main()
