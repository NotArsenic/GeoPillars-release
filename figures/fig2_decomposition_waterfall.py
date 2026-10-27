#!/usr/bin/env python3
"""Fig. 2: decomposition of the 13.76-point AP@0.7 gap (CARLA test), as a waterfall."""

import json
import os.path as osp
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps

Y0, Y1 = 60.0, 84.5
STEP_COLOR = "#5E6E7E"
NGFM_COLOR = "#AAB4BE"
NGFM_NOTE = "within single-run variance"


def break_marks(ax, x, width, y):
    """White zig-zag across a bar (or the y-axis when width is 0) to mark the axis break."""
    h = 0.55
    for dy in (0.0, 0.75):
        xs = [x - width / 2 - 0.02, x + width / 2 + 0.02]
        ax.add_patch(
            Polygon(
                [
                    [xs[0], y + dy - h / 2],
                    [xs[1], y + dy + h / 2],
                    [xs[1], y + dy + h / 2 + 0.28],
                    [xs[0], y + dy - h / 2 + 0.28],
                ],
                closed=True,
                fc="white",
                ec="none",
                zorder=5,
                clip_on=False,
            )
        )


def main():
    ps.apply()
    dec = json.load(open(osp.join(HERE, "data", "paper_metrics.json")))["decomposition"]
    start, end = dec["start"]["ap"], dec["end"]["ap"]
    inc = dec["increments"]
    gap = dec["total_gap"]

    labels = [
        "D\nCP, OSF=4\n(budget)",
        "+ head\nresolution\nOSF 4→1",
        "+ encoder,\nbackbone,\nupsampler",
        "+ NGFM",
        "A\nGeoPillars",
    ]
    fig, ax = plt.subplots(figsize=(ps.COLUMN_W, 2.3))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.25, top=0.97)
    w = 0.62

    # totals
    for x, val, letter in ((0, start, "D"), (4, end, "A")):
        ax.bar(
            x,
            val - Y0 + 3,
            bottom=Y0 - 3,
            width=w,
            color=ps.RUN_STYLE[letter]["color"],
            zorder=3,
            linewidth=0,
        )
        ax.text(
            x,
            val + 0.45,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=ps.INK,
            zorder=6,
        )
        break_marks(ax, x, w, Y0 + 0.9)

    # increments
    level = start
    for i, step in enumerate(inc, start=1):
        is_ngfm = step["to"] == "A"
        color = NGFM_COLOR if is_ngfm else STEP_COLOR
        ax.bar(
            i, step["delta"], bottom=level, width=w, color=color, zorder=3, linewidth=0
        )
        share = 100 * step["delta"] / gap
        ax.text(
            i,
            level + step["delta"] + 0.45,
            f"+{step['delta']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=ps.INK,
            zorder=6,
        )
        if not is_ngfm:
            ax.text(
                i,
                level + step["delta"] / 2,
                f"{share:.0f}%\nof gap",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white",
                zorder=6,
                linespacing=1.0,
            )
        # connector from the previous bar's top edge to this bar
        ax.plot(
            [i - 1 + w / 2, i - w / 2],
            [level, level],
            color=ps.INK_MUTED,
            lw=0.5,
            zorder=2,
        )
        level += step["delta"]
    ax.plot(
        [3 + w / 2, 4 - w / 2], [level, level], color=ps.INK_MUTED, lw=0.5, zorder=2
    )

    # NGFM annotation
    ngfm = inc[-1]
    note = f"{100 * ngfm['delta'] / gap:.1f}% of gap,\n" + NGFM_NOTE.replace(
        "single-run ", "single-run\n"
    )
    ax.annotate(
        note,
        xy=(3, ngfm["from_ap"] - 0.15),
        xytext=(3, 74.6),
        ha="center",
        va="top",
        fontsize=6.5,
        color=ps.INK,
        linespacing=1.1,
        arrowprops=dict(
            arrowstyle="-", color=ps.INK_MUTED, lw=0.5, shrinkA=2, shrinkB=0
        ),
    )

    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(Y0 - 3, Y1)
    ax.set_yticks([60, 65, 70, 75, 80])
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, fontsize=7, linespacing=1.05)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylabel("AP@0.7 (%), CARLA val")
    ax.grid(axis="x", visible=False)
    ps.despine(ax)
    ax.spines["bottom"].set_visible(False)
    # axis break on the y spine
    for dy in (0.0, 0.75):
        ax.plot(
            [-0.62, -0.48],
            [Y0 - 1.6 + dy, Y0 - 1.1 + dy],
            color=ps.INK_MUTED,
            lw=0.6,
            clip_on=False,
            zorder=6,
        )
    ax.add_patch(
        Polygon(
            [
                [-0.62, Y0 - 1.6],
                [-0.48, Y0 - 1.1],
                [-0.48, Y0 - 0.35],
                [-0.62, Y0 - 0.85],
            ],
            closed=True,
            fc="white",
            ec="none",
            clip_on=False,
            zorder=5,
            transform=ax.transData,
        )
    )
    ps.save(fig, "fig2_decomposition_waterfall")


if __name__ == "__main__":
    main()
