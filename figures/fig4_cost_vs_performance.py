#!/usr/bin/env python3
"""Fig. 4: accuracy vs inference cost of the released checkpoints (CARLA test)."""

import json
import os.path as osp
import sys

import matplotlib.pyplot as plt

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps

LETTERS = ["D", "C", "F", "G", "A"]
AREA_PER_MB = 0.55

LABEL = {
    "A": ((-4, 14), "right", "bottom"),
    "G": ((-4, -14), "right", "top"),
    "C": ((-10, 8), "right", "bottom"),
    "F": ((10, -8), "left", "top"),
    "D": ((10, 0), "left", "center"),
}


def main():
    ps.apply()
    doc = json.load(open(osp.join(HERE, "data", "inference_cost.json")))
    runs = doc["runs"]

    fig, ax = plt.subplots(figsize=(ps.COLUMN_W, 2.3))
    fig.subplots_adjust(left=0.135, right=0.975, bottom=0.17, top=0.97)
    for z, letter in enumerate(LETTERS):  # A last, on top
        r, st = runs[letter], ps.RUN_STYLE[letter]
        x, y = r["latency_ms_mean"], 100 * r["ap_0.7_carla_test"]

        ax.scatter(
            [x],
            [y],
            s=AREA_PER_MB * r["peak_vram_allocated_mb"],
            marker="o",
            c=st["color"],
            alpha=0.8,
            edgecolors="white",
            linewidths=0.8,
            zorder=4 + z,
        )
        ax.scatter(
            [x], [y], s=4, marker="o", c=ps.INK, linewidths=0, zorder=10
        )  # exact position
        (dx, dy), ha, va = LABEL[letter]
        ax.annotate(
            f"{letter}  {ps.short(letter)}\n{y:.1f} AP, {x:.1f} ms, {r['peak_vram_allocated_mb']:.0f} MiB",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=6.3,
            color=ps.INK,
            linespacing=1.1,
            zorder=11,
            arrowprops=dict(
                arrowstyle="-", color=ps.INK_MUTED, lw=0.4, shrinkA=1, shrinkB=1.5
            ),
        )

    # marker-size key
    key_x, key_y = 29.6, 70.2
    ax.text(
        key_x - 1.0,
        key_y + 1.6,
        "Peak VRAM, batch 1",
        fontsize=6.3,
        color=ps.INK_MUTED,
        ha="left",
        va="bottom",
    )
    for i, mb in enumerate((100, 200, 300)):
        yy = key_y - i * 2.1
        ax.scatter(
            [key_x],
            [yy],
            s=AREA_PER_MB * mb,
            marker="o",
            facecolors="none",
            edgecolors=ps.INK_MUTED,
            linewidths=0.6,
            zorder=3,
        )
        ax.text(
            key_x + 1.6,
            yy,
            f"{mb} MiB",
            fontsize=6.3,
            color=ps.INK_MUTED,
            ha="left",
            va="center",
        )

    ax.set_xlim(0, 38)
    ax.set_ylim(64, 85)
    ax.set_yticks([65, 70, 75, 80, 85])
    ax.set_xlabel(
        "Latency per sample (ms), batch 1, "
        + doc["environment"]["gpu"].replace("NVIDIA GeForce ", ""),
        labelpad=2,
    )
    ax.set_ylabel("AP@0.7 (%), CARLA val", labelpad=3)
    ps.despine(ax)
    ps.save(fig, "fig4_cost_vs_performance")


if __name__ == "__main__":
    main()
