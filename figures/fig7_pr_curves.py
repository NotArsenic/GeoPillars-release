#!/usr/bin/env python3
"""Fig. 7: car PR curves at IoU 0.7, CARLA val vs zero-shot nuScenes."""

import json
import os.path as osp
import sys

import matplotlib.pyplot as plt

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps

ORDER = ["A", "F", "C", "D"]
PANELS = [("carla_test", "(a) CARLA val"), ("nusc_zeroshot", "(b) nuScenes, zero-shot")]
IOU = "0.7"


def main():
    ps.apply()
    data = json.load(open(osp.join(HERE, "data", "pr_curves.json")))["curves"]

    fig, axes = plt.subplots(
        1, 2, figsize=(ps.COLUMN_W, 1.95), sharex=True, sharey=True
    )
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.2, top=0.74, wspace=0.09)

    handles = {}
    for ax, (ds, title) in zip(axes, PANELS):
        for z, letter in enumerate(reversed(ORDER)):
            st = ps.RUN_STYLE[letter]
            c = data[ds][letter][IOU]["all"]
            (h,) = ax.plot(
                c["recall"],
                c["precision"],
                color=st["color"],
                ls=st["ls"],
                lw=1.1,
                zorder=3 + z,
            )

            ax.plot(
                c["recall"][-1],
                c["precision"][-1],
                "o",
                ms=3.2,
                mfc=st["color"],
                mec="white",
                mew=0.7,
                zorder=10 + z,
                clip_on=False,
            )
            handles[letter] = h
        ax.set_title(title, pad=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0", ".25", ".5", ".75", "1"])
        ax.set_xlabel("Recall", labelpad=1.5)
        ax.tick_params(pad=1.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        lines = [
            f"{letter}  {100 * data[ds][letter][IOU]['all']['ap_r40']:4.1f}"
            for letter in ORDER
        ]
        x, y, va = (0.04, 0.05, "bottom") if ds == "carla_test" else (0.66, 0.97, "top")
        ax.text(
            x,
            y,
            "AP@0.7\n" + "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va=va,
            fontsize=6.5,
            color=ps.INK,
            linespacing=1.15,
        )
    axes[0].set_ylabel("Precision", labelpad=2)

    fig.legend(
        [handles[k] for k in ORDER],
        [f"{k}  {ps.RUN_STYLE[k]['label']}" for k in ORDER],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.55, 1.005),
        handlelength=2.6,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.1,
        labelspacing=0.25,
    )

    out = osp.join(HERE, "fig7_pr_curves")
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=450)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
