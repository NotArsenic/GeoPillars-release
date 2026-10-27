#!/usr/bin/env python3
"""Fig. 8: qualitative BEV detections, GT / GeoPillars (A) / CenterPoint OSF=4 (D)."""

import argparse
import glob
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps

DATA = osp.join(HERE, "data")
DEFAULT_CARLA = osp.join(DATA, "carla_qualitative", "scene_Town10HD_Opt__056653.npz")
DEFAULT_NUSC = osp.join(
    DATA, "nusc_qualitative", "scene_9657675e78c24c19b409ee9c04a135c3.npz"
)
COLUMNS = [
    ("gt", "Ground truth", None),
    ("geopillars", "A  GeoPillars", "A"),
    ("centerpoint_budget_osf4", "D  CenterPoint OSF=4", "D"),
]
A_KEY, D_KEY = "geopillars", "centerpoint_budget_osf4"
GROUND_Z = -1.4
GT_LINE = "#1a1a1a"
GT_REF_LINE = "#b8b8b8"
GROUND_PTS = "#cfcfcf"
OBJECT_PTS = "#6f6f6f"
FLIP_GOOD_DEG, FLIP_BAD_DEG, FLIP_MIN_RANGE = 15.0, 90.0, 20.0


def box_segments(boxes):
    """BEV outline (4 edges) plus a heading tick (centre to front-edge midpoint) per box."""
    segs, ticks = [], []
    for x, y, _, l, w, _, yaw in boxes[:, :7]:
        c, s = np.cos(yaw), np.sin(yaw)
        corners = np.array(
            [[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]]
        )
        rot = corners @ np.array([[c, s], [-s, c]]) + [x, y]
        segs += [rot[[i, (i + 1) % 4]] for i in range(4)]
        ticks.append(np.array([[x, y], [x + c * l / 2, y + s * l / 2]]))
    return segs, ticks


def add_boxes(ax, boxes, color, lw, ls="-", z=3, tick=True):
    if len(boxes) == 0:
        return
    segs, ticks = box_segments(boxes)
    ax.add_collection(
        LineCollection(
            segs,
            colors=color,
            linewidths=lw,
            linestyles=ls,
            zorder=z,
            capstyle="round",
            joinstyle="round",
        )
    )
    if tick:
        ax.add_collection(
            LineCollection(
                ticks, colors=color, linewidths=lw, zorder=z, capstyle="round"
            )
        )


def flipped_cars(z):
    """GT cars >= 20 m that A and D both match (IoU 0.5) with A within 15 deg and D off by > 90 deg."""
    ka, kd = f"gt_heading_err_rad_iou05_{A_KEY}", f"gt_heading_err_rad_iou05_{D_KEY}"
    if ka not in z.files:
        return np.zeros(len(z["gt_boxes"]), bool)
    a, d = np.rad2deg(z[ka]), np.rad2deg(z[kd])
    far = np.hypot(z["gt_boxes"][:, 0], z["gt_boxes"][:, 1]) >= FLIP_MIN_RANGE
    with np.errstate(invalid="ignore"):
        return far & (a < FLIP_GOOD_DEG) & (d > FLIP_BAD_DEG)


def scene_extent(z, margin=6.0, min_half=22.0):
    """Square BEV window around every GT car, every plotted detection and the ego, shared by the row."""
    pts = np.vstack(
        [z["gt_boxes"][:, :2], [[0.0, 0.0]]]
        + [z[f"pred_boxes_{key}"][:, :2] for key, _, letter in COLUMNS if letter]
    )
    lo, hi = pts.min(0) - margin, pts.max(0) + margin
    centre = (lo + hi) / 2
    half = max(min_half, (hi - lo).max() / 2)
    return centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half


def draw_scene(ax, z, key, letter, extent, ptsize=1.0, lw_scale=1.0, ego=True):
    x0, x1, y0, y1 = extent
    p = z["points"]
    m = (p[:, 0] > x0) & (p[:, 0] < x1) & (p[:, 1] > y0) & (p[:, 1] < y1)
    p = p[m]
    ground = p[:, 2] < GROUND_Z
    ax.scatter(
        p[ground, 0],
        p[ground, 1],
        s=0.12 * ptsize,
        c=GROUND_PTS,
        lw=0,
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        p[~ground, 0],
        p[~ground, 1],
        s=0.35 * ptsize,
        c=OBJECT_PTS,
        lw=0,
        rasterized=True,
        zorder=1,
    )
    gt = z["gt_boxes"]
    if key == "gt":
        add_boxes(
            ax,
            z["gt_ignored_boxes"],
            "#9a9a9a",
            0.45 * lw_scale,
            ls=(0, (1.0, 1.2)),
            z=2,
            tick=False,
        )
        add_boxes(ax, gt, GT_LINE, 0.7 * lw_scale, z=3)
    else:
        add_boxes(ax, gt, GT_REF_LINE, 0.5 * lw_scale, z=2, tick=False)
        add_boxes(
            ax,
            z[f"pred_boxes_{key}"],
            ps.RUN_STYLE[letter]["color"],
            0.9 * lw_scale,
            z=4,
        )
    if ego:
        ax.add_patch(
            plt.Polygon(
                [[0, 2.2], [-1.1, -1.3], [1.1, -1.3]],
                closed=True,
                fc=ps.INK,
                ec="none",
                zorder=6,
            )
        )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def panel_note(z, key, flips):
    gt = z["gt_boxes"]
    far = np.hypot(gt[:, 0], gt[:, 1]) >= FLIP_MIN_RANGE
    if key == "gt":
        return (
            f"{len(gt)} scored cars ({int(far.sum())} at ≥20 m)"
            if flips.any()
            else f"{len(gt)} scored cars"
        )
    n = len(z[f"pred_boxes_{key}"])
    det = f"{n} detection{'' if n == 1 else 's'}"
    if flips.any():
        err = np.rad2deg(z[f"gt_heading_err_rad_iou05_{key}"][far])
        err = err[np.isfinite(err)]
        return f"{det}, ≥20 m heading error {err.mean():.1f}°"
    rec07 = z[f"gt_matched_iou07_{key}"].mean() if len(gt) else 0.0
    return f"{det}, recall@0.7 {100 * rec07:.0f}%"


def style_panel(ax, title=None, note=None):
    for s in ax.spines.values():
        s.set_color("#cfcfcf")
        s.set_linewidth(0.5)
    if title:
        ax.set_title(title, pad=2.5)
    if note:
        ax.text(
            0.025,
            0.975,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color=ps.INK,
            bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.85),
            zorder=10,
        )


def add_scale_bar(ax, extent, length=10.0, left=False):
    x0, x1, y0, y1 = extent
    span = x1 - x0
    xs = x0 + 0.05 * span if left else x1 - 0.05 * span - length
    ys = y0 + 0.05 * span
    ax.add_patch(
        Rectangle(
            (xs - 0.02 * span, ys - 0.025 * span),
            length + 0.04 * span,
            0.09 * span,
            fc="white",
            ec="none",
            alpha=0.85,
            zorder=7,
        )
    )
    ax.plot(
        [xs, xs + length],
        [ys, ys],
        color=ps.INK,
        lw=1.0,
        solid_capstyle="butt",
        zorder=8,
    )
    ax.text(
        xs + length / 2,
        ys + 0.015 * span,
        f"{length:g} m",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=ps.INK,
        zorder=8,
    )


def draw_row(axes, z, row_label=None, titles=True):
    extent = scene_extent(z)
    flips = flipped_cars(z)
    zoom = None
    if flips.any():
        c = z["gt_boxes"][flips, :2]
        centre = (c.min(0) + c.max(0)) / 2
        half = max(6.0, (c.max(0) - c.min(0)).max() / 2 + 4.0)
        zoom = (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)
    for ax, (key, title, letter) in zip(axes, COLUMNS):
        draw_scene(ax, z, key, letter, extent)
        style_panel(ax, title if titles else None, panel_note(z, key, flips))
        if zoom is None:
            continue
        zx0, zx1, zy0, zy1 = zoom
        ax.add_patch(
            Rectangle(
                (zx0, zy0), zx1 - zx0, zy1 - zy0, fc="none", ec=ps.INK, lw=0.6, zorder=7
            )
        )
        # inset in the bottom corner on the side away from the zoomed cluster
        right = (zx0 + zx1) / 2 < (extent[0] + extent[1]) / 2
        size = 0.4
        ins = ax.inset_axes([1 - size - 0.02 if right else 0.02, 0.02, size, size])
        draw_scene(ins, z, key, letter, zoom, ptsize=3.0, lw_scale=1.3, ego=False)
        for s in ins.spines.values():
            s.set_color(ps.INK)
            s.set_linewidth(0.6)
        ins.set_zorder(9)
        # on-page magnification: window ratio times the inset's share of the panel width
        ins.text(
            0.04,
            0.04,
            f"×{size * (extent[1] - extent[0]) / (zoom[1] - zoom[0]):.1f}",
            transform=ins.transAxes,
            fontsize=6,
            color=ps.INK,
            ha="left",
            va="bottom",
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85),
        )
    if zoom is None:
        add_scale_bar(axes[-1], extent)
    else:  # the insets take the bottom-right corner
        add_scale_bar(axes[0], extent, left=right)
    if row_label:
        axes[0].set_ylabel(row_label, labelpad=3)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--carla", default=DEFAULT_CARLA)
    ap.add_argument("--nusc", default=DEFAULT_NUSC)
    ap.add_argument("--candidates", action="store_true")
    args = ap.parse_args()
    ps.apply()

    if args.candidates:
        for name in ("carla_qualitative", "nusc_qualitative"):
            files = sorted(
                glob.glob(osp.join(DATA, name, "**", "scene_*.npz"), recursive=True)
            )
            fig, axes = plt.subplots(
                len(files), 3, figsize=(ps.TEXT_W, 2.45 * len(files))
            )
            fig.subplots_adjust(
                left=0.05,
                right=0.995,
                bottom=0.005,
                top=0.995,
                wspace=0.03,
                hspace=0.12,
            )
            for row, f in zip(axes, files):
                z = np.load(f)
                draw_row(
                    row,
                    z,
                    f"{osp.relpath(f, osp.join(DATA, name)).replace('scene_', '')[:26]}\n"
                    f"{z['scene_name']}",
                )
            os.makedirs(osp.join(HERE, "drafts"), exist_ok=True)
            out = osp.join(HERE, "drafts", f"fig8_{name}_candidates")
            fig.savefig(out + ".pdf")
            fig.savefig(out + ".png", dpi=200)
            plt.close(fig)
            print(f"wrote {out}.pdf / .png")
        return

    carla, nusc = np.load(args.carla), np.load(args.nusc)
    fig, axes = plt.subplots(2, 3, figsize=(ps.TEXT_W, 4.85))
    fig.subplots_adjust(
        left=0.03, right=0.997, bottom=0.005, top=0.965, wspace=0.03, hspace=0.03
    )
    draw_row(
        axes[0], carla, f"CARLA val ({str(carla['scene_name']).replace('_Opt', '')})"
    )
    draw_row(axes[1], nusc, f"nuScenes {nusc['scene_name']}, zero-shot", titles=False)
    print(
        f"row 1: {carla['sample_token']}; row 2: {nusc['scene_name']} ({nusc['sample_token']}), "
        f"score >= {float(carla['score_threshold']):.2f} / {float(nusc['score_threshold']):.2f}"
    )
    ps.save(fig, "fig8_qualitative")


if __name__ == "__main__":
    main()
