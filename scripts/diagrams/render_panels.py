#!/usr/bin/env python3
"""End panels for the GeoPillars architecture diagram.

Renders the two photographic endpoints of ``architectureDiagram.tex``:

    panels/lidar_scan.png   the raw single sweep the detector is given
    panels/detections.png   the epoch-8 boxes it returns (score >= 0.25)

Both come from one self-contained scene NPZ written by
``scripts/figure_data/export_carla_qualitative.py``; no dataset is read, which
is the same reproducibility rule ``figures/make_all.py`` enforces.

Usage (repo root)::

    python scripts/diagrams/render_panels.py
    python scripts/diagrams/render_panels.py --scene <npz> --out-dir <dir>
"""

import argparse
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.dirname(osp.dirname(HERE))
DEFAULT_SCENE = osp.join(
    REPO, "figures", "data", "carla_qualitative", "scene_Town10HD_Opt__056653.npz"
)

# Dark canvas, matching the Open3D-style scan panels the draft diagram used.
BG = "#080c18"
PRED_COLOR = "#E63946"  # RUN_STYLE['A'] - GeoPillars
RANGE = 42.0  # m, half-width of the rendered window
GROUND_Z = -1.4  # m, sensor sits 1.84 m above the road
ELEV, AZIM = 26.0, -62.0
# Cold-to-warm ramp over height: reads as depth on black, prints as one hue family.
SCAN_CMAP = LinearSegmentedColormap.from_list(
    "scan", ["#12305e", "#1f6fb2", "#28b5c9", "#6fe3b8", "#d8f5a8"]
)


def load_scene(path):
    z = np.load(path, allow_pickle=True)
    p = z["points"][:, :3].astype(np.float64)
    keep = (np.abs(p[:, 0]) < RANGE) & (np.abs(p[:, 1]) < RANGE) & (p[:, 2] < 4.0)
    return p[keep], z["pred_boxes_geopillars"].astype(np.float64), z


def box_edges(box):
    """8 corners -> 12 edge segments for one (x, y, z, l, w, h, yaw) box."""
    x, y, z, l, w, h, yaw = box[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    sx, sy = np.array([1, 1, -1, -1]) * l / 2, np.array([1, -1, -1, 1]) * w / 2
    cx, cy = x + sx * c - sy * s, y + sx * s + sy * c
    lo, hi = z - h / 2, z + h / 2
    bot = [(cx[i], cy[i], lo) for i in range(4)]
    top = [(cx[i], cy[i], hi) for i in range(4)]
    segs = [(bot[i], bot[(i + 1) % 4]) for i in range(4)]
    segs += [(top[i], top[(i + 1) % 4]) for i in range(4)]
    segs += [(bot[i], top[i]) for i in range(4)]
    return segs


def new_axes():
    fig = plt.figure(figsize=(4.0, 3.0), dpi=420)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlim(-RANGE, RANGE)
    ax.set_ylim(-RANGE, RANGE)
    ax.set_zlim(-6.0, 14.0)  # tall z box flattens the cloud into a disc
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 0.42))
    fig.subplots_adjust(0, 0, 1, 1)
    return fig, ax


def save(fig, path, margin=14):
    """Write the figure cropped to its ink.

    mplot3d reserves a cube-shaped viewport, so a flattened scan occupies maybe a
    third of it. Cropping to the drawn pixels is what makes the panel fill its slot
    in the diagram instead of floating in a field of background.
    """
    os.makedirs(osp.dirname(path), exist_ok=True)
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    bg = np.array([int(BG[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.int16)
    ink = np.abs(rgb.astype(np.int16) - bg).max(axis=2) > 6
    rows, cols = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
    r0, r1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, rgb.shape[0])
    c0, c1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, rgb.shape[1])
    plt.imsave(path, rgb[r0:r1, c0:c1])
    plt.close(fig)
    print(f"wrote {path}  ({c1 - c0}x{r1 - r0} px)")


def render_scan(points, path):
    fig, ax = new_axes()
    order = np.argsort(points[:, 2])  # draw high points last
    p = points[order]
    ax.scatter(
        p[:, 0],
        p[:, 1],
        p[:, 2],
        s=0.45,
        c=p[:, 2],
        cmap=SCAN_CMAP,
        vmin=GROUND_Z - 1.2,
        vmax=2.6,
        linewidths=0,
        alpha=0.95,
        depthshade=False,
    )
    save(fig, path)


def render_detections(points, boxes, path):
    fig, ax = new_axes()
    ground = points[:, 2] < GROUND_Z
    ax.scatter(
        points[ground, 0],
        points[ground, 1],
        points[ground, 2],
        s=0.28,
        c="#2b3757",
        linewidths=0,
        depthshade=False,
    )
    ax.scatter(
        points[~ground, 0],
        points[~ground, 1],
        points[~ground, 2],
        s=0.40,
        c="#7fa8d8",
        linewidths=0,
        depthshade=False,
    )
    for b in boxes:
        for a, c in box_edges(b):
            ax.plot(*zip(a, c), color=PRED_COLOR, lw=0.55, solid_capstyle="round")
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--out-dir", default=osp.join(HERE, "panels"))
    args = ap.parse_args()

    points, boxes, z = load_scene(args.scene)
    print(
        f"{osp.basename(args.scene)}: {len(points):,} points in range, "
        f'{len(boxes)} detections, {str(z["scene_description"])}'
    )
    render_scan(points, osp.join(args.out_dir, "lidar_scan.png"))
    render_detections(points, boxes, osp.join(args.out_dir, "detections.png"))


if __name__ == "__main__":
    main()
