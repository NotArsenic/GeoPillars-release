#!/usr/bin/env python3
"""Fig. 13: 2x2 grid of CARLA scans with their precomputed surface normals."""

import argparse
import glob
import os.path as osp
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as ps

DEFAULT_IMAGES = sorted(
    glob.glob(osp.join(HERE, "data", "normal_screenshots", "*.png"))
)


def load(path, bg_rgb):
    """RGB array with the viewer's flat background colour replaced by `bg_rgb`."""
    a = np.asarray(Image.open(path).convert("RGB")).copy()
    colors, counts = np.unique(a.reshape(-1, 3), axis=0, return_counts=True)
    bg = colors[counts.argmax()]
    a[(a == bg).all(axis=2)] = bg_rgb
    return a


VARIANTS = {
    "": ((255, 255, 255), "white", None, "#bbbbbb"),
    "_dark": ((13, 13, 18), "#0d0d12", "#e6e6e6", "#44444c"),
}


def town_label(path):
    """'Town10HD_Opt__023591_1.png' -> 'Town10HD'."""
    return re.sub(r"_Opt$", "", osp.basename(path).split("__")[0])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--images",
        nargs=4,
        default=DEFAULT_IMAGES,
        help="four screenshots, row-major order",
    )
    args = ap.parse_args()
    if len(args.images) != 4:
        raise SystemExit(f"need 4 images, found {len(args.images)}")

    ps.apply()
    for suffix, style in VARIANTS.items():
        render(args.images, suffix, *style)


def render(paths, suffix, bg_rgb, page, ink, border):
    imgs = [load(p, bg_rgb) for p in paths]
    h, w = imgs[0].shape[:2]
    gap = 0.012
    label_h = 0.12
    panel_w = (ps.TEXT_W * (1 - gap)) / 2
    panel_h = panel_w * h / w
    fig_h = 2 * (panel_h + label_h) + 0.04
    fig = plt.figure(figsize=(ps.TEXT_W, fig_h), facecolor=page)

    for i, (img, path) in enumerate(zip(imgs, paths)):
        r, c = divmod(i, 2)
        x0 = c * (panel_w + gap * ps.TEXT_W) / ps.TEXT_W
        y0 = 1 - (r + 1) * (panel_h + label_h) / fig_h + label_h / fig_h
        ax = fig.add_axes([x0, y0, panel_w / ps.TEXT_W, panel_h / fig_h])
        ax.imshow(img, interpolation="none")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_color(border)
        ax.text(
            0.5,
            -0.012,
            f'({"abcd"[i]}) {town_label(path)}',
            transform=ax.transAxes,
            ha="center",
            va="top",
            **({"color": ink} if ink else {}),
        )

    ps.save(fig, "fig13_normals_grid" + suffix)


if __name__ == "__main__":
    main()
