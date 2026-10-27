"""Shared matplotlib style for the GeoPillars paper"""

import os.path as osp

import matplotlib as mpl
from matplotlib import font_manager

COLUMN_W = 252 / 72.27  # in
TEXT_W = 516 / 72.27  # in

RUN_STYLE = {
    "A": dict(
        run="geo_pillars_h100_intensity",
        label="GeoPillars",
        color="#E63946",
        ls="-",
        marker="o",
    ),
    "F": dict(
        run="centerpoint_osf1_ablation",
        label="CP OSF=1",
        color="#1D3557",
        ls=(0, (4.0, 1.6)),
        marker="s",
    ),
    "C": dict(
        run="centerpoint_h100_intensity_original",
        label="CP native",
        color="#457B9D",
        ls=(0, (6.0, 1.6, 1.2, 1.6)),
        marker="^",
    ),
    "D": dict(
        run="centerpoint_h100_intensity",
        label="CP OSF=4 (budget)",
        color="#2A9D8F",
        ls=(0, (1.2, 1.4)),
        marker="v",
    ),
    "G": dict(
        run="centerpoint_geopillar_encoder_replk_distributed",
        label="CP + encoder + RepLK (no NGFM)",
        short="CP + enc. + RepLK",
        color="#E9C46A",
        ls=(0, (3.0, 1.2, 1.0, 1.2, 1.0, 1.2)),
        marker="D",
    ),
    # intensity ablations
    "B": dict(
        run="geo_pillars_h100_nointensity_final",
        label="GeoPillars, no intensity",
        short="GeoPillars, xyz",
        color="#F0969D",
        ls=(0, (2.0, 1.2)),
        marker="o",
        hollow=True,
    ),
    "E": dict(
        run="centerpoint_h100_no_intensity",
        label="CP OSF=4, no intensity",
        short="CP OSF=4, xyz",
        color="#88CBC2",
        ls=(0, (2.0, 1.2)),
        marker="v",
        hollow=True,
    ),
}
GOLD = RUN_STYLE["G"]["color"]

INK = "#1a1a1a"
INK_MUTED = "#5c5c5c"
GRID = "#e3e3e3"
GT_COLOR = "#1a1a1a"
POINTS_COLOR = "#9a9a9a"

_LIBERATION = "/usr/share/fonts/truetype/liberation"


def apply():
    for name in ("Regular", "Bold", "Italic", "BoldItalic"):
        path = osp.join(_LIBERATION, f"LiberationSerif-{name}.ttf")
        if osp.isfile(path):
            font_manager.fontManager.addfont(path)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Liberation Serif",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Serif",
            "mathtext.it": "Liberation Serif:italic",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK_MUTED,
            "axes.linewidth": 0.5,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 150,
        }
    )


def short(letter):
    st = RUN_STYLE[letter]
    return st.get("short", st["label"])


def despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, stem):
    """Write <figures>/<stem>.pdf (vector, fonts embedded) and a 450 dpi .png preview.

    Blank rows top and bottom are cropped first: the paper includes these at
    ``width=\\columnwidth``, so canvas left empty by ``subplots_adjust`` becomes
    blank page space right where a float's own separation already is.
    """
    out = osp.join(osp.dirname(osp.abspath(__file__)), stem)
    bbox = _tight_rows(fig)
    for ext, kw in ((".pdf", {}), (".png", dict(dpi=450))):
        fig.savefig(out + ext, bbox_inches=bbox, **kw)
    print(f"wrote {out}.pdf / .png")


def _tight_rows(fig, pad=0.01):
    """Bounding box cropped top and bottom only, keeping the canvas width.

    The paper includes these at ``width=\\columnwidth``, so cropping *sideways*
    is counterproductive: it scales the content up to fill the column and the
    figure gets taller. Cropping only the blank rows lowers the aspect ratio,
    which is what actually hands vertical space back to the text.
    """
    from matplotlib.transforms import Bbox

    try:
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
    except Exception:  # backend without a live renderer
        return "tight"
    w, h = fig.get_figwidth(), fig.get_figheight()
    return Bbox.from_extents(
        min(tb.x0, 0.0), max(tb.y0 - pad, 0.0), max(tb.x1, w), min(tb.y1 + pad, h)
    )


def spread(values, min_gap, lo=None, hi=None):
    """Label positions for converging end-points: keep order, push apart to >= min_gap, stay in [lo, hi].

    Returns positions in the same order as ``values``; draw a leader from each label to its point.
    """
    import numpy as np

    v = np.asarray(values, dtype=float)
    order = np.argsort(v)
    pos = v[order].copy()
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            gap = pos[i] - pos[i - 1]
            if gap < min_gap - 1e-9:
                push = (min_gap - gap) / 2
                pos[i - 1] -= push
                pos[i] += push
                moved = True
        if lo is not None and pos[0] < lo:
            pos += lo - pos[0]
            moved = True
        if hi is not None and pos[-1] > hi:
            pos -= pos[-1] - hi
            moved = True
        if not moved:
            break
    out = np.empty_like(pos)
    out[order] = pos
    return out


def run_line(ax, x, y, letter, lw=1.1, ms=3.6, z=3, **kw):
    """Line + markers in the run's colour, dash and marker (hollow for the intensity ablations)."""
    st = RUN_STYLE[letter]
    face = "white" if st.get("hollow") else st["color"]
    return ax.plot(
        x,
        y,
        color=st["color"],
        ls=st["ls"],
        lw=lw,
        marker=st["marker"],
        ms=ms,
        mfc=face,
        mec=st["color"] if st.get("hollow") else "white",
        mew=0.8 if st.get("hollow") else 0.5,
        zorder=z,
        clip_on=False,
        **kw,
    )[0]
