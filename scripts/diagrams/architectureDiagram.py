#!/usr/bin/env python3
"""Full GeoPillars architecture diagram (run A)"""

import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


COLORS = r"""
\definecolor{cIn}      {HTML}{C3CEDC}   % raw points / pillar builder
\definecolor{cLinear}  {HTML}{6E97C7}   % Linear + BN + ReLU
\definecolor{cAttn}    {HTML}{B47CC7}   % grouped vector attention
\definecolor{cPool}    {HTML}{E07B7B}   % masked max-pool
\definecolor{cCanvas}  {HTML}{74B58C}   % BEV canvas
\definecolor{cPrior}   {HTML}{E9C46A}   % learned positional prior
\definecolor{cS0}      {HTML}{93C47D}   % backbone stage 0  (k=13)
\definecolor{cS1}      {HTML}{E8D07A}   % backbone stage 1  (k=11)
\definecolor{cS2}      {HTML}{E2A85F}   % backbone stage 2  (k=9)
\definecolor{cNgfm}    {HTML}{4FA8AC}   % normal-guided modulation
\definecolor{cUp1}     {HTML}{A8D5BA}   % upsample stage 1
\definecolor{cUp2}     {HTML}{7FC4A0}   % upsample stage 2
\definecolor{cHead}    {HTML}{3A5F8F}   % CenterHead
\definecolor{cLoss}    {HTML}{E63946}   % NGFM auxiliary loss
\definecolor{cInk}     {HTML}{1A1A1A}
\definecolor{cMuted}   {HTML}{55606B}
\definecolor{cGrid}    {HTML}{93A0AD}
\definecolor{cAux}     {HTML}{8A6FB0}   % training-only signals
"""

H = 2.40  # standard block height / depth (cm)
PANEL_H = 2.30  # end-panel image height (cm)


def box(name, x, width, height=H, depth=None, fill="cIn", opacity=0.95, y=0.0, z=0.0):
    """A PlotNeuralNet PlainBox whose west face sits at (x, y, z).

    Returns (tex, east_x) so blocks chain along the pipeline.
    """
    depth = height if depth is None else depth
    tex = (
        f"\\pic[shift={{(0,0,0)}}] at ({x:.3f},{y:.3f},{z:.3f})\n"
        f"    {{PlainBox={{name={name}, caption=, fill={fill}, opacity={opacity},\n"
        f"        width={width:.3f}, height={height:.3f}, depth={depth:.3f}, scale=1}}}};\n"
    )
    return tex, x + width


def iso_grid(name, x0, nx, nz, cell, bars, dots=0, seed=0, grid_opacity=0.85, hl=None):
    """Ground grid in the (x, z) plane with 3-D bars rising along +y.

    ``bars`` is a list of ``(i, j, height, colour)`` in cell indices. Faces use
    the same three-visible-face order PlainBox does (near +z, top +y, right +x),
    so a bar and a box read as one solid under one projection. Bars paint
    back-to-front in z, the only depth axis in this projection.
    """
    z0 = -nz * cell / 2.0  # centre the plane on the baseline
    x1, z1 = x0 + nx * cell, z0 + nz * cell
    out = [f"% -- isometric pillar grid `{name}` --"]
    for i in range(nx + 1):
        gx = x0 + i * cell
        out.append(
            f"\\draw[cGrid, opacity={grid_opacity}, line width=0.3pt] "
            f"({gx:.3f},0,{z0:.3f}) -- ({gx:.3f},0,{z1:.3f});"
        )
    for j in range(nz + 1):
        gz = z0 + j * cell
        out.append(
            f"\\draw[cGrid, opacity={grid_opacity}, line width=0.3pt] "
            f"({x0:.3f},0,{gz:.3f}) -- ({x1:.3f},0,{gz:.3f});"
        )

    if dots:
        rng = random.Random(seed)
        taken = {(i, j) for i, j, *_ in bars}
        for _ in range(dots):
            i, j = rng.randrange(nx), rng.randrange(nz)
            if (i, j) in taken:
                continue
            px = x0 + (i + rng.uniform(0.2, 0.8)) * cell
            pz = z0 + (j + rng.uniform(0.2, 0.8)) * cell
            out.append(
                f"\\fill[cMuted, opacity=0.75] ({px:.3f},0,{pz:.3f}) circle (0.55pt);"
            )

    (inset,) = (cell * 0.17,)
    side = cell - 2 * inset
    for i, j, h, colour in sorted(bars, key=lambda b: b[1]):
        bx, bz = x0 + i * cell + inset, z0 + j * cell + inset
        ex, ez = bx + side, bz + side
        lw = "0.7pt" if hl == (i, j) else "0.15pt"
        edge = "cLoss" if hl == (i, j) else "cInk"
        out.append(
            f"\\draw[fill={colour}, draw={edge}, line width={lw}] "
            f"({bx:.3f},0,{ez:.3f}) -- ({ex:.3f},0,{ez:.3f}) -- "
            f"({ex:.3f},{h:.3f},{ez:.3f}) -- ({bx:.3f},{h:.3f},{ez:.3f}) -- cycle;"
        )
        out.append(
            f"\\draw[fill={colour}!75!black, draw={edge}, line width={lw}] "
            f"({ex:.3f},0,{bz:.3f}) -- ({ex:.3f},0,{ez:.3f}) -- "
            f"({ex:.3f},{h:.3f},{ez:.3f}) -- ({ex:.3f},{h:.3f},{bz:.3f}) -- cycle;"
        )
        out.append(
            f"\\draw[fill={colour}!55, draw={edge}, line width={lw}] "
            f"({bx:.3f},{h:.3f},{bz:.3f}) -- ({ex:.3f},{h:.3f},{bz:.3f}) -- "
            f"({ex:.3f},{h:.3f},{ez:.3f}) -- ({bx:.3f},{h:.3f},{ez:.3f}) -- cycle;"
        )
        if hl == (i, j):
            out.append(f"\\coordinate ({name}-hl) at ({ex:.3f},{h / 2:.3f},{ez:.3f});")
            out.append(
                f"\\coordinate ({name}-hltop) at "
                f"({(bx + ex) / 2:.3f},{h:.3f},{(bz + ez) / 2:.3f});"
            )

    out += [
        f"\\coordinate ({name}-w) at ({x0:.3f},0,{(z0 + z1) / 2:.3f});",
        f"\\coordinate ({name}-e) at ({x1:.3f},0,{(z0 + z1) / 2:.3f});",
        f"\\coordinate ({name}-c) at ({(x0 + x1) / 2:.3f},0,{z1:.3f});",
    ]
    return "\n".join(out) + "\n", x1


def bev_slab(name, x0, width, side, cells, fill, opacity, y0=0.0):
    """An upright BEV map: a thin PlainBox whose east face carries the grid.

    In this projection the east face is the one that reads as a square, so the
    grid goes there. Drawn upright the map costs a fraction of the horizontal
    room a ground-plane parallelogram does, which is what keeps the arrows short.
    """
    tex, x1 = box(
        name, x0, width, height=side, depth=side, fill=fill, opacity=opacity, y=y0
    )
    out = [tex.rstrip("\n")]
    lo, hi = y0 - side / 2, y0 + side / 2
    for k in range(1, cells):
        t = lo + side * k / cells
        out.append(
            f"\\draw[white, opacity=0.55, line width=0.2pt] "
            f"({x1:.3f},{t:.3f},{-side / 2:.3f}) -- ({x1:.3f},{t:.3f},{side / 2:.3f});"
        )
        t = -side / 2 + side * k / cells
        out.append(
            f"\\draw[white, opacity=0.55, line width=0.2pt] "
            f"({x1:.3f},{lo:.3f},{t:.3f}) -- ({x1:.3f},{hi:.3f},{t:.3f});"
        )
    return "\n".join(out) + "\n", x1


def bev_plate(name, x0, side, cells, fill, opacity, y0=0.0, z_centre=0.0):
    """A thin filled square lying in the (x, z) ground plane - one BEV map."""
    z0 = z_centre - side / 2.0
    x1, z1 = x0 + side, z0 + side
    q = (
        f"({x0:.3f},{y0:.3f},{z0:.3f}) -- ({x1:.3f},{y0:.3f},{z0:.3f}) -- "
        f"({x1:.3f},{y0:.3f},{z1:.3f}) -- ({x0:.3f},{y0:.3f},{z1:.3f}) -- cycle"
    )
    out = [f"\\fill[{fill}, opacity={opacity}] {q};"]
    for k in range(1, cells):
        t = x0 + side * k / cells
        out.append(
            f"\\draw[white, opacity=0.55, line width=0.15pt] "
            f"({t:.3f},{y0:.3f},{z0:.3f}) -- ({t:.3f},{y0:.3f},{z1:.3f});"
        )
        t = z0 + side * k / cells
        out.append(
            f"\\draw[white, opacity=0.55, line width=0.15pt] "
            f"({x0:.3f},{y0:.3f},{t:.3f}) -- ({x1:.3f},{y0:.3f},{t:.3f});"
        )
    out.append(f"\\draw[cInk, line width=0.2pt] {q};")
    out += [
        f"\\coordinate ({name}-w) at ({x0:.3f},{y0:.3f},{z_centre:.3f});",
        f"\\coordinate ({name}-e) at ({x1:.3f},{y0:.3f},{z_centre:.3f});",
        f"\\coordinate ({name}-c) at ({(x0 + x1) / 2:.3f},{y0:.3f},{z_centre:.3f});",
        f"\\coordinate ({name}-s) at ({(x0 + x1) / 2:.3f},{y0:.3f},{z1:.3f});",
        f"\\coordinate ({name}-n) at ({(x0 + x1) / 2:.3f},{y0:.3f},{z0:.3f});",
    ]
    return "\n".join(out) + "\n", x1


def lbl(at, text, style="dim", anchor="north", dy=-0.10, dx=0.0):
    return (
        f"\\node[{style}, anchor={anchor}, xshift={dx:.2f}cm, yshift={dy:.2f}cm] "
        f"at {at} {{{text}}};\n"
    )


def group_brace(x_a, x_b, y, text, name_y):
    """Brace under [x_a, x_b] at height y, with its name on the shared name tier."""
    return f"\\draw[brace] ({x_b:.3f},{y:.3f},0) -- ({x_a:.3f},{y:.3f},0);\n" + lbl(
        f"({(x_a + x_b) / 2:.3f},{name_y:.3f},0)", text, "mod", dy=0.0
    )


# -----------------------------------------------------------------------------
# Vertical tiers (cm).
#
# One row of names, one row of shape tags, one dashed rail. Everything the
# picture does not need to *show* lives in the caption. Names wrap to two lines
# so a slot is as wide as the widest word rather than the whole name, which is
# what keeps the connecting arrows short.
# -----------------------------------------------------------------------------
TAG_Y = 2.45  # tag row: clears the ghost stack's projected top (2.05)
BRACE_Y = -1.52  # group braces
NAME_Y = -1.98  # the single module-name row, anchored north
RAIL_Y = -4.00  # the one training-only rail; the loss node centres on it
PRIOR_Y = 4.05  # the floating positional-prior slab
LEGEND_Y = -4.00  # legend, on the rail row beside the loss node
GAP = 0.48  # slack added to every slot, i.e. the arrow budget


class Slots:
    """Left-to-right cursor. A slot is as wide as whichever is bigger, the
    graphic or its label, plus one fixed gap - so nothing is spaced out further
    than its own text demands."""

    def __init__(self, x=0.0):
        self.x = x

    def place(self, graphic_w, label_w=0.0, gap=GAP):
        slot = max(graphic_w, label_w) + gap
        left = self.x + (slot - graphic_w) / 2.0
        self.x += slot
        return left


# -----------------------------------------------------------------------------
# The picture
# -----------------------------------------------------------------------------


def build():
    t = []
    add = t.append

    add(
        r"""\documentclass[border=4pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{./PlotNeuralNet/layers/}{init}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usetikzlibrary{positioning,3d,arrows.meta,calc,fit,backgrounds,shapes.geometric,decorations.pathreplacing}
"""
        + COLORS
        + r"""
\begin{document}
\begin{tikzpicture}[
    font=\fontsize{17}{19}\selectfont,
    lbl/.style   ={text=cInk, align=center, inner sep=0.8pt},
    mod/.style   ={lbl, font=\fontsize{19.5}{21.5}\selectfont},
    dim/.style   ={lbl, text=cMuted, font=\fontsize{16}{18}\selectfont},
    tag/.style   ={lbl, text=cMuted, font=\fontsize{17}{19}\selectfont},
    flow/.style  ={-{Stealth[length=5.5pt,width=4pt]}, draw=cInk, line width=1pt},
    aux/.style   ={-{Stealth[length=5.5pt,width=4pt]}, draw=cAux, line width=0.9pt,
                   dash pattern=on 3.5pt off 2.5pt},
    leader/.style={draw=cGrid, line width=0.45pt},
    brace/.style ={decorate, decoration={brace, amplitude=4pt}, draw=cMuted, line width=0.5pt},
  ]
"""
    )
    s = Slots()

    # -- 1. LiDAR sweep -------------------------------------------------------
    scan_w = PANEL_H * 1031 / 504
    scan_x = s.place(scan_w, 1.9)
    add("% ====== input sweep ======\n")
    add(f"""\\node[anchor=west, inner sep=0] (scan) at ({scan_x:.3f},0,0)
    {{\\includegraphics[height={PANEL_H:.2f}cm]{{panels/lidar_scan.png}}}};
""")
    add(
        lbl(
            f"({scan_x + scan_w / 2:.3f},{NAME_Y:.3f},0)",
            r"\textbf{LiDAR}\\\textbf{sweep}",
            "mod",
        )
    )

    # -- 2. Pillar feature builder --------------------------------------------
    add("% ====== pillar feature builder ======\n")
    x0 = s.place(0.38, 2.4)
    tex, _ = box("builder", x0, 0.38, fill="cIn")
    add(tex)
    add(r"\draw[flow] (scan.east) -- (builder-west);" + "\n")
    add(
        f"\\draw[leader] ({x0 + 0.19:.3f},{-H / 2 - 0.58:.3f},0) -- "
        f"({x0 + 0.19:.3f},{NAME_Y + 0.06:.3f},0);\n"
    )
    add(
        lbl(
            f"({x0 + 0.19:.3f},{NAME_Y:.3f},0)",
            r"\textbf{Pillar}\\\textbf{features}",
            "mod",
        )
    )
    add(
        lbl(
            f"({x0 + 0.19:.3f},{TAG_Y:.3f},0)",
            r"0.4\,m, $\le8$ pts",
            "tag",
            anchor="south",
        )
    )

    # -- 3. Per-point pillar tensor, one pillar picked out --------------------
    cell, ngx, ngz = 0.36, 9, 6
    x0 = s.place(ngx * cell, 2.1)
    rng = random.Random(7)
    bars, tints = [], ["cLinear", "cAttn", "cCanvas", "cPrior", "cS0"]
    for n, (ci, cj) in enumerate([(0, 1), (2, 4), (3, 0), (5, 3), (7, 1), (8, 4)]):
        for k in range(2):
            bars.append(
                (
                    min(ci + k, ngx - 1),
                    min(cj + k, ngz - 1),
                    rng.uniform(0.48, 1.15),
                    tints[(n + k) % len(tints)],
                )
            )
    hl_cell = (8, 5)
    tex, _ = iso_grid("gA", x0, ngx, ngz, cell, bars, dots=70, seed=11, hl=hl_cell)
    add(tex)
    add(f"\\draw[flow] (builder-east) -- ({x0 - 0.18:.3f},0,0);\n")
    add(
        lbl(
            f"({x0 + ngx * cell / 2:.3f},{TAG_Y:.3f},0)",
            r"$(M,\,8,\,12)$",
            "tag",
            anchor="south",
        )
    )

    # -- 4. GeoPillar encoder, run once per pillar ----------------------------
    # Two ghost copies sit behind the stack (negative z recedes up-right in this
    # projection) and the flow arrow leaves the single highlighted pillar, so the
    # picture says "these weights are applied M times, once per pillar".
    add("% ====== GeoPillar encoder - shared, applied per pillar ======\n")
    gaps, widths = (0.34, 0.34, 0.60), (0.30, 0.56, 0.30, 0.30)
    enc_w = sum(widths) + sum(gaps)
    enc_x0 = s.place(enc_w + 0.42, 2.9)

    def enc_stack(suffix, z, opacity):
        out, x = [], enc_x0
        tex, x = box(
            f"lin1{suffix}", x, widths[0], fill="cLinear", opacity=opacity, z=z
        )
        out.append(tex)
        tex, x = box(
            f"gva{suffix}", x + gaps[0], widths[1], fill="cAttn", opacity=opacity, z=z
        )
        out.append(tex)
        tex, x = box(
            f"lin2{suffix}",
            x + gaps[1],
            widths[2],
            fill="cLinear",
            opacity=opacity,
            z=z,
        )
        out.append(tex)
        tex, x = box(
            f"pool{suffix}",
            x + gaps[2],
            widths[3],
            height=0.60,
            fill="cPool",
            opacity=opacity,
            z=z,
        )
        out.append(tex)
        return "".join(out), x

    for k, (zz, op) in enumerate(((-1.00, 0.28), (-0.50, 0.52))):
        tex, _ = enc_stack(f"g{k}", zz, op)
        add(tex)
    tex, enc_x1 = enc_stack("", 0.0, 0.95)
    add(tex)
    add(r"\draw[flow] (gA-hl) -- (lin1-west);" + "\n")
    add(r"\draw[flow] (lin1-east) -- (gva-west);" + "\n")
    add(r"\draw[flow] (gva-east) -- (lin2-west);" + "\n")
    add(r"\draw[flow] (lin2-east) -- (pool-west);" + "\n")
    add(
        f"\\draw[brace] ({enc_x1:.3f},{BRACE_Y:.3f},0) -- ({enc_x0:.3f},{BRACE_Y:.3f},0);\n"
    )
    add(
        lbl(
            f"({(enc_x0 + enc_x1) / 2:.3f},{NAME_Y:.3f},0)",
            r"\textbf{GeoPillar}\\\textbf{encoder}",
            "mod",
        )
    )
    add(
        lbl(
            f"({(enc_x0 + enc_x1) / 2:.3f},{TAG_Y:.3f},0)",
            r"GVA, $G=8$\\$\times M$ pillars",
            "tag",
            anchor="south",
        )
    )

    # -- 5. Pooled pillar tensor ----------------------------------------------
    cell, ngx, ngz = 0.36, 7, 6
    x0 = s.place(ngx * cell, 1.5)
    pbars, rng = [], random.Random(3)
    for ci, cj in [(0, 2), (1, 5), (3, 0), (4, 3), (6, 1)]:
        pbars.append((ci, cj, rng.uniform(0.95, 1.35), "cPool"))
    tex, _ = iso_grid("gB", x0, ngx, ngz, cell, pbars, dots=0)
    add(tex)
    add(f"\\draw[flow] (pool-east) -- ({x0 - 0.18:.3f},0,0);\n")
    add(
        lbl(
            f"({x0 + ngx * cell / 2:.3f},{TAG_Y:.3f},0)",
            r"$(M,\,64)$",
            "tag",
            anchor="south",
        )
    )

    # -- 6. Scatter -----------------------------------------------------------
    add("% ====== scatter ======\n")
    x0 = s.place(0.26, 2.5)
    tex, _ = box("scatter", x0, 0.26, fill="cIn", opacity=0.85)
    add(tex)
    add(f"\\draw[flow] (gB-e) -- (scatter-west);\n")
    add(
        f"\\draw[leader] ({x0 + 0.13:.3f},{-H / 2 - 0.58:.3f},0) -- "
        f"({x0 + 0.13:.3f},{NAME_Y + 0.06:.3f},0);\n"
    )
    add(lbl(f"({x0 + 0.13:.3f},{NAME_Y:.3f},0)", r"\textbf{Scatter}", "mod"))

    # -- 7. BEV canvas, upright, and the learned positional prior -------------
    add("% ====== BEV canvas + learned positional prior ======\n")
    cslab = 2.40
    x0 = s.place(0.30 + 0.385 * cslab, 2.7)
    tex, canvas_x1 = bev_slab("canvas", x0, 0.30, cslab, 10, "cCanvas", 0.62)
    add(tex)
    add(r"\draw[flow] (scatter-east) -- (canvas-west);" + "\n")
    add(
        lbl(
            f"({x0 + 0.15:.3f},{NAME_Y:.3f},0)", r"\textbf{BEV}\\\textbf{canvas}", "mod"
        )
    )
    add(
        lbl(
            f"({x0 + 0.15:.3f},{TAG_Y:.3f},0)",
            r"$64\times256^2$",
            "tag",
            anchor="south",
        )
    )

    sum_x = s.place(0.0, 1.0)
    add(
        f"\\node[circle, draw=cInk, line width=0.7pt, inner sep=1pt, fill=white,"
        f" font=\\fontsize{{16}}{{16}}\\selectfont] (sum) at ({sum_x:.3f},0,0) {{$+$}};\n"
    )
    add(r"\draw[flow] (canvas-east) -- (sum);" + "\n")

    # x0 = sum_x - width/2 puts -south directly over the sum, and the depth
    # offsets cancel, so the slab still reads as centred on it.
    pslab = 1.70
    tex, _ = bev_slab("prior", sum_x - 0.15, 0.30, pslab, 8, "cPrior", 0.95, y0=PRIOR_Y)
    add(tex)
    add(r"\draw[flow, draw=cPrior!62!black] (prior-south) -- (sum);" + "\n")
    add(
        lbl(
            "(prior-north)",
            r"\textbf{learned BEV prior}",
            "mod",
            anchor="south",
            dy=1.12,
        )
    )
    add(
        lbl(
            "(prior-north)",
            r"4.19\,M params, 57.6\,\% of the model",
            "tag",
            anchor="south",
            dy=0.50,
        )
    )

    # -- 8. RepLK-BEV backbone ------------------------------------------------
    add("% ====== RepLK-BEV backbone ======\n")
    bb_w = (
        0.22
        + 0.22
        + 0.28
        + 0.13
        + 0.28
        + 0.46
        + 0.36
        + 0.13
        + 0.36
        + 0.46
        + 4 * 0.48
        + 3 * 0.13
    )
    bb_x0 = s.place(bb_w, 2.6)
    tex, x = box("stem", bb_x0, 0.22, height=H, fill="cS0", opacity=0.5)
    add(tex)
    tex, x = box("s0a", x + 0.22, 0.28, height=H, fill="cS0")
    add(tex)
    tex, x = box("s0b", x + 0.13, 0.28, height=H, fill="cS0")
    add(tex)
    tex, x = box("s1a", x + 0.46, 0.36, height=1.70, fill="cS1")
    add(tex)
    tex, x = box("s1b", x + 0.13, 0.36, height=1.70, fill="cS1")
    add(tex)
    for k, nm in enumerate(["s2a", "s2b", "s2c", "s2d"]):
        tex, x = box(nm, x + (0.46 if k == 0 else 0.13), 0.48, height=1.12, fill="cS2")
        add(tex)
    bb_x1 = x
    add(r"\draw[flow] (sum) -- (stem-west);" + "\n")
    add(
        f"\\draw[brace] ({bb_x1:.3f},{BRACE_Y:.3f},0) -- ({bb_x0:.3f},{BRACE_Y:.3f},0);\n"
    )
    add(
        lbl(
            f"({(bb_x0 + bb_x1) / 2:.3f},{NAME_Y:.3f},0)",
            r"\textbf{RepLK}\\\textbf{backbone}",
            "mod",
        )
    )
    add(
        lbl(
            f"({(bb_x0 + bb_x1) / 2:.3f},{TAG_Y:.3f},0)",
            r"$k=13/11/9$\\stride 4",
            "tag",
            anchor="south",
        )
    )

    # -- 9. NGFM - labelled above, so the column below stays free for the
    #        supervision arrow. -----------------------------------------------
    add("% ====== normal-guided feature modulation ======\n")
    x0 = s.place(0.58 + 0.385 * 1.12, 2.7)
    tex, _ = box("ngfm", x0, 0.58, height=1.12, fill="cNgfm")
    add(tex)
    add(r"\draw[flow] (s2d-east) -- (ngfm-west);" + "\n")
    add(lbl("(ngfm-north)", r"\textbf{NGFM}", "mod", anchor="south", dy=0.20))
    add(
        lbl(
            f"({x0 + 0.29:.3f},{TAG_Y:.3f},0)",
            r"$256\times64^2$",
            "tag",
            anchor="south",
        )
    )
    ngfm_cx = x0 + 0.29

    # -- 10. BEV upsampler ----------------------------------------------------
    add("% ====== BEV upsampler ======\n")
    up_x0 = s.place(0.36 + 0.46 + 0.36 + 0.385 * H, 3.3)
    tex, x = box("up1", up_x0, 0.36, height=1.70, fill="cUp1")
    add(tex)
    tex, up_x1 = box("up2", x + 0.46, 0.36, height=H, fill="cUp2")
    add(tex)
    add(r"\draw[flow] (ngfm-east) -- (up1-west);" + "\n")
    add(r"\draw[flow] (up1-east) -- (up2-west);" + "\n")
    add(
        f"\\draw[brace] ({up_x1:.3f},{BRACE_Y:.3f},0) -- ({up_x0:.3f},{BRACE_Y:.3f},0);\n"
    )
    add(
        lbl(f"({(up_x0 + up_x1) / 2:.3f},{NAME_Y:.3f},0)", r"\textbf{Upsampler}", "mod")
    )
    add(
        lbl(
            f"({(up_x0 + up_x1) / 2:.3f},{TAG_Y:.3f},0)",
            r"$128\times256^2$",
            "tag",
            anchor="south",
        )
    )

    # -- 11. CenterPoint head -------------------------------------------------
    add("% ====== CenterPoint head ======\n")
    x0 = s.place(0.46 + 0.385 * H, 3.7)
    tex, _ = box("head", x0, 0.46, fill="cHead", opacity=0.9)
    add(tex)
    add(r"\draw[flow] (up2-east) -- (head-west);" + "\n")
    add(
        f"\\draw[leader] ({x0 + 0.23:.3f},{-H / 2 - 0.58:.3f},0) -- "
        f"({x0 + 0.23:.3f},{NAME_Y + 0.06:.3f},0);\n"
    )
    add(
        lbl(
            f"({x0 + 0.23:.3f},{NAME_Y:.3f},0)",
            r"\textbf{CenterPoint}\\\textbf{head}",
            "mod",
        )
    )
    add(lbl(f"({x0 + 0.23:.3f},{TAG_Y:.3f},0)", r"OSF $=1$", "tag", anchor="south"))

    # -- 12. Detections -------------------------------------------------------
    add("% ====== detections ======\n")
    det_w = PANEL_H * 1031 / 518
    x0 = s.place(det_w, 2.2)
    add(f"""\\node[anchor=west, inner sep=0] (det) at ({x0:.3f},0,0)
    {{\\includegraphics[height={PANEL_H:.2f}cm]{{panels/detections.png}}}};
""")
    add(r"\draw[flow] (head-east) -- (det.west);" + "\n")
    add(
        lbl(
            f"({x0 + det_w / 2:.3f},{NAME_Y:.3f},0)",
            r"\textbf{3-D}\\\textbf{boxes}",
            "mod",
        )
    )

    # -- 13. Auxiliary normal supervision. One rail, dropped at the panel's left
    #        edge so it misses the name, and run at the loss node's own height so
    #        the last segment lands on its west point without doubling back. ---
    add("% ====== auxiliary normal supervision - training only ======\n")
    add(f"""\\node[ellipse, draw=cLoss, fill=cLoss!8, line width=0.8pt, inner sep=2.5pt,
      text=cInk, align=center, font=\\fontsize{{18}}{{20}}\\selectfont]
      (ngfmloss) at ({ngfm_cx:.3f},{RAIL_Y:.3f},0) {{\\textbf{{NGFM loss}}}};
\\draw[aux, draw=cLoss] (ngfm-south) -- (ngfmloss.north);
\\draw[aux] ({scan_x + 0.40:.3f},{-PANEL_H / 2:.3f},0)
      -- ({scan_x + 0.40:.3f},{RAIL_Y:.3f},0) -- (ngfmloss.west);
""")

    # -- 14. Legend -----------------------------------------------------------
    lx = ngfm_cx + 2.75  # the band right of the loss node is empty
    add(f"""% ====== legend ======
\\draw[flow] ({lx:.3f},{LEGEND_Y + 0.34:.3f},0) -- ++(0.85,0);
\\draw[aux]  ({lx:.3f},{LEGEND_Y - 0.34:.3f},0) -- ++(0.85,0);
""")
    add(
        lbl(
            f"({lx + 1.00:.3f},{LEGEND_Y + 0.34:.3f},0)",
            r"inference",
            "tag",
            anchor="west",
            dy=0.0,
        )
    )
    add(
        lbl(
            f"({lx + 1.00:.3f},{LEGEND_Y - 0.34:.3f},0)",
            r"training only: offline surface normals",
            "tag",
            anchor="west",
            dy=0.0,
        )
    )

    add("\n\\end{tikzpicture}\n\\end{document}\n")
    return "".join(t)


def main():
    out = os.path.join(HERE, "architectureDiagram.tex")
    with open(out, "w") as f:
        f.write(build())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
