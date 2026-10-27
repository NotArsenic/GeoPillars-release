#!/usr/bin/env python3
"""Grouped vector attention, in two panels.

Panel (a) is the primitive, one group's worth of vector relation attention;
panel (b) is how the block composes G of them, which is the same split Vaswani
et al. use between scaled dot-product and multi-head attention.

Everything is read from ``src/models/components/encoders/pillar_feature_net.py``
(class ``GroupedVectorAttention``) and cross-checked against the trained checkpoints.
Four things here are not scaled dot-product attention, and the figure is
drawn to make each visible:

  * the logit is a *difference* of two low-rank projections, q_ng - k_mg, so the
    block encodes relative arrangement rather than agreement;
  * there is no sqrt(d) term; the learned per-group scale s_g is the only
    temperature in the block;
  * the mask and the softmax both act on the key axis m alone, so padded query
    rows produce values that the pre-pool mask-fill discards;
  * one attention map is shared by all C/G = 8 channels of its group, which is
    what holds the block to 5,129 parameters.

Layout rule: every tensor shape sits beside the arrow it describes, in the gap
between the two boxes that arrow joins. Gaps carrying a label are opened to
``LGAP``; the rest stay at ``GAP``.

Usage (from scripts/diagrams/)::

    python gvaDiagram.py && latexmk -pdf gvaDiagram.tex
"""

import os

from blockstyle import arrow, blk, chain, footer, group, header, op, shape, txt

HERE = os.path.dirname(os.path.abspath(__file__))

W = 6.2  # standard box width, cm
H1, H2 = 0.575, 0.635  # half-height of a one- and a two-line box
GAP, LGAP = 0.45, 0.80  # clear air between boxes; more where a label sits


def build():
    t = [header()]
    add = t.append

    # ======================================================================
    # Panel (a) - vector relation attention, one group g
    # ======================================================================
    ax = 0.0  # panel (a) column
    vx = 5.80  # the value branch runs up the right side
    # bottom-to-top: sub(2 lines), scl, msk, sfm, agg(2 lines)
    y1 = 1.75
    y2 = y1 + H2 + LGAP + H1
    y3 = y2 + H1 + GAP + H1
    y4 = y3 + H1 + GAP + H1
    y5 = y4 + H1 + LGAP + H2
    y_out = y5 + H2 + 0.95

    add("% ====== panel (a): vector relation attention ======\n")
    add(txt(ax - 2.30, 0.0, r"$q_{\cdot g}$"))
    add(txt(ax + 2.30, 0.0, r"$k_{\cdot g}$"))
    add(txt(vx, 0.0, r"$v_{\cdot g}$"))

    add(blk("sub", ax, y1, r"Subtract\\$q_{ng}-k_{mg}$", "bLin", W))
    add(blk("scl", ax, y2, r"Scale by $s_g$", "bNorm", W))
    add(blk("msk", ax, y3, r"Mask padded keys $m$", "bMask", W))
    add(blk("sfm", ax, y4, r"SoftMax over $m$", "bSoft", W))
    add(blk("agg", ax, y5, r"Aggregate\\$\sum_m \alpha_{nmg}\,v_{mgc}$", "bAgg", 7.4))

    for dx in (-2.30, 2.30):
        add(f"\\draw[flow] ({ax + dx:.2f},0.30) -- ({ax + dx:.2f},{y1 - H2:.2f});\n")
    add(chain(["sub", "scl", "msk", "sfm", "agg"]))
    add(f"\\draw[flow] ({vx:.2f},0.30) -- ({vx:.2f},{y5:.2f}) -- (agg.east);\n")
    add(arrow("agg.north", f"{ax:.2f},{y_out - 0.32:.2f}"))
    add(txt(ax, y_out, r"$o_{n g \cdot}$"))

    # shapes, each beside its own arrow
    add(shape(ax, (0.30 + y1 - H2) / 2, r"$(N,G)$", dx=0.0))  # between q and k
    add(shape(ax, (y1 + H2 + y2 - H1) / 2, r"$(N,N,G)$"))
    add(shape(ax, (y4 + H1 + y5 - H2) / 2, r"$\alpha$: $(N,N,G)$"))
    add(shape(ax, (y5 + H2 + y_out - 0.32) / 2, r"$(N,C/G)$"))
    add(shape(vx, (0.30 + y5) / 2, r"$(N,C/G)$"))

    add(txt(ax + 0.6, -1.85, r"(a)~Vector relation attention, one group", "ttl"))

    # ======================================================================
    # Panel (b) - the block: G groups, residual, zero-initialised gate
    # ======================================================================
    bx = 20.5  # panel (b) column
    rx = bx - 7.4  # the residual runs up the left side
    lin_dx, lin_w = 4.6, 4.2
    GRP = 1.15  # half-height of the dashed group box
    u1 = 1.75  # the three projections
    u2 = u1 + H2 + LGAP + GRP  # the group box, centred
    u3 = u2 + GRP + LGAP + H2  # flatten
    u4 = u3 + H2 + GAP + H1  # gamma
    u5 = u4 + H1 + GAP + 0.33  # the residual sum
    u6 = u5 + 0.33 + 0.80  # y

    add("% ====== panel (b): grouped vector attention ======\n")
    add(txt(bx, 0.0, r"$x$\quad$(P,N,C{=}64)$"))

    add(blk("wq", bx - lin_dx, u1, r"Linear $W_q$\\$C\to G$", "bLin", lin_w))
    add(blk("wk", bx, u1, r"Linear $W_k$\\$C\to G$", "bLin", lin_w))
    add(blk("wv", bx + lin_dx, u1, r"Linear $W_v$\\$C\to C$", "bLin", lin_w))
    for nm, dx in (("wq", -lin_dx), ("wk", 0.0), ("wv", lin_dx)):
        add(f"\\draw[flow] ({bx:.2f},0.32) -- ({bx + dx:.2f},0.32) -- ({nm}.south);\n")

    add(blk("vra", bx, u2, r"Vector relation attention\\\emph{panel (a)}", "bAgg", 7.6))
    add(
        group(
            "grp", bx - 6.4, u2 - GRP, bx + 6.4, u2 + GRP, tag=r"$\times\,G=8$ groups"
        )
    )
    for nm, dx in (("wq", -lin_dx), ("wk", 0.0), ("wv", lin_dx)):
        add(f"\\draw[flow] ({nm}.north) -- ({bx + dx:.2f},{u2 - GRP:.2f});\n")

    add(blk("flt", bx, u3, r"Flatten groups\\$(N,G,C/G)\to(N,C)$", "bSoft", W))
    add(blk("gam", bx, u4, r"Gate $\gamma$\quad(scalar, init $0$)", "bNorm", W))
    add(op("add", bx, u5, r"$+$"))
    add(txt(bx, u6, r"$y$\quad$(P,N,C)$"))

    add(f"\\draw[flow] ({bx:.2f},{u2 + GRP:.2f}) -- (flt.south);\n")
    add(chain(["flt", "gam"]))
    add(arrow("gam.north", "add.south"))
    add(arrow("add.north", f"{bx:.2f},{u6 - 0.32:.2f}"))
    add(
        f"\\draw[flow] ({bx:.2f},0.32) -- ({rx:.2f},0.32) "
        f"-- ({rx:.2f},{u5:.2f}) -- (add.west);\n"
    )
    add(shape(rx, (u1 + u4) / 2, r"identity", "east"))

    # one shape per projection arrow, in the gap they all cross
    yq = (u1 + H2 + u2 - GRP) / 2
    add(shape(bx - lin_dx, yq, r"$(P,N,G)$"))
    add(shape(bx, yq, r"$(P,N,G)$"))
    add(shape(bx + lin_dx, yq, r"$(P,N,G,C/G)$"))
    add(shape(bx, (u2 + GRP + u3 - H2) / 2, r"$(P,N,G,C/G)$"))

    add(txt(bx, -1.85, r"(b)~Grouped vector attention", "ttl"))
    add(txt(bx, -2.55, r"5{,}129 parameters in the whole block", "shp"))

    add(footer())
    return "".join(t)


def main():
    out = os.path.join(HERE, "gvaDiagram.tex")
    with open(out, "w") as f:
        f.write(build())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
