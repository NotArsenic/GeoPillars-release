#!/usr/bin/env python3
"""Normal-guided feature modulation, in two panels.

Panel (a) is the forward branch that runs at inference; panel (b) is the target
construction and loss, which exist only during training. Read from
``src/models/components/modulation/{NGFM.py,normal_target.py}`` and cross-checked
against the paper's Eqs. (8)-(11).

Points the figure is drawn to make:

  * the branch is shape-preserving: it returns the same 256x64x64 map it was
    given, and the only thing that leaves it is the gate;
  * the gate is a sigmoid, so 1 + m lands in (1, 2) and the block cannot zero a
    feature out - the identity path is always live;
  * which orientations get amplified is learned by the 3 -> 256 expansion, not
    imposed by the normal;
  * the target is occupancy-weighted pooling *then* renormalisation, and a patch
    is supervised only when at least 2 of its 16 sub-cells are occupied.

The density weight the implementation also carries is inert (the positional
prior makes every cell non-zero, so it is identically one); the loss is an
unweighted masked mean and is drawn that way.

Layout rule: every tensor shape sits beside the arrow it describes, in the gap
between the two boxes that arrow joins. Gaps carrying a label are opened to
``LGAP``; the rest stay at ``GAP``.

Usage (from scripts/diagrams/)::

    python ngfmDiagram.py && latexmk -pdf ngfmDiagram.tex
"""

import os

from blockstyle import arrow, blk, chain, footer, header, shape, txt

HERE = os.path.dirname(os.path.abspath(__file__))

W = 7.0  # standard box width, cm
H1, H2 = 0.575, 0.635  # half-height of a one- and a two-line box
GAP, LGAP = 0.45, 0.80  # clear air between boxes; more where a label sits


def build():
    t = [header()]
    add = t.append

    # ======================================================================
    # Panel (a) - the forward branch
    # ======================================================================
    ax, idx = 0.0, -5.60  # column, and the identity path's x
    y1 = 1.55  # conv 3x3
    y2 = y1 + H1 + LGAP + H1  # BN + GELU
    y3 = y2 + H1 + GAP + H1  # conv 1x1 -> 3
    y4 = y3 + H1 + LGAP + H2  # l2 normalise (2 lines)
    y5 = y4 + H2 + LGAP + H1  # conv 1x1 -> 256
    y6 = y5 + H1 + GAP + H1  # BN + sigmoid
    y7 = y6 + H1 + LGAP + H1  # modulate
    y_out = y7 + H1 + 0.95

    add("% ====== panel (a): forward modulation ======\n")
    add(txt(ax, 0.0, r"$F$\quad$(B,256,64,64)$"))
    add(blk("c1", ax, y1, r"Conv $3{\times}3$, $256\to128$", "bLin", W))
    add(blk("n1", ax, y2, r"BatchNorm, GELU", "bNorm", W))
    add(blk("c2", ax, y3, r"Conv $1{\times}1$, $128\to3$", "bLin", W))
    add(
        blk(
            "l2",
            ax,
            y4,
            r"$\ell_2$ normalise over channels\\$\hat{n}=r/\max(\Vert r\Vert_2,10^{-6})$",
            "bSoft",
            W,
        )
    )
    add(blk("c3", ax, y5, r"Conv $1{\times}1$, $3\to256$", "bLin", W))
    add(blk("n2", ax, y6, r"BatchNorm, Sigmoid", "bNorm", W))
    add(blk("mod", ax, y7, r"$F' = F \odot (1+m)$", "bAgg", W))
    add(txt(ax, y_out, r"$F'$\quad$(B,256,64,64)$"))

    add(f"\\draw[flow] ({ax:.2f},0.32) -- (c1.south);\n")
    add(chain(["c1", "n1", "c2", "l2", "c3", "n2", "mod"]))
    add(arrow("mod.north", f"{ax:.2f},{y_out - 0.32:.2f}"))
    add(
        f"\\draw[flow] ({ax:.2f},0.32) -- ({idx:.2f},0.32) "
        f"-- ({idx:.2f},{y7:.2f}) -- (mod.west);\n"
    )
    add(shape(idx, (y3 + y4) / 2, r"identity", "east"))

    # shapes, each beside its own arrow
    add(shape(ax, (y1 + H1 + y2 - H1) / 2, r"$(B,128,64^2)$"))
    add(shape(ax, (y3 + H1 + y4 - H2) / 2, r"$r$: $(B,3,64^2)$"))
    add(shape(ax, (y4 + H2 + y5 - H1) / 2, r"$\hat{n}$: unit vectors"))
    add(shape(ax, (y6 + H1 + y7 - H1) / 2, r"$m\in(0,1)$"))

    add(txt(ax, -2.20, r"(a)~Normal-guided feature modulation", "ttl"))
    add(txt(ax, -2.90, r"296{,}832 parameters", "shp"))

    # ======================================================================
    # Panel (b) - target construction and loss, training only
    # ======================================================================
    nx, ox = 15.90, 22.80  # normals column, occupancy column
    bw = 6.2
    v1 = 1.55  # average pool
    v2 = v1 + H1 + GAP + H2  # divide / validity (2 lines)
    v3 = v2 + H2 + GAP + H1  # l2 normalise
    v4 = v3 + H1 + LGAP + H1  # 1 - cos
    v5 = v4 + H1 + GAP + H1  # masked mean
    v6 = v5 + H1 + GAP + H1  # the loss

    add("% ====== panel (b): auxiliary supervision, training only ======\n")
    add(txt(nx, 0.0, r"offline normals\\$(B,3,256,256)$"))
    add(txt(ox, 0.0, r"occupancy\\$(B,1,256,256)$"))

    add(blk("pn", nx, v1, r"Average pool $4{\times}4$", "bPool", bw))
    add(blk("po", ox, v1, r"Average pool $4{\times}4$", "bPool", bw))
    add(blk("dv", nx, v2, r"$\bar{n}/\max(\bar{o},10^{-6})$", "bLin", bw))
    add(
        blk("vm", ox, v2, r"$16\,\bar{o} > 1.6$\\\emph{2 of 16 sub-cells}", "bMask", bw)
    )
    add(blk("ln", nx, v3, r"$\ell_2$ normalise", "bSoft", bw))
    add(blk("cs", nx, v4, r"$1-\cos(\hat{n},n)$", "bAgg", bw))
    add(blk("mm", nx, v5, r"Mean over supervised $V$", "bSoft", bw))
    add(blk("ls", nx, v6, r"$L_{\mathrm{ngfm}}$", "bLoss", 3.6))

    for col, first in ((nx, "pn"), (ox, "po")):
        add(f"\\draw[flow] ({col:.2f},0.62) -- ({first}.south);\n")
    add(chain(["pn", "dv", "ln", "cs", "mm", "ls"]))
    add(arrow("po.north", "vm.south"))
    add(f"\\draw[flow] ({ox:.2f},{v2 + H2:.2f}) -- ({ox:.2f},{v5:.2f}) -- (mm.east);\n")
    add(shape(ox, (v2 + H2 + v5) / 2, r"supervised\\patches $V$"))

    add(shape(nx, (v3 + H1 + v4 - H1) / 2, r"$n$: $(B,3,64,64)$"))

    # the only thing that crosses between the panels, labelled on its own run
    jx = (ax + W / 2 + nx - bw / 2) / 2
    add(
        f"\\draw[aux] (l2.east) -- ({jx:.2f},{y4:.2f}) "
        f"-- ({jx:.2f},{v4:.2f}) -- (cs.west);\n"
    )
    add(shape(jx, (y4 + v4) / 2 + 0.52, r"$\hat{n}$: $(B,3,64,64)$"))

    add(
        txt(
            (nx + ox) / 2,
            -2.20,
            r"(b)~Auxiliary normal supervision, training only",
            "ttl",
        )
    )

    add(footer())
    return "".join(t)


def main():
    out = os.path.join(HERE, "ngfmDiagram.tex")
    with open(out, "w") as f:
        f.write(build())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
