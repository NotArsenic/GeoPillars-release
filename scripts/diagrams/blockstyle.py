#!/usr/bin/env python3
"""Shared TikZ scaffolding for the two block diagrams.

``gvaDiagram.py`` and ``ngfmDiagram.py`` draw bottom-to-top data flows in the
idiom of Vaswani et al.'s Figs. 1-2: rounded pastel boxes, one operation per
box, thin arrows carrying tensor shapes, a dashed rectangle around anything
that repeats, and a circled operator for the residual join.

Unlike ``architectureDiagram.py`` these are plain 2-D pictures, so every
coordinate below is centimetres in the page plane. They are drawn at roughly
1.8x their printed size: a 14 pt label here sets about 7.8 pt at \\textwidth.
"""

PALETTE = r"""
\definecolor{bLin}  {HTML}{CFE2F3}   % linear / convolution
\definecolor{bNorm} {HTML}{FFF2CC}   % normalisation + activation
\definecolor{bSoft} {HTML}{D9EAD3}   % softmax / vector normalisation
\definecolor{bMask} {HTML}{EAD1DC}   % masking
\definecolor{bAgg}  {HTML}{FBD3A8}   % contraction / elementwise product
\definecolor{bPool} {HTML}{E1D5F0}   % pooling
\definecolor{bLoss} {HTML}{F4CCCC}   % objective
\definecolor{bRed}  {HTML}{C0392B}
\definecolor{bInk}  {HTML}{1A1A1A}
\definecolor{bMuted}{HTML}{55606B}
"""

STYLES = r"""
    blk/.style  ={draw=bInk, line width=0.8pt, rounded corners=3pt, align=center,
                  minimum height=1.15cm, inner xsep=6pt, inner ysep=3pt,
                  text=bInk, font=\fontsize{14}{16.5}\selectfont},
    io/.style   ={align=center, text=bInk, font=\fontsize{14.5}{17}\selectfont},
    % inner sep 1pt so a shape label fits in the gap between two boxes
    shp/.style  ={align=center, text=bMuted, inner sep=1pt,
                  font=\fontsize{12}{14}\selectfont},
    ttl/.style  ={align=center, text=bInk, font=\fontsize{15}{17}\selectfont},
    flow/.style ={-{Stealth[length=5pt,width=3.6pt]}, draw=bInk, line width=0.9pt},
    aux/.style  ={-{Stealth[length=5pt,width=3.6pt]}, draw=bRed, line width=0.9pt,
                  dash pattern=on 3pt off 2pt},
    grp/.style  ={draw=bMuted, line width=0.7pt, dash pattern=on 3pt off 2pt,
                  rounded corners=5pt},
    op/.style   ={draw=bInk, line width=0.8pt, circle, inner sep=1.6pt, fill=white,
                  text=bInk, font=\fontsize{14}{14}\selectfont},
"""


def header():
    return (
        r"""\documentclass[border=4pt, tikz]{standalone}
\usepackage{mathptmx}
\usepackage{amsmath,amssymb}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds,shapes.geometric,decorations.pathreplacing}
"""
        + PALETTE
        + "\\begin{document}\n\\begin{tikzpicture}[\n"
        # a blank line inside the option list is a \par, which breaks tikz
        + STYLES.strip("\n")
        + "\n  ]\n"
    )


def footer():
    return "\n\\end{tikzpicture}\n\\end{document}\n"


def blk(name, x, y, text, fill, w):
    """One operation. ``w`` is the box's minimum width in cm."""
    return (
        f"\\node[blk, fill={fill}, minimum width={w:.2f}cm] ({name}) "
        f"at ({x:.2f},{y:.2f}) {{{text}}};\n"
    )


def op(name, x, y, text):
    return f"\\node[op] ({name}) at ({x:.2f},{y:.2f}) {{{text}}};\n"


def txt(x, y, text, style="io", anchor="center"):
    return f"\\node[{style}, anchor={anchor}] at ({x:.2f},{y:.2f}) {{{text}}};\n"


def arrow(a, b, style="flow"):
    return f"\\draw[{style}] ({a}) -- ({b});\n"


def chain(names, style="flow"):
    """Vertical arrows through a bottom-to-top column of node names."""
    return "".join(arrow(a, b, style) for a, b in zip(names, names[1:]))


def shape(x, y, text, anchor="west", dx=0.30):
    """A tensor shape set beside the arrow at ``x``, at that arrow's own height.

    Shapes go next to the line they describe, never out in the margin: a label
    parked at the edge of the panel reads as belonging to nothing in particular.
    ``dx`` is the clearance from the arrow, signed by ``anchor``.
    """
    return txt(x + (dx if anchor == "west" else -dx), y, text, "shp", anchor)


def group(name, x0, y0, x1, y1, tag=None, tag_anchor="north west"):
    """Dashed rectangle marking something that repeats, with an optional tag."""
    out = (
        f"\\node[grp, fit={{({x0:.2f},{y0:.2f}) ({x1:.2f},{y1:.2f})}}] ({name}) {{}};\n"
    )
    if tag:
        ax, ay = (x1 - 0.12, y1 - 0.12)
        out += f"\\node[shp, anchor=north east] at ({ax:.2f},{ay:.2f}) {{{tag}}};\n"
    return out
