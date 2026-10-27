#!/usr/bin/env bash
# Rebuild every paper diagram in this directory, end to end.
#
#   bash scripts/diagrams/make_diagrams.sh
#
#   bash scripts/diagrams/make_diagrams.sh --install <figures dir>
#
# Produces:
#   architectureDiagram.pdf  full pipeline (Fig. 1)   -> fig1_architecture.pdf
#   gvaDiagram.pdf           grouped vector attention -> fig_gva.pdf
#   ngfmDiagram.pdf          normal-guided modulation -> fig_ngfm.pdf
#   architectureDiagram.png  150 dpi copy of Fig. 1 for the README
#
# --install copies the PDFs into <figures dir> under the names on the right. The end
# panels for the architecture figure are re-rendered from the exported scene NPZ,
# so a clean run proves the figure needs no dataset.
#
# Needs latexmk, pdfinfo, pdftoppm and PlotNeuralNet cloned into this directory:
#   git clone https://github.com/HarisIqbal88/PlotNeuralNet scripts/diagrams/PlotNeuralNet
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
cd "$HERE"
if [[ ! -d PlotNeuralNet/layers ]]; then
    echo "PlotNeuralNet not found: git clone https://github.com/HarisIqbal88/PlotNeuralNet $HERE/PlotNeuralNet" >&2
    exit 1
fi

"$PY" render_panels.py
for d in architectureDiagram gvaDiagram ngfmDiagram; do
    "$PY" "$d.py"
    latexmk -pdf -interaction=nonstopmode -halt-on-error "$d.tex" >/dev/null
    latexmk -c "$d.tex" >/dev/null
    printf '%-24s %s\n' "$d.pdf" "$(pdfinfo "$d.pdf" | sed -n 's/^Page size: *//p')"
done
pdftoppm -png -r 150 -singlefile architectureDiagram.pdf architectureDiagram

if [[ "${1:-}" == "--install" ]]; then
    DEST="${2:?--install needs a target directory}"
    cp architectureDiagram.pdf "$DEST/fig1_architecture.pdf"
    cp gvaDiagram.pdf          "$DEST/fig_gva.pdf"
    cp ngfmDiagram.pdf         "$DEST/fig_ngfm.pdf"
    echo "installed into $DEST"
fi
