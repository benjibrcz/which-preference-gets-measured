#!/usr/bin/env bash
# Build a PDF from a Markdown source, substituting Unicode Greek/math glyphs into LaTeX
# math mode (\ensuremath{...}) so they render under tectonic's default Latin Modern fonts
# instead of appearing as missing-glyph boxes.
#
# Usage: src/build_pdf.sh <source.md> <out.pdf> [margin] [fontsize]
#   e.g. src/build_pdf.sh results/SUBMISSION.md results/SUBMISSION.pdf 2.2cm 10pt
set -euo pipefail

SRC_ABS="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT_ABS="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
MARGIN="${3:-2.2cm}"
FS="${4:-10pt}"

DIR="$(dirname "$SRC_ABS")"
cd "$DIR"                      # so relative image paths (figures/...) resolve
TMP="./.build_tmp.md"

sed -e 's/→/\\ensuremath{\\rightarrow}/g'  -e 's/↔/\\ensuremath{\\leftrightarrow}/g' \
    -e 's/◆/\\ensuremath{\\blacklozenge}/g' -e 's/≠/\\ensuremath{\\neq}/g' -e 's/≫/\\ensuremath{\\gg}/g' \
    -e 's/≤/\\ensuremath{\\leq}/g'  -e 's/≥/\\ensuremath{\\geq}/g'  -e 's/≈/\\ensuremath{\\approx}/g' \
    -e 's/±/\\ensuremath{\\pm}/g'   -e 's/✓/(pass)/g'  -e 's/✗/(fail)/g'  -e 's/⟂/\\ensuremath{\\perp}/g' \
    -e 's/˅/v/g'  -e 's/×/\\ensuremath{\\times}/g'  -e 's/β/\\ensuremath{\\beta}/g' \
    -e 's/ρ/\\ensuremath{\\rho}/g'  -e 's/Δ/\\ensuremath{\\Delta}/g'  -e 's/α/\\ensuremath{\\alpha}/g' \
    -e 's/σ/\\ensuremath{\\sigma}/g' -e 's/η/\\ensuremath{\\eta}/g' \
    -e 's/∈/\\ensuremath{\\in}/g' \
    "$(basename "$SRC_ABS")" > "$TMP"

pandoc "$TMP" -o "$OUT_ABS" --pdf-engine=tectonic \
    ${PDF_HEADER:+--include-in-header="$PDF_HEADER"} \
    -V geometry:margin="$MARGIN" -V fontsize="$FS" --resource-path=.

rm -f "$TMP"
echo "built $OUT_ABS"
