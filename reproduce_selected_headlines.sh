#!/usr/bin/env bash
# Reproduce SELECTED headline numbers from cached model outputs — NO API access required.
# Requires Python 3.11+ and the packages in requirements.txt (numpy, pandas, scipy).
#   python -m venv .venv && .venv/bin/pip install -r requirements.txt
#   ./reproduce_selected_headlines.sh
#
# SCOPE — what this DOES reproduce (deterministically; bootstraps are seeded):
#   * the shared-baseline cross-fit of beta + the non-agent normative control (analysis_review)
#   * the concordance triangle + say/do dissociation on the Gemma core grid (analysis_concordance)
#   * post-exit persistence, Gemma (analysis_hysteresis)
#   * the negation-aware recode of the post-exit identity channel (analysis_identity)
#   * automated assertions on the key estimates above (assert_headlines)
# What it does NOT reproduce here: the full cross-model wedge table, the extended interventions and
# their exact CIs, the context-surgery arm, Figure 7 / the paper figures (run src/make_figures.py and
# src/fig_construct.py), the PDFs, and the GPU activation/steering arm (raw .npy dumps are omitted —
# see README). Those require the other scripts and/or a GPU.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
echo "using interpreter: $PY"; $PY --version

echo; echo "=== 1. cross-fit beta + non-agent normative control ==="
$PY src/analysis_review.py

echo; echo "=== 2. concordance triangle + say/do dissociation (Gemma core grid) ==="
$PY src/analysis_concordance.py runs/gridA_gemma

echo; echo "=== 3. post-exit persistence (Gemma) ==="
$PY src/analysis_hysteresis.py runs/hyst_gemma

echo; echo "=== 4. negation-aware post-exit identity recode ==="
$PY src/analysis_identity.py

echo; echo "=== 5. assert key headline estimates (fails loudly if any drift) ==="
$PY src/assert_headlines.py

echo; echo "All selected-headline checks passed."
