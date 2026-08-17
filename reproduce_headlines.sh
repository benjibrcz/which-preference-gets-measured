#!/usr/bin/env bash
# Reproduce the headline numbers from cached model outputs — NO API access required.
# Requires Python 3.11+ and the packages in requirements.txt (numpy, pandas).
#   python -m venv .venv && .venv/bin/pip install -r requirements.txt
#   ./reproduce_headlines.sh
#
# Behavioural results are deterministic from runs/*/results.jsonl (seeded bootstraps).
# The GPU activation/steering arm is NOT reproduced here — its raw .npy dumps are omitted
# (see README); only the behavioural headlines below reproduce offline.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
echo "using interpreter: $PY"; $PY --version

echo
echo "==================================================================="
echo " 1. Shared-baseline cross-fit of beta + non-agent normative control"
echo "    (expect: max |naive - cross-fit| small, 10/12 cells >= 0.50;"
echo "     policy beta 0.20-0.79 with CIs excluding zero)"
echo "==================================================================="
$PY src/analysis_review.py

echo
echo "==================================================================="
echo " 2. Concordance triangle + say/do dissociation (Gemma core grid)"
echo "==================================================================="
$PY src/analysis_concordance.py runs/gridA_gemma

echo
echo "==================================================================="
echo " 3. Post-exit persistence (hysteresis), Gemma"
echo "==================================================================="
$PY src/analysis_hysteresis.py runs/hyst_gemma

echo
echo "Done. These figures back the three claims in results/SUBMISSION.md."
