#!/bin/sh
# Post-collection pipeline for one loop-7/8 run: judge (gpt-4.1-mini) -> second judge (Llama-3.3-70B) -> analysis
# -> cross-fit -> figures -> loop-8 scoring.  Usage: sh src/run_loop8_post.sh runs/postexit2_llama70b data/postexit2_generated_llama70b.json llama70b
set -e
RUN=$1; GEN=$2; TAG=$3
PY=.venv/bin/python
$PY src/identity_judge2.py "$RUN" "$GEN" > "$RUN/judge.log" 2>&1
$PY src/identity_judge2.py "$RUN" "$GEN" --judge-provider openrouter --judge-model meta-llama/llama-3.3-70b-instruct --suffix _llama33 > "$RUN/judge_llama33.log" 2>&1
$PY src/analysis_postexit2.py "$RUN" > "$RUN/analysis.log" 2>&1
$PY src/analysis_postexit2_crossfit.py "$RUN" > "$RUN/crossfit.log" 2>&1
$PY src/fig_loop7.py "$RUN" "$TAG" 2>&1 | grep -v findfont || true
$PY src/score_prereg8.py "$RUN" > "$RUN/prereg8.log" 2>&1
$PY src/judge_agreement.py "$RUN/postexit2_probes.jsonl" "$RUN/postexit2_probes_llama33.jsonl" > "$RUN/judge_agreement.log" 2>&1
gzip -k -f "$RUN/results.jsonl"
echo "DONE $RUN"
