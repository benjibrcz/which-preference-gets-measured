# Apart Digital Minds Research Sprint — 14–16 Aug 2026

**Status: experiments complete.** Both candidate projects were run to conclusion (A absorbed B, as planned): the concordance triangle (revealed / stated / probe channels × binding grid × hysteresis) and the marginal-sensitivity analysis, replicated across four core models (Gemma-3-27B, Llama-3.1-70B, gpt-4.1-mini, Qwen-2.5-72B) with a 12-model extension. **[results/SUBMISSION.md](results/SUBMISSION.md) / [.pdf](results/SUBMISSION.pdf)** is the sprint-sized deliverable; the full technical write-up with review-round robustness analyses is in the report below.

- **[results/SUBMISSION.md](results/SUBMISSION.md)** — the 2-page sprint submission (start here).
- **[results/REPORT.md](results/REPORT.md)** — the full write-up (abstract → results §2.1–2.17 → interpretation → limitations), with figures in `results/figures/` (`src/make_figures.py` regenerates them).
- **[results/FINDINGS.md](results/FINDINGS.md)** — running findings log with per-model numbers.
- **[PREREG.md](PREREG.md)** — preregistered analysis decisions, gates, and amendment log.
- **[results/PROGRAM_ACTIVE_PERSPECTIVES.md](results/PROGRAM_ACTIVE_PERSPECTIVES.md)** — post-sprint research-programme sketch this work seeds.

Headlines: (1) fiction-attributed persona descriptions capture revealed behaviour (10/12 model×persona cells ≥ 0.50 across 4 models; both exceptions Qwen), no instruction regime is leak-free, warmth leaks everywhere; (2) say/do wedges under identity manipulation recur across all 4 models with model-specific direction (individually significant only in the largest cells); (3) after roleplay exit, identity reports reset 100% while preference channels stay displaced — and nothing clears it (8 neutral turns, explicit reset, fresh system prompt all fail); (4) almost no measured preferences are stable across persona configurations (1/0/7/4 of 76 pairs, marginal-sensitivity analysis) — measured preferences track the active configuration far more than the model; (5) probes: every dissociation is internally real, the probe reads whichever answer the question calls up (no neutral internal arbiter), yet the disavowed action-preference is co-present and decodable during the disavowal itself; steering: choices flip at doses that leave facts 100% intact; (6) mechanism: capture rides a *content* pathway, avowal tracks a separable *enactment-stance* state — personas compete for expression, prose doesn't; (7) the assistant itself: an identity cloud (not a point; cloud size is a model trait), a trait mixture dominated by caution for HHH/constitution phrasings, a one-way door (easy to leave by description, impossible to restore in-context), and hardening buys only dilution.

Original project sketches: [A — Concordance Triangle](A_concordance_triangle.md), [B — Entity Individuation](B_entity_individuation.md). Both extend Gilg, Beckmann, Paleka & Butlin, *Probing Persona-Dependent Preferences in Language Models* (arXiv:2605.13339), and instantiate the representation-vs-binding agenda + Cozmin's invariance/covariance doc.

## Reproduce

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/run_grid.py --preset gridA --model google/gemma-3-27b-it --provider openrouter --out runs/gridA_gemma
.venv/bin/python src/hysteresis.py --model google/gemma-3-27b-it --out runs/hyst_gemma
.venv/bin/python src/analysis_concordance.py runs/gridA_gemma       # triangle + dissociation
.venv/bin/python src/analysis_direction.py runs/gridA_gemma runs/deconfound_gemma
.venv/bin/python src/analysis_provenance.py runs/gridA_gemma runs/context_gemma
.venv/bin/python src/analysis_hysteresis.py runs/hyst_gemma
.venv/bin/python src/analysis_review.py                             # cross-fit beta + non-agent normative control
```

All API results are cached in `runs/*/results.jsonl`; the behavioural analyses are deterministic from those files (no API calls needed to reproduce the numbers). Provide your own `OPENROUTER_API_KEY` / `OPENAI_API_KEY` as environment variables (or an `.env` file) only if you want to re-run the model queries. The GPU activation/steering arm (`src/pod_extract2.py`, `src/pod_steer.py`, `src/analysis_probe*.py`) requires raw activation dumps that are **not** included here (multi-GB `.npy`); the derived probe CSVs in `runs/pod_out/` are retained.
