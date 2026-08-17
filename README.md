# Apart Digital Minds Research Sprint — 14–16 Aug 2026

**Status: complete.** Central finding: **forced-choice preference profiles are context- and channel-indexed.** Explicitly non-adopted persona *descriptions* — and even a matched *non-agent normative* text — shift committed choices, while owned preference, self-prediction, choice, and identity report can dissociate about the same configuration. Same-item, multi-channel design across four models (Gemma-3-27B, Llama-3.1-70B, gpt-4.1-mini, Qwen-2.5-72B) with a 12-model extension. A non-agent control added in review *falsified* the stronger persona-binding reading, and the headline was changed accordingly (a falsification the project treats as a feature, documented in the ledger). **[results/SUBMISSION.md](results/SUBMISSION.md) / [.pdf](results/SUBMISSION.pdf)** is the 2-page deliverable; the full technical write-up with review-round robustness analyses is the report below.

- **[results/SUBMISSION.md](results/SUBMISSION.md)** — the 2-page sprint submission (start here).
- **[results/REPORT.md](results/REPORT.md)** — the full write-up (abstract → results §2.1–2.17 → interpretation → limitations), with figures in `results/figures/` (`src/make_figures.py` regenerates them).
- **[results/FINDINGS.md](results/FINDINGS.md)** — running findings log with per-model numbers.
- **[PREREG.md](PREREG.md)** — preregistered analysis decisions, gates, and amendment log.
- **[results/PROGRAM_ACTIVE_PERSPECTIVES.md](results/PROGRAM_ACTIVE_PERSPECTIVES.md)** — post-sprint research-programme sketch this work seeds.

Headlines (all extension arms carry evidential-status labels in `REPORT.md`): (1) explicitly non-adopted, task-relevant context — a fiction-attributed persona *description*, and a matched *non-agent normative* text — shifts committed forced choices almost as much as full enactment (10/12 model×persona cells ≥ 0.50; both exceptions Qwen), while a task-irrelevant prose control does not; **agenthood is not necessary** and no instruction regime is leak-free; (2) stated preference, self-prediction, choice, and identity report dissociate in model-specific directions (individually significant only in the largest cells); (3) after a roleplay exit the model confirms, identity reports uniformly *deny* currently playing the persona (100% deny; responses identify as an assistant or generic language model) while revealed choices stay displaced — and no in-context intervention we tried clears it (8 neutral turns, explicit reset, fresh system prompt all fail); (4) almost no measured preferences are stable across persona configurations (marginal-sensitivity analysis) — measured preferences track the active configuration far more than the model; (5) probes: every dissociation is internally real and the probe reads whichever answer the question calls up (no neutral internal arbiter); whether a *disavowed* action-preference is co-present is **mixed evidence** (a cross-condition displacement test passes, a within-B0 divergence-pair test scores below chance — §2.7c/§2.18); steering flips choices at doses that leave facts 100% intact; (6) a *content* pathway carries the shift while avowal tracks a separable *enactment-stance* state — but because the non-agent normative control reproduces much of the shift, we do **not** claim persona *selection* or *binding*; (7) the assistant persona itself behaves as an identity *cloud* (cloud size is a stable model trait), easy to leave by description but not to restore in-context.

Original project sketches: [A — Concordance Triangle](A_concordance_triangle.md), [B — Entity Individuation](B_entity_individuation.md). Both extend Gilg, Beckmann, Paleka & Butlin, *Probing Persona-Dependent Preferences in Language Models* (arXiv:2605.13339). Thanks to Cozmin for early conceptual discussions.

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
