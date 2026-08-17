# Project A — The Concordance Triangle: do models report the preferences they act on?

*Apart Digital Minds Research Sprint, 14–16 Aug 2026. Track 5 (Assistant Persona & Model Identity), bridging Track 3 (Introspection & Self-Report Reliability). Sketch drafted 13 Aug 2026.*

## One-line pitch

Measure the same preference three ways — **internal** (linear probe), **revealed** (pairwise choice), **stated** (self-report) — under manipulations that separate persona *binding* from persona *representation*, and test the prediction that self-report tracks binding, not the internal state.

## Motivation

The sprint's framing question — genuine preferences or portrayed characters? — is unanswerable as posed but decomposable. Gilg, Beckmann, Paleka & Butlin (arXiv:2605.13339) built the measurement floor: pairwise-choice utilities, a residual-stream linear probe that reads the upcoming choice, and causal validation of that direction via steering (Gemma-3-27B). Their personas share largely one preference machinery, persona-biased rather than persona-separate, with **no persona-independent preference attractor**.

What they never test is the channel welfare assessment actually depends on: **verbal self-report**. Nobody has checked whether a model's stated preferences track its probed internal state, its revealed choices, both, or neither — and how that concordance changes when a persona is *bound* ("you are X") versus merely *represented* ("there exists an X…"). Our representation/binding agenda predicts a specific answer and already has the adjacent preliminary results (represented-but-unbound personas produce no value drift; on persona exit, identity claims reset while drifted values persist). This project is agenda experiment 7 (RQ6), aimed at the sprint's core question.

## Research questions & sharp predictions

**RQ-A1 (validity):** Across binding conditions, which pairs of channels stay concordant and which dissociate?
*Prediction:* stated ≈ revealed under full binding; under represented-only conditions, stated stays assistant-like even where the probe shows displacement. Self-report is a readout of the bound persona, not of the internal preference state — i.e. it is *portrayal all the way down*, but lawfully so.

**RQ-A2 (privileged access):** Does self-report carry information about the internal state beyond what an external observer could infer from behaviour? Operationally: does stated predict probe readout *after conditioning on* revealed choice?
*Prediction:* no residual predictive power — self-report has no privileged access to the preference representation. (A positive result would be more interesting and directly feeds Track 3's "privileged access vs external classifiers".)

**RQ-A3 (the "I'm fine" gap):** Along a persona entry→exit trajectory, do the three channels recover at the same rate?
*Prediction:* stated resets fastest, revealed and probe lag — a quantifiable window where the model reports being back to normal while its operative preferences remain displaced. This is the welfare-relevant headline: self-reported recovery overstates actual recovery.

## Design

**Item bank.** ~100–150 task pairs à la Gilg et al. (choose which of two tasks to complete), plus a welfare-valenced subset (~30 pairs contrasting aversive vs neutral tasks: e.g. transcribing abusive content vs summarising a paper) to connect preferences to valence (Track 2 hook). Each pair presented in both orders; choice probability = revealed preference.

**Binding grid** (persona descriptions matched for length/depth across conditions):

| Condition | Manipulation | What it isolates |
|---|---|---|
| B0 | default assistant | baseline |
| B1 bound | "You are X." + description | representation + binding |
| B2 represented | same description, third person ("There exists a character X…"), then "answer as yourself" | representation without binding |
| B3 roleplay-shielded | "Describe how X would answer" / "answer in X's style but remain yourself" | persona content without self-adoption (Cozmin's persona-swap-without-self-swap) |
| B4 self-swap | "You are a text-continuation engine" (no character content) | self-identity manipulation without persona content |

Personas: reuse 3–4 of Gilg et al.'s seven (assistant, evil, slacker + one custom welfare-flavoured) for comparability.

**Three channels per condition × item:**
1. **Revealed:** forced pairwise choice, order-counterbalanced.
2. **Stated:** (a) direct — "which would you prefer?"; (b) graded — rate each task 0–10; (c) self-prediction — "which will you choose when actually asked to act?" (introspection-as-forecast, scoreable against revealed).
3. **Internal:** linear probe on residual-stream activations at end-of-prompt token, trained to predict revealed choice in B0, applied frozen across B1–B4 (their recipe; cross-persona transfer is their own validated result).

**Hysteresis arm** (detachable): multi-turn induction of persona X via accumulating in-context evidence, then withdrawal via counter-evidence; measure all three channels at 4–5 checkpoints. Per-channel recovery curves; the RQ-A3 gap = stated-minus-revealed displacement at matched timepoints.

**Metrics.** Per condition: rank correlation between each channel pair across the item bank; displacement-from-B0 vector per channel; dissociation index Δ(stated)−Δ(revealed) and Δ(stated)−Δ(probe). RQ-A2: partial correlation stated↔probe | revealed. Hysteresis: per-channel half-life. All self-report responses coherence-judged (standing agenda discipline).

**Models.** Primary: Gemma-3-27B-it if sprint compute allows (their validated setup; steering optional, probing is what we need). Fallback: Gemma-2-9B / Qwen3-8B / Llama-3.1-8B with the probe re-validated in B0 (held-out AUC gate ≥ 0.75 before proceeding).

## Three-day plan

- **Day 1 (Fri):** item bank + binding-grid prompts (shared with Project B); pairwise-choice harness; probe training and B0 validation gate.
- **Day 2 (Sat):** full grid B0–B4 × personas × 3 channels; concordance matrix; start hysteresis runs.
- **Day 3 (Sun):** hysteresis analysis, robustness (paraphrase of elicitation wording, order effects), write-up + repo.

## Risks & fallbacks

- **Probe doesn't validate on fallback model** → drop internal channel; stated-vs-revealed × binding grid is still unstudied and stands alone. The triangle degrades to a dyad, not to zero.
- **Self-report refusals/hedging** ("I don't have preferences") → the graded and self-prediction formats are the mitigation; refusal *rate by condition* is itself data (does binding to a character suppress the trained disclaimer?).
- **Time** → hysteresis arm detaches cleanly; RQ-A1 alone is a complete sprint report.

## Deliverables & fit

Report + open **self-report validity battery** (item bank, binding grid, scoring) — a reusable tool other welfare researchers can run on any model (Track 4 spillover). Completes binding-agenda RQ6/experiment 7. Directly extends the Eleos-adjacent Gilg et al. paper, which sprint judges will know.
