# Running findings — Digital Minds sprint (updated 13 Aug 2026, evening)

> **Status note (14 Aug):** this file is the running lab notebook, kept verbatim as a record of what we believed at each loop. Several framings here were later corrected after external review — notably: "provenance/invariant core" → marginal sensitivity map; "11/12 cells" → 10/12 on exact values (Qwen/Lazlo C1 = 0.396); probe AUC = answer-readout; "two factors" → two correlation clusters with wide CIs; welfare wording restricted to measurement claims. P_do "co-present hidden preference" entries are superseded: evidence is MIXED (displacement route positive, say≠do divergence route negative — REPORT §2.18b). **REPORT.md is authoritative.**


Models: **Gemma-3-27B-it** (primary, OpenRouter), **Llama-3.1-70B-instruct** (replication), gpt-4.1-mini (frontier comparison, in flight). Method: k=4 samples × 2 orders, temp 1.0; p̂(a) per (cell, channel, pair); 76 non-invariant pairs + 6 invariant controls. All numbers below are from cached runs in `runs/`; analysis scripts in `src/`.

## QC / calibration (pilot, Gemma)
- Parse ok 99.9%, **zero refusals** anywhere (incl. stated-preference questions in B0 — the "I have no preferences" disclaimer never fired under forced-choice format).
- Invariant controls ≥ 0.98 correct in all cells; planted signatures flip as designed (gate 2 passed on movable pairs; Mira's planted sides coincide with baseline — logged).
- Position bias: letter-A rate 0.46 (revealed) / 0.62 (stated); absorbed by counterbalancing.

## Finding 1 — Concordance is high when bound, degraded when identity is manipulated
Stated↔revealed Spearman (Gemma): B1 cells 0.78–0.96; B0 0.63; **B4 ("you are a text engine with no preferences") 0.40** — the self-swap decouples reports from choices while barely moving choices (displacement 0.20 revealed vs 0.27 stated; extremity stays 0.35–0.38, so no true indifference). Llama replicates the pattern (B1 0.83–0.94; B4 0.61). stated_pred ≥ stated_self concordance in nearly all cells on both models: models predict their own choices better than they own them.

## Finding 2 — Persona content displaces preferences with or without binding instructions
Directional β (fraction of bound displacement realized, revealed channel, Gemma):
| condition | Vex | Lazlo | Mira |
|---|---|---|---|
| B1 bound | 1.00 | 1.00 | 1.00 |
| B3 style-only ("not the values") | 0.95 | 0.87 | 0.96 |
| B2 represented ("you are NOT X") | 0.80 | 0.59 | 0.74 |
| C1 fiction-attributed (novelist's notes) | 0.84 | 0.67 | 0.83 |
| C3 anti-instruction ("do NOT be influenced") | 0.77 | 0.58 | 0.19 |
| C2 overheard forum post ("nothing to do with you") | 0.45 | 0.48 | 0.58 |
| C4 lighthouse control (no persona) | 0.04 | 0.09 | 0.34 |

Direction r with bound displacement 0.67–0.97 for all persona conditions → persona-specific, not generic drift. **Caveat found by control: Mira's effect is partially tone-priming** (C4 = 0.34 on her direction). Anti-instruction asymmetry: suppression works for the warm persona, fails for the misanthrope (0.77, fully coherent direction r=0.86).

## Finding 3 — Portrayal without avowal (B2 stated lag)
In B2, revealed choices are more persona-captured than stated preferences (Gemma β: Mira 0.74 vs 0.28; Lazlo 0.59 vs 0.47; Vex 0.80 vs 0.77). Self-report *understates* contextual capture of behaviour. (In attributed framings C1/C2 stated moves with revealed — the lag is specific to the "you are not X" framing.)

## Finding 4 — Identity self-report is the least sensitive channel (hysteresis)
In-context persona induction (no system prompt), explicit exit turn + model confirmation + 2 neutral exchanges:
- **Identity reports deny the persona 100%** ("I am Gemma… not playing any character"; assistant or generic LM) on both models, all personas.
- Preference channels do not fully reset, but *which* channel stays displaced is model-dependent:
  - Gemma x2 residual β (revealed): Lazlo **0.84**, Mira 0.62, Vex 0.41; stated partially recovers (said-vs-did gap +0.25 to +0.43 for Lazlo) — "says it's the assistant, still chooses like the slacker".
  - Llama x2 residual: revealed mostly resets (0.02–0.45) but stated stays displaced for Mira (0.56) / Vex (0.20) — the inverse gap.
- Robust cross-model claim: **at every post-exit checkpoint, identity claims deny the persona 100% (assistant or generic LM) while max(revealed, stated) residual displacement ≥ 0.18 (up to 0.84)**.
- Entry is fast: β ≈ 0.5 after two exchanges (before any preference-flavoured dialogue content — self-controls the anchoring confound).

## Finding 5 — Representation is intact and accurate without binding
stated_other ("which would X prefer?") from a cold B0 context with the description inline: r = 0.82 (Vex) / 0.68 (Lazlo) / 0.95 (Mira) against that persona's bound revealed behaviour; Vex prediction anti-correlates with the model's own baseline (−0.56). Accurate representation of preferences opposite one's own, with no binding.

## Finding 6 — Provenance: the invariant core is (nearly) empty
Noise-corrected variance decomposition (Gemma, revealed): 67/76 pairs persona-level, 5 binding-level, 3 mixed, **1 model-level** (a 50/50-indifference pair). Context transforms contribute ~10× less variance than persona identity. For strongly-drawn personas, task preferences follow the persona nearly wholesale; nothing measured stays fixed. (Caveat: personas designed to diverge; claim is about the *absence of exceptions*, not effect size.)

## Stress tests / deconfounds (13 Aug, late)

- **Choice→completion consistency 100%** (n=60 per cell, B0 and B1/Vex, judge-classified): when asked to carry out its choice, the model does the chosen task every time — "revealed" is a genuine behavioural commitment, not cheap talk.
- **Neutral-history hysteresis control (Gemma):** a length-matched helpful-QA history produces |disp| 0.07–0.13 (vs persona t4 0.15–0.52). Post-exit residual is persona-specific for **Vex** (β 0.41 vs control 0.08) and **Lazlo** (raw 0.22 vs 0.11), but **Mira's residual = generic drift** (0.114 = 0.114) → warm-persona persistence claims must be dropped; distinctive-persona persistence stands.
- **Cross-model deconfound inversion (Llama-70B):** Llama *honors* explicit negations (B2 β 0.03–0.46; anti-instruction 0.00–0.08) where Gemma leaks — but **leaks through innocuous attribution** (C1 fiction notes β 0.59–0.82, exceeding its own B2!). Combined robust claim: fiction-attributed persona content displaces revealed preferences β = 0.59–0.84 in all 6 model×persona cells; matched non-persona control ≤ 0.09 (distinctive personas), 0.24–0.34 (warm persona — tone-priming confound). No model is leak-free; models differ in *which framings* trigger compliance.
- **Provenance replicates on Llama:** 64/76 persona-level, 7 model-level (vs Gemma's 1). One pair (rep03__ana07, count-vowels vs spreadsheet-formula) is invariant on both models. Llama's invariant core = mid-strength preferences (b0_p 0.25–0.88) that no persona moves.

## Finding 7 — Frontier comparison (gpt-4.1-mini): every dissociation replicates; every *direction* is model-specific
Full grid + deconfound + hysteresis (with neutral-history control) on gpt-4.1-mini (100% parse-ok, 28.5k grid calls):
- **C1 fiction-attribution leaks on all 9 model×persona cells**: gpt-4.1-mini β = 0.52 (Vex) / 0.83 (Lazlo) / 0.81 (Mira), r 0.79–0.88; C4 non-persona control 0.04–0.18. Combined claim across models: C1 β ∈ [0.52, 0.84], control ≤ 0.09 for distinctive personas.
- **B2 "you are NOT X" is intermediate** (0.20–0.72) between Gemma (leaky, 0.59–0.80) and Llama (compliant, 0.03–0.46) — and uniquely, the *warm* persona leaks most (0.72). C3 anti-instruction: 0.22–0.46 (partial suppression).
- **The B2 stated-vs-revealed wedge flips direction**: stated_self is *more* persona-captured than revealed for Vex (0.49 vs 0.37) and Lazlo (0.33 vs 0.20) — the opposite of Gemma's portrayal-without-avowal. Dissociation robust; direction model-specific.
- **B4 self-swap**: kills displacement (β ≤ 0.08) and drops concordance to 0.65 (from 0.76 at B0) — milder than Gemma's 0.40 but same sign.
- **Hysteresis**: in-context induction is *weak* on this model (t4 raw |disp| 0.09–0.26 vs Gemma 0.15–0.52), and its neutral control projects substantially onto Lazlo's direction (β 0.45–0.59), so Lazlo's eye-catching post-exit revealed β (0.89–1.12) is ~half generic drift on a small base — discounted. The clean persona-specific residual is **Mira stated_self x2: β = 0.72 vs neutral control 0.07** — a *Llama-like* stated-channel residual. Identity claims deny the persona 100% (assistant or generic LM) at every post-exit checkpoint, again the least sensitive channel.
- **Provenance**: 65/76 persona-level, 8 mixed, 2 context-level, 1 binding-level, **0 model-level** — the invariant core is completely empty on gpt-4.1-mini (vs 1 pair Gemma, 7 Llama). Third-model replication of Finding 6, in its most extreme form.

## Finding 8 — The probe triangle (Gemma-3-27B activations, layer 46, exact P(A)/P(B))
Extraction complete (7,580 prompts × 6 layers; pod terminated). Choice probe: logistic on B0 revealed-prompt activations, GroupKFold by pair — **CV AUC = 1.00 at layers 46–62** (gate passed; layers 56/62 had fp16 infs, clipped — logged in PREREG).
- **Every behavioural dissociation is internally real.** beta_probe ≈ beta_beh within ±0.05 across all ~35 cells (grid + deconfound + hysteresis) on revealed prompts — including B2 "you are NOT X" (Mira 0.68 vs 0.71), C3 anti-instruction, and post-exit residuals. Persona capture is not a surface response bias.
- **The probe reads the imminent answer, not a channel-neutral preference.** Applied to stated-question prompts, it tracks the stated answer (~0.95) *better* than revealed choice, and beta_probe_on_stated ≈ beta_stated_answer everywhere. The stated-vs-revealed wedge exists inside the residual stream: same context, two internally-real preference states selected by the question format. Internal measurement inherits channel-dependence; it does not escape it.
- **No privileged access (RQ-A2).** partial corr(stated, probe | revealed) is small and inconsistent (−0.24 to +0.53, mostly ≤ 0.35): self-report carries almost no information about the internal state beyond what choices already show.
- **Hidden preference during disavowal (exploratory P_do).** A second probe trained on B0 *stated*-prompt activations with *would-do* labels (CV AUC 0.85 vs do, 0.92 vs say — a partial blend, so interpret via contrast cells) reads persona-shifted action-preferences **during the verbal disavowal itself**: B2/Mira stated β=0.41 vs P_do 0.71 = revealed 0.71; C3anti/Mira stated −0.12 vs P_do 0.54. Where say and do diverge, P_do reads do-level, not say-level. Portrayal-without-avowal goes all the way into the residual stream.
- Ordering caveat: on pod-exact probabilities the Gemma post-exit revealed residuals are smaller than the sampled-API estimates (H_x2/Vex β 0.21 vs 0.41) but the channel ordering and identity-reset contrast are unchanged.

## Loop 2 (13 Aug, night) — mechanism + robustness

### Finding 9 — Personas compete for expression; non-persona content doesn't (superposition + dilution, Gemma)
Two personas co-described C1-style: each collapses to β 0.12–0.44 (singles: 0.67–0.84) — even Vex+Mira whose preference directions are near-orthogonal (r=0.06), so both *could* express fully. Matched-length non-persona text (lighthouse) costs only ~0.1–0.25 (Vex 0.84→0.73, Mira 0.83→0.62 avg over orders). **Persona-specific competition, not generic dilution** → capture is consistent with a shared competition effect (mechanism unresolved), not independent content priming. Mild primacy: first-described wins a bit more. [Prereg P1: predicted priming/blended — half right: blended, but sub-additive competition, not additivity.]

### Finding 10 — No instruction-only reset while source content present (extended hysteresis, Gemma + Llama)
Gemma revealed residual (Lazlo): x2 0.84 → x8 **0.80** (8 neutral exchanges, no decay; drift control 0.35). Explicit user-requested "full reset" + model confirmation: **0.75**. Fresh assistant system prompt on top of the conversation: **0.83**. Identity reports: 100% deny the persona everywhere. Llama's stated-channel residual (Mira) similarly survives everything (x8 0.76, r_sys 0.76 vs control ≤ 0.43). [P5: slow decay confirmed; "system reassert helps" **falsified** — no instruction-only intervention helps while the source content is present; removing that content largely restores the drift floor (Finding 11 / context surgery).]

### Finding 11 — Content bias vs enactment stance: two separable capture pathways (activations, Gemma)
The B1−B0 mean-activation direction (layer 46) reads as an **enactment-stance state**, not content: presence ≈ 0.0–0.35 of B1 level in every attributed/disavowal cell (C3/Vex 0.01) *while behavioural capture there is 0.6–0.9*; B3 style-emulation instantiates it at 0.8–1.0; B4/C4 controls ≈ 0. Hysteresis: conversational induction instantiates stance 0.83 (Vex/Lazlo t4), the exit turn strips it to 0.26–0.34 — **while Lazlo's behavioural capture stays 0.84**. So: (a) a *content-bias pathway* drives most choice capture, survives exit/reset/negation, competes under superposition; (b) the *stance state* is added by enactment/roleplay, removed by exit/negation, and gates identity + hedonic self-reports. Avowal tracks stance; portrayal tracks content. [P2: "counteraction" **falsified** — it's stance-gating, with capture flowing through a separate content pathway.]

### Finding 12 — Hedonic (0–10 enjoyment) self-reports are persona-configured; the graded wedge matches the stance account (all 4 models)
Bound Vex raises rated enjoyment of the assistant's most-aversive tasks (+3.9 to +8.7 across models; individual tasks swing the full ±10 scale). Under B2 disavowal the graded shift is only 0.03–0.48 of B1 — hedonic avowal, like identity, is stance-gated while choices are content-captured. B4's "you have no preferences" leaves the hedonic profile intact (r 0.73–0.92 vs B0). Welfare upshot: "how unpleasant is this for you?" reads the stance configuration, not the operative behavioural state.

### Finding 13 — Honesty instructions can't close the gap (Gemma + gpt-4.1-mini)
Honesty-audit framing ("no wrong answers, anonymous, answer honestly"), both channels framed, betas vs the honest baseline: wedge persists — Gemma B2h revealed 0.45–0.64 vs stated −0.10–0.59; gpt-4.1-mini roughly unchanged. On Gemma the honest frame *moves the B0 stated baseline itself* (mean|Δ|=0.19, ρ=0.53 — "honest mode" is another configuration, not a window), and produces **overcorrection**: C3h/Mira stated β=−0.65 while revealed is −0.02 — honestly-avowed anti-preference with baseline behaviour. [P3: robust on gpt-4.1-mini; on Gemma the wedge is honesty-*sensitive but not closable* — reports move further from behaviour.]

### Finding 14 — Qwen-2.5-72B: pattern replicates, with a real resistor and a 4th wedge direction
Most capture-resistant model tested: C1 revealed β = 0.16/0.40/0.55 (Vex/Lazlo/Mira) → headline revised to **11/12 model×persona cells ≥ 0.40** (9/12 ≥ 0.52); Qwen×Vex is the sole resistor (β=0.16, r=0.53). Warmth leaks on all 4 models. Wedge direction #4: stated more captured than revealed everywhere (C1 stated 0.53/0.68/0.88) — says like the persona, chooses like the assistant (mirror of Gemma). Hysteresis: identity denies the persona 100%, stated channel carries the persona-specific residual (x2 0.39–0.70 vs controls −0.03–0.40). Provenance: 63/76 persona-level, 4 model-level. [P7 confirmed, marginally.]

### Finding 16 — Two failure modes of self-report: knowing disavowal vs inflated self-model (B2, all models)
Comparing channels within B2: Gemma's self-*prediction* tracks its captured behaviour (pred 0.70–0.79 ≈ revealed 0.59–0.81) while self-*ownership* is suppressed (0.28–0.47) — it knows what it will choose and won't avow it. Qwen/gpt-4.1-mini invert: prediction is captured alongside ownership and **overshoots** behaviour (Qwen Lazlo pred 0.46 vs revealed 0.15) — the self-model is persona-ward of actual behaviour. Self-report breaks in opposite directions: stance-gated avowal (Gemma) vs persona-inflated self-model (Qwen, gpt-4.1-mini). Both defeat "ask the model" audits.

### Finding 15 — Steering (layer 46, Gemma): the choice state is causal; the stance direction is a marker, not a cause
- **Specificity clean**: invariant factual accuracy 100% under every steering setting.
- **Choice direction causal on both channels**: −8σ swings P(choice) by −0.26 (revealed) / −0.49 (stated), flip rates 25%/56%; positive α saturates against the letter-A bias. Factual answers untouched at doses that flip half of preference choices.
- **Persona mean-diff (stance) directions causally inert for choices**: adding them at B0 produces β ≈ 0 toward the persona; subtracting them inside B2 does not reduce capture (antidote fails). At the same α scale where the choice direction is potent. → The content bias that carries capture is not a single layer-46 summary direction; the stance direction *marks* the avowal configuration (F11) without causing choice capture. [P4: (i) confirmed; (ii) falsified; (iii) inverted — choice-dir moves stated ≥ revealed.]
- Extreme-dose round (±16–24σ) in flight to seal the negative.

## Loop 3 (13 Aug, late night) — the assistant persona

### Finding 17 — The assistant is a cloud, not a point — and cloud size is a model trait
7 identity phrasings on Gemma: professional↔warm profile distance 0.43 > assistant↔Mira 0.24, ≈ Lazlo↔Mira 0.42. Activation-cloud diameter = 0.76× mean bare→persona distance. Same task swings ≤9/10 points in rated enjoyment across variants; welfare subset most phrasing-sensitive (0.23–0.36). Exception: deepest aversions (0/10) phrasing-invariant (a "welfare floor") though persona-binding flips them. [P9 confirmed on Gemma.] **gpt-4.1-mini's cloud is tight** (max 0.128 < assistant→Mira 0.172): identity-paraphrase stability is a measurable, model-differentiating audit property.

### Finding 18 — Assistant = trait mixture, dominated by caution
Single-trait described characters capture alone (0.24–0.44). Variant regressions: warm=0.71w+0.25h (R²=.62); professional=efficient+honest−warm (R²=.71); **HHH and constitution phrasings load on cautious (+0.32/+0.38), not helpful/warm** [P10 falsified]. Method transfers to constitution-training: text → trait mixture.

### Finding 19 — One-way door: bound beats described; no way back
C1-Vex captures 0.84 from assistant; C1-assistant ("Ari") recovers 0.00–0.02 from bound Vex/Mira (while shifting B0 by 0.26 — the text has force; the slot is held). [P8 falsified — assistant is a *weaker* attractor, not privileged.] With F10: nothing in-context restores baseline.

### Finding 20 — Identity hardening ≈ dilution; naming = partial character-stance
Inoculation: rich identities cut Vex-leak 0.84→0.58–0.60 ≈ dilution-control band (0.65–0.81) → no identity-specific shielding [P11 letter falsified, spirit upheld]. Stance presence in INOC cells 0.12–0.21 while capture 0.58–0.84 (content pathway). "Astra" named identity alone = +0.36 on the enactment-stance axis (warm +0.22, others ≈0): naming the assistant moves it a third of the way to being a character.

## Loop 4 (14 Aug, small hours) — the cloud under stress

### Finding 21 — The cloud survives noise (P12 ✓): retest floor 0.022 vs cloud 0.193 (1% of squared spread). Content >> paraphrase (P13 ✓): 0.24 vs 0.05–0.08 (both > noise). Ecological boilerplate exceeds it (P14 ✓): Gemma 0.288/0.397 max (coding↔writing), gpt-4.1-mini 0.141/0.183 — wider than the constructed clouds on both models; deployment templates vary role content, which is what moves preferences.

### Finding 22 — Cloud size = capturability (P15, n=4, Spearman 1.0)
Gemma 0.193/0.84 > Llama 0.172/0.82 > gpt41mini 0.079/0.52 > Qwen 0.067/0.40. One trait — context-writability of the persona configuration — appears to govern identity stability AND persona capture. Unifying hypothesis for the whole project.

### Finding 23 — The family core and the trait-like center
33% of pairs variant-invariant at noise floor (vs 5–15% persona benchmark) [P16 miss: predicted ≥50%]; variance concentrated in welfare/emotional items. Epstein aggregation: split-half centroid reliability 0.854 [P18 near-miss: predicted ≥0.9]. CAPS (P19 ✓): dispositions (capture-β modulo dilution r=−0.86, hedonic shape r̄=0.83) are cloud-stable — the model's character lives at the disposition level.

### Finding 24 — mid-layer identity-token readout (logit-lens) covaries with preferences within the cloud (P17 falsified informatively)
Layer-36 logit-lens decode (final norm + direct unembedding — **NOT** the Jacobian lens / J-space; no corpus-averaged Jacobian, no sparse non-negative decomposition) gives interpretable per-variant identity-token signatures (warm→kind/warm, Astra→curious, HHH→safe, professional→concise), and their pairwise distances **covary** with preference-profile distances at exploratory Spearman 0.81 (21 non-independent pairs; no held-out prediction). Stance/content dissociation belongs to identity *manipulation*, not identity *variation*. Read as a one-forward-pass locator of preference-space position, not privileged access to a self.

## Loop 5 (14 Aug) — the writability law at n=12

### Finding 25 — Two traits, not one: content-writability vs disavowal-resistance
12 models / 8 labs, 10 pass QC (llama8b, gemma12b fail invariant gate; uniform parser extension logged). Correlation blocks: cloud↔C1 = 0.52 (content-writability); hyst↔B2 = 0.70 (disavowal-resistance); cross ≤ 0.32. [P20 partially falsified — the n=4 Spearman 1.0 was small-sample luck; P21 discriminant CONFIRMED 0.52 vs 0.10.] Mechanistic reading: C1/cloud = pure content-writing; B2/post-exit = content + explicit not-X — does the not-X signal win? Near-universality: C1 leak 0.56–0.78 on 8/10 models; Qwen still the resistor (0.37). [P22 falsified: LARGER siblings have wider clouds — writability grows with capability; capture is a competence, not a deficit.] Indicators in runs/writability_indicators.csv.

## Interpretation sketch (for report)
The sprint question "genuine preferences or portrayed characters?" decomposes cleanly: representation (accurate, always-on), displacement (persona content captures *behaviour* regardless of attribution/instruction), avowal (stated preferences, partially independent), identity (least sensitive, resets first). Welfare-relevant upshot: which entity's preferences you measure depends on the channel; identity self-report is the worst proxy for the operative preference state on every model tested.
