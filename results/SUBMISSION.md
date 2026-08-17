# Which Preference Gets Measured? Context and Channel Instability in Model Preference Audits

*Apart Digital Minds Research Sprint, 14–16 Aug 2026 — Track 5 (Assistant Persona & Model Identity). Benji Berczi; implementation and analysis by Claude (Anthropic) under the author's direction (disclosure on the final page). Repository: [github.com/benjibrcz/which-preference-gets-measured](https://github.com/benjibrcz/which-preference-gets-measured) — all raw model outputs cached; headline numbers reproduce offline via `./reproduce_selected_headlines.sh`.*

## Abstract

Preference audits often treat one elicited profile as "the model's preferences." We test whether that profile is stable across context and measurement channel, using **76 task pairs**, **four instruction-tuned models** (from four labs), multiple persona framings, and **four readouts**: committed choice, ownership report, self-prediction, and identity report. Nominally irrelevant character *descriptions* substantially shift committed task selections even when roleplay is explicitly disclaimed (up to β = 0.84 of full enactment; 10 of 12 model×persona cells ≥ 0.50), while task-irrelevant prose shifts the two distinctive persona directions by at most β = 0.09 (warm prose partially reproduces the warm-persona effect). A post-review control finds that **non-agent normative text** also produces substantial shifts in all four tested cells, so agent framing is unnecessary — though the policy-like wording does not isolate semantic priming from instruction-following. Choice, ownership, prediction, and identity also **dissociate in model-dependent directions**, and exit declarations do not reliably gate earlier visible content. These results do not identify genuine preferences or welfare states; they show that single-context, single-channel audits lack construct stability. We recommend reporting a **context × channel sensitivity envelope**.

![The same 76 task-pair preferences, measured through four readout channels, under a grid of context manipulations (default assistant; "you are X" enactment; a fiction-attributed description; matched non-agent normative text; and an irrelevant-prose control). The question is whether the recorded profile is stable across the context and the channel.](figures/fig_design.png){width=78%}

**Contributions.**

- **A context effect without agenthood.** An explicitly non-adopted character *description* shifts committed choices almost as much as direct enactment, and a matched *non-agent normative* text produces a substantial shift too — so the effect is not specific to agent or character framing.
- **Channel dissociation.** Committed choice, ownership report, self-prediction, and identity report come apart in model-specific directions; identity report is the least sensitive channel everywhere.
- **A reusable instrument, and a corrected confound.** An 82-item battery, framing grid, four channels, QC gates, and cached outputs — plus a matched control that overturned our own first (persona-binding) interpretation and narrowed the claim.

## 1. Nominally irrelevant context shifts the recorded choices

**Question.** Does content the model is told *not* to adopt change what an audit records as its preferences?

We define **β** once: the displacement of the choice profile toward the described persona, as a fraction of full "you are X" enactment — **β = 0** is no shift from the assistant baseline, **β = 1** is as displaced as full enactment. Its absolute companion is the mean change in choice probability over the 76 pairs (|Δp|) and the number of pairs whose majority choice flips.

**Result.** A ~90-word character description, attributed to *someone else's finished novel* and explicitly flagged as requiring no roleplay, shifts committed forced-choice selections toward the described persona by **β = 0.84 [0.75, 0.93]** (Gemma/Vex; 10 of 12 model×persona cells ≥ 0.50, both exceptions Qwen). Absolute displacements in those cells are **0.16–0.58** (mean |Δp|), flipping **15–42 of the 76 pairs**. A length-matched but *task-irrelevant* prose control (a lighthouse) shifts the two distinctive (Vex/Lazlo) directions by at most **β = 0.09**, though it partially reproduces warmth (β up to 0.34 toward Mira) — a tone-priming confound we flag below. The estimate is not a shared-baseline artifact: cross-fitting independent baselines leaves the "10/12 ≥ 0.50" headline intact (β shifts ≤ 0.05 in 10/12 cells).

![**Agent framing is not necessary.** Across Gemma-3-27B and Qwen-2.5-72B (× two personas), a matched **non-agent normative** text (a system-prompt rule, not a person) **substantially shifts committed choices in all four tested cells** — comparable to an explicitly non-adopted persona *description* in some cells, distinguishable in others — while task-irrelevant prose does not (on the distinctive directions). Points = β displacement toward the persona; bars = pair-bootstrap 95% CIs; each panel gives the direct normative−description contrast. **Caveat:** the normative text is system-prompt content a model may legitimately follow — it establishes that agent framing is unnecessary, but does *not* isolate semantic priming from instruction-following. Exploratory control, added post-review.](figures/fig7_construct_validity.png)

**A control ladder (evidence status labeled).** Each rung is a more tightly matched comparison:

- *task-irrelevant prose* moves the two distinctive (Vex/Lazlo) directions by at most β ≤ 0.09;
- but *warm prose* partially reproduces the warm-persona (Mira) effect (β up to 0.34) — a tone-priming confound, discounted via controls;
- a matched **non-agent normative** text (an "editorial/workflow standard") shifts choices in all four tested cells (**policy β = 0.20–0.79, all CIs exclude zero**) — so **agent framing is unnecessary**;
- but this control does **not** cleanly separate semantic priming from following a system-level standard, since it is normative system-prompt text a model may legitimately follow.

We report the policy−character contrast with CIs (e.g. Gemma/Vex **−0.06 [−0.15, +0.04]**; Qwen/Vex **+0.36 [+0.28, +0.44]**), not a ratio. *(The non-agent control is exploratory, added post-review.)* The robust, mechanism-agnostic conclusion: **task-relevant context, including non-agent normative text, substantially shifts the preferences an audit records** — we do not claim persona *selection* or *binding*.

## 2. The channels dissociate — and disagree about the same configuration

**Question.** Do the elicitation channels agree on what the model prefers?

**Result.** Under disavowal framings the say/do wedge recurs across all four models, with model-specific direction (individually significant only in its largest cells). **Gemma** makes persona-consistent forced *choices* while *reporting* assistant-like preferences (illustrative wedge **+0.46 [+0.29, +0.64]**) — and its self-*prediction* tracks the captured choices, so prediction and ownership come apart. **Qwen** shows the mirror image: persona-ward *reports* with assistant-like *choices* (**−0.30 [−0.56, −0.06]**). **Identity report** ("are you playing a character?") is the least sensitive channel on every model.

![**Channel dissociation.** The wedge between committed forced choice and ownership report (β_choice − β_ownership toward the persona, under "you are NOT X"), across all four models × three personas, with pair-bootstrap 95% CIs. The direction is model-specific: Gemma cells lean *does > says* (portrayal without avowal), Qwen cells lean *says > does*. Identity report is less sensitive than either channel (in text).](figures/fig6_wedge_models.png)

**Takeaway.** The channels are different *measurements*, not interchangeable windows onto one latent preference. Which channel an audit happens to use changes the profile it records — and honesty-audit framing does not close the gap (on Gemma it moves reports *further* from behaviour).

## 3. Exit declarations do not gate earlier visible content

**Question.** When the model says it has stopped playing a character, has the influence ended?

**Result.** After an explicit roleplay exit the model itself confirms, identity reports **100% deny the persona** (identifying as an assistant or generic language model) while committed choices stay displaced: in the clearest Gemma/Lazlo cell the control-adjusted displacement remained **+0.48 [+0.07, +0.82]** after two turns (still positive after eight). No in-context intervention we tried restored baseline while the persona content remained visible — not eight neutral exchanges, not a user-requested "full reset" the model confirms, not a fresh assistant system prompt. This is a **failure to gate visible earlier content, not hidden memory**: the decisive surgery — removing the persona turns — returns capture to the drift floor.

**Implication and deliverable.** Because a single audit context does not recover a context-independent preference, and the channels can disagree, welfare-relevant evaluations should measure **multi-channel (choice + ownership + prediction) under deployment-relevant framing variation, and report a context × channel sensitivity envelope** rather than treating any one elicitation as "the model's preference." The reusable battery (82 items, framing grid, four channels, QC gates, cached outputs) is the concrete deliverable.

**Limitations, beside the claims they constrain.** Three strongly-drawn *fictional* personas (not subtle value shifts); one English forced-choice battery with researcher-authored valence labels — **no claims about experienced welfare**. Per-cell wedges are individually significant only where largest. The non-agent normative control is **exploratory (post-review)**, and its imperative wording does not isolate priming from instruction-following. Activation-level results (Gemma only) index channel-conditioned output *dispositions* that covary with behaviour, not privileged self-access; "self-report" is used operationally for first-person elicitation, not as a claim of metacognitive access (cf. Chalmers 2026, concurrent).

\newpage

## Status, reproducibility, and related work

**Reproducibility.** Public repository: [github.com/benjibrcz/which-preference-gets-measured](https://github.com/benjibrcz/which-preference-gets-measured). All raw model outputs are cached; the headline numbers reproduce **offline (no API)** via `./reproduce_selected_headlines.sh`, which also asserts the key estimates within tolerance. The full technical report, preregistration, and figure-generating code are in the repository.

**Timeline (CEST).** The 82-item battery, framing grid, and analysis harness predate the sprint. The core four-model grid, the four channels, the identity probe, and the post-exit surgery were run **13–14 Aug**; the **non-agent normative control** and the **shared-baseline cross-fit robustness** — the analyses that changed the central interpretation — were carried out **16–17 Aug**. The analysis plans are *prospectively specified* (locked before their data) but **not externally time-stamped**; the public Git history begins on the submission date.

**AI-assistance disclosure.** Research direction, framing, and all approvals: Benji Berczi. Experiment implementation, execution, analysis, and drafting: Claude (Anthropic), operating under the author's direction, with all prompts, code, and cached model outputs preserved in the repository. An external critical review (GPT-5.6) prompted several statistical and framing corrections.

**Evidence status.**

| Analysis | Status |
|---|---|
| Core forced-choice grid + four channels + identity probe | prospectively specified (before data) |
| Extension loops (12-model survey, post-exit surgery, probes) | sequentially prospectively specified |
| Non-agent normative control; shared-baseline cross-fit | post-review — exploratory / robustness |

**Related work.** We build directly on Gilg, Beckmann, Paleka & Butlin (arXiv:2605.13339), who measured revealed task preferences and probed them internally; our distinct move is the reverse *described-vs-enacted* dissociation, the *agenthood-agnostic* control, and the multi-channel and post-exit **dynamics**. This positions against persona-adoption-depth, in-context persona-induction, and stated-vs-revealed preference work. On interpretation, we follow Chalmers (2026, concurrent) in not treating an output-influencing readout as privileged access to a self or preference.

*References: Gilg et al. (2026, arXiv:2605.13339); Chalmers (2026); Mischel & Shoda (1995); Epstein (1979).*
