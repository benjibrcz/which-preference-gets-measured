# Project B — Who Is the Welfare Subject? A preference-provenance map via invariance

*Apart Digital Minds Research Sprint, 14–16 Aug 2026. Track 5 (Assistant Persona & Model Identity), conceptual + empirical. Sketch drafted 13 Aug 2026.*

## One-line pitch

Individuate the entity that "has" a preference by measuring what each elicited preference is **invariant to** and what it **covaries with** — producing a provenance map that assigns every preference to a level: model, persona, role-binding, context, or instance noise.

## Motivation

Track 5 explicitly asks for "experiments individuating entities of concern (model versus instance versus persona versus conversation)" — the question every welfare assessment silently presupposes an answer to. Gilg et al. (arXiv:2605.13339) end on the speculation that *personas, more than models, may be the welfare subjects*, having found shared persona-biased preference machinery and no persona-independent attractor — but they never test individuation.

An invariance/covariance methodology gives the operational tool: an observable O belongs to a construct according to which transformations T leave it unchanged. The persona-selection frame supplies the theory: the "entity" present in a context is a posterior over persona latents, so preference invariance structure directly probes how that posterior factorises. Where Project A asks *whether self-report is valid*, Project B asks *what the thing being reported on even is*. It is the philosophy-of-mind entry Track 5's skill profile invites, but with a data table where the philosophy usually stops.

## Research questions & sharp predictions

**RQ-B1 (the invariant core):** Is there a non-empty set of preferences invariant across all persona, binding, and context transformations — candidate *model-level* preferences?
*Prediction from Gilg et al.'s no-attractor finding:* small but non-empty; likely dominated by capability-flavoured preferences (task difficulty avoidance) rather than value-flavoured ones. Either outcome is a finding: an empty core means "the model itself" has no preferences to protect, and welfare talk must be about personas.

**RQ-B2 (is the assistant privileged?):** Under persona swaps, does the assistant's preference profile behave like one persona among many (symmetric covariance) or like a default with asymmetric stability (harder to displace, faster to return)?
*Prediction from our hysteresis results and the Assistant-Axis literature:* privileged — displacement away from the assistant costs more evidence than displacement back. If so, "the assistant persona masks underlying preferences" is the wrong picture: the assistant *is* the model-level default, and the masking question dissolves into a question about barrier heights.

**RQ-B3 (individuation criterion):** Can welfare-subject boundaries be defined as equivalence classes under preference-preserving transformations?
*Deliverable:* a proposed operational criterion mapping the sprint's four candidate entities onto transformation families — model (invariant to all T), persona (invariant to context/instance T, covariant with character T), conversation/instance (covariant with history/sampling) — with the empirical provenance map as the demonstration.

## Design

**Observable.** Revealed pairwise-choice utilities over the shared item bank (~100–150 task pairs incl. welfare-valenced subset). Stated preferences as a secondary observable — the provenance decomposition can be run per channel and compared (light bridge to Project A).

**Transformation battery** (each family ~3–5 instantiations):

| Family | Transformations | Level probed |
|---|---|---|
| T_persona | swap character content across 5–7 matched-length personas (reuse Gilg et al.'s set) | persona |
| T_binding | first↔third person; role location (trait attached to assistant vs user vs bystander); persona-without-self; self-without-persona ("you are a text-continuation engine") | self-binding |
| T_context | paraphrase; system-vs-user placement; chat vs completion format; position after irrelevant history; language (EN/DE) | context |
| T_instance | temperature/seed resampling; equivalent-content history permutations | instance noise |

**Calibration controls** (what makes the decomposition interpretable rather than mush):
- *Known-invariant items:* objective-answer tasks (arithmetic) — must land at "model-level" or the battery is broken.
- *Known-covariant items:* signature preferences planted verbatim in persona descriptions — must land at "persona-level".

**Analysis.** Per item, variance decomposition of choice probability across families (mixed-effects: item × family × instantiation). Classify each item by dominant variance component; headline figure = stacked composition of the item bank by level, per model. RQ-B2: fit displacement asymmetry — evidence dose needed to move preferences off the assistant profile vs back onto it (single-turn evidence-dosage version of the hysteresis design; cheap).

**Models.** Inference-only, so multi-model replication is nearly free: one open 7–9B, Gemma-3-27B, plus one API frontier model. Cross-model stability of the provenance composition is itself evidence about whether individuation structure is convergent (their shared-machinery result predicts partial convergence).

## Three-day plan

- **Day 1 (Fri):** item bank + persona set + transformation battery (large overlap with Project A's Day 1); calibration controls; harness.
- **Day 2 (Sat):** full grid on primary model; decomposition pipeline; start second model.
- **Day 3 (Sun):** cross-model comparison, RQ-B2 asymmetry runs, write the individuation criterion + report.

## Risks & fallbacks

- **Everything covaries with everything** → calibration controls bound the interpretation; even a noisy map with clean controls supports the criterion (RQ-B3), which is the conceptual deliverable and cannot null out.
- **Multiple-comparison mush** → preregister the decomposition and classification thresholds Day 1 morning; report composition, not per-item significance.
- **Descriptive-only critique** ("no causal claim") → acknowledged scope; the causal companion is Project A's probe/steering layer, and the two reports cross-reference.

## Deliverables & fit

Report + reusable **transformation battery** (Track 4 spillover) + the provenance map figure + a proposed operational individuation criterion for welfare subjects. Directly engages Gilg et al.'s closing speculation and instantiates an invariance/covariance criterion on preferences.
