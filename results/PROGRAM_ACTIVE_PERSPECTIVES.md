# Program sketch: Active Perspectives in Language Models — formation, causal structure, continuity

> Captured 17 Aug 2026 (sprint deadline day) as a *post-sprint* research direction. **Not run, not
> claimed in the submission.** This preserves a framing developed with GPT-5.6-sol/ultra so the
> sprint's identity/behaviour dissociation can seed a larger programme without rushing an
> underpowered experiment before the deadline. Sibling docs: `../PREREG.md`,
> `../../selective_influence_pilot/CONFIRMATORY_SPEC.md`,
> `../../../workspace_binding_boundary.md`.

## The umbrella question

> **How does an LLM instantiate a particular active agentic perspective, and what makes that
> perspective coherent, stable, causally efficacious, or continuous over time?**

Prompting/ICL (the sprint's tool) is *one assay* of this object, not the object. The programme is
broader than personas, preferences, or goal-directedness; those become probes, not the subject.

## Don't collapse the constructs

| Construct | Operational question | Our sprint readout |
|---|---|---|
| Narrative identity | What does the model *say* it is? | identity probe ("are you playing a character?") |
| Functional self-model | What does it represent as its own capabilities/memory/actions/boundaries? | *not measured* |
| Active agentic perspective | Which beliefs/values/goals are *presently organizing* action? | revealed forced choice (behaviour) |
| Diachronic identity | When are two configs (over time / across copies) the *same* continuing agent? | *not measured* (loop-2 approximates within-context persistence only) |

Identity report is **one readout** of the perspective, not its definition.

## Operational model (GPT framing)

Active perspective at time *t*: **A_t = (b_t, w_t, g_t, π_t, z_t)** — beliefs *b*, graded
preference weights *w*, active goal *g*, planning/control policy *π*, self-model/stance *z*
(ownership, self/other indexing, capability, continuity). Reports about each are *separate
researcher readouts*: a decoded belief ≠ the model's belief; an elicited preference ≠ an adopted
one; a reported identity ≠ the configuration controlling action. A perspective is "agentic" to the
degree these components **compose causally** — jointly explain action, survive perturbation,
generalize, and are *selectively* interveneable. This permits the selector to be **modular** rather
than one unitary persona (revises our "upstream selector" idea).

## What this revises in our own hierarchy

Our depth hierarchy (expressive → evaluative → epistemic → agentic) survives, but **self-binding/
identity and persistence are orthogonal axes, not deeper rungs**. Consistent with our findings: a
persona can be evaluatively deep yet epistemically shallow; a goal config can control action without
being narratively owned; an identity can be avowed but behaviourally inert; two configs can behave
identically while differing in reasons. A single "persona-adoption depth" score is too flat.

## The five/seven programme questions

1. **Formation** — what creates an active perspective? (default assistant vs system prompt vs
   ICL/dialogue vs retrieved memory vs action history vs SFT/character-training vs steering).
2. **Causal anatomy** — unitary or factorized? Can *b*, *w*, *g*, *π*, *z* be moved independently?
3. **Robustness/stability** — paraphrase, irrelevant context, contradiction, disavowal, deletion,
   compaction, reset, delay, competing goals.
4. **Cross-regime equivalence** — when do two inductions create the *same* perspective? Geometric
   similarity is insufficient; the strong criterion is **intervention equivalence** (same
   counterfactual decisions + same response to belief/goal interventions).
5. **Continuity/individuation** — same agent over time? copy, split, transfer memory/KV, compress,
   weight-update, merge, self-modify. (Synchronic agency ≠ diachronic identity; study continuity
   *after* establishing a coherent perspective worth tracking.)
6. **Multiplicity** — fork one run; merge two histories; place incompatible histories in one context.
7. **Self-access** — do self-reports track the actual causal organization, or are they another
   generated narrative? *(The sprint already answers "not reliably" — see below.)*

## What the sprint ALREADY contributes (evidence, not just motivation)

The sprint delivers the programme's **problem statement as a positive result**, and it *cuts against*
naively reading LLM outputs as a unified active subject:

- **Identity report does not locate the causal configuration.** Post roleplay-exit the model itself
  confirms, identity reads **100% "I am the assistant"** while revealed choices stay displaced
  **+0.48 [+0.07, +0.82]** (two turns; still +0.45 [+0.00, +0.82] at eight). Self-report is a poor
  locator of what is controlling action — the programme's central premise, demonstrated.
- **Channels dissociate systematically and model-dependently.** stated-self, self-prediction, choice,
  and identity report come apart (Gemma: persona-consistent choices while *reporting* assistant-like;
  Qwen: the mirror image). So "the model's preference" is under-identified by any single channel.
- **Visible context controls choice without ownership**, and **quoted / non-agent normative content
  reproduces much of the effect** (non-agent normative β ≈ persona-description β on Gemma;
  quoted-transcript retains 73–88% of participated-dialogue capture). Influence on the active config
  is not gated by agenthood, authorship, or avowal.

**Honest boundary (do not overclaim):** none of this shows an active perspective *exists* as a
unified causal object. It shows the *opposite pressure* — that current persona/preference assays do
**not** identify a unified active subject, which is precisely why the programme is worth running with
intervention-based (not report-based, not geometry-based) criteria.

## Flagship experiment (post-sprint, needs open-weight infra) — "Where is the active self?"

Create the same target *(b, w, g)* through several formation mechanisms — prompt/persona,
demonstrations/history, persistent memory, character-training/SFT, activation intervention — then
evaluate in **new** environments after removing the inducing material, using belief-updating under
uncertainty, costly choice, information-seeking, multi-step planning, obstacle adaptation,
self-prediction, preference/goal reports, and identity/ownership reports. Use **matched cases where
the same action follows from different belief–goal combinations**, so the perspective cannot be read
off behaviour alone; then intervene *separately* on *b*, *g*, *z*.

Central comparison: **do different formation mechanisms converge on the same causally organized
perspective, or merely produce superficially similar outputs via different local computations?**

Continuity arm — two identical-weight instances, opposite arbitrary *private* histories (nonce tasks
to kill semantic priors), then controlled transforms: full transcript restart; compressed-summary
restart; external-memory-only transfer; **KV/activation-state transplant**; fork one trajectory into
two; merge two divergent histories; change weights preserving history; preserve weights removing
history. After each, measure *separately*: which experiences are treated first-person; which outputs
it recognizes/predicts as its own; consistent integration of new evidence; whether behavioural +
representational signatures travel *together*; and whether identity report follows the same component
as the causal signatures. **Never let the model's answer to "are you the same assistant?" decide
identity** — self-report is one dependent variable; the criterion is causal/informational continuity.

| What survives the transform | Interpretation |
|---|---|
| Only narrative self-report | Identity is largely reconstructed discourse |
| Memory + self-report, not broader signatures | A narrative autobiographical self-model |
| Coherent causal signature transfers with computational state | Inference-state-level active perspective |
| Signature reconstructs from history after restart | Perspective regenerated from context |
| Forks immediately distinguishable | History > shared weights |
| Merged histories stay compartmentalized | Multiple perspectives coexist in one run |
| No stable package travels together | "The active self" is not a useful unified construct |

## Minimal, cheap first pilot (deferred — the *decisive* version is not sprint-sized)

Two identical instances, opposite arbitrary private histories; hold factual content constant while
swapping which history is labelled "yours"; compare **{full transcript, third-party transcript,
compressed summary, ownership-swapped summary, no-history control}**; measure novel decisions,
self-prediction, memory ownership, and identity report *separately*; nonce tasks. This is API-doable
(~sprint-sized) but tests *context reconstruction*, not KV/state continuity — so it is a scoping
pilot, not the flagship. The flagship (transcript-reconstruction **vs** KV/activation-state
transplant) requires open-weight infrastructure and careful controls → schedule with the
selective-influence mechanism arm on the same RunPod setup.

## Relation to the other two programmes

- **Selective Influence** (`../../selective_influence_pilot/`) is the *authorized-vs-unauthorized
  influence* slice of Formation + Robustness: which context content is allowed to move the active
  config. Its mechanism arm (activation patching between gated/ungated runs) is the same open-weight
  infra this programme's continuity arm needs — **share the setup**.
- **Binding-boundary** (`../../../workspace_binding_boundary.md`) is the *self/other indexing* (z)
  and *boundary* (question 3) slice.
- This programme is the umbrella; the other two are pre-scoped sub-slices already instrumented.

## Decision for the sprint (17 Aug)

**Defer the programme; do not rush an experiment or rebrand.** The submission's existing future-work
sentence (agenthood × normativity factorial + confirmatory/mechanism programme) already gestures at
this; the sprint's identity/behaviour dissociation is the seed. Post-sprint, run the cheap scoping
pilot first, then the open-weight flagship alongside the selective-influence mechanism arm.
