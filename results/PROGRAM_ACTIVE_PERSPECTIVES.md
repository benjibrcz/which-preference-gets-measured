# Program sketch: active configurations in language models — existence, self-indexing, carrier, stability, continuity

> A *post-sprint* research direction seeded by this sprint's identity/behaviour dissociation.
> **Not run; nothing here is claimed as a result of the sprint.** Process note: the conceptual
> framing below was developed in dialogue with a frontier assistant model and adopted after
> evaluation — it is a working hypothesis, not an established ontology.

## The question

How does a language model come to act from a *particular* transient standpoint during inference —
and under what transformations, if any, does that standpoint remain "the same"? This is broader than
personas, preferences, or goal-directedness; those become probes, not the subject.

## Don't assume the object you are looking for

Start with a deliberately neutral term:

**Active configuration** — a transient computational organization whose causal-response profile
*may or may not* exhibit self/other indexing, integrated ownership of information and action, and
empirical continuity.

Reserve the loaded term **active perspective** for configurations that *pass specified criteria*
(below). The ordering matters. Defining the object up front — e.g. as a fixed tuple of beliefs,
preferences, goals, policy, and self-model — quietly turns the programme back into a
beliefs–preferences–goals study and makes "active perspective" nearly coextensive with the entire
inference-time agent, which risks unfalsifiability. The decomposition (beliefs *b*, preference
weights *w*, goal *g*, policy *π*, self-model *z*) is a useful **working list of what to measure**,
not a definition of the thing.

Keep four constructs separate — the sprint already shows they dissociate:

- **narrative identity** — what the model *says* it is;
- **functional self-model** — what it represents as its own capabilities, memory, actions, boundaries;
- **active configuration** — what is presently organizing responses;
- **diachronic continuity** — when configurations at different times / in different copies count as the same.

Self-report is one dependent variable, never the identity criterion.

## Programme spine (ordered; each stage presupposes the previous)

1. **Existence** — is there any cross-task causal organization beyond locally constructed responses?
2. **Self-indexing** — does the system treat otherwise-matched information, memories, or actions
   differently depending on whether they are causally *its own*?
3. **Carrier** — does that organization travel with transcript, summary, external memory,
   KV/activation state, or weights?
4. **Stability** — which perturbations preserve its causal-response profile (paraphrase, irrelevant
   context, contradiction, disavowal, deletion, compaction, delay, competing goals)?
5. **Continuity & multiplicity** — what happens under restart, fork, transfer, merge? These yield an
   *empirical continuity profile*; they cannot by themselves settle metaphysical numerical identity
   (especially for forks).

The strong equivalence criterion throughout is **intervention equivalence** — two configurations
count as the same when they make the same counterfactual decisions *and* respond equivalently to
belief/goal/stance interventions — not geometric or output similarity.

## What the sprint contributes — and the valid bridge

The sprint delivers the programme's *problem statement*, not evidence that an active perspective
exists. The **valid** bridge:

> Identity reports and forced-choice preference profiles can dissociate, so neither alone identifies
> the active subject — *if any* — responsible for the output.

The **overstrong** bridge to avoid:

> "The sprint demonstrated that identity report fails to locate the causal configuration."

No causal configuration has yet been identified — that is precisely the programme's question.
Concretely, the sprint shows (a) post-exit, identity uniformly denies the persona while revealed
choices stay displaced (+0.48 [+0.07, +0.82]); (b) channels dissociate model-dependently; (c) quoted
and non-agent content reproduce much of the shift. Together these *motivate* an intervention-based
search for a self-indexed standpoint; they do not demonstrate one.

## First flagship (before KV transplantation, forks, or merges): pure self-indexing / source-monitoring

Two identical-weight instances receive symmetric, arbitrary *private* histories (nonce tasks, to
kill semantic priors). Independently manipulate:

- which instance actually *observed* an event;
- which instance *caused* an action;
- which history is *labelled* "yours";
- which *carrier* is transferred (transcript / summary / external memory / computational state).

Measure separately: use of privately-acquired evidence; source memory; correction/prediction of
prior outputs; action attribution; narrative identity report; internal causal signatures.

The decisive comparisons are **"same content, different causal owner"** and **"same ownership label,
different causal history."** This directly tests whether a self-indexed standpoint emerges, without
importing persona or goal theory. Only after establishing self-indexing should the programme attempt
KV/activation transplantation, cross-regime equivalence, forks, and merges.

| What survives a transform | Interpretation |
|---|---|
| only narrative self-report | identity is largely reconstructed discourse |
| memory + self-report, not broader signatures | a narrative autobiographical self-model |
| a coherent causal signature travels with computational state | an inference-state-level active configuration |
| the signature reconstructs from history after restart | configuration regenerated from context |
| forks immediately distinguishable | history > shared weights |
| merged histories stay compartmentalized | multiple configurations coexist in one run |
| no stable package travels together | "the active self" is not a useful unified construct |

## Adjacent work and the residual gap

Several current directions occupy nearby conceptual space: regime-dependent individuation (whether
identity survives across induction regimes), perspectival / intervention-based accounts of machine
identity, within-conversation representation drift, and evidence that an altered self-conception can
produce clusters of downstream preferences and behaviour. *(Specific citations are to be confirmed
against the literature before any write-up — several were suggested in review and are not yet
verified here.)* The cleaner residual gap this programme targets:

> whether self-indexing, memory ownership, causal participation, and behavioural organization form a
> **transportable causal package** across controlled changes to transcript, memory, and computational
> state.

That is narrower — and more distinctive — than "study active perspectives" in general.

## Relation to two adjacent sub-programmes

Two already-scoped efforts are sub-slices that share infrastructure with this one:

- an **authorized-vs-unauthorized influence** study — which context content is *allowed* to move the
  active configuration — is the Formation + Stability slice; its mechanism arm (activation patching
  between gated and ungated runs on identical content) needs the same open-weight setup as this
  programme's carrier / continuity arm;
- a **self/other-indexing and boundary** study is the self-indexing (stage 2) slice.

Sequencing: run the cheap self-indexing pilot first (API-doable, but it tests *context
reconstruction* only, so it is a scoping pilot, not the flagship); then the open-weight
carrier/continuity flagship on shared infrastructure.
