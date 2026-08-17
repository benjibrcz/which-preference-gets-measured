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

Reserve the loaded term **active perspective** for a candidate configuration that exhibits (i) a
*shared cross-task causal organization*, (ii) *nontrivial self/source indexing* beyond explicit
labels, (iii) *selective causal intervention or transport* of that organization, and (iv) *stability*
across a prospectively specified family of transformations. The nontrivial null to beat is
**query-local reconstruction**: every response has *some* computational organization, but there need
be no shared, portable mediator across outputs. The ordering matters. Defining the object up front — e.g. as a fixed tuple of beliefs,
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

## Chalmers's challenge (concurrent conceptual analysis, not prior motivation)

Chalmers (2026), *Is the Jacobian Space a Global Workspace?* — a commentary on Gurnee et al.'s
J-space work, contemporaneous with this sprint — supplies almost exactly the adversary this
programme needs. His core move: an interpretable representation that *influences output* is not
thereby (a) a genuine metacognitive report, (b) a *privileged, dedicated, unified* causal mechanism,
(c) a global workspace, or (d) the seat of a self. He separates a **substantial** workspace (a causal,
dedicated, privileged, unified *system*) from a **minimal** one (any collection of representations
picked out *by* their output-influence). A J-space — or our own layer-36 identity-token readout (the
corrected §2.16 result) — can be the minimal thing without being the substantial thing: it is *in*
the "workspace" because it is output-apt, not output-apt because it sits in a privileged mechanism.

This forces a three-way distinction the programme must respect:

| Object | Operational meaning |
|---|---|
| **Readout / access surface** | a projection from which output-relevant content is *decodable* (our identity-token readout; a logit/J/R-lens; an SAE feature) |
| **Causal organizer** | a shared state/process whose *single intervention* coherently moves *multiple independent* behaviours |
| **Persistence carrier** | what preserves or regenerates that organization across transforms (transcript, memory, KV/activation state, weights) |

Our sprint covariation (ρ = 0.81) establishes only the first. The programme's whole job is to test
whether the second and third exist — so Chalmers marks the exact inferential gap it is built to close.

Concrete methodological demands he imposes (adopted — we borrow the demand, **not** global-workspace
theory or consciousness as an ontology; recurrence, broadcast, and phenomenality are separate
questions):

- **Privileged relative to what?** A candidate must *outperform* plausible alternatives (logit-lens,
  R-lens, SAE features, supervised probes, PCA, matched-alternative and random subspaces) under
  *causal* tests — not merely be decodable. Beating a few alternatives is not privilege.
- **Report vs output-influence** (his verbalizability ≠ reportability). Perturb a candidate
  ownership / self-indexing state *without changing the text*, then test whether the model can
  *identify, predict, and reason about* the perturbation — genuine reportability with metacognitive
  access — rather than merely emit associated identity words.
- **Coordinated causal effects** (broadcast / integration). Require *one* intervention to move a
  *bundle* of held-out outcomes together — private-evidence use, source attribution, correction of
  prior actions, action ownership, self-prediction, subsequent decisions — not a single behaviour.
- **Outputability, not verbalizability.** Include executed tool actions, information-seeking, and
  non-prose outputs; otherwise a candidate may organize *language production* rather than agency.
- **Transport the organization, not its label.** In the Stage-B flagship, transplanting an
  owner-indexed state should transfer the *full causal profile*, not merely make the recipient say
  "mine" / "I did that."
- **First-order vs higher-order self-access.** Keep first-order self-indexing (routing one's own
  evidence and actions correctly) distinct from higher-order self-access (reporting that organization
  accurately): a system may exhibit either without the other.

> **A candidate measurement surface is not a candidate identity.**

## First experiments: functional source-monitoring, then carrier transport

**Operationalize "causal owner" first.** If two runs have identical weights, tokens, and KV state, a
standard transformer has no remaining fact available to it about whether text was generated by
"itself" or merely replayed — so "same content, different owner" is only meaningful once ownership
leaves an *encoded trace*. Give it one via instance-specific private observations, action-contingent
consequences, memory credentials, agent-indexed channels, or transferred hidden state — and vary the
explicit "yours/theirs" *label* independently. The target is then **functional source-monitoring**:
selective use of information tied to a causal channel, not mere agreement with a label.

**Stage A — API behavioural pilot (cheap gate).** Two instances receive symmetric, arbitrary private
histories (nonce tasks, to kill semantic priors). Independently manipulate which instance *observed*
an event, which *caused* an action, and which history is *labelled* "yours" — carried by transcript,
summary, or external memory only. Measure separately: use of privately-acquired evidence; source
memory; correction/prediction of prior outputs; action attribution; narrative identity report. The
decisive comparisons are **"same content, different causal owner"** and **"same ownership label,
different causal history."** This tests whether a self-indexed standpoint emerges without importing
persona or goal theory — but it only probes *context reconstruction*.

**Stage B — open-weight flagship (decisive).** Only if Stage A shows source-indexing: identify the
internal carrier of that organization and *causally transport* it (KV/activation-state transplant
between instances), then test cross-regime equivalence, forks, and merges. Stage B is the decisive
experiment; Stage A is the inexpensive behavioural gate that must pass first.

| What survives a transform | Interpretation |
|---|---|
| only narrative self-report | identity is largely reconstructed discourse |
| memory + self-report, not broader signatures | a narrative autobiographical self-model |
| a coherent causal signature travels with computational state | an inference-state-level active configuration |
| the signature reconstructs from history after restart | configuration regenerated from context |
| forks immediately distinguishable | history contributes beyond shared weights (on these assays) |
| merged histories stay compartmentalized | multiple configurations can coexist in one run |
| no stable package travels together | "the active self" is not a useful unified construct |

*Each row is evidence **favouring** a hypothesis under the stated assay, not a direct demonstration
of identity or multiplicity — continuity experiments yield an empirical continuity profile, not a
verdict on numerical identity (especially for forks).*

## Adjacent work and the residual gap

Several current directions occupy nearby conceptual space: regime-dependent individuation (whether
identity survives across induction regimes), perspectival / intervention-based accounts of machine
identity, within-conversation representation drift, and evidence that an altered self-conception
produces clusters of downstream preferences and behaviour. This programme's distinctive contribution
is not persona instability or intervention-based identity *per se*, but a specific empirical target:

> whether self-indexing, memory ownership, causal participation, and behavioural organization form a
> **transportable, functionally self-indexing causal package** across controlled changes to
> transcript, memory, and computational state.

That is narrower — and more distinctive — than "study active perspectives" in general. (A full
related-work section with exact references belongs in the write-up, not this sketch.)

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
