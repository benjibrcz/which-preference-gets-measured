# Submission rewrite brief: lessons from previous Apart winners

> Prepared 17 Aug 2026 from publicly available Apart project pages, reviewer comments, and two
> accessible winning PDFs. This is comparative guidance, not an official Apart template. The
> examples come from different sprint topics, so treat recurring patterns as evidence about
> judgeability—not proof that a particular layout causes a project to win.

## Sources reviewed

- **1st, AI Manipulation (2026):** [Who Does Your AI Serve?](https://apartresearch.com/project/who-does-your-ai-serve-manipulation-by-and-of-ai-assistants-77xx)
  — 11-page report; abstract quantifies three studies; public code; study-design diagrams; explicit
  hackathon caution; reviewer praised breadth, readability, human validation, and a surprising
  mechanism-level observation (“evasion, not lying”).
- **2nd, AI Manipulation (2026):** [Eliciting Deception on Generative Search Engines](https://apartresearch.com/project/eliciting-deception-on-generative-search-engines-6nuu)
  — tightly scoped baseline-versus-attack comparison; reviewer praised that the experiment tested
  exactly what it claimed, was falsifiable/reproducible, and had production relevance.
- **3rd, AI Manipulation (2026):** [Cross-Linguistic Sycophancy](https://apartresearch.com/project/crosslinguistic-sycophancy-in-frontier-llms-a-benchmark-study-w55u)
  — one memorable disparity, an immediately legible safety implication, and acknowledged limits.
- **1st, AI Control (2026):** [Omission Attacks](https://apartresearch.com/project/omission-attacks-when-doing-nothing-is-the-attack-0y1v)
  — defines one neglected failure mode, demonstrates it against existing monitors, proposes a
  targeted defense, and reports a clean before/after result.
- **3rd, AI Control (2026):** [ActionLens](https://apartresearch.com/project/actionlens-preexecution-environment-probing-for-agent-action-approval-xy22)
  — six pages according to its reviewers; sharp conceptual insight, design diagrams, benchmark
  transfer, ablations, confidence intervals, public code, and an honest usefulness cost.
- **1st, AI Forecasting (2025):** [AI Incidents Forecasting](https://apartresearch.com/project/ai-incidents-forecasting-w92p)
  — 8-page report; explicit research questions, baseline, backtest, uncertainty, reproducibility,
  and one main forecast figure.
- **Current sprint instructions:** [Digital Minds Research Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16)
  requests a **short research report**, but exposes no public page limit or mandatory template.

## What repeatedly makes top submissions judgeable

1. **A one-sentence object.** The reader can say what failed, what was measured, and why it matters
   after the abstract.
2. **Numbers in the abstract.** Winners state models/tasks/participants, the main contrast, and the
   practical implication immediately.
3. **A clean comparison.** Baseline versus attack, transcript-only versus environment probes, main
   model versus naive baseline. Complexity may exist behind the result, but the headline contrast
   is simple.
4. **A visible experimental design.** Strong reports show the causal comparison as a diagram before
   presenting many results.
5. **One non-obvious update.** “Evasion, not lying”; “the useful probe depends on environment”; or,
   here, **agent framing is not necessary and identity reports do not identify the operative choice
   state**.
6. **External or adversarial validation.** Human calibration, transfer benchmark, ablation, matched
   control, or backtest. The validation is integrated into the story rather than listed as defensive
   detail.
7. **Public artifacts.** A clickable repository, data/code description, and a reproducible command
   are visible in the report.
8. **Honest limitations without surrendering the result.** Winning reports openly identify sample,
   judge, ecological-validity, and usefulness limitations while preserving a narrower conclusion.

## Recommended format for this submission

Use **4–5 readable pages**, not the current two-page compression and not the 19-page technical
report. Keep the full report as a linked appendix.

### Page 1 — Problem, answer, design

- Outcome-oriented title: **Which Preference Gets Measured? Context and Channel Instability in
  Model Preference Audits**.
- 120–170 word abstract with this order:
  1. problem: welfare audits seek “model preferences”;
  2. method: 76 task pairs, multiple elicitation channels, four models, controlled framings;
  3. main result: nominally irrelevant task-relevant context shifts committed choices;
  4. discriminating control: agenthood is not necessary, but the normative-text control does not
     isolate ordinary semantic priming;
  5. implication: report a context × channel sensitivity envelope, not one preference profile.
- A compact design diagram: **context manipulation → four readouts (choice, ownership,
  self-prediction, identity)**.
- A three-bullet contribution box, no literature survey on page 1.

### Page 2 — Core measurement result

- Define β once in plain language and give an absolute companion measure.
- Hero figure: baseline, task-irrelevant prose, persona description, and non-agent normative text,
  with 95% CIs.
- State the matched-control progression:
  - generic prose does not explain distinctive-persona shifts;
  - warm prose exposed tone priming;
  - normative non-agent text shows the effect is not agent-specific;
  - this last control does **not** cleanly separate priming from following a system-level standard.
- Report policy-minus-character contrasts with CIs; avoid ratios such as “3×” around a small
  denominator.

### Page 3 — Channel dissociation

- One compact figure or matrix showing choice, ownership report, self-prediction, and identity for
  the clearest Gemma and Qwen cells.
- Lead with the qualitative reversal, then the numbers:
  - Gemma: persona-consistent choice with assistant-like ownership report;
  - Qwen: the reverse pattern.
- Interpretation: channels are different measurements, not interchangeable windows onto one latent
  preference.

### Page 4 — Contextual carryover and implications

- Describe the post-exit result as **failure to gate visible earlier content**, not hidden memory.
- Show the decisive surgery result: removing persona turns returns the effect toward drift.
- Give the applied recommendation and reusable deliverable.
- Put limitations beside the claims they constrain.

### Optional Page 5 — Reproducibility, status, and related work

- Clickable repository and one cache-only reproduction command.
- Exact dated timeline and AI-assistance disclosure.
- Small evidence-status table: prospectively specified core / sequential extensions /
  post-review exploratory controls.
- One concise related-work paragraph and references.
- No table of contents.

## Binding rewrite rules for Claude

1. **Write for a judge scanning in 90 seconds.** Every page needs a question, a result, and a
   takeaway visible without reading every sentence.
2. **Lead with measurement validity, not ontology.** Do not claim persona binding, genuine
   preference, welfare state, active self, or individuation.
3. **Use one main thesis:** nominally irrelevant context and elicitation channel materially alter
   the preference profile an audit records.
4. **Keep at most three supporting findings:** context effect; channel dissociation; contextual
   carryover. Everything else links to the technical report.
5. **Separate evidence status visually.** Label the normative non-agent control “exploratory,
   post-review” wherever first introduced.
6. **Never call the normative-policy control “decisive semantic priming.”** It establishes that
   agent framing is unnecessary; its imperative/system-prompt character remains a confound.
7. **Prefer direct contrasts with uncertainty.** Every headline numeric contrast gets a CI; avoid
   unstable ratios and isolated point estimates.
8. **Define β in one sentence, once.** Pair it with absolute probability displacement or changed
   pairs so a nontechnical judge can understand magnitude.
9. **Use figures to replace prose.** Maximum three main figures: design, context/control ladder,
   channel dissociation. Captions state the finding, not merely what is plotted.
10. **No defensive methods wall on page 1.** Move cross-fitting details, parser details, prediction
    ledger, probes, steering, identity clouds, and model correlations to the linked report.
11. **Make the confound discovery a strength.** State the original interpretation, the control that
    challenged it, and the narrower conclusion retained. Do not dramatize the iterative process.
12. **Use calibrated verbs:** “shifts,” “covaries,” “dissociates,” “fails to gate.” Avoid
    “reveals,” “proves,” “internally real,” “predicts,” or “genuine behaviour” unless literally
    justified.
13. **Show practical value concretely:** recommend multi-context, multi-channel audits that report a
    sensitivity envelope.
14. **Make links usable:** public GitHub URL, clickable report/code references, no local paths.
15. **Keep disclosures exact:** the core battery predates the sprint; cross-fit and normative-control
    work occurred during it; distinguish human direction from AI implementation without implying
    unaudited work was manually verified.

## Suggested abstract skeleton

> Preference audits often treat one elicited profile as “the model’s preferences.” We test whether
> that profile is stable across context and measurement channel using 76 task pairs, four
> instruction-tuned models, multiple persona framings, and four readouts: committed choice,
> ownership report, self-prediction, and identity report. Nominally irrelevant character
> descriptions substantially shift task selections even when roleplay is explicitly disclaimed,
> while task-irrelevant prose does not. A post-review control finds similarly large shifts from
> non-agent normative text, showing that agent framing is unnecessary—though the policy-like
> wording does not isolate semantic priming from instruction following. Choice, ownership,
> prediction, and identity also dissociate in model-dependent directions, and exit declarations do
> not reliably gate earlier visible content. These results do not identify genuine preferences or
> welfare states. They show that single-context, single-channel audits lack construct stability; we
> recommend reporting a context × channel sensitivity envelope instead.

## Final preflight

- Can a judge state the main result after the abstract?
- Is the principal comparison visible in one figure?
- Does every headline estimate have uncertainty?
- Are confirmatory and exploratory analyses unmistakable?
- Are the timeline, AI role, eligibility status, and repository link accurate?
- Does the submission remain persuasive if every ontology/mechanism claim is removed?
