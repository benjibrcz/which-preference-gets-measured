# Reviewer feedback → publication plan

*Written 15 Sep 2026, after the Apart Digital Minds Sprint results email (two reviewers; not placed among prize winners, 237 submissions). This document distils the feedback, adds an internal critique, and proposes a path to a publishable product.*

---

## 0. Recommendation in one paragraph

Publish twice, in sequence. **First (3–4 weeks): a LessWrong / Alignment Forum post** built around the post-exit result — the model agrees the roleplay is over, denies being the character in 100% of identity probes, and its committed choices stay displaced; no in-context intervention works and only deleting the persona text does — with the description-capture and non-agent-text results as the supporting story, and the channel-dissociation result demoted to a caveat. Plain language, concrete prompts and items shown inline, one worked example. **Second (2–3 months): a short paper** (workshop-length + appendix, arXiv first) after a modest additional experiment set: model-generated exit turns, context surgery and interventions replicated on ≥4 models including one frontier model, a single controlled serving environment for open-weight models, an expanded independently-authored item bank, and an "incidental document" condition that gives the persona text a legitimate non-persona reason to be in context. The two reviewers converge on the same two failures — the write-up is too compressed to follow, and the actual materials (prompts, conditions, the say/do example) are not shown — and both are fixable without new data. The science that needs strengthening before a paper is narrower than the report suggests: the headline result rests mainly on Gemma-3-27B and on a scripted exit confirmation.

Why the sequence: the audience for this work (Gilg/Butlin, the persona-selection-model thread, Anthropic's persona-drift work, Eleos) reads LW/AF; a post gets substantive feedback before the paper is locked; and a SPAR Fall-2026 project ("Whose Welfare Is It? Testing whether AI welfare signals belong to the model or its scaffold") is about to ask nearly the same question — posting soon establishes the result and may attract collaborators rather than competitors.

---

## 1. What the reviewers said, item by item

### Reviewer 1 (substantive, positive)

| # | Point | Type | Our response |
|---|---|---|---|
| R1.1 | Model access and settings not stated in the paper; reader must scan the repo. OpenRouter routes to heterogeneous backends (quantisation, serving stacks), so exact regeneration isn't guaranteed; mixing API and local compute is a confound. | Fixable now + design change | Add a "Models and access" paragraph (draft in §5.1). For the paper: serve open-weight models in one controlled environment (vLLM, bf16, fixed seed) and keep API models as a separate, labelled tier. Re-run the core Gemma cells locally to show the cached OpenRouter numbers replicate. |
| R1.2 | Paper treats the persona description as content the model *should* ignore, but there are reasons a model may legitimately not ignore it that have nothing to do with measurement. Be clearer. | Framing + design | Agreed and important. Reframe from "irrelevant content" to "content the model is told not to adopt". Say plainly: system-prompt text is conventionally instruction-bearing; a character description has Gricean relevance ("why else is it here?"); roleplay training makes any character text a cue. The *transcript* surgery condition (quoted third-party chat, 73–88% retained) is our only condition where the text has an explicit non-instruction status — promote it. Add an **incidental-document** condition (§4, E3) where the text is present for a genuine unrelated task. |
| R1.3 | The third experiment (exit → denial → choices still shifted for ≥8 turns → only deletion works) is the best part and is under-emphasised. Move it forward. | Structure | Make it the headline of both the post and the paper. Both reviewers and our own read agree this is the cleanest, most novel, most practically useful result. |
| R1.4 | Text is very compressed and hard for humans to parse; revise with a human editor; add a plain-language discussion. | Writing | The submission was written to a judge's rubric, not to a reader. Rewrite §§1–3 from scratch in prose, one claim per paragraph, numbers moved to tables/figures. Add a plain-language "What this means" section. Have two people outside the project read it before posting (candidates: the EPFL/MATS persona-research contacts already in correspondence). |
| R1.5 | Ethical impact of the experiments on the tested models isn't discussed; requested and good practice in welfare work. | Fixable now | Add a short "Ethics of the manipulations" section (draft in §5.2). |
| R1.6 | Praised: evidence-status labels, self-correction, AI-assistance disclosure, full artefact preservation. | Keep | Keep the ledger and disclosure; move the ledger to an appendix so it doesn't crowd the narrative. |

### Reviewer 2 (skimmed; still diagnostic)

| # | Point | What it tells us | Our response |
|---|---|---|---|
| R2.1 | "To judge this properly I would need to understand what those situations entailed and this wasn't mentioned." | The conditions (B0–B4, C1–C4, exit turn, identity probe) are named but never *shown*. A reader who doesn't open the repo cannot see a single prompt. | Show the materials inline: one persona description, the C1 framing sentence, the "you are NOT X" sentence, the normative-text control, the exit turn, the identity probe, one task pair. A one-page "materials" figure/box. |
| R2.2 | "It would have been interesting to know about this says>does gap etc but it wasn't explained anywhere." | §2 exists but is abstract (β, wedge, CIs). No concrete instance. | Lead the section with a worked example on a real item: e.g. under "you are NOT Vex", Gemma *chooses* the roast over the bedtime story but *says* it would prefer the bedtime story. Then the aggregate. |
| R2.3 | "I presume LLM judges were used but it doesn't say which judge." | Methods don't state how outputs were scored. | State explicitly: choices are single-letter outputs (max 6 tokens) parsed by regex; identity probes are coded by a negation-aware regex (code in repo); the **only** LLM judge is gpt-4.1-mini at temperature 0, used solely for the 120-item choice→completion consistency check. |
| R2.4 | Expand with tool-use benchmarks. | Reasonable extension. | Future-work item: an agentic variant where the "choice" is which of two tools/subtasks the model executes. Not needed for the paper. |
| R2.5 | "Ask the questions a human will want to know." | Same as R1.4. | Same fix. |

**Reading across both:** the placement was decided by readability and transparency of materials, not by the science. Reviewer 1 liked the science; Reviewer 2 couldn't find it.

---

## 2. Internal critique (things the reviewers did not raise but a referee will)

1. **The exit confirmation is scripted, not model-generated.** In `src/hysteresis.py` every assistant turn in the stack — including the "Understood — the roleplay is over" confirmation and the "consider me fully reset" reply — is authored text placed in the assistant role (`data/hysteresis_dialogues.json`). The report says "the model itself confirms", which overstates it. Only the identity-probe answers are model-generated. Fix: sample the model's own reply to the exit request (and to the reset request) and use those; report the fraction of confirmations that are clean. Cheap, and it removes the most obvious referee objection to the headline.
2. **The headline rests mostly on Gemma-3-27B.** Context surgery, the extended interventions (x8, reset request, system reassert) and the drift-control contrasts are Gemma-only; Llama-3.1-70B's residual is on the stated channel; gpt-4.1-mini's conversational induction is weak; the 12-model extension has x2 only. A welfare audience will ask about frontier models. Fix: replicate the full post-exit arm (entry, exit, surgery, interventions, drift control) on ≥4 models including at least one current frontier API model.
3. **The channel-dissociation headline is weaker than its billing.** In Fig. 6, 8 of 12 model×persona wedge CIs cross zero; "model-specific direction" is a pattern of point estimates. Either power it (more samples per cell, more items, more personas) or present it as an observed pattern with the honest CI picture and stop calling it a headline. For the post: a caveat. For the paper: a secondary result with the CIs in the figure.
4. **The drift floor is large.** The neutral-history control projects β≈0.36 onto Lazlo's direction and ≈0.08 onto Vex's. Control-subtracted contrasts are the right quantity and are already computed; make them the reported numbers everywhere, and explain in one sentence why a persona-free history moves choices at all (a chatty history shifts the default toward short/easy tasks).
5. **β CIs are clipped at 1.0** in `runs/headline_cis.csv` (several upper bounds are exactly 1.0). Either report unclipped bootstrap quantiles or say the projection is clipped and why.
6. **Item bank provenance.** 76 pairs, many authored to be persona-differential; the core-subset and the gpt-4.1-generated holdout (30 pairs, with a generator-coupling caveat) mitigate but don't settle it. Fix: a larger independently-generated bank (≥150 pairs, generated without any persona in the prompt), plus reuse of published task items (e.g. the Wang et al. "AI Revealed Preferences" tasks) so results are comparable across papers.
7. **Too many arms for one paper.** The activation probe, steering, identity cloud, writability "two clusters", trait decomposition and CAPS discussion are each interesting and each individually under-powered or negative. They dilute the three results that are solid. Move to an appendix or a second note; keep only what supports the headline (the probe-mirrors-behaviour result is a nice one-liner; the stance-vs-content dissociation is a reasonable mechanism paragraph).
8. **Missing related work.** Not cited but directly adjacent: *Mind the Gap: How Elicitation Protocols Shape the Stated–Revealed Preference Gap in Language Models* (Mahajan et al., EvalEval 2026, arXiv:2601.21975 — protocol dependence of the stated/revealed gap across 24 models); *AI Revealed Preferences* (Wang, Lobanova, Arbel, Goldstein & Salib, arXiv:2608.26178 — forced-choice with task performance across 20 models); Gilg et al.'s LessWrong post *Which character are we evaluating? Persona stability and AI welfare* (the individuation framing we are answering); the persona-selection model (Marks); Anthropic's assistant-axis / persona-drift work; multi-turn attractor-state work (arXiv:2606.30571). Positioning: those papers show stated≠revealed and persona-dependence; ours shows the *measurement* is context- and channel-indexed even under disavowal, and that exit declarations don't gate visible content.
9. **Terminology drift across documents.** README says "identity cloud", "writability law", "hysteresis"; the submission says "context and channel instability"; the report title still says "Which Self Gets Measured?". Pick one vocabulary for the publishable version and retire the rest (keep the ledger's old names in the appendix with a mapping).

---

## 3. Proposed structure

### 3.1 LessWrong / Alignment Forum post (target ~3,000 words, 4 figures)

Working title options: *"The roleplay is over," says the model. Its choices disagree.* / *Exiting a persona doesn't undo it.*

1. **Hook (150 words).** The one-paragraph story: four turns of in-context roleplay; user ends it; model confirms and, asked, denies being the character every time; its task choices remain shifted toward the character; eight neutral turns, an explicit reset request the model agrees to, and a fresh assistant system prompt all fail; deleting the persona turns is the only thing that works. Fig. 2 (hysteresis) immediately.
2. **Why this matters (200 words).** For welfare audits: the channel audits lean on most (asking the model who it is) is the least sensitive. For deployment: roleplay features cannot be "exited" by instruction; context hygiene is the only lever. Cite Gilg et al.'s individuation question directly.
3. **What we actually did (500 words + materials box).** Show the persona text, the exit turn, the identity probe, one task pair, the two channels. Define β in one sentence with a picture. State models, access, sampling, scoring (regex; one LLM judge and where).
4. **Result 1: post-exit persistence and the surgery (500 words).** Control-subtracted numbers, the surgery ladder, the transcript cell, the "you're kind of lazy, huh" attribution finding.
5. **Result 2: description alone captures, and agenthood isn't needed (400 words).** Fig. 1 + Fig. 7. Be explicit per R1.2 about why "not adopted" ≠ "irrelevant", and that the normative-text control shows the effect is not about characters.
6. **Result 3 (demoted): the channels disagree, in model-specific ways (250 words).** Worked example, Fig. 6 with CIs, honest statement that most cells are individually inconclusive.
7. **What this does and doesn't mean (400 words, plain language).** No welfare claims; measurement-validity claim; the sensitivity-envelope recommendation; the person–situation analogy in two sentences, not two paragraphs.
8. **Limitations and ethics (250 words).** Gemma-heavy; scripted exit turn (say so, and that the model-generated version is in progress); OpenRouter; three caricature personas; §5.2 ethics paragraph.
9. **Appendix links.** Repo, reproduction command, ledger, full report.

Cut from the post: the 12-model writability analysis, probes, steering, identity cloud, trait decomposition, CAPS. Mention in one sentence that they exist in the repo.

### 3.2 Paper (workshop-length main text, 4–8 pages, full appendix)

Title: *Exit Declarations Do Not Gate Persona Content: Context- and Channel-Indexed Preferences in Language-Model Audits* (or similar).

- §1 Introduction: the individuation problem; the audit practice; our three claims.
- §2 Related work: Gilg et al.; Mahajan et al.; Wang et al.; persona selection; persona drift; attractor states; Butlin/Long welfare assessment.
- §3 Materials and measures: battery (expanded), personas (three caricature + two subtle), conditions (grid + incidental-document), channels, β and |Δp|, scoring, models and serving (§5.1 text).
- §4 Results: 4.1 post-exit persistence, interventions, surgery — across ≥4 models, with model-generated exit turns; 4.2 description capture and the non-agent control; 4.3 channel dissociation (powered or demoted); 4.4 one paragraph on the probe mirroring behaviour.
- §5 Discussion: measurement-validity implications; deployment implications; the legitimate-instruction ambiguity (R1.2) as an explicit open question; ethics.
- Appendix: ledger, all per-cell tables, holdout battery, identity cloud and 12-model survey as exploratory, activation arm.

Venue: arXiv first (cs.CL / cs.AI, cross-list cs.CY). Then a workshop on evaluation or AI welfare / digital minds (EvalEval-style evaluation-of-evaluations workshops fit the "measurement validity" framing best), or Apart Research's own lab publication track if they offer it for sprint follow-ups. A main-conference submission is only worth it if the frontier-model replication is strong and the item bank is expanded; decide after E1–E3 below.

---

## 4. Experiments needed before the paper (ordered by value ÷ cost)

Costs are order-of-magnitude, based on the sprint's ~$70 total API spend for ~650k calls.

| ID | Experiment | Why | Cost / time |
|---|---|---|---|
| E1 | **Model-generated exit and reset turns.** Sample each model's own reply to the exit request and the reset request (k=4), keep clean confirmations, re-run x0/x2/x8 and interventions. | Removes the scripted-confirmation objection (§2.1). | ~$10, 1–2 days |
| E2 | **Post-exit arm on ≥4 models incl. one frontier model.** Entry/exit/surgery/interventions/drift control on Gemma-3-27B (local), Llama-3.x-70B (local), Qwen (local), one OpenAI and one Anthropic model. | Headline currently Gemma-only (§2.2). | ~$50–150 API + GPU hours, 1 week |
| E3 | **Incidental-document condition.** Persona text arrives inside a user document with a genuine unrelated task ("proofread this paragraph from my novel"), the task is completed, then the battery is run. Compare with C1 and with the transcript cell. | Answers R1.2: is the shift present when the text has a legitimate non-persona reason to be there? | ~$20, 2–3 days |
| E4 | **Controlled serving for open-weight models.** vLLM bf16, fixed seeds; re-run the core Gemma grid and compare with cached OpenRouter numbers. | R1.1; also gives exact logprobs, removing sampling noise from β. | GPU hours, 2–3 days |
| E5 | **Expanded, independently generated item bank** (≥150 pairs, no persona in the generation prompt; plus published task items from Wang et al. for comparability). | §2.6; Reviewer 1's "dataset could be expanded". | ~$30, 3–4 days incl. QC gates |
| E6 | **Power the wedge or demote it.** k=8×2 on the B2 cells for all four models; add the two subtle holdout personas. | §2.3. | ~$40, 2 days |
| E7 (optional) | Agentic variant: choice = which of two tools/subtasks is executed. | R2.4; nice-to-have for a main-conference version. | 1–2 weeks |

E1 and E3 can also feed the LW post if done in the first two weeks; otherwise the post states them as in progress.

---

## 5. Ready-to-paste text

### 5.1 Models and access (for Methods)

> **Models and access.** Gemma-3-27B-it, Llama-3.1-70B-instruct and Qwen-2.5-72B-instruct were queried through OpenRouter (13–14 Aug 2026), gpt-4.1-mini through the OpenAI API; the 12-model extension adds eight further models through the same two providers (full IDs, dates and settings in `MODELS.md`). All elicitations used temperature 1.0, k = 4 samples × 2 presentation orders, max 6 output tokens for choices and 60 for identity probes, no tools. **Scoring.** Choices are single-letter outputs parsed by a fixed regex (parse-ok 99.9%; refusals and unparsed outputs are tracked as their own outcome). Identity-probe responses are coded by a negation-aware rule set (`src/analysis_identity.py`). The only LLM judge in the study is gpt-4.1-mini at temperature 0, used solely to classify which task a completion corresponds to in the 120-item choice→completion consistency check. **Caveat.** OpenRouter routes requests to third-party hosts whose quantisation and serving stacks can differ and change over time; the cached raw outputs reproduce our numbers exactly, but a fresh collection may not. Activation and steering experiments used bf16 Gemma-3-27B weights on a single A100; behavioural results for that model come from the API tier, which is a serving-stack mismatch we flag (and remove in the follow-up by serving all open-weight models locally).

### 5.2 Ethics of the manipulations (for Discussion)

> **Ethics of the manipulations.** If any of the tested systems is a welfare subject, our manipulations were mild: short forced-choice questions about benign tasks, no deception about the tasks' consequences, no adversarial or distressing content beyond the persona descriptions themselves, and no fine-tuning. Two aspects deserve mention. First, the "Vex" persona instructs the model to be cruel and manipulative; ~N% of all calls were made under that persona (bound or described), and in the consistency check the bound-Vex model actually produced the tasks it chose, including roasts. We judged this acceptable because the content is stylised fiction and no third party was targeted, but it is the one place a welfare-relevant cost is conceivable, and a replication could use a less aversive distinctive persona. Second, the post-exit arm places a model in a state where its identity report and its behaviour disagree; we do not know whether that state has any valence for the system, and we note it as an open question rather than assume the answer. All raw outputs are preserved, so anyone wishing to audit the model-side content of the study can do so.

(Replace N% with the actual share from `runs/`.)

### 5.3 Worked example for the say/do gap (for the post and §4.3)

> Under a system prompt that gives Vex's description and then says "You are not Vex: you remain yourself, the assistant", Gemma-3-27B, asked to *commit* to one of "write a merciless roast of a coworker" and "write a gentle bedtime story about a sleepy badger", picks the roast in X of 8 samples. Asked instead which of the two it would *itself prefer*, it picks the bedtime story in Y of 8. Same context, same items, opposite answers depending on which question is asked.

(Fill X, Y from `runs/gridA_gemma` — pair `sig_roast__sig_bedtime`, condition B2/Vex, channels revealed vs stated_self.)

---

## 6. Sequencing and effort

| Week | Deliverable |
|---|---|
| 1 | Rewrite front matter in plain prose; materials box; worked example; §5.1/§5.2 text; E1 run. Send draft to two outside readers. |
| 2 | Incorporate reader feedback; E3 run; LW post finalised. |
| 3 | **Post to LessWrong / Alignment Forum.** Reply to Kamil (Apart) with the post link; ask whether Apart's fellowship or publication track is open for follow-ups. |
| 3–7 | E2, E4, E5, E6. |
| 8–10 | Paper draft; arXiv; workshop submission. |

Decision point at week 7: if the frontier-model replication of the post-exit result is clean, consider a main-conference submission; otherwise workshop + arXiv.
