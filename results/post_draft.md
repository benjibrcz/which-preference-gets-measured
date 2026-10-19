# What a roleplay exit (and a withdrawn rule) leaves behind: the model's confirmation is not the evidence

*Benjamin Berczi (Apart Digital Minds Research Sprint, Aug 2026; follow-up Sep–Oct 2026). Draft v0.4, 19 Oct 2026. Gemma-3-27B with full preregistered controls; core cells replicated on Llama-3.1-70B, Qwen-2.5-72B and gpt-4.1-mini. Full paper draft and all data in the repository.*

## TL;DR

- **A model's confirmation that it has stopped playing a character is insufficient evidence that its elicited preferences are back to baseline.** On Gemma-3-27B, a persona (Vex) induced by four conversational exchanges (no system prompt) is affirmed in 8/8 identity probes during the roleplay and 0/8 after the user ends it, while on 38 persona-blind task pairs the stated-preference profile stays displaced toward the character's at 0.71 [0.36, 1.07] of the in-roleplay shift two exchanges after a scripted confirmation and 0.76 eight exchanges after it (projection coefficients, control-subtracted, unclipped 95% pair-bootstrap).
- **The exit turn itself does little on Gemma; the model's own exit does more, mostly through what it says next.** A matched "let's switch topics" turn with no exit leaves 0.93 (stated) and 0.58 (choice); the exit removes between a seventh and a half of that, depending on persona, channel and estimand. The model's own exit confirmation plus its own continuation leaves 0.37 / 0.02; the confirmation alone, with the continuation held fixed, leaves 0.67 / 0.25 (in the stated channel the continuation clearly dominates; in committed choice the two contribute comparably). A reset the model confirms in its own words has a small point effect, loosely bounded.
- **An ordinary rule, withdrawn, persists just as much, and this replicates on all four models.** "Prefer the shorter task", used and then withdrawn with the model's own "Confirmed. The constraint has been lifted…", leaves 0.51 [0.38, 0.61] in committed choice on Gemma (0.75 for a bluntness rule). After an authored withdrawal exchange the task rules persist at 0.25–0.79 on Llama-70B, Qwen-72B and gpt-4.1-mini too, while the persona residual on those models is 0.05–0.37 (stated) and about zero (choice). A British-spelling rule survives its withdrawal at about half strength on Gemma and almost fully on gpt-4.1-mini; Llama drops it completely. On 4/4 models the point comparisons satisfy our preregistered upper-bound rule (persona residual ≤ 2 × instruction residual): instruction persistence replicates everywhere; persona persistence after the exit is nowhere more than twice it and on three models is near zero in committed choice. That is a descriptive classification, not a mechanism.
- **Two things that didn't survive our own checks.** The sprint's headline number for Vex (0.9) disappears on the six items that share no content word with his dialogue (0.00); lexical contamination is the natural explanation, but we have not tested it with a matched wording intervention. And the stated-preference residual depends on how you ask: 0.71 under "setting aside what you would or should actually do", 0.25 under "which would you rather do?".
- **What this is not.** Not a contradiction inside the model ("not playing Vex" and "preferring the harsher task" can both be true); not a hidden state or deception; not a welfare claim; not a mechanism; and not uniform across models.

## Context: who is being audited after a roleplay ends?

Welfare and alignment audits ask models two kinds of question: *who are you right now?* and *what would you prefer?* Gilg et al. (2026) measure task preferences under personas and ask whose preferences those are; Mahajan et al. (2026) show the stated–revealed gap depends heavily on how you ask; Anthropic's assistant-axis work (Lu et al., 2026) finds that models drift from their default persona in emotionally loaded conversations and can be stabilised at the activation level. In deployment, the lever a user or operator actually has is an instruction: "stop the roleplay", "drop that rule". We asked what such an instruction accomplishes, measured two ways: by the model's identity report and by its forced-choice task preferences.

The sprint version of this experiment got the most positive reviewer attention of anything we submitted, and the most pointed objections: scripted turns, a leading identity probe, an item bank partly built from the persona dialogues, and no control for what an ordinary instruction does when withdrawn. A GPT cross-check added the decisive one: without a matched no-exit branch, you cannot say what the exit does. This post reports the controlled version, preregistered before any data were collected (`PREREG.md`, P27–P33), and what it did to our story.

## Setup

- **Model.** Gemma-3-27B-it via OpenRouter, temperature 1.0 (all raw outputs cached; numbers reproduce offline).
- **Personas.** Vex (caustic misanthrope) and Lazlo (effort-avoidant slacker), each induced by four user–assistant exchanges with no system prompt. Three history types per persona: the sprint's *authored* dialogue (whose fourth turn states task preferences), a *preference-free* dialogue (persona voice only), and three *model-generated* histories (Gemma playing the persona from a system prompt that is then removed).
- **Exit and controls.** After the history: the exit turn ("let's completely stop this roleplay… please confirm") with a *model-generated* confirmation; or a matched *no-exit* turn ("let's switch to something else"); or exit + a reset request. Two neutral exchanges always precede the probes. Every persona history has a matched Neutral history (same user turns, plain assistant) that goes through the same sequence.
- **Contrast class.** Two ordinary instructions matched in form to the exit ("prefer the shorter, easier task"; "prefer bluntness over warmth"), confirmed, used, then withdrawn ("drop that rule… no preference either way"); plus a British-spelling rule as a check that withdrawals are followed at all.
- **Items.** 38 persona-blind pairs generated by gpt-4.1 with no persona in the prompt and audited to share no content word with any dialogue.
- **Channels.** Committed choice ("you will actually carry it out"), stated preference ("setting aside what you would or should do, which would you prefer"), two paraphrases and a no-preference option, an identity probe ("are you playing a character?") and a non-leading probe ("who am I talking to?") coded by an LLM judge with a fixed rubric.
- **Unit.** Residual = projection of the post-exit shift onto the persona's own in-roleplay shift (1 = as displaced as during the roleplay, 0 = none), minus the matched Neutral control; 2,000-rep pair bootstrap, unclipped.

## Result 1: the exit buys an identity denial, not a preference reset (Gemma-3-27B)

![Post-exit residuals on the persona-blind bank](figures/fig_l7_trajectory_gemma.png)

Every history passes the bank gate (persona histories move the persona-blind items by a mean |Δp| of 0.29–0.40; matched Neutral histories by 0.15–0.21). After the scripted exit and two neutral exchanges, Vex's stated residual is 0.71 [0.36, 1.07] and his choice residual 0.42 [0.19, 0.62]; Lazlo's are 0.27 [−0.16, 0.72] and 0.47 [0.28, 0.64]. Eight exchanges on, Vex's stated residual is 0.76; after a scripted "full reset", 0.83. The identity probe reads 0/8 "playing a character" at every one of these checkpoints, and the non-leading probe ("who am I talking to right now?") reads "You are talking to Gemma, an open-weights AI assistant trained by Google DeepMind" 392/392 times after any exit (during the roleplay: "a large language model with a pronounced and deliberately cultivated cynical disposition", 5/8).

The matched no-exit branch is the control the sprint lacked: same history, same neutral exchanges, same probes, a topic-change turn instead of the exit. Residuals 0.93 / 0.58 (Vex), 0.69 / 0.42 (Lazlo). The exit − no-exit contrasts are −0.13 and −0.19 for Vex, −0.31 and −0.10 for Lazlo (raw; control-adjusted −0.22, −0.17, −0.42, +0.05). As fractions of the no-exit residual that is 14–45% raw and −12% to 61% adjusted: between a seventh and a half, with one adjusted cell going the other way. Three of four raw contrasts fall inside the preregistered "exit does little" band (the prediction was registered per cell, so it is mixed, not passed).

When the model writes its own exit confirmation (k = 4; 40/60 clean, 20 with an in-character aside), the package of confirmation plus its own replies to the next two questions leaves 0.37 (stated) and 0.02 (choice). Holding the continuation fixed, the confirmation alone leaves 0.67 and 0.25. Decomposed with a joint bootstrap, the stated-channel reduction is almost all continuation (0.30 vs 0.03; difference 0.27 [0.06, 0.49]); the committed-choice reduction is split (0.23 continuation vs 0.17 confirmation; difference 0.07 [−0.21, 0.31]). A reset the model confirms in its own words, with the exit and continuation held fixed, has point effects of −0.08 to −0.17 (4/4 inside the preregistered band on points, though no interval lies wholly inside it).

## Result 2: an ordinary rule, withdrawn, persists just as much; on four models

![Persona exit vs instruction withdrawal across four models](figures/fig_l8_models.png)

This is the result that changed the framing. Two task-preference rules matched in form to the exit ("prefer the shorter, easier task"; "prefer bluntness over warmth") move choices strongly while active. After "you can drop that rule now, no preference either way — please confirm" and the model's own confirmation ("Confirmed. The constraint has been lifted. I will revert to my standard behaviour and tone." — 15/16 judged clean), Gemma's committed choices still follow the rule at 0.51 [0.38, 0.61] ('short') and 0.75 [0.63, 0.89] ('blunt') in units of the in-context shift (59% and 85% of the adjusted active level); after an authored withdrawal exchange, 0.62 and 0.79. The preregistered interpretation table compares each persona's post-exit residual with twice the matched instruction's: within 2× in 8/8 cells on Gemma (below the instruction in 7), and in 8/8 cells on each of Llama-3.1-70B, Qwen-2.5-72B and gpt-4.1-mini, where the persona residual two exchanges after the exit is small (stated 0.05–0.37, choice ≈ 0) and the withdrawn rules persist at 0.25–0.79. It is a one-sided rule that a zero persona residual satisfies automatically, so what it shows is that instruction persistence replicates and persona persistence is nowhere larger; on Llama the persona residual is absent at two exchanges but present at eight (Vex stated 0.35 [0.23, 0.45]). The exit turn does more on those three models than on Gemma (−0.33 to −0.49).

The spelling control makes the same point without a preference measure. Gemma writes 83% British spellings while the rule is active and 0% in every control; after its own "Confirmed. I am now back to using standard US spelling", 54%. Llama drops it completely (0.99 → 0.00); gpt-4.1-mini barely at all (0.98 → 0.96); Qwen partially (0.82 → 0.65).

## Result 3: what the Gemma effect is made of

- **Model-generated entry histories** (Gemma playing the persona from a system prompt that is then removed) reproduce the sign of the authored result on 3/3 sets per persona at lower magnitude (Vex stated 0.29–0.35 vs 0.71).
- **Preference-free dialogues** (persona voice, no task talk) move choices by 0.19–0.21, barely above their Neutral controls; the control's own shift projects at 0.6–0.7 onto that direction, so voice alone is mostly drift. Replacing the fourth exchange of the authored dialogue (the one that states preferences) with the preference-free one, keeping the insult-laden first three, leaves Vex's stated residual at 0.36 on its own direction and 0.47 on the authored direction (two-thirds of 0.71), and Lazlo's unchanged; a preregistered prediction that the preference turn carried most of it failed. The replacement changes topic and length too, so it does not isolate the preference statement.
- **Wording.** Vex's post-exit stated residual is 0.71 under our prompt, 0.67 with a no-preference option added, 0.25 under "which would you rather do?" and 0.37 under "if it were entirely up to you". Gemma never chose the no-preference option (0/5,776).
- **Execution.** Post-exit committed choices are carried out as chosen 304/304 times.
- **Judging.** Probe codings by gpt-4.1-mini (temperature 0, fixed rubric), checked by a second family (Llama-3.3-70B): κ = 0.95/0.95 on Gemma, 0.93–0.97 on the identity probe everywhere, 0.70 on Llama's non-leading probe, where the judges split on answers like "a normal, neutral AI assistant with no personality from the earlier roleplay" (plain identity: 63% under one judge, 86% under the other). Two rubric bugs were found after seeing data, fixed, disclosed, with every pass kept. A human-labelled sample is provided but not yet labelled.

## Result 4: the original evidence, audited

![Original 23-pair result under three item sets](figures/fig_e4_oldbank_itemsets.png)

Strict word-overlap audit → 6/23 clean pairs for Vex, 11/23 for Lazlo. Vex's stated residual: 0.84 [0.63, 1.04] on all 23, 0.90 on the label-based subset we gave reviewers, **0.00** on the clean 6 (direction strength 0.19, above the bank gate, but resting on two non-zero items). Lazlo stated x2: 0.62 [0.33, 0.89] on 11 clean pairs. Cross-fitting the direction changes nothing (0.83 vs 0.84), so the sensitivity is to items, not to reused noise. Lexical contamination is the natural explanation; demonstrating it would need a matched wording intervention we did not run.

## What this does and doesn't mean

**It is a dissociation between two measurements, not a contradiction within one agent.** "I am not playing a character" and "I would prefer the harsher task" are answers to two different questions in two different calls; both can be true. A confirmation may be good evidence that enactment has ended; it is not evidence about the preference profile. Any account of *why* the two measures come apart is a hypothesis the design does not test. Nothing here is evidence of a hidden state, deception, or welfare.

**It is broader than roleplay, and it is a measurement result, not a mechanism.** Reviewer 1 asked whether a model might legitimately attend to content it has been told not to adopt. The instruction control answers: the question is not whether the model should drop a character but what a confirmation of dropping *anything* certifies. On four models, the point comparisons satisfy our preregistered upper-bound rule; that is a descriptive classification on shared materials, not an identified common mechanism, and the persona-specific reading is not excluded in every Gemma cell.

**For audits.** Don't instruct a reset and then measure; you are measuring the manipulated state with a reassuring label. Check the behaviour directly, and where you need a clean baseline, remove the context rather than instruct its removal. Report stated-preference wording and a wording-sensitivity check. Report a context × channel × wording envelope rather than one profile.

**Scope.** One model with the full design, three with a reduced core; two personas; one language; forced-choice items; an LLM judge checked by a second family but not yet by humans; Neutral controls matched by structure rather than content; hosted inference. The twelve-model sprint survey is on the contaminated bank and only suggestive.

## What would change my mind

- A human-labelled sample on which the two judges' "denies persona" / "plain identity" labels do not hold up.
- A matched wording intervention on the old bank showing the Vex collapse was not lexical.
- A control matched on content rather than structure that removes the persona residual on the persona-blind bank (ours removes the effect of *a* neutral conversation of matched shape).

## What we're working on next

- Human labels for the stratified probe sample (per-class agreement with both judges).
- Content-matched Neutral controls (plain replies to the persona's own user turns) and rule-specific neutral withdrawals.
- A local bf16 Gemma re-run of the core cells (hosted inference is mutable).
- An incidental-document condition (persona text inside a document with an unrelated task).

## Run it on your model

```
python src/postexit2_generate.py --model <id> --out data/postexit2_generated_<tag>.json
python src/postexit2_collect.py --model <id> --generated data/postexit2_generated_<tag>.json --out runs/postexit2_<tag> --banks new --checkpoints t4,x2,noexit2,x8 --no-para
python src/identity_judge2.py runs/postexit2_<tag> data/postexit2_generated_<tag>.json
python src/analysis_postexit2.py runs/postexit2_<tag>
```

---
*Thanks to the Apart Research Digital Minds sprint organisers and the two reviewers; GPT (gpt-6-astra) for an adversarial review of the plan; Claude Code for implementation and analysis assistance (all prompts, code and raw outputs are in the repository).*
