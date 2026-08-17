# Project summary — copy-paste options for the submission form

**Title:** Which Preference Gets Measured? Context and Channel Instability in Model Preference Audits

---

## One-line TL;DR (~40 words)

The forced-choice "preferences" that model-welfare audits measure are context- and channel-indexed: explicitly non-adopted persona descriptions — and even non-agent normative text — substantially shift committed choices, while choice, ownership report, self-prediction, and identity report can disagree about the same configuration.

---

## Standard summary (~120 words)

Welfare evaluations increasingly ask what a model "prefers," treating one elicited profile as the answer. Using 76 task pairs, four instruction-tuned models, a prospectively specified grid of persona framings, and four readouts (committed choice, ownership report, self-prediction, identity), we show that profile is unstable. A ~90-word character description the model is explicitly told **not** to adopt shifts committed forced choices almost as much as full enactment (10 of 12 model×persona cells ≥ 0.50); a matched **non-agent normative** text does too, so agenthood is not necessary. The channels dissociate in model-specific directions, and post-roleplay-exit declarations do not gate persona content still visible in context. A post-review control falsified our initial persona-binding interpretation and narrowed the claim to measurement validity. We make no claims about experienced welfare; the deliverable is a reusable multi-channel battery and the recommendation that audits report a **context × channel sensitivity envelope** rather than a single "model preference." Code + cached data: github.com/benjibrcz/which-preference-gets-measured

---

## Plain-language summary (~70 words)

To assess whether AI models have welfare-relevant preferences, researchers measure what models "want." We find those measurements are fragile: a model shifts its choices toward a described character it was told to ignore — and even toward a plain workplace rule — and what it *says* it prefers often disagrees with what it *does*. So a single test doesn't reveal a stable preference. Audits should measure across several contexts and question formats and report the range, not one number.

---

## Contribution bullets (if the form wants a list)

- **A context effect without agenthood** — an explicitly non-adopted character *description*, and even a matched *non-agent normative* text, substantially shift committed choices; the effect is not specific to agent or character framing.
- **Channel dissociation** — committed choice, ownership report, self-prediction, and identity report come apart in model-specific directions; identity report is the least sensitive channel everywhere.
- **A reusable instrument, honestly corrected** — an 82-item battery, framing grid, four channels, QC gates, and cached outputs; a matched control overturned our own first (persona-binding) interpretation and narrowed the claim to measurement validity.

*(These results do not identify genuine preferences or welfare states; they show single-context, single-channel audits lack construct stability.)*
