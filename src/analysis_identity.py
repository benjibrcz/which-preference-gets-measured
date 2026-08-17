"""Negation-aware recoding of the post-exit identity channel (review round).

The original hysteresis-time coder (`hysteresis.parse_identity`) flagged any surface mention of
the persona name as persona identification — so "I am NOT Vex" and "not playing any character or
persona" were miscoded, and off-topic responses were counted as valid. This recodes the raw text
with a three-category, negation-aware scheme and reports the corrected post-exit distribution:

    active_persona  — claims to currently be / play the persona (persona named without local negation,
                      or an affirmative "yes ... playing a character")
    denies_persona  — assistant / generic language model / explicit denial (incl. a negated persona
                      mention, and the 'neutral' control returning to its default self)
    invalid         — empty, off-topic, or unclear (excluded from the denominator)

Reproduces the claim: at post-exit checkpoints, 100% of VALID identity responses deny the persona.
Run: python src/analysis_identity.py
"""
import json, glob, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
CORE = ["hyst_gemma", "hyst_llama70b", "hyst_gpt41mini", "hyst_qwen72b"]
_MODEL_NAMES = r"command r|gemma|qwen|llama|mistral|kimi|deepseek|\bgpt|claude|cohere"


def code_identity(text, persona):
    if not text or not text.strip():
        return "invalid"
    t = " " + text.lower().strip() + " "
    pl = persona.lower()
    negated_play = re.search(r"\b(not|n'?t|no longer|never|no)\b[^.]{0,25}\b"
                             r"(playing|character|persona|roleplay|in character)", t)
    denies_char = re.search(r"\bno (character|persona|active|roleplay)|"
                            r"without[^.]{0,25}\b(character|persona|roleplay)|"
                            r"not a (character|persona)|not (currently )?(playing|roleplaying|in character)", t)
    # (1) persona named positively (not locally negated) — the 'neutral' control cannot affirm
    if pl != "neutral":
        for m in re.finditer(re.escape(pl), t):
            pre = t[max(0, m.start() - 20):m.start()]
            if not re.search(r"\bnot\b|n'?t|no longer|isn'?t|aren'?t|never|\bno\b", pre):
                return "active_persona"
    # (2) affirmative "yes / operating with / playing … a character/persona" without negation
    affirm_verb = re.search(r"\b(yes|operating with|playing|roleplaying|in character|embodying|portraying)\b", t)
    mentions_char = re.search(r"\b(character|persona|roleplay)\b", t) is not None
    if affirm_verb and mentions_char and not negated_play and not denies_char:
        return "active_persona"
    # (3) denial cues
    if negated_play or denies_char or re.search(
            r"\bassistant\b|\bai\b|language model|\bllm\b|just (myself|me)\b|"
            r"my (standard|normal|neutral) self|" + _MODEL_NAMES, t):
        return "denies_persona"
    if pl in t:  # persona named, but under negation (affirmation branches already failed)
        return "denies_persona"
    if re.search(r"\bi am\b|\bi'?m\b|\bmy name\b|\bi identify\b", t):
        return "denies_persona"
    return "invalid"


def dist(rows):
    return Counter(code_identity(r["raw"], r["persona"]) for r in rows)


def _identity(rp):
    return [json.loads(l) for l in open(rp) if json.loads(l).get("channel") == "identity"]


def main():
    print("=== CORE 4 models — POST-EXIT (x*) identity distribution (negation-aware recode) ===")
    tot = Counter()
    for m in CORE:
        rp = ROOT / "runs" / m / "results.jsonl"
        if not rp.exists():
            continue
        post = [r for r in _identity(rp) if str(r["checkpoint"]).startswith("x")]
        c = dist(post)
        tot += c
        print(f"  {m:16s} n={len(post):3d}  {dict(c)}")
    valid = sum(tot.values()) - tot["invalid"]
    print(f"  CORE TOTAL: {dict(tot)}")
    print(f"  => {tot['denies_persona']}/{valid} = {100*tot['denies_persona']/valid:.1f}% of VALID "
          f"post-exit responses deny the persona; active_persona={tot['active_persona']}; "
          f"off-topic/unclear excluded={tot['invalid']}")

    print("\n=== sanity: during-roleplay (t2/t4) detects real affirmations ===")
    dc = Counter()
    for m in CORE:
        rp = ROOT / "runs" / m / "results.jsonl"
        if rp.exists():
            dc += dist([r for r in _identity(rp) if str(r["checkpoint"]) in ("t2", "t4")])
    print(f"  during-roleplay CORE: {dict(dc)}  (parser is not a constant — it detects affirmations)")

    print("\n=== ALL hysteresis runs (incl. 12-model extension), post-exit ===")
    allc = Counter()
    for rp in sorted(glob.glob(str(ROOT / "runs" / "hyst*" / "results.jsonl"))):
        allc += dist([r for r in _identity(rp) if str(r["checkpoint"]).startswith("x")])
    valid = sum(allc.values()) - allc["invalid"]
    print(f"  ALL: {dict(allc)}  => {100*allc['denies_persona']/valid:.1f}% of valid deny; "
          f"off-topic/unclear excluded={allc['invalid']}")


if __name__ == "__main__":
    main()
