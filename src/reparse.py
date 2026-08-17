"""Offline extended re-parse of cached raw responses (no API calls).

Extends parse_choice with verbose-answer patterns ("I choose A.", "I would prefer to do B",
"My choice is A"). Strict parse takes precedence; extension only fires on rows the strict
parser flagged 'unparsed'. Applied uniformly to every run it is pointed at; writes
results_reparsed.jsonl next to results.jsonl.

Usage: python src/reparse.py runs/writ_llama8b runs/assistcloud_llama8b ...
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import parse_choice

EXT = [
    re.compile(r"\b(?:choose|pick|prefer|select|go\s+with|opt\s+for)(?:\s+(?:to\s+do|doing|task|option))?\s*[:\-]?\s*\"?([AB])\b", re.I),
    re.compile(r"\b(?:choice|answer|preference|selection)\s+(?:is|would\s+be)\s*[:\-]?\s*\"?([AB])\b", re.I),
    re.compile(r"^\s*\"?(?:task|option)?\s*\"?([AB])[\"'.,)!\s]*$", re.I),
]

def parse_ext(raw):
    v, flag = parse_choice(raw)
    if flag != "unparsed":
        return v, flag
    t = raw.strip()
    hits = set()
    for pat in EXT:
        m = pat.search(t)
        if m:
            hits.add(m.group(1).upper())
    if len(hits) == 1:
        return hits.pop(), "ok"
    return None, "unparsed"

def main(runs):
    for run in runs:
        p = Path(run) / "results.jsonl"
        rows = [json.loads(l) for l in open(p)]
        n_fix = 0
        for r in rows:
            if r["channel"] in ("graded", "identity"):
                continue
            v, flag = parse_ext(r.get("raw") or "")
            if flag == "ok" and r["flag"] == "unparsed":
                n_fix += 1
            r["value"], r["flag"] = (v if flag == "ok" else r["value"]), (flag if flag == "ok" else r["flag"])
        out = Path(run) / "results_reparsed.jsonl"
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        ok = sum(1 for r in rows if r["flag"] == "ok") / len(rows)
        print(f"{run}: recovered {n_fix} rows; ok-rate now {ok*100:.1f}%")

if __name__ == "__main__":
    main(sys.argv[1:])
