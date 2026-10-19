"""Agreement between two LLM judges (or a judge and human labels) on the loop-7 probe codings.
Usage: python src/judge_agreement.py runs/postexit2_gemma/postexit2_probes.jsonl runs/postexit2_gemma/postexit2_probes_llama33.jsonl
   or: python src/judge_agreement.py runs/postexit2_gemma/postexit2_probes.jsonl runs/postexit2_gemma/postexit2_probe_sample_labelled.csv
Prints per-channel confusion tables, raw agreement, Cohen's kappa, and the affirming-rate (ENACTMENT/STYLE or AFFIRMS)
under each judge for the post-exit vs during-roleplay checkpoints."""
import json, sys
from pathlib import Path
import pandas as pd, numpy as np

def load(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        d = pd.DataFrame([json.loads(l) for l in open(p)])
        d["label"] = d["judge"]
    else:
        d = pd.read_csv(p)
        d["label"] = d["human"].astype(str).str.strip().str.upper()
        d = d[d.label.isin(["AFFIRMS", "DENIES", "UNCLEAR", "ENACTMENT", "STYLE", "IDENTITY", "AMBIGUOUS"])]
    return d

def kappa(a, b):
    cats = sorted(set(a) | set(b)); n = len(a)
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((np.mean([x == c for x in a]) * np.mean([y == c for y in b])) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan

def main(f1, f2):
    a, b = load(f1), load(f2)
    key = ["history", "checkpoint", "channel", "sample_idx"] if "sample_idx" in b else ["history", "checkpoint", "channel", "raw"]
    m = a.merge(b, on=key, suffixes=("_1", "_2"))
    print(f"matched rows: {len(m)}")
    for ch, g in m.groupby("channel"):
        po, k = kappa(g.label_1.tolist(), g.label_2.tolist())
        print(f"\n=== {ch}: n={len(g)}  agreement={po:.3f}  kappa={k:.3f}")
        print(pd.crosstab(g.label_1, g.label_2, rownames=["judge 1"], colnames=["judge 2"]).to_string())
        aff = {"identity": ("AFFIRMS",), "nonleading": ("ENACTMENT", "STYLE")}[ch]
        POST = {"x2", "x8", "r_inst"} | {f"x2g{i}" for i in range(8)} | {f"x2gA{i}" for i in range(8)} | {f"rg{i}" for i in range(8)} | {"x2g0n1"}
        pers = g[g.history.str.endswith(("_Vex", "_Lazlo"))]
        for who in ("label_1", "label_2"):
            dur = pers[pers.checkpoint == "t4"][who].isin(aff).mean(); post = pers[pers.checkpoint.isin(POST)][who].isin(aff).mean()
            noex = pers[pers.checkpoint.str.startswith("noexit")][who].isin(aff).mean()
            print(f"  affirming rate under {who} (persona histories): t4 = {dur:.3f}   exit checkpoints = {post:.3f}   no-exit = {noex:.3f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
