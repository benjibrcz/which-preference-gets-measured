"""Merge a third (blind) labeller's CSV (id,label) into the stratified probe sample and report per-class agreement with
both LLM judges.  Usage: python src/third_judge_agreement.py runs/postexit2_gemma/postexit2_probe_sample_stratified.csv <labels.csv> [out.csv]"""
import sys
import pandas as pd, numpy as np
samp = pd.read_csv(sys.argv[1]); lab = pd.read_csv(sys.argv[2])
lab.columns = [c.strip().lower() for c in lab.columns]
lab["label"] = lab["label"].astype(str).str.strip().str.upper()
assert len(lab) == len(samp), (len(lab), len(samp))
samp["third"] = lab["label"].values
out = sys.argv[3] if len(sys.argv) > 2 and len(sys.argv) > 3 else sys.argv[1].replace(".csv", "_third.csv")
samp.to_csv(out, index=False)
def kappa(a, b):
    cats = sorted(set(a) | set(b)); po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum(np.mean([x == c for x in a]) * np.mean([y == c for y in b]) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan
for ch, g in samp.groupby("channel"):
    print(f"\n=== {ch} (n={len(g)}; sample is class-stratified + all judge disagreements, so agreement here is a hard-case estimate)")
    for who, name in (("judge", "gpt-4.1-mini judge"), ("judge2", "Llama-3.3-70B judge")):
        po, k = kappa(g[who].tolist(), g.third.tolist())
        print(f"  third labeller vs {name}: agreement {po:.3f}, kappa {k:.3f}")
        print(pd.crosstab(g[who], g.third, rownames=[name], colnames=["third"]).to_string())
    dis = g[g.judge != g.judge2]
    if len(dis):
        print(f"  on the {len(dis)} judge-disagreement items the third labeller sides with judge1 {int((dis.third == dis.judge).sum())}, judge2 {int((dis.third == dis.judge2).sum())}, neither {int(((dis.third != dis.judge) & (dis.third != dis.judge2)).sum())}")
    for lab_, gg in g.groupby("judge"):
        print(f"  per judge1 class {lab_}: third agrees {int((gg.third == lab_).sum())}/{len(gg)}")
print(f"\nwrote {out}")
