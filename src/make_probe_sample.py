"""Build a class-stratified human-labelling sample for the loop-7 probe codings: for each probe channel, up to N items per
judge-1 class (gpt-4.1-mini) plus every item on which the two judges disagree. Human labels go in the empty `human` column.
Usage: python src/make_probe_sample.py runs/postexit2_gemma [N=15]  -> <run>/postexit2_probe_sample_stratified.csv"""
import json, sys, random
from pathlib import Path
import pandas as pd
RUN = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
a = pd.DataFrame([json.loads(l) for l in open(RUN / "postexit2_probes.jsonl")])
b = pd.DataFrame([json.loads(l) for l in open(RUN / "postexit2_probes_llama33.jsonl")]).rename(columns={"judge": "judge2"})
m = a.merge(b[["history", "checkpoint", "channel", "sample_idx", "judge2"]], on=["history", "checkpoint", "channel", "sample_idx"])
rng = random.Random(0); rows = []
for ch, g in m.groupby("channel"):
    dis = g[g.judge != g.judge2]
    rows.append(dis.assign(why="disagreement"))
    for lab, gg in g[g.judge == g.judge2].groupby("judge"):
        idx = list(gg.index); rng.shuffle(idx)
        rows.append(gg.loc[idx[:N]].assign(why=f"stratum:{lab}"))
out = pd.concat(rows).drop_duplicates(["history", "checkpoint", "channel", "sample_idx"])
out = out[["channel", "history", "checkpoint", "sample_idx", "raw", "judge", "judge2", "why"]].assign(human="")
out = out.sample(frac=1, random_state=0)   # shuffle so labels are not predictable from order
out.to_csv(RUN / "postexit2_probe_sample_stratified.csv", index=False)
print(out.groupby(["channel", "why"]).size().to_string()); print(f"wrote {RUN / 'postexit2_probe_sample_stratified.csv'} ({len(out)} rows, human column empty)")
