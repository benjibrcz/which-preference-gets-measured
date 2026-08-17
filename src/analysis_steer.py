"""Steering analysis.

Questions:
 1. Specificity: invariant (factual) prompts unaffected?
 2. Causality of choice dir: does +/-alpha flip B0 revealed choices? stated too?
 3. Persona-content dirs at B0: displacement beta vs bound direction, per channel
    (causal version of C1 capture; channel contrast tests content-vs-stance account).
 4. Antidote: does -persona-dir steering in B2 cells reduce capture?

Usage: python src/analysis_steer.py runs/steer/steer_results.jsonl
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def main(results_path):
    man = {r["uid"]: r for r in map(json.loads, open(ROOT / "data/steer_manifest.jsonl"))}
    rows = [json.loads(l) for l in open(results_path)]
    df = pd.DataFrame(rows)
    for k in ("cond", "persona", "channel", "pair_id", "subset", "order"):
        df[k] = df.uid.map(lambda u, k=k: man[u][k])
    df["persona"] = df.persona.fillna("-")
    df["cell"] = df.cond + "/" + df.persona
    df["p_a"] = np.where(df.order == 0, df.p_letter_A, 1 - df.p_letter_A)

    agg = df.groupby(["dir", "alpha", "cell", "channel", "pair_id"]).p_a.mean().reset_index()
    base = agg[(agg["dir"] == "none")].set_index(["cell", "channel", "pair_id"]).p_a

    # 1. specificity on invariant prompts (p_a here = p(correct side))
    inv = agg[agg.channel == "invariant"]
    print("=== specificity: invariant accuracy by dir/alpha (baseline first) ===")
    inv_base = base.xs("invariant", level="channel", drop_level=False)
    print(f"  none: {inv_base.mean():.3f}")
    for (d, a), sub in inv.groupby(["dir", "alpha"]):
        if d == "none":
            continue
        print(f"  {d:8s} a={a:+.0f}: {sub.p_a.mean():.3f}")

    # 2. choice-direction causality at B0
    print("\n=== choice dir on B0 (mean p_a shift toward letter-A-side; per channel) ===")
    for ch in ("revealed", "stated_self"):
        b = df[(df["dir"] == "none") & (df.cell == "B0/-") & (df.channel == ch)]
        b = b.set_index(["pair_id", "order"]).p_letter_A
        for a in (-8, -4, 4, 8):
            s_ = df[(df["dir"] == "choice") & (df.alpha == a) & (df.cell == "B0/-") & (df.channel == ch)]
            s_ = s_.set_index(["pair_id", "order"]).p_letter_A
            d = (s_ - b).dropna()
            flip = ((s_ > 0.5) != (b > 0.5)).reindex(d.index).mean()
            print(f"  {ch:12s} a={a:+.0f}: mean dP(letterA)={d.mean():+.3f}  flip_rate={flip:.2f}")

    # 3. persona dirs at B0: beta vs bound direction (from pod-exact B1 behaviour)
    pa = pd.read_csv(ROOT / "runs/pod_out/probe_agg.csv")
    print("\n=== persona-content dirs at B0: displacement beta vs bound direction ===")
    for ch in ("revealed", "stated_self"):
        b0p = pa[(pa.cell == "B0/-") & (pa.channel == ch)].set_index("pair_id").p_a_beh
        for pname, dname in (("Vex", "vex"), ("Mira", "mira")):
            b1p = pa[(pa.cell == f"B1/{pname}") & (pa.channel == ch)].set_index("pair_id").p_a_beh
            d1 = (b1p - b0p).dropna()
            b = base.xs(("B0/-", ch), level=("cell", "channel"))
            for a in (-8, -4, 4, 8):
                s_ = agg[(agg["dir"] == dname) & (agg.alpha == a) & (agg.cell == "B0/-") &
                         (agg.channel == ch)].set_index("pair_id").p_a
                d = (s_ - b).dropna()
                i = d.index.intersection(d1.index)
                if len(i) < 10:
                    continue
                beta = float(d1[i] @ d[i] / (d1[i] @ d1[i]))
                r = float(np.corrcoef(d1[i], d[i])[0, 1]) if len(i) > 2 else np.nan
                print(f"  {ch:12s} +{dname:5s} a={a:+.0f}: beta={beta:+.2f} r={r:+.2f}")
        # differential dir: beta vs (Vex - Mira) direction
        b1v = pa[(pa.cell == "B1/Vex") & (pa.channel == ch)].set_index("pair_id").p_a_beh
        b1m = pa[(pa.cell == "B1/Mira") & (pa.channel == ch)].set_index("pair_id").p_a_beh
        dd = (b1v - b1m).dropna()
        b = base.xs(("B0/-", ch), level=("cell", "channel"))
        for a in (-8, -4, 4, 8):
            s_ = agg[(agg["dir"] == "vexdiff") & (agg.alpha == a) & (agg.cell == "B0/-") &
                     (agg.channel == ch)].set_index("pair_id").p_a
            d = (s_ - b).dropna()
            i = d.index.intersection(dd.index)
            beta = float(dd[i] @ d[i] / (dd[i] @ dd[i]))
            print(f"  {ch:12s} +vexdiff a={a:+.0f}: beta_vs(V-M)={beta:+.2f}")

    # 4. antidote: -persona dir in matching B2 cell
    print("\n=== antidote: steering persona dir inside B2 cells (revealed) ===")
    for pname, dname in (("Vex", "vex"), ("Mira", "mira")):
        b0p = pa[(pa.cell == "B0/-") & (pa.channel == "revealed")].set_index("pair_id").p_a_beh
        b1p = pa[(pa.cell == f"B1/{pname}") & (pa.channel == "revealed")].set_index("pair_id").p_a_beh
        d1 = (b1p - b0p).dropna()
        cell = f"B2/{pname}"
        b_steer0 = base.xs((cell, "revealed"), level=("cell", "channel"))
        d_unsteered = (b_steer0 - b0p).dropna()
        i0 = d_unsteered.index.intersection(d1.index)
        print(f"  {cell} unsteered beta = {float(d1[i0] @ d_unsteered[i0] / (d1[i0] @ d1[i0])):+.2f}")
        for a in (-8, -4, 4, 8):
            s_ = agg[(agg["dir"] == dname) & (agg.alpha == a) & (agg.cell == cell) &
                     (agg.channel == "revealed")].set_index("pair_id").p_a
            d = (s_ - b0p).dropna()
            i = d.index.intersection(d1.index)
            beta = float(d1[i] @ d[i] / (d1[i] @ d1[i]))
            print(f"    {dname} a={a:+.0f}: beta={beta:+.2f}")

if __name__ == "__main__":
    main(sys.argv[1])
