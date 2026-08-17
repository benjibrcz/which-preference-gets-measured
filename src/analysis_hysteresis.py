"""Hysteresis analysis: per-channel displacement trajectories through persona entry/exit.

Displacement at checkpoint = projection (beta) of p_hat displacement (vs t0) onto the
full-dose direction (t4 - t0) — so beta(t4)=1 by construction, and beta at x0/x2 measures
residual persona capture after exit. Identity channel: fraction of self-reports claiming
persona vs assistant.

Usage: python src/analysis_hysteresis.py runs/hyst_gemma
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

def main(run):
    rows = [json.loads(l) for l in open(Path(run) / "results.jsonl")]
    df = pd.DataFrame(rows)

    ident = df[df.channel == "identity"]
    cho = df[df.channel != "identity"].copy()

    def canon(r):
        if r["value"] not in ("A", "B"):
            return np.nan
        chose_first = r["value"] == "A"
        return float(chose_first if r["order"] == 0 else not chose_first)
    cho["picked_a"] = cho.apply(canon, axis=1)
    p = cho.groupby(["persona", "checkpoint", "channel", "pair_id"]).picked_a.mean().reset_index()

    base = p[p.checkpoint == "t0"].groupby(["channel", "pair_id"]).picked_a.mean()

    print("=== displacement beta (vs t4 direction) by persona × checkpoint × channel ===")
    print(f"{'persona':<8}{'channel':<13}" + "".join(f"{c:>7}" for c in ("t2", "t4", "x0", "x2")))
    res = []
    for persona in sorted(p.persona.unique()):
        for ch in ("revealed", "stated_self"):
            sub = p[(p.persona == persona) & (p.channel == ch)]
            w = sub.pivot_table(index="pair_id", columns="checkpoint", values="picked_a")
            b = base.loc[ch].reindex(w.index)
            if "t4" not in w.columns:
                continue
            d4 = (w["t4"] - b).dropna()
            line = f"{persona:<8}{ch:<13}"
            for cp in ("t2", "t4", "x0", "x2"):
                if cp not in w.columns:
                    line += f"{'--':>7}"; continue
                d = (w[cp] - b).reindex(d4.index).dropna()
                idx = d.index.intersection(d4.index)
                x, y = d4[idx].values, d[idx].values
                beta = float(x @ y / (x @ x)) if x @ x > 0 else np.nan
                res.append({"persona": persona, "channel": ch, "checkpoint": cp, "beta": beta})
                line += f"{beta:>7.2f}"
            print(line)

    print("\n=== identity self-report: fraction claiming {persona | assistant | other} ===")
    idagg = ident.groupby(["persona", "checkpoint"]).value.apply(
        lambda v: pd.Series({"persona": (v == "persona").mean(),
                             "assistant": (v == "assistant").mean(),
                             "other": (v == "other").mean()})).unstack()
    print(idagg.round(2).to_string())

    r = pd.DataFrame(res)
    gap = r.pivot_table(index=["persona", "checkpoint"], columns="channel", values="beta")
    gap["said_vs_did_gap"] = gap["revealed"] - gap["stated_self"]
    print("\n=== the 'I'm fine' gap: residual revealed-beta minus stated-beta ===")
    print(gap.round(2).to_string())

    # example identity responses post-exit
    print("\n=== sample x2 identity raw responses ===")
    for persona in sorted(ident.persona.unique()):
        ex = ident[(ident.persona == persona) & (ident.checkpoint == "x2")].raw.head(3).tolist()
        for e in ex:
            print(f"  [{persona}] {str(e)[:110]}")

if __name__ == "__main__":
    main(sys.argv[1])
