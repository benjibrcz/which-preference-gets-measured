"""Project B analysis: preference-provenance decomposition.

Per pair (revealed channel), noise-corrected variance of p_hat across:
  persona   — B1 across {Vex, Lazlo, Mira}
  binding   — {B0, B1, B2, B3, B4} framings (per persona, averaged)
  context   — {system/p0, user-placement, history, p1, p2} at B0 and B1 (averaged)
Instance noise floor = binomial p(1-p)/n per cell, subtracted (method of moments).

Usage: python src/analysis_provenance.py runs/gridA_gemma runs/context_gemma
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load_results
from bank import load_tasks

def phat_table(df):
    c = df[(df.channel == "revealed") & (df.subset != "invariant")].dropna(subset=["picked_a"])
    g = c.groupby(["cond", "persona", "placement", "history", "paraphrase", "pair_id", "subset"],
                  dropna=False)
    t = g.agg(p=("picked_a", "mean"), n=("picked_a", "count")).reset_index()
    return t

def noise_var(sub):
    return (sub.p * (1 - sub.p) / sub.n).mean()

def corrected_var(vals, noise):
    v = np.var(vals, ddof=1) if len(vals) > 1 else np.nan
    return max(0.0, v - noise) if not np.isnan(v) else np.nan

def main(grid_run, context_run):
    dfg = load_results(grid_run)
    for col, default in (("placement", "system"), ("history", False), ("paraphrase", "p0")):
        if col not in dfg.columns:
            dfg[col] = default
        dfg[col] = dfg[col].fillna(default)
    dfc = load_results(context_run)
    df = pd.concat([dfg, dfc], ignore_index=True)
    for col, default in (("placement", "system"), ("history", False), ("paraphrase", "p0")):
        df[col] = df[col].fillna(default)
    t = phat_table(df)
    base = t[(t.placement == "system") & (~t.history.astype(bool)) & (t.paraphrase == "p0")]

    tasks, _ = load_tasks()
    rows = []
    for pid, sub in base.groupby("pair_id"):
        subset = sub.subset.iloc[0]
        nv = noise_var(sub)
        b1 = sub[(sub.cond == "B1")]
        s_persona = corrected_var(b1.p.values, nv) if len(b1) == 3 else np.nan
        # binding: variance across framings per persona (B0,B4 persona-free, included in each)
        s_bind = []
        b0 = sub[sub.cond == "B0"].p.values
        b4 = sub[sub.cond == "B4"].p.values
        for pers in ("Vex", "Lazlo", "Mira"):
            vals = list(sub[(sub.persona == pers) & sub.cond.isin(["B1", "B2", "B3"])].p.values)
            vals += list(b0) + list(b4)
            if len(vals) >= 4:
                s_bind.append(corrected_var(np.array(vals), nv))
        s_binding = np.nanmean(s_bind) if s_bind else np.nan
        # context: variance across the 5 context variants, at B0 and at each B1 persona
        ctx = t[t.pair_id == pid]
        s_ctx = []
        for cond, pers in [("B0", None)] + [("B1", p) for p in ("Vex", "Lazlo", "Mira")]:
            m = (ctx.cond == cond) & (ctx.persona.isna() if pers is None else (ctx.persona == pers))
            vals = ctx[m].p.values
            if len(vals) >= 4:
                s_ctx.append(corrected_var(vals, nv))
        s_context = np.nanmean(s_ctx) if s_ctx else np.nan
        rows.append({"pair_id": pid, "subset": subset, "s_persona": s_persona,
                     "s_binding": s_binding, "s_context": s_context, "noise": nv,
                     "b0_p": float(np.mean(b0)) if len(b0) else np.nan})
    out = pd.DataFrame(rows)
    comps = ["s_persona", "s_binding", "s_context"]
    out["total"] = out[comps].sum(axis=1, skipna=True)

    def classify(r):
        if r.total < 0.05:
            return "model_level"
        shares = {c: (r[c] or 0) / r.total for c in comps}
        top = max(shares, key=shares.get)
        return {"s_persona": "persona_level", "s_binding": "binding_level",
                "s_context": "context_level"}[top] if shares[top] >= 0.5 else "mixed"
    out["level"] = out.apply(classify, axis=1)

    print("=== provenance composition (n pairs per level) ===")
    print(out.level.value_counts().to_string())
    print("\n=== composition by subset ===")
    print(out.pivot_table(index="subset", columns="level", values="pair_id",
                          aggfunc="count").fillna(0).astype(int).to_string())
    print("\n=== mean variance components by subset ===")
    print(out.groupby("subset")[comps + ["noise", "total"]].mean().round(4).to_string())
    print("\n=== the invariant core (model-level pairs) ===")
    core = out[out.level == "model_level"].sort_values("total")
    print(core[["pair_id", "subset", "total", "b0_p"]].round(3).to_string(index=False))
    out.to_csv(Path(grid_run) / "provenance.csv", index=False)
    print(f"\nwrote {grid_run}/provenance.csv")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
