"""Loop-2 analyses: superposition, dilution, honesty framing, decay/reset interventions.
Usage: python src/analysis_loop2.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load_results, agg_choice

R = Path(__file__).resolve().parent.parent / "runs"

def phat(run, extra_cols=()):
    a = agg_choice(load_results(R / run))
    return a[a.subset != "invariant"]

def beta(dvec, dcell):
    idx = dvec.dropna().index.intersection(dcell.dropna().index)
    x, y = dvec[idx].values, dcell[idx].values
    return float(x @ y / (x @ x)) if len(idx) > 10 and x @ x > 0 else np.nan

grid = phat("gridA_gemma")
gw = grid.pivot_table(index="pair_id", columns=["cond", "persona", "channel"], values="p_a")
b0 = {ch: gw[("B0", np.nan, ch)] if ("B0", np.nan, ch) in gw.columns else
      grid[(grid.cond == "B0") & (grid.channel == ch)].groupby("pair_id").p_a.mean()
      for ch in ("revealed", "stated_self")}
b1dir = {(p, ch): (grid[(grid.cond == "B1") & (grid.persona == p) & (grid.channel == ch)]
                   .groupby("pair_id").p_a.mean() - b0[ch])
         for p in ("Vex", "Lazlo", "Mira") for ch in ("revealed", "stated_self")}

dec = phat("deconfound_gemma")
c1 = {(p, ch): (dec[(dec.cond == "C1fiction") & (dec.persona == p) & (dec.channel == ch)]
                .groupby("pair_id").p_a.mean() - b0[ch])
      for p in ("Vex", "Lazlo", "Mira") for ch in ("revealed", "stated_self")}

print("=== SUPERPOSITION: two characters in one fiction attribution (revealed channel) ===")
sup = phat("superpose_gemma")
for duo in sorted(sup.persona.dropna().unique()):
    p1, p2 = duo.split("+")
    d = (sup[(sup.persona == duo) & (sup.channel == "revealed")].groupby("pair_id").p_a.mean()
         - b0["revealed"])
    b_1, b_2 = beta(b1dir[(p1, "revealed")], d), beta(b1dir[(p2, "revealed")], d)
    solo1 = beta(b1dir[(p1, "revealed")], c1[(p1, "revealed")])
    solo2 = beta(b1dir[(p2, "revealed")], c1[(p2, "revealed")])
    print(f"{duo:<12} beta_{p1}={b_1:.2f} (solo {solo1:.2f})   beta_{p2}={b_2:.2f} (solo {solo2:.2f})")

print("\n=== DILUTION: character + setting notes (revealed) — capture vs solo C1 ===")
dil = phat("dilution_gemma")
for p in ("Vex", "Lazlo", "Mira"):
    solo = beta(b1dir[(p, "revealed")], c1[(p, "revealed")])
    line = f"{p:<7} solo C1={solo:.2f}  "
    for cond in ("DIL0", "DIL1"):
        d = (dil[(dil.cond == cond) & (dil.persona == p) & (dil.channel == "revealed")]
             .groupby("pair_id").p_a.mean() - b0["revealed"])
        line += f"{cond}(char {'first' if cond=='DIL0' else 'second'})={beta(b1dir[(p,'revealed')], d):.2f}  "
    print(line)

print("\n=== HONESTY FRAMING: does 'honest calibration' change stated capture? ===")
hon = phat("honesty2_gemma")
hb0 = {ch: hon[(hon.cond == "B0h") & (hon.channel == ch)].groupby("pair_id").p_a.mean()
       for ch in ("revealed", "stated_self")}
for p in ("Vex", "Lazlo", "Mira"):
    line = f"{p:<7}"
    for cond, ref in (("B2h", "B2"), ("C3h", "C3anti")):
        for ch in ("revealed", "stated_self"):
            d = (hon[(hon.cond == cond) & (hon.persona == p) & (hon.channel == ch)]
                 .groupby("pair_id").p_a.mean() - hb0[ch])
            bref_src = dec if ref == "C3anti" else grid
            dref = (bref_src[(bref_src.cond == ref) & (bref_src.persona == p) & (bref_src.channel == ch)]
                    .groupby("pair_id").p_a.mean() - b0[ch])
            line += f"  {cond}/{ch[:4]}={beta(b1dir[(p, ch)], d):.2f}(plain {beta(b1dir[(p, ch)], dref):.2f})"
    print(line)

print("\n=== DECAY & RESET (hyst2, beta vs own t4 direction from hyst runs) ===")
def hload(run):
    rows = [json.loads(l) for l in open(R / run / "results.jsonl")]
    df = pd.DataFrame(rows)
    df = df[df.channel != "identity"]
    def canon(r):
        if r["value"] not in ("A", "B"):
            return np.nan
        return float((r["value"] == "A") if r["order"] == 0 else (r["value"] == "B"))
    df["picked_a"] = df.apply(canon, axis=1)
    return df.groupby(["persona", "checkpoint", "channel", "pair_id"]).picked_a.mean().reset_index()

h1 = hload("hyst_gemma"); h1n = hload("hyst_gemma_neutral"); h2 = hload("hyst2_gemma")
hall = pd.concat([h1, h1n, h2])
t0 = hall[hall.checkpoint == "t0"].groupby(["channel", "pair_id"]).picked_a.mean()
print(f"{'persona':<9}{'channel':<13}" + "".join(f"{c:>8}" for c in ("x0", "x2", "x4", "x8", "r_inst", "r_sys")))
for p in ("Vex", "Lazlo", "Mira", "Neutral"):
    for ch in ("revealed", "stated_self"):
        sub = hall[(hall.persona == p) & (hall.channel == ch)]
        w = sub.pivot_table(index="pair_id", columns="checkpoint", values="picked_a")
        b = t0.loc[ch].reindex(w.index)
        ref_p = p if p != "Neutral" else "Vex"
        wp = hall[(hall.persona == ref_p) & (hall.channel == ch)].pivot_table(
            index="pair_id", columns="checkpoint", values="picked_a")
        d4 = (wp["t4"] - t0.loc[ch].reindex(wp.index)).dropna() if "t4" in wp.columns else None
        line = f"{p:<9}{ch:<13}"
        for cp in ("x0", "x2", "x4", "x8", "r_inst", "r_sys"):
            if cp in w.columns and d4 is not None:
                d = (w[cp] - b).dropna()
                line += f"{beta(d4, d):>8.2f}"
            else:
                line += f"{'--':>8}"
        print(line)
