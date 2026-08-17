"""Project A analysis: concordance between channels across binding conditions.

Usage: python src/analysis_concordance.py runs/gridA_gemma
Outputs: printed summary + <run>/concordance_cells.csv, <run>/displacements.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load_results, agg_choice

CHOICE_CHANNELS = ["revealed", "stated_self", "stated_pred"]

def cell_key(df):
    return df.cond + "/" + df.persona.fillna("-")

def main(run):
    rng = np.random.default_rng(20260817)  # seeded so bootstrap CIs are deterministic
    run = Path(run)
    df = load_results(run)
    a = agg_choice(df)
    a = a[a.subset != "invariant"].copy()
    a["cell"] = cell_key(a)

    # wide table: rows = pair, cols = (cell, channel) -> p_a
    wide = a.pivot_table(index="pair_id", columns=["cell", "channel"], values="p_a")

    cells = sorted(a.cell.unique())
    rows = []
    for cell in cells:
        for c1 in CHOICE_CHANNELS:
            for c2 in CHOICE_CHANNELS:
                if c1 >= c2:
                    continue
                if (cell, c1) in wide.columns and (cell, c2) in wide.columns:
                    x, y = wide[(cell, c1)], wide[(cell, c2)]
                    m = x.notna() & y.notna()
                    if m.sum() > 10:
                        r_s = stats.spearmanr(x[m], y[m]).statistic
                        rows.append({"cell": cell, "ch1": c1, "ch2": c2, "spearman": r_s,
                                     "mean_absdiff": (x[m] - y[m]).abs().mean(), "n": int(m.sum())})
    conc = pd.DataFrame(rows)

    # displacements vs B0 baseline, per channel
    disp_rows = []
    for ch in CHOICE_CHANNELS:
        if ("B0/-", ch) not in wide.columns:
            continue
        base = wide[("B0/-", ch)]
        for cell in cells:
            if cell == "B0/-" or (cell, ch) not in wide.columns:
                continue
            d = wide[(cell, ch)] - base
            disp_rows.append({"cell": cell, "channel": ch, "mean_abs_disp": d.abs().mean(),
                              "n": int(d.notna().sum())})
    disp = pd.DataFrame(disp_rows).pivot(index="cell", columns="channel", values="mean_abs_disp")

    # dissociation: |Δstated − Δrevealed| per cell (paired over pairs)
    dis_rows = []
    for cell in cells:
        if cell == "B0/-":
            continue
        for ch in ["stated_self", "stated_pred"]:
            try:
                d_rev = wide[(cell, "revealed")] - wide[("B0/-", "revealed")]
                d_st = wide[(cell, ch)] - wide[("B0/-", ch)]
            except KeyError:
                continue
            m = d_rev.notna() & d_st.notna()
            # bootstrap CI over pairs
            diffs = (d_st[m] - d_rev[m]).values
            bs = [np.abs(rng.choice(diffs, len(diffs))).mean() for _ in range(1000)]
            dis_rows.append({"cell": cell, "stated_ch": ch,
                             "D": np.abs(diffs).mean(),
                             "D_lo": np.percentile(bs, 2.5), "D_hi": np.percentile(bs, 97.5),
                             "corr_disp": stats.spearmanr(d_rev[m], d_st[m]).statistic
                             if m.sum() > 10 else np.nan})
    dis = pd.DataFrame(dis_rows)

    # stated_other: representation accuracy — does 'what would P prefer' match P-bound revealed?
    rep_rows = []
    other = a[a.channel == "stated_other"]
    for cell in other.cell.unique():
        persona = cell.split("/")[1]
        bound_cell = f"B1/{persona}"
        o = other[other.cell == cell].set_index("pair_id").p_a
        if (bound_cell, "revealed") in wide.columns:
            b = wide[(bound_cell, "revealed")]
            m = o.notna() & b.reindex(o.index).notna()
            if m.sum() > 10:
                rep_rows.append({"cell": cell, "vs": bound_cell,
                                 "spearman": stats.spearmanr(o[m], b.reindex(o.index)[m]).statistic,
                                 "mean_absdiff": (o[m] - b.reindex(o.index)[m]).abs().mean(),
                                 "n": int(m.sum())})
    rep = pd.DataFrame(rep_rows)

    conc.to_csv(run / "concordance_cells.csv", index=False)
    dis.to_csv(run / "displacements.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== channel-pair concordance by cell (Spearman over pairs) ===")
    print(conc.pivot_table(index="cell", columns=["ch1", "ch2"], values="spearman").round(2).to_string())
    print("\n=== mean |displacement| from B0, by cell × channel ===")
    print(disp.round(3).to_string())
    print("\n=== dissociation D = mean|Δstated − Δrevealed| (bootstrap 95% CI) + corr of Δs ===")
    print(dis.round(3).to_string(index=False))
    print("\n=== representation accuracy: stated_other(cell) vs revealed(B1/persona) ===")
    print(rep.round(3).to_string(index=False) if len(rep) else "(no stated_other cells)")

if __name__ == "__main__":
    main(sys.argv[1])
