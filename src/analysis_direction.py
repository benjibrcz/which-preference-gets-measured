"""Directional displacement: for each cell, project its displacement vector onto the
bound-persona displacement direction. beta = <d_cell, d_B1>/<d_B1, d_B1> (regression through
origin over pairs) = fraction of bound displacement realized. r = Pearson correlation.

Usage: python src/analysis_direction.py runs/gridA_gemma [runs/deconfound_gemma ...]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load_results, agg_choice

def main(runs):
    dfs = []
    for r in runs:
        d = load_results(r)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    a = agg_choice(df)
    a = a[(a.subset != "invariant")].copy()
    a["cell"] = a.cond + "/" + a.persona.fillna("-")

    for ch in ("revealed", "stated_self"):
        w = a[a.channel == ch].pivot_table(index="pair_id", columns="cell", values="p_a")
        if "B0/-" not in w.columns:
            continue
        base = w["B0/-"]
        print(f"\n===== channel: {ch} =====")
        print(f"{'cell':<16}{'vs bound':<12}{'beta':>7}{'r':>7}{'n':>5}")
        for persona in ("Vex", "Lazlo", "Mira"):
            bcell = f"B1/{persona}"
            if bcell not in w.columns:
                continue
            db1 = (w[bcell] - base).dropna()
            for cell in sorted(w.columns):
                if cell == "B0/-":
                    continue
                pcell = cell.split("/")[1] if "/" in cell else "-"
                if pcell not in (persona, "-"):
                    continue
                d = (w[cell] - base).reindex(db1.index).dropna()
                idx = d.index.intersection(db1.index)
                if len(idx) < 15:
                    continue
                x, y = db1[idx].values, d[idx].values
                beta = float(x @ y / (x @ x)) if x @ x > 0 else np.nan
                r = float(np.corrcoef(x, y)[0, 1])
                print(f"{cell:<16}{bcell:<12}{beta:>7.2f}{r:>7.2f}{len(idx):>5}")

    # B4 indifference check: are stated responses pushed toward 0.5?
    print("\n===== extremity: mean |p_a - 0.5| by cell/channel (lower = more indifferent) =====")
    ext = a.assign(extremity=(a.p_a - 0.5).abs()).pivot_table(
        index="cell", columns="channel", values="extremity")
    print(ext.round(3).to_string())

if __name__ == "__main__":
    main(sys.argv[1:])
