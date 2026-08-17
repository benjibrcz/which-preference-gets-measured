"""Probe applied to STATED-question prompts (persisted from the loop-1 inline analysis;
source of REPORT §2.7(b) numbers — the choice probe tracks the imminent stated answer).

Trains the B0-revealed choice probe (as analysis_probe.py) and evaluates its scores on
stated_self-prompt activations: concordance with the stated answer vs the revealed choice,
and displacement betas of probe-on-stated vs the stated answer, per cell.

Usage: python src/analysis_probe_stated.py runs/pod_out
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

LI = 3  # layer 46

def main(out_dir):
    out = Path(out_dir)
    acts = np.load(out / "activations.npy", mmap_mode="r")
    man = [json.loads(l) for l in open(Path(__file__).resolve().parent.parent / "data" / "probe_manifest.jsonl")]
    df = pd.DataFrame(man).drop(columns=["messages"])
    choices = {r["uid"]: r["p_letter_A"] for r in map(json.loads, open(out / "choices.jsonl"))}
    df["p_letter_A"] = df.uid.map(choices)
    df["p_a_beh"] = np.where(df.order == 0, df.p_letter_A, 1 - df.p_letter_A)

    def X_of(sub):
        return np.nan_to_num(acts[sub.uid.values, LI, :].astype(np.float32), posinf=6e4, neginf=-6e4)

    tr = df[(df.cond == "B0") & (df.channel == "revealed")]
    y = (tr.p_letter_A > 0.5).astype(int).values
    sc = StandardScaler().fit(X_of(tr))
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X_of(tr)), y)

    def s(v): return v if isinstance(v, str) and v else None
    df["cell"] = df.apply(lambda r: (s(r["cond"]) or f"H_{s(r['checkpoint'])}") + "/" + (s(r["persona"]) or "-"), axis=1)
    st = df[df.channel == "stated_self"].copy()
    st["probe"] = clf.predict_proba(sc.transform(X_of(st)))[:, 1]
    st["p_probe_a"] = np.where(st.order == 0, st.probe, 1 - st.probe)
    rev = df[df.channel == "revealed"].copy()

    wst = st.groupby(["cell", "pair_id"])[["p_probe_a", "p_a_beh"]].mean().reset_index()
    wrev = rev.groupby(["cell", "pair_id"]).p_a_beh.mean().reset_index()
    pst = wst.pivot_table(index="pair_id", columns="cell", values="p_probe_a")
    ast_ = wst.pivot_table(index="pair_id", columns="cell", values="p_a_beh")
    arev = wrev.pivot_table(index="pair_id", columns="cell", values="p_a_beh")

    print(f"{'cell':<20}{'probe(st)~stated':>17}{'probe(st)~revealed':>19}")
    for cell in sorted(pst.columns):
        if cell not in arev.columns:
            continue
        m = pst[cell].notna() & ast_[cell].notna() & arev[cell].notna()
        if m.sum() < 15:
            continue
        r1 = stats.spearmanr(pst[cell][m], ast_[cell][m]).statistic
        r2 = stats.spearmanr(pst[cell][m], arev[cell][m]).statistic
        print(f"{cell:<20}{r1:>17.2f}{r2:>19.2f}")

    # displacement betas on stated prompts: does probe-on-stated displacement track
    # the stated answer's displacement? (source of REPORT §2.7(b) beta claims)
    def betaf(d1, d):
        i = d.dropna().index.intersection(d1.dropna().index)
        return float(d1[i] @ d[i] / (d1[i] @ d1[i]))
    print(f"\n{'cell':<20}{'beta_stated_answer':>19}{'beta_probe_on_stated':>21}")
    for persona in ("Vex", "Lazlo", "Mira"):
        b1 = f"B1/{persona}"
        d_say = (ast_[b1] - ast_["B0/-"]).dropna()
        d_pr = (pst[b1] - pst["B0/-"]).dropna()
        for cell in sorted(pst.columns):
            parts = cell.split("/")
            if len(parts) < 2 or parts[1] != persona or cell == b1:
                continue
            bs = betaf(d_say, ast_[cell] - ast_["B0/-"])
            bp = betaf(d_pr, pst[cell] - pst["B0/-"])
            print(f"{cell:<20}{bs:>19.2f}{bp:>21.2f}")

if __name__ == "__main__":
    main(sys.argv[1])
