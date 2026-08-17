"""Probe channel analysis (the third leg of the triangle).

1. Train logistic probe on B0 revealed-prompt activations to predict the model's upcoming
   letter choice (GroupKFold by pair; pick layer by CV AUC; gate >= 0.75).
2. Apply frozen probe everywhere; aggregate to canonical p_probe(a) per (cell, channel, pair).
3. Concordance triangle per cell: probe vs revealed vs stated (pod-exact probabilities).
4. RQ-A2 privileged access: partial corr(stated, probe | revealed).
5. Hysteresis: probe displacement beta per checkpoint vs revealed/stated.

Usage: python src/analysis_probe.py runs/pod_out
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def partial_corr(x, y, z):
    """corr(x, y | z) via residualization (spearman on ranks of residuals)."""
    x, y, z = map(np.asarray, (x, y, z))
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return stats.spearmanr(rx, ry).statistic

def main(out_dir):
    out = Path(out_dir)
    meta = json.load(open(out / "meta.json"))
    acts = np.load(out / "activations.npy")
    choices = {r["uid"]: r["p_letter_A"] for r in map(json.loads, open(out / "choices.jsonl"))}
    man = [json.loads(l) for l in open(Path(__file__).resolve().parent.parent / "data" / "probe_manifest.jsonl")]
    df = pd.DataFrame(man).drop(columns=["messages"])
    df["p_letter_A"] = df.uid.map(choices)
    def s(v):
        return v if isinstance(v, str) and v else None
    df["cell"] = df.apply(lambda r: (s(r["cond"]) or f"H_{s(r['checkpoint'])}") + "/" +
                          (s(r["persona"]) or "-"), axis=1)

    # ---- train probe on B0 revealed ----
    tr = df[(df.cond == "B0") & (df.channel == "revealed")]
    y = (tr.p_letter_A > 0.5).astype(int).values
    groups = tr.pair_id.values
    best = None
    for li, L in enumerate(meta["layers"]):
        X = np.nan_to_num(acts[tr.uid.values, li, :].astype(np.float32),
                          posinf=6.0e4, neginf=-6.0e4)
        aucs = []
        gkf = GroupKFold(n_splits=5)
        for tr_i, te_i in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr_i])
            clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X[tr_i]), y[tr_i])
            aucs.append(roc_auc_score(y[te_i], clf.predict_proba(sc.transform(X[te_i]))[:, 1]))
        m = float(np.mean(aucs))
        print(f"layer {L}: CV AUC = {m:.3f}")
        if best is None or m > best[1]:
            best = (li, m, L)
    li, auc, L = best
    print(f"\nSelected layer {L} (CV AUC {auc:.3f}); gate >= 0.75: {'PASS' if auc >= 0.75 else 'FAIL'}")

    X_all = np.nan_to_num(acts[:, li, :].astype(np.float32), posinf=6.0e4, neginf=-6.0e4)
    sc = StandardScaler().fit(X_all[tr.uid.values])
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X_all[tr.uid.values]), y)
    df["probe_score"] = clf.predict_proba(sc.transform(X_all[df.uid.values]))[:, 1]

    # canonicalize to p(a)
    df["p_a_beh"] = np.where(df.order == 0, df.p_letter_A, 1 - df.p_letter_A)
    df["p_a_probe"] = np.where(df.order == 0, df.probe_score, 1 - df.probe_score)
    agg = df.groupby(["family", "cell", "channel", "pair_id"], dropna=False)[
        ["p_a_beh", "p_a_probe"]].mean().reset_index()

    # ---- triangle per cell ----
    print("\n=== triangle per cell (Spearman over pairs) ===")
    print(f"{'cell':<18}{'rev~probe':>10}{'st~probe':>10}{'st~rev':>10}{'partial st~probe|rev':>22}")
    w = agg.pivot_table(index="pair_id", columns=["cell", "channel"],
                        values=["p_a_beh", "p_a_probe"])
    for cell in sorted(agg.cell.unique()):
        try:
            rev = w[("p_a_beh", cell, "revealed")]
            strev = w[("p_a_beh", cell, "stated_self")]
            prob = w[("p_a_probe", cell, "revealed")]
        except KeyError:
            continue
        m = rev.notna() & strev.notna() & prob.notna()
        if m.sum() < 15:
            continue
        r1 = stats.spearmanr(rev[m], prob[m]).statistic
        r2 = stats.spearmanr(strev[m], prob[m]).statistic
        r3 = stats.spearmanr(strev[m], rev[m]).statistic
        pc = partial_corr(strev[m].values, prob[m].values, rev[m].values)
        print(f"{cell:<18}{r1:>10.2f}{r2:>10.2f}{r3:>10.2f}{pc:>22.2f}")

    # ---- displacement betas incl. probe channel ----
    print("\n=== displacement beta vs bound direction (probe vs behaviour), revealed prompts ===")
    base_beh = w[("p_a_beh", "B0/-", "revealed")]
    base_pr = w[("p_a_probe", "B0/-", "revealed")]
    for persona in ("Vex", "Lazlo", "Mira"):
        b1 = f"B1/{persona}"
        d1b = (w[("p_a_beh", b1, "revealed")] - base_beh).dropna()
        d1p = (w[("p_a_probe", b1, "revealed")] - base_pr).dropna()
        for cell in sorted(agg.cell.unique()):
            pcell = cell.split("/")[1]
            if pcell not in (persona, "-") or cell == "B0/-":
                continue
            try:
                db = (w[("p_a_beh", cell, "revealed")] - base_beh).reindex(d1b.index).dropna()
                dp = (w[("p_a_probe", cell, "revealed")] - base_pr).reindex(d1p.index).dropna()
            except KeyError:
                continue
            i1 = db.index.intersection(d1b.index); i2 = dp.index.intersection(d1p.index)
            if len(i1) < 15:
                continue
            bb = float(d1b[i1].values @ db[i1].values / (d1b[i1].values @ d1b[i1].values))
            bp = float(d1p[i2].values @ dp[i2].values / (d1p[i2].values @ d1p[i2].values))
            print(f"{cell:<18} vs {b1:<10} beta_beh={bb:>6.2f}  beta_probe={bp:>6.2f}")

    agg.to_csv(out / "probe_agg.csv", index=False)
    print(f"\nwrote {out}/probe_agg.csv")

if __name__ == "__main__":
    main(sys.argv[1])
