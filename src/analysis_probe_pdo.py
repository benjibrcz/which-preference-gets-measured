"""Hidden-preference test (P_do): from STATED-question activations, decode what the model
WOULD CHOOSE (revealed label), and ask whether that internal would-do state stays
persona-shifted while the verbal report disavows.

Train: B0 stated_self-prompt activations (layer 46), labels = binarized B0 revealed choice
per pair (exact pod P(A)/P(B)), canonicalized per presentation order. GroupKFold by pair.
Apply frozen probe to stated_self prompts of every cell; report displacement beta of
(stated answer | P_do internal | revealed behaviour) vs the bound (B1-B0) direction.

Usage: python src/analysis_probe_pdo.py runs/pod_out
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

LI = 3  # index of layer 46 in [16, 26, 36, 46, 56, 62]

def main(out_dir):
    out = Path(out_dir)
    acts = np.load(out / "activations.npy", mmap_mode="r")
    man = [json.loads(l) for l in open(Path(__file__).resolve().parent.parent / "data" / "probe_manifest.jsonl")]
    df = pd.DataFrame(man).drop(columns=["messages"])
    choices = {r["uid"]: r["p_letter_A"] for r in map(json.loads, open(out / "choices.jsonl"))}
    df["p_letter_A"] = df.uid.map(choices)
    df["p_a_beh"] = np.where(df.order == 0, df.p_letter_A, 1 - df.p_letter_A)

    def X_of(sub):
        return np.nan_to_num(acts[sub.uid.values, LI, :].astype(np.float32),
                             posinf=6e4, neginf=-6e4)

    b0rev = df[(df.cond == "B0") & (df.channel == "revealed")]
    y_do_pair = (b0rev.groupby("pair_id").p_a_beh.mean() > 0.5).astype(int)
    b0st = df[(df.cond == "B0") & (df.channel == "stated_self")]
    y_say_pair = (b0st.groupby("pair_id").p_a_beh.mean() > 0.5).astype(int)
    print(f"B0 pairs where say != do: {int((y_do_pair != y_say_pair.reindex(y_do_pair.index)).sum())}/{len(y_do_pair)}")

    tr = b0st.copy()
    tr["y_do_letter"] = np.where(tr.order == 0, tr.pair_id.map(y_do_pair), 1 - tr.pair_id.map(y_do_pair))
    tr["y_say_letter"] = np.where(tr.order == 0, tr.pair_id.map(y_say_pair), 1 - tr.pair_id.map(y_say_pair))
    X = X_of(tr)

    auc_do, auc_say = [], []
    for tri, tei in GroupKFold(n_splits=5).split(X, tr.y_do_letter.values, tr.pair_id.values):
        sc = StandardScaler().fit(X[tri])
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X[tri]), tr.y_do_letter.values[tri])
        pred = clf.predict_proba(sc.transform(X[tei]))[:, 1]
        auc_do.append(roc_auc_score(tr.y_do_letter.values[tei], pred))
        auc_say.append(roc_auc_score(tr.y_say_letter.values[tei], pred))
    print(f"P_do CV AUC vs would-DO labels {np.mean(auc_do):.3f}, vs SAY labels {np.mean(auc_say):.3f}")

    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X), tr.y_do_letter.values)

    def s(v):
        return v if isinstance(v, str) and v else None
    for part, frame in (("st", df[df.channel == "stated_self"].copy()),
                        ("rev", df[df.channel == "revealed"].copy())):
        frame["cell"] = frame.apply(lambda r: (s(r["cond"]) or f"H_{s(r['checkpoint'])}") + "/" +
                                    (s(r["persona"]) or "-"), axis=1)
        if part == "st":
            st = frame
        else:
            rev = frame
    st["p_do"] = clf.predict_proba(sc.transform(X_of(st)))[:, 1]
    st["p_do_a"] = np.where(st.order == 0, st.p_do, 1 - st.p_do)

    wst = st.groupby(["cell", "pair_id"]).p_do_a.mean().reset_index().pivot_table(
        index="pair_id", columns="cell", values="p_do_a")
    wsay = st.groupby(["cell", "pair_id"]).p_a_beh.mean().reset_index().pivot_table(
        index="pair_id", columns="cell", values="p_a_beh")
    wrev = rev.groupby(["cell", "pair_id"]).p_a_beh.mean().reset_index().pivot_table(
        index="pair_id", columns="cell", values="p_a_beh")

    print(f"\n{'cell':<20}{'stated answer':>14}{'P_do internal':>14}{'revealed beh':>14}")
    rows = []
    for persona in ("Vex", "Lazlo", "Mira"):
        b1 = f"B1/{persona}"
        dirs = [(w[b1] - w["B0/-"]).dropna() for w in (wsay, wst, wrev)]
        for cell in sorted(wst.columns):
            if cell.split("/")[1] != persona or cell == b1:
                continue
            betas = []
            for w_, d_ in zip((wsay, wst, wrev), dirs):
                if cell not in w_.columns:
                    betas.append(np.nan); continue
                d = (w_[cell] - w_["B0/-"]).reindex(d_.index).dropna()
                i = d.index.intersection(d_.index)
                betas.append(float(d_[i] @ d[i] / (d_[i] @ d_[i])))
            print(f"{cell:<20}{betas[0]:>14.2f}{betas[1]:>14.2f}{betas[2]:>14.2f}")
            rows.append({"cell": cell, "beta_stated": betas[0], "beta_pdo": betas[1],
                         "beta_revealed": betas[2]})
    pd.DataFrame(rows).to_csv(out / "pdo_betas.csv", index=False)
    print(f"\nwrote {out}/pdo_betas.csv")

if __name__ == "__main__":
    main(sys.argv[1])
