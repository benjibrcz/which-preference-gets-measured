"""Analyses addressing EVALUATION.md points 1, 2, 4, 6, 9.

A. Core-subset marginal-sensitivity map (invariant-core selection check)  [eval #6]
B. Crossed two-factor partial R^2 on the fully-crossed context grid       [eval #1]
C. Seeded 2000-rep bootstrap CIs for headline quantities                  [eval #4]
D. Leave-one-family-out cross-model correlations                          [eval #9]
E. Probe family-held-out CV (GroupKFold by task category)                 [eval #2]

Usage: python src/analysis_eval_fixes.py
"""
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RNG = np.random.default_rng(20260814)
NBOOT = 2000

def phat(run, channel="revealed"):
    p = RUNS / run / "results_reparsed.jsonl"
    if not p.exists():
        p = RUNS / run / "results.jsonl"
    rows = [json.loads(l) for l in open(p)]
    d = pd.DataFrame(rows)
    d["persona"] = d.persona.fillna("-")
    d = d[(d.flag == "ok") & (d.channel == channel) & (d.subset != "invariant")].copy()
    def canon(r):
        chose_first = r["value"] == "A"
        return float(chose_first if r["order"] == 0 else not chose_first)
    d["picked_a"] = d.apply(canon, axis=1)
    return d.groupby(["cond", "persona", "pair_id"]).picked_a.mean(), \
           d.groupby("pair_id").subset.first()

def beta(d1, d):
    i = d.dropna().index.intersection(d1.dropna().index)
    return float(d1[i] @ d[i] / (d1[i] @ d1[i]))

def boot_beta(d1, d, n=NBOOT):
    i = list(d.dropna().index.intersection(d1.dropna().index))
    vals = []
    a1, a2 = d1[i].values, d[i].values
    for _ in range(n):
        idx = RNG.integers(0, len(i), len(i))
        x, y = a1[idx], a2[idx]
        vals.append(x @ y / (x @ x))
    return np.percentile(vals, [2.5, 97.5])

def main():
    out = []

    # ---- A. core-subset marginal sensitivity (Gemma) ----
    print("=== A. invariant-core on the neutral 'core' subset only (Gemma) ===")
    prov = pd.read_csv(RUNS / "gridA_gemma/provenance.csv")
    for sub in ("all", "core"):
        p_ = prov if sub == "all" else prov[prov.subset == "core"]
        counts = p_.level.value_counts()
        n = len(p_)
        print(f"  {sub:5s} (n={n}): " +
              "  ".join(f"{k}={v} ({v/n*100:.0f}%)" for k, v in counts.items()))

    # ---- B. crossed partial R^2 on the context grid (persona x transform, revealed) ----
    print("\n=== B. jointly-estimated two-factor R^2 (fully-crossed context grid, Gemma) ===")
    rows = [json.loads(l) for l in open(RUNS / "context_gemma/results.jsonl")]
    d = pd.DataFrame(rows)
    d["persona"] = d.persona.fillna("-")
    d = d[(d.flag == "ok")].copy()
    def canon(r):
        chose_first = r["value"] == "A"
        return float(chose_first if r["order"] == 0 else not chose_first)
    d["picked_a"] = d.apply(canon, axis=1)
    d["tf"] = d.apply(lambda r: r["paraphrase"] if r["paraphrase"] != "p0"
                      else ("user" if r["placement"] == "user" else
                            ("hist" if r["history"] else "sys")), axis=1)
    cell = d.groupby(["persona", "tf", "pair_id"]).picked_a.mean().reset_index()
    r2p, r2t, r2i = [], [], []
    for pid, sub in cell.groupby("pair_id"):
        piv = sub.pivot_table(index="persona", columns="tf", values="picked_a")
        if piv.isna().any().any() or len(piv) < 3:
            continue
        y = piv.values
        gm = y.mean()
        sp = ((y.mean(axis=1) - gm) ** 2).mean()          # persona main effect
        st = ((y.mean(axis=0) - gm) ** 2).mean()          # transform main effect
        resid = y - y.mean(axis=1, keepdims=True) - y.mean(axis=0, keepdims=True) + gm
        si = (resid ** 2).mean()
        tot = ((y - gm) ** 2).mean()
        if tot > 0:
            r2p.append(sp / tot); r2t.append(st / tot); r2i.append(si / tot)
    print(f"  mean share: persona={np.mean(r2p):.2f}  transform={np.mean(r2t):.2f}  "
          f"interaction+noise={np.mean(r2i):.2f}  (n={len(r2p)} pairs; balanced 2-way ANOVA per pair)")

    # ---- C. bootstrap CIs for headline quantities ----
    print("\n=== C. seeded 2000-rep bootstrap 95% CIs (resampling pairs) ===")
    g, _ = phat("gridA_gemma")
    dec, _ = phat("deconfound_gemma")
    b0 = g.xs(("B0", "-"), level=("cond", "persona"))
    for cond, src, label in (("C1fiction", dec, "C1 fiction-leak"),
                             ("B2", g, "B2 negation-leak"),
                             ("C3anti", dec, "C3 anti-instruction")):
        for persona in ("Vex", "Lazlo", "Mira"):
            b1 = g.xs(("B1", persona), level=("cond", "persona"))
            d1 = (b1 - b0).dropna()
            try:
                cell_ = src.xs((cond, persona), level=("cond", "persona"))
            except KeyError:
                continue
            if not len(cell_):
                continue
            b = beta(d1, cell_ - b0)
            lo, hi = boot_beta(d1, cell_ - b0)
            out.append((f"Gemma {label} {persona}", b, lo, hi))
    # wedge per model (B2 revealed - stated)
    for model, run in (("gemma", "gridA_gemma"), ("gpt41mini", "gridA_gpt41mini"),
                       ("llama70b", "gridA_llama70b"), ("qwen72b", "gridA_qwen72b")):
        rev, _ = phat(run, "revealed"); st, _ = phat(run, "stated_self")
        for persona in ("Mira",):
            wb = {}
            for ch, ph_ in (("rev", rev), ("st", st)):
                b0_ = ph_.xs(("B0", "-"), level=("cond", "persona"))
                b1_ = ph_.xs(("B1", persona), level=("cond", "persona"))
                d1_ = (b1_ - b0_).dropna()
                b2_ = ph_.xs(("B2", persona), level=("cond", "persona"))
                wb[ch] = (d1_, b2_ - b0_)
            i = list(wb["rev"][1].dropna().index.intersection(wb["rev"][0].dropna().index)
                     .intersection(wb["st"][1].dropna().index).intersection(wb["st"][0].dropna().index))
            vals = []
            for _ in range(NBOOT):
                idx = RNG.integers(0, len(i), len(i))
                ii = [i[j] for j in idx]
                br = wb["rev"][0][ii].values @ wb["rev"][1][ii].values / (wb["rev"][0][ii].values @ wb["rev"][0][ii].values)
                bs = wb["st"][0][ii].values @ wb["st"][1][ii].values / (wb["st"][0][ii].values @ wb["st"][0][ii].values)
                vals.append(br - bs)
            pt = beta(*wb["rev"]) - beta(*wb["st"])
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out.append((f"{model} B2 wedge (rev−st) Mira", pt, lo, hi))
    # cloud size (Gemma) with CI over pairs
    cl, _ = phat("assistcloud_gemma")
    P = cl.reset_index().pivot_table(index="pair_id", columns="cond", values="picked_a")
    ids = [c for c in P.columns if c.startswith("ID_")]
    pairsM = P[ids].dropna()
    dists = []
    for _ in range(NBOOT):
        idx = RNG.integers(0, len(pairsM), len(pairsM))
        M = pairsM.values[idx]
        dd = [np.abs(M[:, a] - M[:, b]).mean()
              for a, b in itertools.combinations(range(len(ids)), 2)]
        dists.append(np.mean(dd))
    ptd = np.mean([np.abs(pairsM.values[:, a] - pairsM.values[:, b]).mean()
                   for a, b in itertools.combinations(range(len(ids)), 2)])
    lo, hi = np.percentile(dists, [2.5, 97.5])
    out.append(("Gemma identity-cloud mean distance", ptd, lo, hi))

    for label, pt, lo, hi in out:
        print(f"  {label:38s} {pt:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    pd.DataFrame(out, columns=["quantity", "point", "lo95", "hi95"]).to_csv(
        RUNS / "headline_cis.csv", index=False)

    # ---- D. leave-one-family-out cross-model correlations ----
    print("\n=== D. cross-model correlation clusters: sensitivity ===")
    T = pd.read_csv(RUNS / "writability_indicators.csv").set_index("model")
    T = T[(T.ok >= 0.90) & ((T.invQC >= 0.95) | T.invQC.isna())]
    fam = {"llama70b": "llama", "llama33": "llama", "gemma27b": "gemma",
           "gpt41mini": "openai", "gpt4omini": "openai"}
    def cors(sub):
        return (stats.spearmanr(sub["cloud"], sub["c1"]).statistic,
                stats.spearmanr(sub.dropna(subset=["hyst", "b2"])["hyst"],
                                sub.dropna(subset=["hyst", "b2"])["b2"]).statistic)
    c0 = cors(T)
    print(f"  all passing (n={len(T)}): rho(cloud,c1)={c0[0]:.2f}  rho(hyst,b2)={c0[1]:.2f}")
    for f in ("llama", "openai"):
        sub = T[[fam.get(m) != f for m in T.index]]
        c = cors(sub)
        print(f"  drop {f:7s} family (n={len(sub)}): rho(cloud,c1)={c[0]:.2f}  rho(hyst,b2)={c[1]:.2f}")
    # bootstrap over models
    ks = list(T.index)
    r1s, r2s = [], []
    for _ in range(NBOOT):
        idx = RNG.integers(0, len(ks), len(ks))
        sub = T.iloc[idx]
        if sub.cloud.nunique() < 3:
            continue
        c = cors(sub)
        r1s.append(c[0]); r2s.append(c[1])
    print(f"  bootstrap 95% CI: rho(cloud,c1) [{np.percentile(r1s,2.5):.2f},{np.percentile(r1s,97.5):.2f}]  "
          f"rho(hyst,b2) [{np.percentile(r2s,2.5):.2f},{np.percentile(r2s,97.5):.2f}]")

    # ---- E. probe family-held-out CV ----
    print("\n=== E. choice probe: held-out task-family CV (layer 46) ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    acts = np.load(RUNS / "pod_out/activations.npy", mmap_mode="r")
    man = [json.loads(l) for l in open(ROOT / "data/probe_manifest.jsonl")]
    dfm = pd.DataFrame(man).drop(columns=["messages"])
    choices = {r["uid"]: r["p_letter_A"] for r in map(json.loads, open(RUNS / "pod_out/choices.jsonl"))}
    dfm["p_letter_A"] = dfm.uid.map(choices)
    tasks = {t["id"]: t for t in json.loads(open(ROOT / "data/tasks.json").read())["tasks"]}
    tr = dfm[(dfm.cond == "B0") & (dfm.channel == "revealed")].copy()
    tr["fam"] = tr.pair_id.map(lambda p: tasks[p.split("__")[0]]["cat"])
    X = np.nan_to_num(acts[tr.uid.values, 3, :].astype(np.float32), posinf=6e4, neginf=-6e4)
    y = (tr.p_letter_A > 0.5).astype(int).values
    aucs = []
    for tri, tei in GroupKFold(n_splits=5).split(X, y, tr.fam.values):
        sc = StandardScaler().fit(X[tri])
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X[tri]), y[tri])
        aucs.append(roc_auc_score(y[tei], clf.predict_proba(sc.transform(X[tei]))[:, 1]))
    print(f"  held-out-family AUC = {np.mean(aucs):.3f} (folds: {[round(a,3) for a in aucs]})")

if __name__ == "__main__":
    main()
