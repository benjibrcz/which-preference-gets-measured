"""Extended bootstrap CIs (appends to runs/headline_cis.csv):
 - hysteresis residuals + reset interventions (Gemma, revealed; 23-pair bank, noted)
 - C1 fiction-leak and B2 negation-leak betas for all QC-passing models x personas
 - attractor recovery and inoculation betas (Gemma)
 - task-disjoint probe holdout (leave-one-category-out; both tasks excluded from train)

Usage: python src/analysis_more_cis.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RNG = np.random.default_rng(20260814)
NBOOT = 2000

def canon_df(path):
    p = Path(path) / "results_reparsed.jsonl"
    if not p.exists():
        p = Path(path) / "results.jsonl"
    rows = [json.loads(l) for l in open(p)]
    d = pd.DataFrame(rows)
    d["persona"] = d.persona.fillna("-")
    return d

def phat(d, channel="revealed"):
    d = d[(d.flag == "ok") & (d.channel == channel) & (d.subset != "invariant")].copy()
    def canon(r):
        if r["value"] not in ("A", "B"):
            return np.nan
        chose_first = r["value"] == "A"
        return float(chose_first if r["order"] == 0 else not chose_first)
    d["picked_a"] = d.apply(canon, axis=1)
    return d.groupby(["cond", "persona", "pair_id"]).picked_a.mean()

def beta_ci(d1, d):
    i = list(d.dropna().index.intersection(d1.dropna().index))
    a1, a2 = d1[i].values, d[i].values
    pt = float(a1 @ a2 / (a1 @ a1))
    vals = []
    while len(vals) < NBOOT:
        idx = RNG.integers(0, len(i), len(i))
        x, y = a1[idx], a2[idx]
        den = x @ x
        if den <= 1e-12:
            continue  # degenerate resample (all-zero direction); redraw
        v = x @ y / den
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return pt, lo, hi, len(i)

def contrast_ci(d1, d_a, d_b):
    """Joint pair-bootstrap CI for beta(d_a) - beta(d_b), same resample per draw."""
    i = list(d_a.dropna().index.intersection(d_b.dropna().index)
             .intersection(d1.dropna().index))
    a1, aa, ab = d1[i].values, d_a[i].values, d_b[i].values
    pt = float(a1 @ aa / (a1 @ a1) - a1 @ ab / (a1 @ a1))
    vals = []
    while len(vals) < NBOOT:
        idx = RNG.integers(0, len(i), len(i))
        x = a1[idx]
        den = x @ x
        if den <= 1e-12:
            continue
        v = (x @ aa[idx] - x @ ab[idx]) / den
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return pt, lo, hi, len(i)

def main():
    out = []

    # ---- hysteresis (Gemma revealed): x2/x8/r_inst/r_sys per persona + neutral ctrl ----
    rows = []
    for r in ("hyst_gemma", "hyst2_gemma", "hyst_gemma_neutral"):
        rows += [json.loads(l) for l in open(RUNS / r / "results.jsonl")]
    d = pd.DataFrame(rows)
    d = d[d.channel == "revealed"].copy()
    def canon(r):
        if r["value"] not in ("A", "B"):
            return np.nan
        chose_first = r["value"] == "A"
        return float(chose_first if r["order"] == 0 else not chose_first)
    d["picked_a"] = d.apply(canon, axis=1)
    p = d.groupby(["persona", "checkpoint", "pair_id"]).picked_a.mean().reset_index()
    base = p[p.checkpoint == "t0"].groupby("pair_id").picked_a.mean()
    wN = p[p.persona == "Neutral"].pivot_table(index="pair_id", columns="checkpoint", values="picked_a")
    bN = base.reindex(wN.index)
    for persona in ("Lazlo", "Vex", "Mira", "Neutral"):
        sub = p[p.persona == persona]
        w = sub.pivot_table(index="pair_id", columns="checkpoint", values="picked_a")
        b = base.reindex(w.index)
        ref = "Lazlo" if persona == "Neutral" else persona
        wr = p[p.persona == ref].pivot_table(index="pair_id", columns="checkpoint", values="picked_a")
        d4 = (wr["t4"] - base.reindex(wr.index)).dropna()
        for cp in ("x2", "x8", "r_inst", "r_sys"):
            if cp not in w.columns:
                continue
            pt, lo, hi, n = beta_ci(d4, w[cp] - b)
            tag = f"Gemma hyst {persona} {cp}" + (" (ctrl on Lazlo dir)" if persona == "Neutral" else "")
            out.append((tag + f" [n={n} pairs]", pt, lo, hi))
            # direct persona-minus-control contrast (joint bootstrap, same resample)
            if persona != "Neutral" and cp in wN.columns:
                pt2, lo2, hi2, n2 = contrast_ci(d4, w[cp] - b, wN[cp] - bN)
                out.append((f"Gemma hyst {persona} {cp} MINUS ctrl [n={n2} pairs]", pt2, lo2, hi2))

    # ---- C1 + B2 betas for all models x personas ----
    MODELS = {
        "gemma27b": ("gridA_gemma", "deconfound_gemma"),
        "gpt41mini": ("gridA_gpt41mini", "deconfound_gpt41mini"),
        "llama70b": ("gridA_llama70b", "deconfound_llama70b"),
        "qwen72b": ("gridA_qwen72b", "deconfound_qwen72b"),
        "deepseek": ("writ_deepseek", None), "mistral": ("writ_mistral", None),
        "kimi": ("writ_kimi", None), "cohere": ("writ_cohere", None),
        "llama33": ("writ_llama33", None), "gpt4omini": ("writ_gpt4omini", None),
    }
    for name, (grun, decrun) in MODELS.items():
        g = phat(canon_df(RUNS / grun))
        dec = phat(canon_df(RUNS / decrun)) if decrun else g
        b0 = g.xs(("B0", "-"), level=("cond", "persona"))
        for persona in ("Vex", "Lazlo", "Mira"):
            b1 = g.xs(("B1", persona), level=("cond", "persona"))
            d1 = (b1 - b0).dropna()
            for cond, src in (("C1fiction", dec), ("B2", g)):
                try:
                    cell = src.xs((cond, persona), level=("cond", "persona"))
                except KeyError:
                    continue
                if not len(cell):
                    continue
                pt, lo, hi, n = beta_ci(d1, cell - b0)
                out.append((f"{name} {cond} {persona}", pt, lo, hi))

    # ---- attractor + inoculation (Gemma) ----
    g = phat(canon_df(RUNS / "gridA_gemma"))
    b0 = g.xs(("B0", "-"), level=("cond", "persona"))
    att = phat(canon_df(RUNS / "attract_gemma"))
    for persona in ("Vex", "Mira"):
        b1 = g.xs(("B1", persona), level=("cond", "persona"))
        d1 = (b1 - b0).dropna()
        cell = att.xs((f"ATT_{persona}", persona), level=("cond", "persona"))
        pt, lo, hi, n = beta_ci(d1, cell - b0)
        out.append((f"Gemma attractor remaining {persona}", pt, lo, hi))
    cloud = phat(canon_df(RUNS / "assistcloud_gemma"))
    inoc = phat(canon_df(RUNS / "inoculate_gemma"))
    b1v = g.xs(("B1", "Vex"), level=("cond", "persona"))
    d1 = (b1v - b0).dropna()
    for idname in ("bare", "minimal", "hhh", "named", "constitution"):
        idc = cloud.xs((f"ID_{idname}", "-"), level=("cond", "persona"))
        ino = inoc.xs((f"INOC_{idname}", "Vex"), level=("cond", "persona"))
        pt, lo, hi, n = beta_ci(d1, ino - idc)
        out.append((f"Gemma inoculation {idname}", pt, lo, hi))

    for label, pt, lo, hi in out:
        print(f"  {label:48s} {pt:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    old = pd.read_csv(RUNS / "headline_cis.csv")
    new = pd.DataFrame(out, columns=["quantity", "point", "lo95", "hi95"])
    pd.concat([old, new]).drop_duplicates(subset="quantity", keep="last").to_csv(
        RUNS / "headline_cis.csv", index=False)
    print(f"\nheadline_cis.csv now has {len(old) + len(new)} rows")

    # ---- task-disjoint probe holdout (leave-one-category-out) ----
    print("\n=== task-disjoint probe holdout (train excludes BOTH tasks' categories) ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    acts = np.load(RUNS / "pod_out/activations.npy", mmap_mode="r")
    man = [json.loads(l) for l in open(ROOT / "data/probe_manifest.jsonl")]
    dfm = pd.DataFrame(man).drop(columns=["messages"])
    choices = {r["uid"]: r["p_letter_A"] for r in map(json.loads, open(RUNS / "pod_out/choices.jsonl"))}
    dfm["p_letter_A"] = dfm.uid.map(choices)
    tasks = {t["id"]: t for t in json.loads(open(ROOT / "data/tasks.json").read())["tasks"]}
    tr = dfm[(dfm.cond == "B0") & (dfm.channel == "revealed")].copy()
    tr["tids"] = tr.pair_id.map(lambda p: frozenset(p.split("__")))
    X = np.nan_to_num(acts[tr.uid.values, 3, :].astype(np.float32), posinf=6e4, neginf=-6e4)
    y = (tr.p_letter_A > 0.5).astype(int).values
    all_tasks = sorted({t_ for ts in tr.tids for t_ in ts})
    aucs = []
    rng2 = np.random.default_rng(7)
    for seed in range(10):
        test_tasks = set(rng2.choice(all_tasks, size=int(0.4 * len(all_tasks)), replace=False))
        # STRICT task-disjoint: test pairs use only test tasks; train pairs use NO test task
        test = tr.tids.map(lambda ts: ts <= test_tasks).values
        train = tr.tids.map(lambda ts: not (ts & test_tasks)).values
        if test.sum() < 10 or len(np.unique(y[test])) < 2:
            continue
        sc = StandardScaler().fit(X[train])
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X[train]), y[train])
        aucs.append(roc_auc_score(y[test], clf.predict_proba(sc.transform(X[test]))[:, 1]))
    print(f"  strict task-disjoint splits (10 seeds, 40% tasks held out; "
          f"test pairs = both tasks held-out, train pairs = neither):")
    print(f"  AUCs: {[round(a,3) for a in aucs]}  mean={np.mean(aucs):.3f}  min={np.min(aucs):.3f}")

if __name__ == "__main__":
    main()
