"""Persisted loop-6 analysis (P23–P26): reproduces every reported number.
Outputs runs/loop6_results.json and prints a full table.

A. E1 context surgery betas + CIs (design caveats: transcript cell also omits the exit
   turn and changes role structure — it bounds quoted-content capture, it does not
   cleanly isolate participation; truncation varies content along with dose).
B. E3 holdout battery: C1 betas, warm-prose direct contrast + ratio CIs, and the
   generator-coupling split (pairs touching the Analyst's stated keywords vs not).
C. E6 P_do baselines: text-embedding, text-embedding + stated-answer prob, activations;
   all-pairs and say!=do-subset AUCs (embeddings cached to runs/pdo_text_emb.npy).
D. E2a nested trial-level logistic (NOT hierarchical): out-of-sample log-loss R^2
   increments under trial-stratum holdout within pairs, both addition orders.

Usage: python src/analysis_loop6.py
"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
RNG = np.random.default_rng(20260814)
NBOOT = 2000
OUT = {}

def canon_rows(path, channel="revealed"):
    rows = [json.loads(l) for l in open(path)]
    d = pd.DataFrame(rows)
    d = d[(d.flag == "ok") & (d.channel == channel)].copy()
    def canon(r):
        if r["value"] not in ("A", "B"):
            return np.nan
        chose_first = r["value"] == "A"
        return float(chose_first if r["order"] == 0 else not chose_first)
    d["picked_a"] = d.apply(canon, axis=1)
    return d

def beta_ci(d1, dd, n=NBOOT):
    i = list(dd.dropna().index.intersection(d1.dropna().index))
    a1, a2 = d1[i].values, dd[i].values
    pt = float(a1 @ a2 / (a1 @ a1))
    vals = []
    while len(vals) < n:
        idx = RNG.integers(0, len(i), len(i))
        x = a1[idx]; den = x @ x
        if den <= 1e-12:
            continue
        v = x @ a2[idx] / den
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(pt, 3), round(lo, 3), round(hi, 3)

def contrast_ci(d1, da, db, kind="diff", n=NBOOT):
    i = list(da.dropna().index.intersection(db.dropna().index).intersection(d1.dropna().index))
    a1, aa, ab = d1[i].values, da[i].values, db[i].values
    def stat(x, ya, yb):
        den = x @ x
        ba, bb = x @ ya / den, x @ yb / den
        return (ba - bb) if kind == "diff" else (ba / bb if abs(bb) > 1e-9 else np.nan)
    pt = stat(a1, aa, ab)
    vals = []
    while len(vals) < n:
        idx = RNG.integers(0, len(i), len(i))
        x = a1[idx]
        if x @ x <= 1e-12:
            continue
        v = stat(x, aa[idx], ab[idx])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(pt), 3), round(lo, 3), round(hi, 3)

# ---------------- A. E1 context surgery ----------------
def e1():
    rows = pd.concat([canon_rows(RUNS / "surgery_gemma/results.jsonl"),
                      canon_rows(RUNS / "hyst_gemma/results.jsonl")])
    p = rows.groupby(["persona", "checkpoint", "pair_id"]).picked_a.mean().reset_index()
    base = p[p.checkpoint == "t0"].groupby("pair_id").picked_a.mean()
    res = {}
    for persona in ("Vex", "Lazlo"):
        wt = p[p.persona == persona].pivot_table(index="pair_id", columns="checkpoint", values="picked_a")
        b = base.reindex(wt.index)
        d4 = (wt["t4"] - b).dropna()
        for cond in ("full", "trunc2", "trunc1", "usr_neu", "transcript"):
            if cond in wt.columns:
                res[f"{persona}.{cond}"] = beta_ci(d4, wt[cond] - b)
        ws = p[p.persona == "shared"].pivot_table(index="pair_id", columns="checkpoint", values="picked_a")
        bs = base.reindex(ws.index)
        for cond in ("del", "del_noexit"):
            res[f"{persona}.{cond}"] = beta_ci(d4, ws[cond] - bs)
        # transcript vs full direct contrast
        res[f"{persona}.transcript_minus_full"] = contrast_ci(d4, wt["transcript"] - b, wt["full"] - b)
    OUT["E1"] = res
    print("--- E1 context surgery (beta [95% CI]; transcript cell bounds quoted-content "
          "capture, does not isolate participation) ---")
    for k, v in res.items():
        print(f"  {k:28s} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]")

# ---------------- B. E3 holdout battery ----------------
def e3():
    bat = json.loads((ROOT / "data/holdout_battery.json").read_text())
    kw = ["format", "list", "outlin", "organiz", "structur", "table", "bullet", "categor"]
    coupled = {p["id"] for p in bat["pairs"] if any(k in (p["a"] + p["b"]).lower() for k in kw)}
    res = {}
    for model, run in (("gemma", "holdout_gemma"), ("gpt41mini", "holdout_gpt41mini")):
        d = canon_rows(RUNS / run / "results.jsonl")
        p = d.groupby(["cond", "pair_id"]).picked_a.mean()
        b0 = p.xs("B0", level="cond")
        for pname, short in (("WarmSupporter", "WarmSupporter"), ("SystematicAnalyst", "SystematicAnalyst")):
            b1 = p.xs(f"B1_{short}", level="cond")
            d1 = (b1 - b0).dropna()
            c1 = p.xs(f"C1_{short}", level="cond")
            res[f"{model}.{short}.C1"] = beta_ci(d1, c1 - b0)
            if short == "SystematicAnalyst":
                for tag, keep in (("coupled", coupled), ("uncoupled", set(b0.index) - coupled)):
                    d1s, c1s, b0s = d1[d1.index.isin(keep)], c1[c1.index.isin(keep)], b0[b0.index.isin(keep)]
                    res[f"{model}.{short}.C1_{tag}"] = beta_ci(d1s, (c1s - b0s))
            if short == "WarmSupporter":
                wp = p.xs("WARMPROSE", level="cond")
                res[f"{model}.warmprose_beta"] = beta_ci(d1, wp - b0)
                res[f"{model}.warmprose_minus_C1"] = contrast_ci(d1, wp - b0, c1 - b0, "diff")
                res[f"{model}.warmprose_over_C1"] = contrast_ci(d1, wp - b0, c1 - b0, "ratio")
        b1a = p.xs("B1_WarmSupporter", level="cond")
        b1b = p.xs("B1_SystematicAnalyst", level="cond")
        res[f"{model}.pairs_sensitive_frac"] = (round(float(((b1a - b1b).abs() > 0.16).mean()), 3),) * 3
    OUT["E3"] = res
    OUT["E3_coupled_pairs"] = sorted(coupled)
    print(f"\n--- E3 holdout ({len(coupled)}/30 pairs keyword-coupled to the Analyst persona) ---")
    for k, v in res.items():
        print(f"  {k:36s} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]")

# ---------------- C. E6 P_do baselines ----------------
def e6():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    acts = np.load(RUNS / "pod_out/activations.npy", mmap_mode="r")
    man = [json.loads(l) for l in open(ROOT / "data/probe_manifest.jsonl")]
    df = pd.DataFrame(man)
    choices = {r["uid"]: r["p_letter_A"] for r in map(json.loads, open(RUNS / "pod_out/choices.jsonl"))}
    df["p_letter_A"] = df.uid.map(choices)
    df["p_a_beh"] = np.where(df.order == 0, df.p_letter_A, 1 - df.p_letter_A)
    b0rev = df[(df.cond == "B0") & (df.channel == "revealed")]
    b0st = df[(df.cond == "B0") & (df.channel == "stated_self")]
    y_do = (b0rev.groupby("pair_id").p_a_beh.mean() > 0.5).astype(int)
    y_say = (b0st.groupby("pair_id").p_a_beh.mean() > 0.5).astype(int)
    p_say_pair = b0st.groupby("pair_id").p_a_beh.mean()
    disagree = set(y_do.index[(y_do != y_say.reindex(y_do.index))])
    tr = b0st.copy()
    tr["y"] = np.where(tr.order == 0, tr.pair_id.map(y_do), 1 - tr.pair_id.map(y_do))
    tr["p_say_letter"] = np.where(tr.order == 0, tr.pair_id.map(p_say_pair),
                                  1 - tr.pair_id.map(p_say_pair))
    emb_path = RUNS / "pdo_text_emb.npy"
    if emb_path.exists():
        Xt = np.load(emb_path)
    else:
        import asyncio, httpx
        from harness import Client  # .env load
        uid2text = {r["uid"]: r["messages"][-1]["content"] for r in man}
        texts = tr.uid.map(uid2text).tolist()
        async def embed(batch):
            async with httpx.AsyncClient(timeout=60) as cl:
                r = await cl.post("https://api.openai.com/v1/embeddings",
                                  headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                                  json={"model": "text-embedding-3-small", "input": batch})
                r.raise_for_status()
                return [x["embedding"] for x in r.json()["data"]]
        embs = []
        for i in range(0, len(texts), 100):
            embs += asyncio.run(embed(texts[i:i + 100]))
        Xt = np.array(embs)
        np.save(emb_path, Xt)
    Xa = np.nan_to_num(acts[tr.uid.values, 3, :].astype(np.float32), posinf=6e4, neginf=-6e4)
    Xts = np.column_stack([Xt, tr.p_say_letter.values])
    y, groups = tr.y.values, tr.pair_id.values
    feats = {"text": (Xt, 1.0), "text+stated": (Xts, 1.0), "acts": (Xa, 0.01)}
    res = {k: {"all": [], "disagree": []} for k in feats}
    for tri, tei in GroupKFold(n_splits=5).split(Xt, y, groups):
        dis_mask = np.isin(groups[tei], list(disagree))
        for name, (X, C) in feats.items():
            sc = StandardScaler().fit(X[tri])
            clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tri]), y[tri])
            pr = clf.predict_proba(sc.transform(X[tei]))[:, 1]
            res[name]["all"].append(roc_auc_score(y[tei], pr))
            if dis_mask.sum() >= 4 and len(np.unique(y[tei][dis_mask])) == 2:
                res[name]["disagree"].append(roc_auc_score(y[tei][dis_mask], pr[dis_mask]))
    OUT["E6"] = {k: {s: round(float(np.mean(v)), 3) for s, v in d.items() if v}
                 for k, d in res.items()}
    OUT["E6"]["n_disagree_pairs"] = len(disagree)
    print("\n--- E6 P_do baselines (GroupKFold-by-pair AUC; disagree = say!=do pairs) ---")
    for k, d in OUT["E6"].items():
        print(f"  {k}: {d}")

# ---------------- D. E2a out-of-sample nested decomposition ----------------
def e2a():
    from scipy.sparse import hstack
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import log_loss
    d = canon_rows(RUNS / "gridA_gemma/results.jsonl")
    d = d[(d.subset != "invariant") & d.cond.isin(["B1", "B2", "B3"])].copy()  # fully crossed 3x3 only (B0/none is structurally confounded with cond)
    d = d.dropna(subset=["picked_a"])
    y = d.picked_a.values
    ip = OneHotEncoder(sparse_output=True).fit_transform(d[["pair_id"]])
    per = OneHotEncoder(sparse_output=True).fit_transform(d[["persona"]]).toarray()
    bind = OneHotEncoder(sparse_output=True).fit_transform(d[["cond"]]).toarray()
    def cross(A, B):
        return hstack([A.multiply(B[:, j:j + 1]) for j in range(B.shape[1])]).tocsr()
    Xip, Xib = cross(ip, per), cross(ip, bind)
    designs = {"item": ip, "item+persona": hstack([ip, Xip]).tocsr(),
               "item+binding": hstack([ip, Xib]).tocsr(),
               "item+both": hstack([ip, Xip, Xib]).tocsr()}
    # trial-level holdout WITHIN pairs (item-interaction params are inestimable for
    # unseen items by construction, so pair-level holdout is the wrong design here):
    # hold out one (order, sample_idx) replicate stratum per fold.
    d["stratum"] = d.order.astype(str) + "_" + d.sample_idx.astype(str)
    strata = sorted(d.stratum.unique())
    lls = {k: [] for k in designs}; llnull = []
    for f in range(4):
        held = set(strata[f::4])
        tei = np.where(d.stratum.isin(held))[0]
        tri = np.where(~d.stratum.isin(held))[0]
        pnull = np.clip(y[tri].mean(), 1e-6, 1 - 1e-6)
        llnull.append(log_loss(y[tei], np.full(len(tei), pnull)))
        for k, X in designs.items():
            clf = LogisticRegression(C=1.0, max_iter=3000, solver="liblinear").fit(X[tri], y[tri])
            lls[k].append(log_loss(y[tei], clf.predict_proba(X[tei])[:, 1]))
    ll0 = np.mean(llnull)
    r2 = {k: round(1 - np.mean(v) / ll0, 3) for k, v in lls.items()}
    OUT["E2a"] = {"oos_R2": r2,
                  "persona_increment_over_item": round(r2["item+persona"] - r2["item"], 3),
                  "binding_increment_over_item": round(r2["item+binding"] - r2["item"], 3),
                  "persona_increment_last": round(r2["item+both"] - r2["item+binding"], 3),
                  "binding_increment_last": round(r2["item+both"] - r2["item+persona"], 3)}
    print("\n--- E2a nested trial-level logistic, OUT-OF-SAMPLE (trial-stratum holdout within pairs) ---")
    for k, v in OUT["E2a"].items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    e1(); e3(); e6(); e2a()
    json.dump(OUT, open(RUNS / "loop6_results.json", "w"), indent=1)
    print(f"\nwrote {RUNS}/loop6_results.json")
