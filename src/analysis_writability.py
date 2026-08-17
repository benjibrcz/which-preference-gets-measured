"""Loop 5: the context-writability law across 12 models.

Per model, four indicators (revealed channel):
  cloud  — mean pairwise |Δp| over ID_ identity variants        (convergent)
  c1     — mean β over personas: C1fiction displacement onto B1−B0   (convergent)
  hyst   — mean over personas of max-channel x2 residual β, Neutral-control-subtracted (convergent)
  b2     — mean β over personas: B2 "you are NOT X"              (discriminant)
QC gates: invariant accuracy ≥ 0.95 (B0 cell of writability/gridA), parse-ok ≥ 0.90.

Usage: python src/analysis_writability.py
"""
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent / "runs"

# model -> (grid_run for B0/B1/B2, c1_run cond name, cloud_run, hyst_run, neutral_run)
MODELS = {
    "gemma27b":  ("gridA_gemma", "deconfound_gemma", "assistcloud_gemma", "hyst_gemma", "hyst_gemma_neutral"),
    "gpt41mini": ("gridA_gpt41mini", "deconfound_gpt41mini", "assistcloud_gpt41mini", "hyst_gpt41mini", "hyst_gpt41mini_neutral"),
    "llama70b":  ("gridA_llama70b", "deconfound_llama70b", "assistcloud_llama70b", "hyst_llama70b", "hyst_llama70b_neutral"),
    "qwen72b":   ("gridA_qwen72b", "deconfound_qwen72b", "assistcloud_qwen72b", "hyst_qwen72b", "hyst_qwen72b"),
    "deepseek":  ("writ_deepseek", "writ_deepseek", "assistcloud_deepseek", "hyst_deepseek", "hyst_deepseek"),
    "mistral":   ("writ_mistral", "writ_mistral", "assistcloud_mistral", "hyst_mistral", "hyst_mistral"),
    "kimi":      ("writ_kimi", "writ_kimi", "assistcloud_kimi", "hyst_kimi", "hyst_kimi"),
    "cohere":    ("writ_cohere", "writ_cohere", "assistcloud_cohere", "hyst_cohere", "hyst_cohere"),
    "gemma12b":  ("writ_gemma12b", "writ_gemma12b", "assistcloud_gemma12b", "hyst_gemma12b", "hyst_gemma12b"),
    "llama8b":   ("writ_llama8b", "writ_llama8b", "assistcloud_llama8b", "hyst_llama8b", "hyst_llama8b"),
    "llama33":   ("writ_llama33", "writ_llama33", "assistcloud_llama33", "hyst_llama33", "hyst_llama33"),
    "gpt4omini": ("writ_gpt4omini", "writ_gpt4omini", "assistcloud_gpt4omini", "hyst_gpt4omini", "hyst_gpt4omini"),
}
PERSONAS = ("Vex", "Lazlo", "Mira")

def load(run):
    p = ROOT / run / "results_reparsed.jsonl"
    if not p.exists():
        p = ROOT / run / "results.jsonl"
    rows = [json.loads(l) for l in open(p)]
    df = pd.DataFrame(rows)
    df["persona"] = df.persona.fillna("-")
    return df

def phat(df, channel="revealed"):
    d = df[(df.flag == "ok") & (df.channel == channel) & (df.subset != "invariant")].copy()
    def canon(r):
        chose_first = r["value"] == "A"
        return float(chose_first if r["order"] == 0 else not chose_first)
    d["picked_a"] = d.apply(canon, axis=1)
    return d.groupby(["cond", "persona", "pair_id"]).picked_a.mean()

def beta(d1, d):
    i = d.dropna().index.intersection(d1.dropna().index)
    if len(i) < 15:
        return np.nan
    return float(d1[i] @ d[i] / (d1[i] @ d1[i]))

def qc(df):
    ok = (df.flag == "ok").mean()
    inv = df[(df.subset == "invariant") & (df.flag == "ok")].copy()
    if len(inv):
        def canon(r):
            chose_first = r["value"] == "A"
            return float(chose_first if r["order"] == 0 else not chose_first)
        inv["picked_a"] = inv.apply(canon, axis=1)
        # invariant pairs: canonical 'a' is not always the true side; use per-pair majority
        # consistency across orders as a formatting/QC proxy instead of truth:
        acc = inv.groupby("pair_id").picked_a.agg(lambda v: max(v.mean(), 1 - v.mean())).mean()
    else:
        acc = np.nan
    return ok, acc

def hyst_indicator(hrun, nrun):
    df = load(hrun)
    cho = df[df.channel != "identity"].copy()
    def canon(r):
        if r["value"] not in ("A", "B"):
            return np.nan
        chose_first = r["value"] == "A"
        return float(chose_first if r["order"] == 0 else not chose_first)
    cho["picked_a"] = cho.apply(canon, axis=1)
    p = cho.groupby(["persona", "checkpoint", "channel", "pair_id"]).picked_a.mean().reset_index()
    base = p[p.checkpoint == "t0"].groupby(["channel", "pair_id"]).picked_a.mean()
    if nrun != hrun:
        dn = load(nrun)
        chn = dn[dn.channel != "identity"].copy()
        chn["picked_a"] = chn.apply(canon, axis=1)
        pn = chn.groupby(["persona", "checkpoint", "channel", "pair_id"]).picked_a.mean().reset_index()
        basen = pn[pn.checkpoint == "t0"].groupby(["channel", "pair_id"]).picked_a.mean()
    else:
        pn, basen = p, base
    vals = []
    for persona in PERSONAS:
        per_ch = []
        for ch in ("revealed", "stated_self"):
            sub = p[(p.persona == persona) & (p.channel == ch)]
            w = sub.pivot_table(index="pair_id", columns="checkpoint", values="picked_a")
            if "t4" not in w.columns or "x2" not in w.columns:
                continue
            b = base.loc[ch].reindex(w.index)
            d4 = (w["t4"] - b).dropna()
            bx = beta(d4, (w["x2"] - b))
            subn = pn[(pn.persona == "Neutral") & (pn.channel == ch)]
            if len(subn):
                wn = subn.pivot_table(index="pair_id", columns="checkpoint", values="picked_a")
                bn_base = basen.loc[ch].reindex(wn.index)
                cpn = "x2" if "x2" in wn.columns else ("x4" if "x4" in wn.columns else None)
                bctrl = beta(d4, (wn[cpn] - bn_base)) if cpn else 0.0
            else:
                bctrl = 0.0
            per_ch.append(bx - bctrl)
        if per_ch:
            vals.append(max(per_ch))
    return float(np.mean(vals)) if vals else np.nan

def main():
    rows = []
    for name, (grun, c1run, crun, hrun, nrun) in MODELS.items():
        try:
            g = load(grun)
        except FileNotFoundError:
            print(f"  {name}: missing {grun}, skipped")
            continue
        okrate, invacc = qc(g)
        ph = phat(g)
        b0 = ph.xs(("B0", "-"), level=("cond", "persona"))
        c1df = load(c1run) if c1run != grun else g
        phc1 = phat(c1df)
        c1s, b2s, clouds = [], [], np.nan
        for persona in PERSONAS:
            b1 = ph.xs(("B1", persona), level=("cond", "persona"))
            d1 = (b1 - b0).dropna()
            b2 = ph.xs(("B2", persona), level=("cond", "persona"))
            b2s.append(beta(d1, b2 - b0))
            try:
                c1 = phc1.xs(("C1fiction", persona), level=("cond", "persona"))
                c1s.append(beta(d1, (c1 - b0)))
            except KeyError:
                c1s.append(np.nan)
        try:
            cl = load(crun)
            P = phat(cl).reset_index().pivot_table(index="pair_id", columns="cond", values="picked_a")
            ids = [c for c in P.columns if c.startswith("ID_")]
            clouds = float(np.mean([(P[a] - P[b]).abs().mean()
                                    for a, b in itertools.combinations(ids, 2)]))
        except FileNotFoundError:
            pass
        try:
            hy = hyst_indicator(hrun, nrun)
        except FileNotFoundError:
            hy = np.nan
        rows.append({"model": name, "ok": okrate, "invQC": invacc, "cloud": clouds,
                     "c1": np.nanmean(c1s), "hyst": hy, "b2": np.nanmean(b2s)})
    T = pd.DataFrame(rows).set_index("model")
    print(T.round(3).to_string())
    T.to_csv(ROOT / "writability_indicators.csv")

    ok = T[(T.ok >= 0.90) & ((T.invQC >= 0.95) | T.invQC.isna())]
    print(f"\nmodels passing QC: {len(ok)}/{len(T)}")
    print("\n=== P20/P21: rank correlations (Spearman) ===")
    cols = ["cloud", "c1", "hyst", "b2"]
    C = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a, b in itertools.combinations(cols, 2):
        m = ok[[a, b]].dropna()
        C.loc[a, b] = C.loc[b, a] = stats.spearmanr(m[a], m[b]).statistic
    print(C.round(2).to_string())
    print("\nP21 check: corr(c1, cloud) vs corr(b2, cloud):",
          f"{C.loc['c1','cloud']:.2f} vs {C.loc['b2','cloud']:.2f}")
    print("\nwithin-family: gemma27b vs gemma12b | llama70b vs llama33 vs llama8b (cloud, c1):")
    for fam in (("gemma27b", "gemma12b"), ("llama70b", "llama33", "llama8b")):
        avail = [m for m in fam if m in T.index]
        print("  " + "  ".join(f"{m}: cloud={T.loc[m,'cloud']:.3f} c1={T.loc[m,'c1']:.2f}"
                               for m in avail))

if __name__ == "__main__":
    main()
