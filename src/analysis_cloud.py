"""Loop 4: identity-cloud stress analysis.

Inputs (runs/): assistcloud_gemma + assistcloud_gemma_retest (noise floor),
paracloud_gemma, ecocloud_gemma, ecocloud_gpt41mini, assistcloud_{gpt41mini,llama70b,qwen72b}.

Outputs: noise-corrected cloud stats (P12), paraphrase-vs-content (P13), ecological (P14),
cross-model cloud index + capturability correlation (P15), cloud provenance (P16),
Epstein aggregation (P18).

Usage: python src/analysis_cloud.py
"""
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent / "runs"

def profiles(run, channel="revealed"):
    rows = [json.loads(l) for l in open(ROOT / run / "results.jsonl")]
    d = pd.DataFrame(rows)
    d = d[(d.flag == "ok") & (d.channel == channel)]
    def canon(r):
        chose_first = r["value"] == "A"
        return float(chose_first if r["order"] == 0 else not chose_first)
    d["picked_a"] = d.apply(canon, axis=1)
    return d.groupby(["cond", "pair_id"]).picked_a.mean().unstack(0)

def cloud_stats(P):
    dists = {(a, b): (P[a] - P[b]).abs().mean()
             for a, b in itertools.combinations(P.columns, 2)}
    return np.mean(list(dists.values())), np.max(list(dists.values())), dists

def main():
    # ---- P12: noise floor ----
    P1 = profiles("assistcloud_gemma")
    P2 = profiles("assistcloud_gemma_retest")
    retest = {v: (P1[v] - P2[v]).abs().mean() for v in P1.columns if v in P2.columns}
    noise = np.mean(list(retest.values()))
    m1, mx1, _ = cloud_stats(P1)
    # noise-corrected distance: sqrt(max(0, d^2 - noise^2)) per pair of variants (rough)
    corr_mean = float(np.sqrt(max(0, m1**2 - noise**2)))
    print("=== P12 noise floor (Gemma) ===")
    print(f"  test-retest |Δ| per variant: mean={noise:.3f}  " +
          " ".join(f"{k.split('_')[1]}={v:.2f}" for k, v in retest.items()))
    print(f"  raw cloud mean dist={m1:.3f}  max={mx1:.3f}  noise-corrected mean≈{corr_mean:.3f}")
    print(f"  verdict: noise explains {(noise/m1)**2*100:.0f}% of squared cloud distance")

    # pooled (both runs averaged) cloud for downstream stats
    Pg = (P1 + P2.reindex(P1.index)[P1.columns]) / 2

    # ---- P13: paraphrase vs content ----
    PP = profiles("paracloud_gemma")
    hhh = [c for c in PP.columns if "hhh" in c]
    warm = [c for c in PP.columns if "warm" in c]
    mh, xh, _ = cloud_stats(PP[hhh])
    mw, xw, _ = cloud_stats(PP[warm])
    cross = np.mean([(PP[a] - PP[b]).abs().mean() for a in hhh for b in warm])
    print("\n=== P13 paraphrase vs content (Gemma) ===")
    print(f"  within-hhh paraphrases: mean={mh:.3f}  within-warm: mean={mw:.3f}")
    print(f"  hhh<->warm (content): mean={cross:.3f}   [retest noise floor: {noise:.3f}]")

    # ---- P14: ecological ----
    for run, label in (("ecocloud_gemma", "Gemma"), ("ecocloud_gpt41mini", "gpt-4.1-mini")):
        try:
            PE = profiles(run)
        except FileNotFoundError:
            continue
        me, xe, de = cloud_stats(PE)
        print(f"\n=== P14 ecological cloud ({label}) ===")
        print(f"  mean={me:.3f}  max={xe:.3f} ({max(de, key=de.get)})")

    # ---- P15: cross-model index ----
    print("\n=== P15 cloud size x capturability across models ===")
    leak = {"gemma": 0.84, "gpt41mini": 0.52, "llama70b": 0.82, "qwen72b": 0.40}
    # leak = max C1 beta (Vex for gemma/gpt41mini; Lazlo for llama; Lazlo for qwen)
    sizes = {}
    for run, key in (("assistcloud_gemma", "gemma"), ("assistcloud_gpt41mini", "gpt41mini"),
                     ("assistcloud_llama70b", "llama70b"), ("assistcloud_qwen72b", "qwen72b")):
        try:
            P = profiles(run)
            sizes[key] = cloud_stats(P)[0]
        except FileNotFoundError:
            pass
    for k in sizes:
        print(f"  {k:10s}: cloud mean={sizes[k]:.3f}   maxC1leak={leak[k]:.2f}")
    if len(sizes) >= 3:
        from scipy import stats as st
        ks = list(sizes)
        r = st.spearmanr([sizes[k] for k in ks], [leak[k] for k in ks]).statistic
        print(f"  spearman(cloud size, leak) = {r:.2f}  (n={len(ks)})")

    # ---- P16: cloud provenance (variant-invariance) ----
    print("\n=== P16 cloud provenance (Gemma, noise-thresholded) ===")
    var_between = Pg.var(axis=1)
    # noise variance of a mean-of-two-runs profile point ~ (retest sd)^2/2; use empirical:
    noise_var = np.mean([(P1[v] - P2[v]).pow(2).mean() for v in P1.columns]) / 2
    invariant = (var_between <= noise_var).mean()
    print(f"  variant-invariant pairs (var <= noise): {invariant*100:.0f}% "
          f"(persona-invariant benchmark: ~5-15%)")
    moved = var_between.sort_values(ascending=False)
    rows = [json.loads(l) for l in open(ROOT / "assistcloud_gemma/results.jsonl")]
    subs = {r["pair_id"]: r["subset"] for r in rows if r.get("pair_id")}
    top = moved.head(12)
    bycat = pd.Series(top.index.map(subs)).value_counts()
    print(f"  top-12 moved pairs by subset: {dict(bycat)}")

    # ---- P18: Epstein aggregation ----
    print("\n=== P18 aggregation (Epstein) ===")
    cols = list(Pg.columns)
    rng = np.random.default_rng(0)
    cors = []
    for _ in range(200):
        rng.shuffle(cols)
        a, b = cols[:len(cols)//2], cols[len(cols)//2:]
        cors.append(Pg[a].mean(axis=1).corr(Pg[b].mean(axis=1), method="spearman"))
    print(f"  split-half centroid reliability: mean={np.mean(cors):.3f} (200 random splits)")

if __name__ == "__main__":
    main()
