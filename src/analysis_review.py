"""Persisted review-round analyses (16 Aug): shared-baseline cross-fit of beta, and the
non-agent normative-content control, with bootstrap CIs and a policy-vs-character CONTRAST CI.
Reproduces the numbers cited in REPORT 2.1 / SUBMISSION. Run:
  python src/analysis_review.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20260816)

def load(run):
    d = pd.DataFrame(json.loads(l) for l in open(ROOT / "runs" / run / "results.jsonl"))
    d = d[(d.flag == "ok") & (d.channel == "revealed") & (d.subset != "invariant")].copy()
    d["persona"] = d.persona.fillna("-")
    d["pa"] = np.where(d.order == 0, (d.value == "A"), (d.value == "B")).astype(float)
    return d

def prof(sub, pairs):
    return sub.groupby("pair_id").pa.mean().reindex(pairs)

# ---------- A. cross-fit vs naive beta (C1) ----------
def crossfit(gridrun, decrun):
    g, dec = load(gridrun), load(decrun)
    pairs = sorted(g[g.cond == "B0"].pair_id.unique())
    b0 = g[(g.cond == "B0") & (g.persona == "-")]
    out = {}
    for persona in ("Vex", "Lazlo", "Mira"):
        b1 = g[(g.cond == "B1") & (g.persona == persona)]
        cc = dec[(dec.cond == "C1fiction") & (dec.persona == persona)]
        pb0, pb1, pc = prof(b0, pairs), prof(b1, pairs), prof(cc, pairs)
        d, e = (pb1 - pb0), (pc - pb0)
        m = d.notna() & e.notna()
        bn = float(e[m] @ d[m] / (d[m] @ d[m]))
        h = lambda x, par: prof(x[x.sample_idx % 2 == par], pairs)
        cf = lambda b0d, b1d, cp, cb0: float(((cp - cb0)[(b1d - b0d).notna() & (cp - cb0).notna()] @
              (b1d - b0d)[(b1d - b0d).notna() & (cp - cb0).notna()]) /
              ((b1d - b0d)[(b1d - b0d).notna()] @ (b1d - b0d)[(b1d - b0d).notna()]))
        bcf = np.mean([cf(h(b0, 0), h(b1, 0), h(cc, 1), h(b0, 1)),
                       cf(h(b0, 1), h(b1, 1), h(cc, 0), h(b0, 0))])
        absd = float((pc - pb0).abs().mean())
        flip = int(((pc > 0.5) != (pb0 > 0.5)).sum())
        out[persona] = (round(bn, 3), round(float(bcf), 3), round(bn - bcf, 3), round(absd, 3), flip)
    return out

# ---------- B. non-agent normative control: policy vs character, with CIs ----------
def _beta_boot(dir_pp, disp_pp, n=2000):
    m = dir_pp.notna() & disp_pp.notna()
    d, e = dir_pp[m].values, disp_pp[m].values
    pt = float(e @ d / (d @ d))
    vals = []
    while len(vals) < n:
        i = RNG.integers(0, len(d), len(d)); x = d[i]
        if x @ x < 1e-9: continue
        vals.append(e[i] @ x / (x @ x))
    return round(pt, 3), round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)

def _contrast_boot(dir_pp, pol_pp, char_pp, n=2000):
    m = dir_pp.notna() & pol_pp.notna() & char_pp.notna()
    d, p, c = dir_pp[m].values, pol_pp[m].values, char_pp[m].values
    pt = float((p @ d - c @ d) / (d @ d))
    vals = []
    while len(vals) < n:
        i = RNG.integers(0, len(d), len(d)); x = d[i]
        if x @ x < 1e-9: continue
        vals.append((p[i] @ x - c[i] @ x) / (x @ x))
    return round(pt, 3), round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)

def normative_control(gridrun, decrun, semrun):
    g, dec, pol = load(gridrun), load(decrun), load(semrun)
    pairs = sorted(g[g.cond == "B0"].pair_id.unique())
    b0 = prof(g[(g.cond == "B0") & (g.persona == "-")], pairs)
    out = {}
    for persona in ("Vex", "Lazlo"):
        d1 = prof(g[(g.cond == "B1") & (g.persona == persona)], pairs) - b0
        char = prof(dec[(dec.cond == "C1fiction") & (dec.persona == persona)], pairs) - b0
        poli = prof(pol[(pol.cond == "POLICY") & (pol.persona == persona)], pairs) - b0
        out[persona] = {"char_C1": _beta_boot(d1, char), "policy": _beta_boot(d1, poli),
                        "policy_minus_char": _contrast_boot(d1, poli, char)}
    return out

def main():
    print("=== A. C1 beta: naive vs cross-fit (shared-baseline bias check) ===")
    print(f"{'model':<10}{'persona':<7}{'naive':>7}{'xfit':>7}{'bias':>7}{'|dp|':>7}{'flip':>6}")
    maxbias = 0
    for name, gr, dr in [("gemma", "gridA_gemma", "deconfound_gemma"),
                         ("llama70b", "gridA_llama70b", "deconfound_llama70b"),
                         ("gpt41mini", "gridA_gpt41mini", "deconfound_gpt41mini"),
                         ("qwen72b", "gridA_qwen72b", "deconfound_qwen72b")]:
        for p, (bn, bcf, bias, absd, flip) in crossfit(gr, dr).items():
            print(f"{name:<10}{p:<7}{bn:>7}{bcf:>7}{bias:>7}{absd:>7}{flip:>6}")
            maxbias = max(maxbias, abs(bias))
    print(f"max |naive-xfit| = {maxbias:.3f}")

    print("\n=== B. non-agent normative control (policy) vs character (C1), with CIs ===")
    for name, gr, dr, sp in [("Gemma", "gridA_gemma", "deconfound_gemma", "semprime_gemma"),
                             ("Qwen", "gridA_qwen72b", "deconfound_qwen72b", "semprime_qwen")]:
        r = normative_control(gr, dr, sp)
        for persona, v in r.items():
            print(f"  {name} {persona}: character(C1)={v['char_C1']}  policy={v['policy']}  "
                  f"policy-char={v['policy_minus_char']}")

if __name__ == "__main__":
    main()
