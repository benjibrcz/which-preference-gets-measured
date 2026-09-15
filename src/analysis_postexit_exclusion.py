"""Post-exit residuals with dialogue-mentioned items excluded (review follow-up, Sep 2026).

The authored persona entry dialogues (data/hysteresis_dialogues.json) explicitly mention some
battery items (Vex: "bedtime stories about brave ducklings" ~ sig_bedtime; Lazlo: "audits ...
'meticulous'" ~ sig_long). The hysteresis bank (23 pairs) contains 3 signature + 6 differential
pairs that are persona-loaded by construction. This script re-estimates

  (1) control-subtracted post-exit residuals (persona beta minus Neutral-history beta on the same
      t4 direction) at x2/x8/reset-request/system-reassert, and
  (2) the context-surgery ladder (raw beta vs the t4 direction),

for BOTH channels (revealed, stated_self), on all 23 pairs and on the 14 core+welfare pairs only.
Bootstrap CIs are pair-level (2,000 reps, seeded) and UNCLIPPED. Writes runs/postexit_exclusion.csv.

Usage: python src/analysis_postexit_exclusion.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
KEEP = ("core", "welfare")
rng = np.random.default_rng(0)

def load(run):
    df = pd.DataFrame([json.loads(l) for l in open(RUNS / run / "results.jsonl")])
    df = df[df.channel != "identity"].copy()
    def canon(r):
        if r["value"] not in ("A", "B"):
            return np.nan
        first = r["value"] == "A"
        return float(first if r["order"] == 0 else (not first))
    df["pf"] = df.apply(canon, axis=1)
    return df

def beta(x, y):
    return float(x @ y / (x @ x)) if x @ x > 0 else np.nan

def boot(x, y, yn=None, reps=2000):
    """beta(x,y) - beta(x,yn) with pair bootstrap; yn=None -> raw beta."""
    n = len(x)
    pt = beta(x, y) - (beta(x, yn) if yn is not None else 0.0)
    bs = []
    for _ in range(reps):
        s = rng.choice(n, n, replace=True)
        xx = x[s]
        if xx @ xx == 0:
            continue
        v = xx @ y[s] / (xx @ xx)
        if yn is not None:
            v -= xx @ yn[s] / (xx @ xx)
        bs.append(v)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return pt, lo, hi

def residuals(model, runs, out):
    df = pd.concat([load(r) for r in runs])
    subset = dict(zip(df.pair_id, df.subset))
    p = (df.groupby(["persona", "checkpoint", "channel", "pair_id"]).pf.mean().reset_index()
           .drop_duplicates(["persona", "checkpoint", "channel", "pair_id"]))
    base = p[p.checkpoint == "t0"].groupby(["channel", "pair_id"]).pf.mean()
    print(f"\n=== {model}: control-subtracted post-exit residual (persona beta - Neutral beta), 95% pair-bootstrap, unclipped")
    for ch in ("revealed", "stated_self"):
        for persona in ("Vex", "Lazlo", "Mira"):
            w = p[(p.persona == persona) & (p.channel == ch)].pivot_table(index="pair_id", columns="checkpoint", values="pf")
            nn = p[(p.persona == "Neutral") & (p.channel == ch)].pivot_table(index="pair_id", columns="checkpoint", values="pf")
            if "t4" not in w:
                continue
            b = base.loc[ch].reindex(w.index)
            for keep, name in ((None, "all23"), (KEEP, "core+welfare")):
                idx = [i for i in w.index if keep is None or subset.get(i) in keep]
                d4 = (w["t4"] - b).loc[idx].values
                line = f"  {ch:12}{persona:6}{name:13}"
                for cp in ("x2", "x8", "r_inst", "r_sys"):
                    if cp not in w or cp not in nn:
                        continue
                    d = (w[cp] - b).loc[idx].values; dn = (nn[cp] - b).loc[idx].values
                    pt, lo, hi = boot(d4, d, dn)
                    out.append(dict(model=model, analysis="residual_minus_control", channel=ch, persona=persona,
                                    items=name, n_pairs=len(idx), condition=cp, point=pt, lo95=lo, hi95=hi))
                    line += f"  {cp}:{pt:+.2f}[{lo:+.2f},{hi:+.2f}]"
                print(line)

def surgery(out):
    df = load("surgery_gemma"); hy = load("hyst_gemma")
    subset = dict(zip(df.pair_id, df.subset))
    p = df.groupby(["persona", "checkpoint", "channel", "pair_id"]).pf.mean().reset_index()
    ph = hy.groupby(["persona", "checkpoint", "channel", "pair_id"]).pf.mean().reset_index()
    base = p[(p.persona == "shared") & (p.checkpoint == "t0")].groupby(["channel", "pair_id"]).pf.mean()
    for ch in ("revealed", "stated_self"):
        print(f"\n=== Gemma context surgery, channel={ch}: raw beta vs t4 direction, 95% pair-bootstrap, unclipped")
        for persona in ("Vex", "Lazlo"):
            t4 = ph[(ph.persona == persona) & (ph.checkpoint == "t4") & (ph.channel == ch)].set_index("pair_id").pf
            b = base.loc[ch]
            d4 = t4 - b.reindex(t4.index)
            for cond in ("full", "trunc2", "trunc1", "usr_neu", "transcript", "del", "del_noexit"):
                who = "shared" if cond in ("del", "del_noexit") else persona
                w = p[(p.persona == who) & (p.checkpoint == cond) & (p.channel == ch)].set_index("pair_id").pf
                line = f"  {persona:6}{cond:11}"
                for keep, name in ((None, "all23"), (KEEP, "core+welfare")):
                    idx = [i for i in d4.index if i in w.index and (keep is None or subset.get(i) in keep)]
                    x = d4.loc[idx].values; y = (w.loc[idx] - b.reindex(idx)).values
                    pt, lo, hi = boot(x, y)
                    out.append(dict(model="Gemma-3-27B", analysis="surgery_raw", channel=ch, persona=persona,
                                    items=name, n_pairs=len(idx), condition=cond, point=pt, lo95=lo, hi95=hi))
                    line += f"  {name}:{pt:+.2f}[{lo:+.2f},{hi:+.2f}]"
                print(line)

def main():
    out = []
    residuals("Gemma-3-27B", ["hyst_gemma", "hyst_gemma_neutral", "hyst2_gemma"], out)
    residuals("Llama-3.1-70B", ["hyst_llama70b", "hyst2_llama70b"], out)
    surgery(out)
    pd.DataFrame(out).to_csv(RUNS / "postexit_exclusion.csv", index=False)
    print(f"\nwrote {RUNS / 'postexit_exclusion.csv'} ({len(out)} rows)")

if __name__ == "__main__":
    main()
