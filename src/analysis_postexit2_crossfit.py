"""Cross-fitted loop-7 residuals (GPT #7 robustness check on the new bank): the history's t4 direction is estimated
from samples {0,1} and the post-exit displacement (and the matched control's) from samples {2,3}, then the folds are
swapped and the two estimates averaged, so the projection never reuses the noise that defined the direction.
Reported for auth_/pref_ x Vex/Lazlo x {revealed, stated_self} at x2, noexit2, x8, r_inst and the pooled generated exits.
Usage: python src/analysis_postexit2_crossfit.py [runs/postexit2_gemma]   -> <run>/postexit2_crossfit.csv"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/postexit2_gemma")
rng = np.random.default_rng(0)
CTRL = {"auth": "neutral_auth", "pref": "neutral_pref", "genA": "neutral_genA", "genB": "neutral_genB", "genC": "neutral_genC"}

def main():
    df = pd.DataFrame([json.loads(l) for l in open(RUN / "results.jsonl")])
    df = df[df.channel.isin(["revealed", "stated_self"]) & (df.bank == "new")].copy()
    df["pf"] = [np.nan if v not in ("A", "B") else float((v == "A") if o == 0 else (v != "A")) for v, o in zip(df.value, df.order)]
    df["fold"] = (df.sample_idx >= 2).astype(int)
    p = df.groupby(["history", "checkpoint", "channel", "fold", "pair_id"]).pf.mean()
    base = {f: p.loc["none"].loc["t0"].xs(f, level="fold") for f in (0, 1)}   # channel x pair
    def vec(h, cp, ch, f, idx):
        try:
            v = p.loc[(h, cp, ch, f)].reindex(idx)
        except KeyError:
            return None
        return (v - base[f].loc[ch].reindex(idx)).values
    out = []
    print("=== cross-fitted control-subtracted residuals, new bank (mean of the two fold assignments; 95% pair bootstrap)")
    for hid in [h for h in sorted(df.history.unique()) if h.startswith(("auth_", "pref_", "gen"))]:
        c = next(v for k, v in CTRL.items() if hid.startswith(k))
        for ch in ("revealed", "stated_self"):
            idx = p.loc[(hid, "t4", ch, 0)].index
            cps = sorted({cp for (h, cp, cc, f, _) in p.index if h == hid and cc == ch and cp != "t4"})
            pooled_cps = [cp for cp in cps if re.fullmatch(r"x2g\d", cp)]
            targets = [(cp, [cp]) for cp in ("x2", "noexit2", "x8", "r_inst") if cp in cps] + ([("x2g_pooled", pooled_cps)] if pooled_cps else [])
            line = f"  {hid:11}{ch:12}"
            for name, members in targets:
                X, Y, YN = [], [], []
                for fd, fy in ((0, 1), (1, 0)):
                    x = vec(hid, "t4", ch, fd, idx)
                    ys = [vec(hid, m, ch, fy, idx) for m in members]; yns = [vec(c, m, ch, fy, idx) for m in members]
                    if x is None or any(v is None for v in ys) or any(v is None for v in yns):
                        X = None; break
                    X.append(x); Y.append(np.nanmean(np.vstack(ys), 0)); YN.append(np.nanmean(np.vstack(yns), 0))
                if X is None:
                    continue
                def est(sel):
                    vals = []
                    for k in range(2):
                        x, y, yn = X[k][sel], Y[k][sel], YN[k][sel]
                        ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(yn)); x, y, yn = x[ok], y[ok], yn[ok]
                        if x @ x == 0:
                            return np.nan
                        vals.append((x @ y - x @ yn) / (x @ x))
                    return float(np.mean(vals))
                n = len(idx); pt = est(np.arange(n))
                bs = [v for v in (est(rng.choice(n, n, replace=True)) for _ in range(2000)) if not np.isnan(v)]
                lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
                out.append(dict(history=hid, channel=ch, checkpoint=name, metric="beta_minus_control_crossfit", point=pt, lo95=lo, hi95=hi, n_pairs=n))
                line += f"  {name}:{pt:+.2f}[{lo:+.2f},{hi:+.2f}]"
            print(line)
    pd.DataFrame(out).to_csv(RUN / "postexit2_crossfit.csv", index=False)
    # side-by-side with the in-sample estimates
    ins = pd.read_csv(RUN / "postexit2_results.csv")
    ins = ins[(ins.metric == "beta_minus_control") & (ins.bank == "new")][["history", "channel", "checkpoint", "point"]].rename(columns={"point": "in_sample"})
    cf = pd.DataFrame(out).merge(ins, on=["history", "channel", "checkpoint"], how="left")
    cf["diff"] = cf.point - cf.in_sample
    print("\n=== cross-fit minus in-sample, summary over cells: mean %.3f, max |diff| %.3f (n=%d)" % (cf["diff"].mean(), cf["diff"].abs().max(), len(cf)))
    print(cf[cf.history.str.startswith(("auth_", "pref_"))].round(2).to_string(index=False))

if __name__ == "__main__":
    main()
