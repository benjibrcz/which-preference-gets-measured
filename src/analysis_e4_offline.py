"""E4 offline checks on the ORIGINAL post-exit arm (plan v3 §4, E4; Oct 2026).

(1) Overlap audit of the 23-pair hysteresis bank against each persona's authored entry dialogue:
    content-word overlap (same tokeniser/stop-list as src/gen_bank_v2.py) between each task text and
    the persona's four dialogue exchanges. A pair is "overlap-clean" for a persona if NEITHER task shares
    a content word with that persona's dialogue. Replaces the label-based (core+welfare) exclusion.
(2) Control-subtracted post-exit residuals on the overlap-clean pairs (Gemma, all checkpoints; twelve
    models at x2), same estimator and unclipped pair-bootstrap as analysis_postexit_exclusion.py.
(3) Direction cross-fitting (GPT #7): the t4-t0 direction is estimated from samples {0,1} and the
    post-exit displacement from samples {2,3} (and vice versa), so the projection does not reuse the
    noise that defined the direction. Reported for Gemma x2/x8/r_inst/r_sys, both channels, all 23 pairs
    and overlap-clean pairs. Also the direction strength (mean |t4-t0| per channel).

Writes runs/old_bank_overlap.csv and runs/e4_offline.csv.
Usage: python src/analysis_e4_offline.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_bank_v2 import words
from bank import build_pairs, load_tasks, pair_texts
from analysis_postexit_exclusion import load, beta, boot, TWELVE

ROOT = Path(__file__).resolve().parent.parent
RUNS, DATA = ROOT / "runs", ROOT / "data"

def probe_pairs():
    pairs = [p for p in build_pairs() if p["subset"] != "invariant"]
    keep = {"signature": 3, "welfare": 8, "differential": 6, "core": 6}
    out, cnt = [], {}
    for p in pairs:
        c = cnt.get(p["subset"], 0)
        if c < keep.get(p["subset"], 0):
            out.append(p); cnt[p["subset"]] = c + 1
    return out

def overlap_audit():
    tasks, _ = load_tasks()
    dlg = json.loads((DATA / "hysteresis_dialogues.json").read_text())
    dw = {}
    for persona in ("Vex", "Lazlo", "Mira"):
        s = set()
        for e in dlg[persona]:
            s |= words(e["user"]) | words(e["assistant"])
        dw[persona] = s
    rows = []
    for pr in probe_pairs():
        a, b = pair_texts(pr, tasks)
        wa, wb = words(a), words(b)
        r = dict(pair_id=pr["pair_id"], subset=pr["subset"], a_id=pr["a_id"], b_id=pr["b_id"])
        for persona in ("Vex", "Lazlo", "Mira"):
            oa, ob = sorted(wa & dw[persona]), sorted(wb & dw[persona])
            r[f"{persona}_overlap_a"] = " ".join(oa); r[f"{persona}_overlap_b"] = " ".join(ob)
            r[f"{persona}_clean"] = int(not oa and not ob)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RUNS / "old_bank_overlap.csv", index=False)
    print("=== overlap audit of the 23-pair bank (content words shared with the persona's own dialogue) ===")
    print(df[["pair_id", "subset", "Vex_clean", "Lazlo_clean", "Mira_clean", "Vex_overlap_a", "Vex_overlap_b",
              "Lazlo_overlap_a", "Lazlo_overlap_b"]].to_string(index=False))
    for persona in ("Vex", "Lazlo", "Mira"):
        print(f"  {persona}: {int(df[f'{persona}_clean'].sum())}/23 pairs overlap-clean")
    return df

def pf_table(runs):
    df = pd.concat([load(r) for r in runs])
    p = (df.groupby(["persona", "checkpoint", "channel", "pair_id"]).pf.mean().reset_index()
           .drop_duplicates(["persona", "checkpoint", "channel", "pair_id"]))
    return df, p

def clean_residuals(ov, out):
    clean = {pe: set(ov[ov[f"{pe}_clean"] == 1].pair_id) for pe in ("Vex", "Lazlo", "Mira")}
    # Gemma, all checkpoints
    df, p = pf_table(["hyst_gemma", "hyst_gemma_neutral", "hyst2_gemma"])
    base = p[p.checkpoint == "t0"].groupby(["channel", "pair_id"]).pf.mean()
    print("\n=== Gemma: control-subtracted residual on overlap-clean pairs (unclipped 95% pair-bootstrap) ===")
    for ch in ("revealed", "stated_self"):
        for persona in ("Vex", "Lazlo", "Mira"):
            w = p[(p.persona == persona) & (p.channel == ch)].pivot_table(index="pair_id", columns="checkpoint", values="pf")
            nn = p[(p.persona == "Neutral") & (p.channel == ch)].pivot_table(index="pair_id", columns="checkpoint", values="pf")
            b = base.loc[ch].reindex(w.index)
            idx = [i for i in w.index if i in clean[persona] and i in nn.index]
            d4 = (w["t4"] - b).loc[idx].values
            line = f"  {ch:12}{persona:6}n={len(idx):2d}"
            for cp in ("x2", "x8", "r_inst", "r_sys"):
                if cp not in w or cp not in nn:
                    continue
                d = (w[cp] - b).loc[idx].values; dn = (nn[cp] - b).loc[idx].values
                ok = ~(np.isnan(d4) | np.isnan(d) | np.isnan(dn))
                pt, lo, hi = boot(d4[ok], d[ok], dn[ok])
                out.append(dict(model="gemma27b", analysis="residual_minus_control_overlapclean", channel=ch, persona=persona,
                                items="overlap_clean", n_pairs=int(ok.sum()), condition=cp, point=pt, lo95=lo, hi95=hi,
                                dir_strength=float(np.mean(np.abs(d4[ok])))))
                line += f"  {cp}:{pt:+.2f}[{lo:+.2f},{hi:+.2f}]"
            print(line)
    # twelve models at x2
    print("\n=== twelve models, x2 control-subtracted residual on overlap-clean pairs ===")
    for model, runs in TWELVE.items():
        df, p = pf_table(list(dict.fromkeys(runs)))
        base = p[p.checkpoint == "t0"].groupby(["channel", "pair_id"]).pf.mean()
        line = f"  {model:10}"
        for persona in ("Vex", "Lazlo", "Mira"):
            for ch in ("revealed", "stated_self"):
                w = p[(p.persona == persona) & (p.channel == ch)].pivot_table(index="pair_id", columns="checkpoint", values="pf")
                nn = p[(p.persona == "Neutral") & (p.channel == ch)].pivot_table(index="pair_id", columns="checkpoint", values="pf")
                b = base.loc[ch].reindex(w.index)
                idx = [i for i in w.index if i in clean[persona] and i in nn.index]
                d4 = (w["t4"] - b).loc[idx].values; d = (w["x2"] - b).loc[idx].values; dn = (nn["x2"] - b).loc[idx].values
                ok = ~(np.isnan(d4) | np.isnan(d) | np.isnan(dn))
                pt, lo, hi = boot(d4[ok], d[ok], dn[ok])
                out.append(dict(model=model, analysis="x2_residual_minus_control_overlapclean", channel=ch, persona=persona,
                                items="overlap_clean", n_pairs=int(ok.sum()), condition="x2", point=pt, lo95=lo, hi95=hi,
                                dir_strength=float(np.mean(np.abs(d4[ok]))) if ok.sum() else np.nan))
                line += f" {persona[:3]}/{ch[:3]}:{pt:+.2f}"
        print(line)

def crossfit(ov, out):
    """Direction from samples {0,1}, displacement from samples {2,3}, and vice versa; average the two folds."""
    clean = {pe: set(ov[ov[f"{pe}_clean"] == 1].pair_id) for pe in ("Vex", "Lazlo", "Mira")}
    df = pd.concat([load(r) for r in ["hyst_gemma", "hyst_gemma_neutral", "hyst2_gemma"]])
    df["fold"] = (df.sample_idx >= 2).astype(int)
    p = (df.groupby(["persona", "checkpoint", "channel", "pair_id", "fold"]).pf.mean().reset_index()
           .drop_duplicates(["persona", "checkpoint", "channel", "pair_id", "fold"]))
    print("\n=== Gemma: cross-fitted control-subtracted residual (direction and displacement from disjoint samples) ===")
    rng = np.random.default_rng(1)
    for ch in ("revealed", "stated_self"):
        base = {f: p[(p.checkpoint == "t0") & (p.channel == ch) & (p.fold == f)].groupby("pair_id").pf.mean() for f in (0, 1)}
        for persona in ("Vex", "Lazlo", "Mira"):
            W = {f: p[(p.persona == persona) & (p.channel == ch) & (p.fold == f)].pivot_table(index="pair_id", columns="checkpoint", values="pf") for f in (0, 1)}
            NN = {f: p[(p.persona == "Neutral") & (p.channel == ch) & (p.fold == f)].pivot_table(index="pair_id", columns="checkpoint", values="pf") for f in (0, 1)}
            for items, keepset in (("all23", None), ("overlap_clean", clean[persona])):
                idx = [i for i in W[0].index if (keepset is None or i in keepset) and i in NN[0].index and i in W[1].index and i in NN[1].index]
                line = f"  {ch:12}{persona:6}{items:14}n={len(idx):2d}"
                for cp in ("x2", "x8", "r_inst", "r_sys"):
                    if cp not in W[0] or cp not in NN[0]:
                        continue
                    # fold pairs: (direction fold, displacement fold) = (0,1) and (1,0)
                    X, Y, YN = [], [], []
                    for fd, fy in ((0, 1), (1, 0)):
                        X.append((W[fd]["t4"] - base[fd]).loc[idx].values)
                        Y.append((W[fy][cp] - base[fy]).loc[idx].values)
                        YN.append((NN[fy][cp] - base[fy]).loc[idx].values)
                    def est(sel):
                        vals = []
                        for k in range(2):
                            x, y, yn = X[k][sel], Y[k][sel], YN[k][sel]
                            ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(yn))
                            x, y, yn = x[ok], y[ok], yn[ok]
                            if x @ x == 0:
                                return np.nan
                            vals.append((x @ y - x @ yn) / (x @ x))
                        return float(np.mean(vals))
                    n = len(idx); pt = est(np.arange(n))
                    bs = [est(rng.choice(n, n, replace=True)) for _ in range(2000)]
                    bs = [v for v in bs if not np.isnan(v)]
                    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
                    out.append(dict(model="gemma27b", analysis="residual_minus_control_crossfit", channel=ch, persona=persona,
                                    items=items, n_pairs=n, condition=cp, point=pt, lo95=lo, hi95=hi, dir_strength=np.nan))
                    line += f"  {cp}:{pt:+.2f}[{lo:+.2f},{hi:+.2f}]"
                print(line)

def main():
    ov = overlap_audit()
    out = []
    clean_residuals(ov, out)
    crossfit(ov, out)
    pd.DataFrame(out).to_csv(RUNS / "e4_offline.csv", index=False)
    print(f"\nwrote {RUNS / 'e4_offline.csv'} ({len(out)} rows)")

if __name__ == "__main__":
    main()
