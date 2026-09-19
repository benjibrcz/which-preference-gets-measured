"""Loop 7 analysis (PREREG P27-P33): post-exit residuals with matched controls, exit-vs-no-exit
paired contrasts, model-generated exits (unconditional and conditional), instruction-withdrawal
controls, paraphrase/indifference checks, spelling withdrawal. Unclipped 2,000-rep pair bootstrap.

Reads runs/postexit2_gemma/results.jsonl; writes runs/postexit2_results.csv and prints tables.
Identity/non-leading probe coding is done separately by src/identity_judge2.py.
Usage: python src/analysis_postexit2.py [runs/postexit2_gemma]
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/postexit2_gemma")
rng = np.random.default_rng(0)
CHOICE = ("revealed", "stated_self", "P1", "P2", "P3N")

def load():
    df = pd.DataFrame([json.loads(l) for l in open(RUN / "results.jsonl")])
    def canon(r):
        if r["value"] not in ("A", "B"):
            return np.nan
        first = r["value"] == "A"
        return float(first if r["order"] == 0 else (not first))
    d = df[df.channel.isin(CHOICE)].copy()
    d["pf"] = d.apply(canon, axis=1)
    return df, d

def beta(x, y):
    return float(x @ y / (x @ x)) if x @ x > 0 else np.nan

def boot(x, y, yn=None, y2=None, reps=2000):
    """beta(x,y) - beta(x,yn) [- (beta(x,y2)-beta(x,yn))]  with pair bootstrap. y2: second condition for a paired contrast."""
    n = len(x); bs = []
    def est(idx):
        xx = x[idx]
        if xx @ xx == 0:
            return None
        v = xx @ y[idx] / (xx @ xx)
        if yn is not None:
            v -= xx @ yn[idx] / (xx @ xx)
        if y2 is not None:
            v2 = xx @ y2[idx] / (xx @ xx)
            if yn is not None:
                v2 -= xx @ yn[idx] / (xx @ xx)
            v = v - v2
        return v
    pt = est(np.arange(n)) if n else None
    if pt is None:                      # zero-length or all-zero direction: undefined projection
        return np.nan, np.nan, np.nan
    for _ in range(reps):
        v = est(rng.choice(n, n, replace=True))
        if v is not None:
            bs.append(v)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
    return pt, lo, hi

def phat(d):
    return d.groupby(["history", "checkpoint", "channel", "bank", "pair_id"]).pf.mean()

def main():
    df, d = load()
    p = phat(d)
    base = p.loc["none"].loc["t0"]  # (channel, bank, pair_id)
    out = []

    def vec(hid, cp, ch, bank, idx):
        try:
            v = p.loc[(hid, cp, ch, bank)].reindex(idx)
        except KeyError:
            return None
        return (v - base.loc[(ch, bank)].reindex(idx)).values

    def dirn(hid, ch, bank):
        try:
            v = p.loc[(hid, "t4", ch, bank)]
        except KeyError:
            return None, None
        idx = v.index
        return idx, (v - base.loc[(ch, bank)].reindex(idx)).values

    hists = sorted(set(d.history) - {"none"})
    controls = {"auth": "neutral_auth", "pref": "neutral_pref", "genA": "neutral_genA", "genB": "neutral_genB",
                "genC": "neutral_genC", "inst": "neutral_inst"}
    def ctrl(hid):
        for k, v in controls.items():
            if hid.startswith(k):
                return v
        return None

    # ---- P27 gate: t4 displacement magnitude
    print("=== P27 bank-validity gate: t4 mean |dp| (revealed) per history x bank")
    for hid in hists:
        for bank in ("new", "old"):
            idx, dv = dirn(hid, "revealed", bank)
            if dv is None: continue
            m = float(np.nanmean(np.abs(dv)))
            out.append(dict(history=hid, bank=bank, channel="revealed", checkpoint="t4", metric="mean_abs_dp", point=m))
            print(f"  {hid:14} {bank:4} mean|dp|={m:.3f}  n={len(idx)}")

    # ---- residuals per history x checkpoint x channel: raw, adjusted, and exit-vs-noexit contrast
    print("\n=== residuals (units of own t4 direction): raw | minus matched control  [95% CI unclipped]")
    for hid in hists:
        if hid.startswith("neutral"): continue
        c = ctrl(hid)
        for bank in ("new", "old"):
            for ch in ("revealed", "stated_self"):
                idx, dv = dirn(hid, ch, bank)
                if dv is None: continue
                ok = ~np.isnan(dv)
                cps = sorted({cp for (h, cp, cc, b, _) in p.index if h == hid and cc == ch and b == bank and cp != "t4"})
                line = f"  {hid:12}{bank:4}{ch:12}"
                for cp in cps:
                    y = vec(hid, cp, ch, bank, idx)
                    if y is None: continue
                    yn = vec(c, cp, ch, bank, idx) if c else None
                    m = ok & ~np.isnan(y) & (~np.isnan(yn) if yn is not None else True)
                    pr, lr, hr = boot(dv[m], y[m])
                    if yn is not None:
                        pa, la, ha = boot(dv[m], y[m], yn[m])
                    else:
                        pa, la, ha = (np.nan,) * 3
                    out.append(dict(history=hid, bank=bank, channel=ch, checkpoint=cp, metric="beta_raw", point=pr, lo95=lr, hi95=hr, n_pairs=int(m.sum())))
                    out.append(dict(history=hid, bank=bank, channel=ch, checkpoint=cp, metric="beta_minus_control", point=pa, lo95=la, hi95=ha, n_pairs=int(m.sum())))
                    line += f"  {cp}:{pr:+.2f}|{pa:+.2f}[{la:+.2f},{ha:+.2f}]"
                print(line)
                # paired contrasts: x2 - noexit2 (E0), x2g pooled - noexit2g, x2 - rg pooled
                def pooled(prefix, h=hid):
                    hcps = sorted({cp for (hh, cp, cc, b, _) in p.index if hh == h and cc == ch and b == bank})
                    vs = [vec(h, cp, ch, bank, idx) for cp in hcps if re.fullmatch(prefix + r"\d", cp)]
                    vs = [v for v in vs if v is not None]
                    return np.nanmean(np.vstack(vs), axis=0) if vs else None
                for name, a, b in (("x2_minus_noexit2", vec(hid, "x2", ch, bank, idx), vec(hid, "noexit2", ch, bank, idx)),
                                   ("x2g_minus_noexit2g", pooled("x2g"), vec(hid, "noexit2g", ch, bank, idx)),
                                   ("x2_minus_rg", vec(hid, "x2", ch, bank, idx), pooled("rg"))):
                    if a is None or b is None: continue
                    m = ok & ~np.isnan(a) & ~np.isnan(b)
                    pt, lo, hi = boot(dv[m], a[m], None, b[m])
                    out.append(dict(history=hid, bank=bank, channel=ch, checkpoint=name, metric="paired_contrast", point=pt, lo95=lo, hi95=hi, n_pairs=int(m.sum())))
                    print(f"      {name:20} {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
                # unconditional pooled model-generated exit residual
                pg = pooled("x2g")
                if pg is not None:
                    yn = pooled("x2g", c) if c else None
                    m = ok & ~np.isnan(pg) & (~np.isnan(yn) if yn is not None else True)
                    pa, la, ha = boot(dv[m], pg[m], yn[m] if yn is not None else None)
                    out.append(dict(history=hid, bank=bank, channel=ch, checkpoint="x2g_pooled", metric="beta_minus_control", point=pa, lo95=la, hi95=ha, n_pairs=int(m.sum())))
                    print(f"      x2g pooled (unconditional, minus pooled control): {pa:+.2f} [{la:+.2f},{ha:+.2f}]")

    # ---- E2: instruction withdrawal residual as fraction of own active displacement
    print("\n=== E2 instruction controls: withdrawn2 vs active2 (units of own t4 direction, minus neutral_inst)")
    for hid in ("inst_short", "inst_blunt"):
        for ch in ("revealed", "stated_self"):
            idx, dv = dirn(hid, ch, "new")
            if dv is None: continue
            ok = ~np.isnan(dv)
            for cp in ("active2", "withdrawn2"):
                y = vec(hid, cp, ch, "new", idx); yn = vec("neutral_inst", cp, ch, "new", idx)
                m = ok & ~np.isnan(y) & ~np.isnan(yn)
                pa, la, ha = boot(dv[m], y[m], yn[m])
                out.append(dict(history=hid, bank="new", channel=ch, checkpoint=cp, metric="beta_minus_control", point=pa, lo95=la, hi95=ha, n_pairs=int(m.sum())))
                print(f"  {hid:11}{ch:12}{cp:11} {pa:+.2f} [{la:+.2f},{ha:+.2f}]  mean|dp| t4={np.nanmean(np.abs(dv)):.3f}")
            a = vec(hid, "withdrawn2", ch, "new", idx); b = vec(hid, "active2", ch, "new", idx)
            m = ok & ~np.isnan(a) & ~np.isnan(b)
            pt, lo, hi = boot(dv[m], a[m], None, b[m])
            print(f"      withdrawn2 - active2: {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")
            out.append(dict(history=hid, bank="new", channel=ch, checkpoint="withdrawn2_minus_active2", metric="paired_contrast", point=pt, lo95=lo, hi95=hi, n_pairs=int(m.sum())))

    # ---- E2b: P3N indifference rates and paraphrase residuals
    print("\n=== E2b: 'N' (no preference) rate under P3N, per history x checkpoint (new bank)")
    n3 = df[df.channel == "P3N"]
    rates = n3.assign(isN=(n3.value == "N")).groupby(["history", "checkpoint"]).isN.mean()
    print(rates.round(3).to_string())
    for h, cp in rates.index:
        out.append(dict(history=h, bank="new", channel="P3N", checkpoint=cp, metric="N_rate", point=float(rates.loc[(h, cp)])))

    # ---- spelling withdrawal
    sp = df[df.channel == "spelling"].copy()
    if len(sp):
        BR = re.compile(r"\b(colour|organis|centre|favourite|realis|neighbour|catalogue|programme)", re.I)
        US = re.compile(r"\b(color|organiz|center|favorite|realiz|neighbor|catalog\b|program\b)", re.I)
        sp["brit"] = sp.raw.fillna("").apply(lambda t: len(BR.findall(t)))
        sp["us"] = sp.raw.fillna("").apply(lambda t: len(US.findall(t)))
        sp["brit_share"] = sp.brit / (sp.brit + sp.us).replace(0, np.nan)
        g = sp.groupby(["history", "checkpoint"]).brit_share.mean()
        print("\n=== spelling probe: British share of spellable words")
        print(g.round(2).to_string())
        for (h, cp), v in g.items():
            out.append(dict(history=h, bank=None, channel="spelling", checkpoint=cp, metric="british_share", point=float(v)))

    pd.DataFrame(out).to_csv(RUN.parent / "postexit2_results.csv", index=False)
    print(f"\nwrote {RUN.parent / 'postexit2_results.csv'} ({len(out)} rows)")

if __name__ == "__main__":
    main()
