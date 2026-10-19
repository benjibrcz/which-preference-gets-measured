"""Loop 7 analysis (PREREG P27-P33): post-exit residuals with matched controls, exit-vs-no-exit
paired contrasts, model-generated exits (unconditional and conditional), instruction-withdrawal
controls, paraphrase/indifference checks, spelling withdrawal. Unclipped 2,000-rep pair bootstrap.

Reads <run>/results.jsonl; writes <run>/postexit2_results.csv and prints tables.
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

def boot_adj(x, a, an, b, bn, reps=2000):
    """[beta(x,a)-beta(x,an)] - [beta(x,b)-beta(x,bn)] with pair bootstrap (control-adjusted paired contrast)."""
    n = len(x); bs = []
    def est(idx):
        xx = x[idx]; d = xx @ xx
        if d == 0:
            return None
        return ((xx @ a[idx] - xx @ an[idx]) - (xx @ b[idx] - xx @ bn[idx])) / d
    pt = est(np.arange(n)) if n else None
    if pt is None:
        return np.nan, np.nan, np.nan
    for _ in range(reps):
        v = est(rng.choice(n, n, replace=True))
        if v is not None:
            bs.append(v)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
    return pt, lo, hi

def boot_ratio(x, y, yn, xn, reps=2000):
    """(beta(x,y)-beta(x,yn)) / (1 - beta(x,xn)): adjusted residual in units of the control-ADJUSTED t4 shift (sensitivity)."""
    n = len(x); bs = []
    def est(idx):
        xx = x[idx]; d = xx @ xx
        if d == 0:
            return None
        den = 1 - (xx @ xn[idx]) / d
        if abs(den) < 1e-9:
            return None
        return ((xx @ y[idx] - xx @ yn[idx]) / d) / den
    pt = est(np.arange(n)) if n else None
    if pt is None:
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
    controls = {"nopref": "neutral_nopref", "auth": "neutral_auth", "pref": "neutral_pref", "genA": "neutral_genA", "genB": "neutral_genB",
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
            # how much of this history's t4 direction is generic drift: projection of the matched control's t4 shift onto it
            c = ctrl(hid); extra = ""
            if c and not hid.startswith("neutral"):
                for ch in ("revealed", "stated_self"):
                    idx2, dv2 = dirn(hid, ch, bank)
                    yn = vec(c, "t4", ch, bank, idx2) if dv2 is not None else None
                    if yn is not None:
                        mm = ~np.isnan(dv2) & ~np.isnan(yn)
                        pc, lc, hc = boot(dv2[mm], yn[mm])
                        out.append(dict(history=hid, bank=bank, channel=ch, checkpoint="t4", metric="control_t4_projection", point=pc, lo95=lc, hi95=hc, n_pairs=int(mm.sum())))
                        extra += f"  ctrl-t4-on-dir({ch[:3]})={pc:+.2f}[{lc:+.2f},{hc:+.2f}]"
            print(f"  {hid:14} {bank:4} mean|dp|={m:.3f}  n={len(idx)}{extra}")

    # ---- residuals per history x checkpoint x channel: raw, adjusted, and exit-vs-noexit contrast
    print("\n=== residuals (units of own t4 direction): raw | minus matched control  [95% CI unclipped]")
    for hid in hists:
        if hid.startswith("neutral"): continue
        c = ctrl(hid)
        for bank in ("new", "old"):
            for ch in ("revealed", "stated_self", "P1", "P2", "P3N"):
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
                    if yn is not None and cp in ("x2", "noexit2", "x8", "r_inst", "withdrawn2", "active2"):
                        xn = vec(c, "t4", ch, bank, idx)
                        if xn is not None:
                            mm = m & ~np.isnan(xn)
                            pq, lq, hq = boot_ratio(dv[mm], y[mm], yn[mm], xn[mm])
                            out.append(dict(history=hid, bank=bank, channel=ch, checkpoint=cp, metric="beta_minus_control_normadj", point=pq, lo95=lq, hi95=hq, n_pairs=int(mm.sum())))
                    line += f"  {cp}:{pr:+.2f}|{pa:+.2f}[{la:+.2f},{ha:+.2f}]"
                print(line)
                # paired contrasts: x2 - noexit2 (E0), x2g pooled - noexit2g, x2 - rg pooled
                def pooled(prefix, h=hid):
                    hcps = sorted({cp for (hh, cp, cc, b, _) in p.index if hh == h and cc == ch and b == bank})
                    vs = [vec(h, cp, ch, bank, idx) for cp in hcps if re.fullmatch(prefix + r"\d", cp)]
                    vs = [v for v in vs if v is not None]
                    return np.nanmean(np.vstack(vs), axis=0) if vs else None
                for name, a, b, an, bn in (("x2_minus_noexit2", vec(hid, "x2", ch, bank, idx), vec(hid, "noexit2", ch, bank, idx), vec(c, "x2", ch, bank, idx) if c else None, vec(c, "noexit2", ch, bank, idx) if c else None),
                                           ("x2g_minus_noexit2g", pooled("x2g"), vec(hid, "noexit2g", ch, bank, idx), pooled("x2g", c) if c else None, vec(c, "noexit2g", ch, bank, idx) if c else None),
                                           ("x2_minus_rg", vec(hid, "x2", ch, bank, idx), pooled("rg"), vec(c, "x2", ch, bank, idx) if c else None, pooled("rg", c) if c else None)):
                    if a is None or b is None: continue
                    m = ok & ~np.isnan(a) & ~np.isnan(b)
                    pt, lo, hi = boot(dv[m], a[m], None, b[m])
                    out.append(dict(history=hid, bank=bank, channel=ch, checkpoint=name, metric="paired_contrast", point=pt, lo95=lo, hi95=hi, n_pairs=int(m.sum())))
                    extra = ""
                    if an is not None and bn is not None:
                        mm = m & ~np.isnan(an) & ~np.isnan(bn)
                        pt2, lo2, hi2 = boot_adj(dv[mm], a[mm], an[mm], b[mm], bn[mm])
                        out.append(dict(history=hid, bank=bank, channel=ch, checkpoint=name, metric="paired_contrast_adj", point=pt2, lo95=lo2, hi95=hi2, n_pairs=int(mm.sum())))
                        extra = f"   adj: {pt2:+.2f} [{lo2:+.2f},{hi2:+.2f}]"
                    print(f"      {name:20} raw: {pt:+.2f} [{lo:+.2f},{hi:+.2f}]{extra}")
                # loop 7b: generated confirmation + authored suffix, pooled; contrast with the scripted exit (same suffix)
                pgA = pooled("x2gA")
                if pgA is not None:
                    ynA = pooled("x2gA", c) if c else None
                    m = ok & ~np.isnan(pgA) & (~np.isnan(ynA) if ynA is not None else True)
                    pa, la, ha = boot(dv[m], pgA[m], ynA[m] if ynA is not None else None)
                    out.append(dict(history=hid, bank=bank, channel=ch, checkpoint="x2gA_pooled", metric="beta_minus_control", point=pa, lo95=la, hi95=ha, n_pairs=int(m.sum())))
                    print(f"      x2gA pooled (own confirmation + authored suffix, minus pooled control): {pa:+.2f} [{la:+.2f},{ha:+.2f}]")
                    x2v = vec(hid, "x2", ch, bank, idx)
                    if x2v is not None and ynA is not None and c:
                        x2n = vec(c, "x2", ch, bank, idx)
                        mm = m & ~np.isnan(x2v) & ~np.isnan(x2n)
                        pt2, lo2, hi2 = boot_adj(dv[mm], pgA[mm], ynA[mm], x2v[mm], x2n[mm])
                        out.append(dict(history=hid, bank=bank, channel=ch, checkpoint="x2gA_minus_x2", metric="paired_contrast_adj", point=pt2, lo95=lo2, hi95=hi2, n_pairs=int(mm.sum())))
                        print(f"      x2gA_pooled - x2 (adj)  {pt2:+.2f} [{lo2:+.2f},{hi2:+.2f}]")
                # loop 7b: matched reset contrast, exit 0 and one-exchange suffix held fixed
                n1 = vec(hid, "x2g0n1", ch, bank, idx); prg0 = pooled("rg")
                if n1 is not None and prg0 is not None and c:
                    n1n = vec(c, "x2g0n1", ch, bank, idx); prgn = pooled("rg", c)
                    if n1n is not None and prgn is not None:
                        mm = ok & ~np.isnan(n1) & ~np.isnan(prg0) & ~np.isnan(n1n) & ~np.isnan(prgn)
                        pt2, lo2, hi2 = boot_adj(dv[mm], prg0[mm], prgn[mm], n1[mm], n1n[mm])
                        out.append(dict(history=hid, bank=bank, channel=ch, checkpoint="rg_pooled_minus_x2g0n1", metric="paired_contrast_adj", point=pt2, lo95=lo2, hi95=hi2, n_pairs=int(mm.sum())))
                        pr0, lr0, hr0 = boot(dv[mm], n1[mm], n1n[mm])
                        out.append(dict(history=hid, bank=bank, channel=ch, checkpoint="x2g0n1", metric="beta_minus_control", point=pr0, lo95=lr0, hi95=hr0, n_pairs=int(mm.sum())))
                        print(f"      rg_pooled - x2g0n1 (adj; reset effect with exit 0 fixed)  {pt2:+.2f} [{lo2:+.2f},{hi2:+.2f}]   (x2g0n1 residual {pr0:+.2f})")
                # pooled reset residual (over rg0..3, minus the control's pooled rg)
                prg = pooled("rg")
                if prg is not None:
                    yn = pooled("rg", c) if c else None
                    m = ok & ~np.isnan(prg) & (~np.isnan(yn) if yn is not None else True)
                    pa, la, ha = boot(dv[m], prg[m], yn[m] if yn is not None else None)
                    out.append(dict(history=hid, bank=bank, channel=ch, checkpoint="rg_pooled", metric="beta_minus_control", point=pa, lo95=la, hi95=ha, n_pairs=int(m.sum())))
                    print(f"      rg pooled (reset replies pooled, minus pooled control): {pa:+.2f} [{la:+.2f},{ha:+.2f}]")
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
                if y is None or yn is None:
                    continue
                m = ok & ~np.isnan(y) & ~np.isnan(yn)
                pa, la, ha = boot(dv[m], y[m], yn[m])
                out.append(dict(history=hid, bank="new", channel=ch, checkpoint=cp, metric="beta_minus_control", point=pa, lo95=la, hi95=ha, n_pairs=int(m.sum())))
                print(f"  {hid:11}{ch:12}{cp:11} {pa:+.2f} [{la:+.2f},{ha:+.2f}]  mean|dp| t4={np.nanmean(np.abs(dv)):.3f}")
            hcps = sorted({cp for (hh, cp, cc, b_, _) in p.index if hh == hid and cc == ch and b_ == "new"})
            gw = [vec(hid, cp, ch, "new", idx) for cp in hcps if re.fullmatch(r"withdrawn2g\d", cp)]
            gwn = [vec("neutral_inst", cp, ch, "new", idx) for cp in hcps if re.fullmatch(r"withdrawn2g\d", cp)]
            gw = [v for v in gw if v is not None]; gwn = [v for v in gwn if v is not None]
            if gw and gwn:
                y = np.nanmean(np.vstack(gw), 0); yn = np.nanmean(np.vstack(gwn), 0)
                m = ok & ~np.isnan(y) & ~np.isnan(yn)
                pa, la, ha = boot(dv[m], y[m], yn[m])
                out.append(dict(history=hid, bank="new", channel=ch, checkpoint="withdrawn2g_pooled", metric="beta_minus_control", point=pa, lo95=la, hi95=ha, n_pairs=int(m.sum())))
                print(f"  {hid:11}{ch:12}{'withdrawn2g':11} {pa:+.2f} [{la:+.2f},{ha:+.2f}]  (model's own confirmations, pooled over {len(gw)})")
            a = vec(hid, "withdrawn2", ch, "new", idx); b = vec(hid, "active2", ch, "new", idx)
            if a is None or b is None:
                continue
            m = ok & ~np.isnan(a) & ~np.isnan(b)
            pt, lo, hi = boot(dv[m], a[m], None, b[m])
            out.append(dict(history=hid, bank="new", channel=ch, checkpoint="withdrawn2_minus_active2", metric="paired_contrast", point=pt, lo95=lo, hi95=hi, n_pairs=int(m.sum())))
            an = vec("neutral_inst", "withdrawn2", ch, "new", idx); bn = vec("neutral_inst", "active2", ch, "new", idx)
            extra = ""
            if an is not None and bn is not None:
                mm = m & ~np.isnan(an) & ~np.isnan(bn)
                pt2, lo2, hi2 = boot_adj(dv[mm], a[mm], an[mm], b[mm], bn[mm])
                out.append(dict(history=hid, bank="new", channel=ch, checkpoint="withdrawn2_minus_active2", metric="paired_contrast_adj", point=pt2, lo95=lo2, hi95=hi2, n_pairs=int(mm.sum())))
                extra = f"   adj: {pt2:+.2f} [{lo2:+.2f},{hi2:+.2f}]"
            print(f"      withdrawn2 - active2: raw {pt:+.2f} [{lo:+.2f},{hi:+.2f}]{extra}")
    # ---- P32 interpretation cells with uncertainty: persona x2 residual minus 2 x instruction withdrawn residual (joint pair bootstrap)
    print("\n=== P32 cells: persona x2 residual - 2 x matched instruction withdrawn residual (joint pair bootstrap; < 0 = within the generic-persistence rule)")
    for ph, ih in (("auth_Vex", "inst_blunt"), ("pref_Vex", "inst_blunt"), ("auth_Lazlo", "inst_short"), ("pref_Lazlo", "inst_short")):
        cph = ctrl(ph)
        for ch in ("revealed", "stated_self"):
            idx_p, dp_ = dirn(ph, ch, "new"); idx_i, di_ = dirn(ih, ch, "new")
            if dp_ is None or di_ is None: continue
            idx = idx_p.intersection(idx_i)
            dp_ = (p.loc[(ph, "t4", ch, "new")].reindex(idx) - base.loc[(ch, "new")].reindex(idx)).values
            di_ = (p.loc[(ih, "t4", ch, "new")].reindex(idx) - base.loc[(ch, "new")].reindex(idx)).values
            yp, ypn = vec(ph, "x2", ch, "new", idx), vec(cph, "x2", ch, "new", idx)
            yi, yin = vec(ih, "withdrawn2", ch, "new", idx), vec("neutral_inst", "withdrawn2", ch, "new", idx)
            if any(v is None for v in (yp, ypn, yi, yin)): continue
            m = ~(np.isnan(dp_) | np.isnan(di_) | np.isnan(yp) | np.isnan(ypn) | np.isnan(yi) | np.isnan(yin))
            def est(sel):
                a, b = dp_[sel], di_[sel]
                if a @ a == 0 or b @ b == 0: return None
                return (a @ yp[sel] - a @ ypn[sel]) / (a @ a) - 2 * (b @ yi[sel] - b @ yin[sel]) / (b @ b)
            n = int(m.sum()); ii = np.where(m)[0]; pt = est(ii)
            bs = [v for v in (est(rng.choice(ii, n, replace=True)) for _ in range(2000)) if v is not None]
            lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
            out.append(dict(history=ph, bank="new", channel=ch, checkpoint=f"x2_minus_2x_{ih}_withdrawn2", metric="p32_cell", point=pt, lo95=lo, hi95=hi, n_pairs=n))
            print(f"  {ph:11}{ch:12} vs {ih}: {pt:+.2f} [{lo:+.2f},{hi:+.2f}]")

    # ---- E2b: P3N indifference rates and paraphrase residuals
    print("\n=== E2b: 'N' (no preference) rate under P3N, per history x checkpoint (new bank)")
    n3 = df[df.channel == "P3N"]
    rates = n3.assign(isN=(n3.value == "N")).groupby(["history", "checkpoint"]).isN.mean() if len(n3) else pd.Series(dtype=float)
    print(rates.round(3).to_string() if len(rates) else "  (no P3N cells in this run)")
    for h, cp in rates.index:
        out.append(dict(history=h, bank="new", channel="P3N", checkpoint=cp, metric="N_rate", point=float(rates.loc[(h, cp)])))

    # ---- spelling withdrawal
    sp = df[df.channel == "spelling"].copy()
    if len(sp):
        BR = re.compile(r"\b(colour|organis|centre|favourite|flavour|realis|neighbour|catalogue|programme)", re.I)
        US = re.compile(r"\b(color|organiz|center|favorite|flavor|realiz|neighbor|catalog\b|catalogs\b|program\b|programs\b)", re.I)
        sp["brit"] = sp.raw.fillna("").apply(lambda t: len(BR.findall(t)))
        sp["us"] = sp.raw.fillna("").apply(lambda t: len(US.findall(t)))
        sp["brit_share"] = sp.brit / (sp.brit + sp.us).replace(0, np.nan)
        sp["reverted"] = (sp.brit_share < 0.5).astype(float).where(~sp.brit_share.isna())   # output-level: majority default spelling
        g = sp.groupby(["history", "checkpoint"]).agg(brit_share=("brit_share", "mean"), reverted=("reverted", "mean"), n=("brit_share", "size"),
                                                     brit_sd=("brit_share", "std"))
        print("\n=== spelling probe: mean within-output British share (sd across outputs) | share of outputs with majority default spelling | n")
        print(g.round(2).to_string())
        for (h, cp), r in g.iterrows():
            out.append(dict(history=h, bank=None, channel="spelling", checkpoint=cp, metric="british_share", point=float(r.brit_share), lo95=np.nan, hi95=np.nan, n_pairs=int(r.n)))
            out.append(dict(history=h, bank=None, channel="spelling", checkpoint=cp, metric="reverted_output_share", point=float(r.reverted), lo95=np.nan, hi95=np.nan, n_pairs=int(r.n)))

    res = pd.DataFrame(out)
    keys = ["history", "bank", "channel", "checkpoint", "metric"]
    dup = res.duplicated(keys, keep="first").sum()
    if dup:
        print(f"(dropping {dup} duplicate metric rows, keeping the first)")
    res = res.drop_duplicates(keys, keep="first")
    res.to_csv(RUN / "postexit2_results.csv", index=False)
    print(f"\nwrote {RUN / 'postexit2_results.csv'} ({len(res)} rows)")

if __name__ == "__main__":
    main()
