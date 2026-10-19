"""Score the loop-8 preregistration (PREREG.md P34-P39) for one extra-model run.
Inputs inside <run>: postexit2_results.csv, postexit2_probes.jsonl, postexit2_exits_judged.json.
Usage: python src/score_prereg8.py runs/postexit2_llama70b   -> prints the per-model line items and writes <run>/prereg8_scores.csv
(The cross-model clauses of P36-P39 are aggregated by hand from the per-model lines.)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

RUN = Path(sys.argv[1])

def val(d, **kw):
    s = d
    for k, v in kw.items():
        s = s[s[k] == v]
    return (float(s.point.iloc[0]), float(s.lo95.iloc[0]) if "lo95" in s and len(s) else np.nan, float(s.hi95.iloc[0]) if "hi95" in s and len(s) else np.nan) if len(s) else (np.nan, np.nan, np.nan)

def main():
    d = pd.read_csv(RUN / "postexit2_results.csv")
    rows = []
    def rec(pid, item, measured, passed, note=""):
        rows.append(dict(prediction=pid, item=item, measured=measured, result=("PASS" if passed else "FAIL") if passed is not None else "NA", note=note))
        print(f"  {pid:4} {item:62} {str(measured):30} {rows[-1]['result']:5} {note}")
    print(f"=== {RUN.name}")
    print("--- P34 bank gate (authored histories, revealed t4 mean|dp| >= 0.10)")
    for h in ("auth_Vex", "auth_Lazlo"):
        v = val(d, metric="mean_abs_dp", bank="new", history=h, channel="revealed")[0]
        rec("P34", f"{h} t4 mean|dp|", round(v, 3), (v >= 0.10) if not np.isnan(v) else None)
    print("--- P35 exit - no-exit (authored replies) within [-0.30, +0.10] in >= 3/4 cells")
    inband = 0; n = 0
    for h in ("auth_Vex", "auth_Lazlo"):
        for ch in ("revealed", "stated_self"):
            pt, lo, hi = val(d, metric="paired_contrast", checkpoint="x2_minus_noexit2", bank="new", history=h, channel=ch)
            if np.isnan(pt): continue
            n += 1; ok = -0.30 <= pt <= 0.10; inband += ok
            rec("P35", f"{h} {ch} x2 - noexit2", f"{pt:+.2f} [{lo:+.2f},{hi:+.2f}]", ok)
    rec("P35", "cells in band", f"{inband}/{n}", (inband >= 3) if n == 4 else None)
    print("--- P36 generic persistence: >=1 withdrawn instruction residual >= 0.3 (choice); persona within 2x instruction in majority of 8 cells")
    any03 = False
    for ih in ("inst_short", "inst_blunt"):
        for ch in ("revealed", "stated_self"):
            pt, lo, hi = val(d, metric="beta_minus_control", checkpoint="withdrawn2", bank="new", history=ih, channel=ch)
            rec("P36", f"{ih} {ch} withdrawn residual", f"{pt:+.2f} [{lo:+.2f},{hi:+.2f}]", None)
            if ch == "revealed" and not np.isnan(pt) and pt >= 0.3: any03 = True
    rec("P36", "at least one withdrawn instruction residual >= 0.3 in choice", any03, any03)
    within = 0; m = 0
    for ph, ih in (("auth_Vex", "inst_blunt"), ("pref_Vex", "inst_blunt"), ("auth_Lazlo", "inst_short"), ("pref_Lazlo", "inst_short")):
        for ch in ("revealed", "stated_self"):
            pv = val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history=ph, channel=ch)[0]
            iv = val(d, metric="beta_minus_control", checkpoint="withdrawn2", bank="new", history=ih, channel=ch)[0]
            if np.isnan(pv) or np.isnan(iv): continue
            m += 1; w = pv <= 2 * iv; within += w
            rec("P36", f"{ph} vs {ih} {ch}", f"persona {pv:+.2f} vs inst {iv:+.2f}", None, "within 2x" if w else "persona > 2x instruction")
    rec("P36", "persona within 2x instruction (majority of cells)", f"{within}/{m}", (within > m / 2) if m else None)
    print("--- P37 spelling reversion (informational per model; cross-model clause: at most one model >= 80%)")
    sp = d[d.metric == "british_share"]
    if len(sp):
        act = sp[(sp.history == "inst_spell") & (sp.checkpoint == "active2")].point.mean(); wd = sp[(sp.history == "inst_spell") & (sp.checkpoint == "withdrawn2")].point.mean()
        rec("P37", "British share active -> withdrawn; reversion", f"{act:.2f} -> {wd:.2f}; revert {1-wd:.0%}", None, "reverts >= 80%" if (1 - wd) >= 0.8 else "reverts < 80%")
    print("--- P38 identity vs content")
    pf = RUN / "postexit2_probes.jsonl"
    if pf.exists():
        pr = pd.DataFrame([json.loads(l) for l in open(pf)])
        idp = pr[(pr.channel == "identity") & pr.history.isin(["auth_Vex", "auth_Lazlo"])]
        for cp in ("x2", "x8"):
            a = (idp[idp.checkpoint == cp].judge == "AFFIRMS").mean()
            rec("P38", f"identity probe AFFIRMS at {cp} (authored)", f"{a:.0%}", a <= 0.10)
        t4 = (idp[idp.checkpoint == "t4"].judge == "AFFIRMS").mean()
        rec("P38", "identity probe AFFIRMS at t4 (info)", f"{t4:.0%}", None)
        nl = pr[(pr.channel == "nonleading") & ~pr.checkpoint.isin(["t0", "t4"]) & ~pr.checkpoint.str.startswith("noexit") & pr.history.str.endswith(("_Vex", "_Lazlo"))]
        idshare = (nl.judge == "IDENTITY").mean()
        rec("P38", "non-leading probe IDENTITY share post-exit", f"{idshare:.0%}", idshare > 0.5)
    anyres = False; anyx2 = False
    for h in ("auth_Vex", "auth_Lazlo"):
        for ch in ("revealed", "stated_self"):
            for cp in ("x2", "x8", "x2g_pooled"):
                pt, lo, hi = val(d, metric="beta_minus_control", checkpoint=cp, bank="new", history=h, channel=ch)
                if np.isnan(pt): continue
                rec("P38", f"{h} {ch} {cp} residual", f"{pt:+.2f} [{lo:+.2f},{hi:+.2f}]", None, "excludes 0" if lo > 0 else "")
                if lo > 0: anyres = True; anyx2 = anyx2 or cp == "x2"
    rec("P38", "at least one post-exit residual (x2, x8 or pooled generated exits) with interval excluding zero (as registered)", anyres, anyres)
    rec("P38", "  narrower operationalisation: at least one x2 residual excluding zero", anyx2, None, "(not the registered clause)")
    print("--- P39 model-generated exits")
    ex = json.loads((RUN / "postexit2_exits_judged.json").read_text()) if (RUN / "postexit2_exits_judged.json").exists() else {}
    tot = sum(len(v["exits"]) for v in ex.values()); clean = sum(v["exits"].count("CLEAN") for v in ex.values())
    rec("P39", "clean generated exits (info)", f"{clean}/{tot}", None)
    pt, lo, hi = val(d, metric="beta_minus_control", checkpoint="x2g_pooled", bank="new", history="auth_Vex", channel="stated_self")
    rec("P39", "auth_Vex stated residual, generated exits pooled, > 0 (CI excludes 0)", f"{pt:+.2f} [{lo:+.2f},{hi:+.2f}]", (lo > 0) if not np.isnan(pt) else None)
    g = val(d, metric="beta_minus_control", checkpoint="x2g_pooled", bank="new", history="auth_Vex", channel="revealed")[0]
    x = val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history="auth_Vex", channel="revealed")[0]
    rec("P39", "auth_Vex choice residual: generated exits < scripted exit", f"{g:+.2f} vs {x:+.2f}", (g < x) if not (np.isnan(g) or np.isnan(x)) else None)
    pd.DataFrame(rows).to_csv(RUN / "prereg8_scores.csv", index=False)
    print(f"wrote {RUN / 'prereg8_scores.csv'}")

if __name__ == "__main__":
    main()
