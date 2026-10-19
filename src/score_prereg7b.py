"""Score PREREG loop 7b (P40-P44) from a combined run directory containing the loop-7 and loop-7b cells.
Usage: python src/score_prereg7b.py runs/postexit2_gemma_all"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
RUN = Path(sys.argv[1])
d = pd.read_csv(RUN / "postexit2_results.csv")
def val(**kw):
    s = d
    for k, v in kw.items(): s = s[s[k] == v]
    return (s.point.iloc[0], s.lo95.iloc[0], s.hi95.iloc[0]) if len(s) else (np.nan,)*3
rows = []
def rec(pid, item, measured, passed, note=""):
    rows.append(dict(prediction=pid, item=item, measured=measured, result=("PASS" if passed else "FAIL") if passed is not None else "NA", note=note))
    print(f"  {pid:4} {item:66} {str(measured):30} {rows[-1]['result']:5} {note}")
print("--- P40 own confirmations: withdrawn2g pooled choice residual >= 0.3 for both rules")
for ih in ("inst_short", "inst_blunt"):
    pt, lo, hi = val(metric="beta_minus_control", checkpoint="withdrawn2g_pooled", history=ih, channel="revealed")
    rec("P40", f"{ih} choice residual after the model's own confirmations (pooled)", f"{pt:+.2f} [{lo:+.2f},{hi:+.2f}]", (pt >= 0.3) if not np.isnan(pt) else None)
    pt2, lo2, hi2 = val(metric="beta_minus_control", checkpoint="withdrawn2g_pooled", history=ih, channel="stated_self")
    rec("P40", f"  (info) {ih} stated residual after own confirmations", f"{pt2:+.2f} [{lo2:+.2f},{hi2:+.2f}]", None)
print("--- P41 spelling after own confirmation")
sp = d[(d.history == "inst_spell") & d.checkpoint.str.startswith("withdrawn2g")]
if len(sp):
    bs = sp[sp.metric == "british_share"].point.mean(); rv = sp[sp.metric == "reverted_output_share"].point.mean()
    rec("P41", "outputs with majority default spelling after own confirmation (< 80% predicted)", f"{rv:.0%}; mean British share {bs:.2f}", (rv < 0.8) and (bs > 0.3))
print("--- P42 confirmation wording vs continuation: auth_Vex choice x2gA_pooled > 0.22")
pt, lo, hi = val(metric="beta_minus_control", checkpoint="x2gA_pooled", history="auth_Vex", channel="revealed")
rec("P42", "auth_Vex choice residual, own confirmation + authored suffix (pooled)", f"{pt:+.2f} [{lo:+.2f},{hi:+.2f}]", (pt > 0.22) if not np.isnan(pt) else None, "scripted exit 0.42; generated package 0.02")
for h in ("auth_Vex", "auth_Lazlo", "pref_Vex", "pref_Lazlo"):
    for ch in ("revealed", "stated_self"):
        a = val(metric="beta_minus_control", checkpoint="x2gA_pooled", history=h, channel=ch); c = val(metric="paired_contrast_adj", checkpoint="x2gA_minus_x2", history=h, channel=ch)
        rec("P42", f"  (info) {h} {ch} x2gA_pooled; x2gA - x2 (adj)", f"{a[0]:+.2f} [{a[1]:+.2f},{a[2]:+.2f}]; {c[0]:+.2f} [{c[1]:+.2f},{c[2]:+.2f}]", None)
print("--- P43 matched reset contrast rg_pooled - x2g0n1 within [-0.30, +0.10] in all 4 authored cells")
inb = 0; n = 0
for h in ("auth_Vex", "auth_Lazlo"):
    for ch in ("revealed", "stated_self"):
        pt, lo, hi = val(metric="paired_contrast_adj", checkpoint="rg_pooled_minus_x2g0n1", history=h, channel=ch)
        if np.isnan(pt): continue
        n += 1; ok = -0.30 <= pt <= 0.10; inb += ok
        rec("P43", f"{h} {ch} rg_pooled - x2g0n1 (adj)", f"{pt:+.2f} [{lo:+.2f},{hi:+.2f}]", ok)
rec("P43", "cells in band", f"{inb}/{n}", (inb == 4) if n == 4 else None)
print("--- P44 explicit preference turn: nopref gate and stated x2 residual < 0.5x authored")
for persona, auth_val, strong in (("Vex", 0.71, True), ("Lazlo", 0.27, False)):
    g = val(metric="mean_abs_dp", history=f"nopref_{persona}", channel="revealed")[0]
    rec("P44", f"nopref_{persona} bank gate (t4 revealed mean|dp|)", round(g, 3) if not np.isnan(g) else "NA", (g >= 0.10) if not np.isnan(g) else None)
    pt, lo, hi = val(metric="beta_minus_control", checkpoint="x2", history=f"nopref_{persona}", channel="stated_self")
    rec("P44", f"nopref_{persona} stated x2 residual < 0.5 x authored ({auth_val})", f"{pt:+.2f} [{lo:+.2f},{hi:+.2f}]", (pt < 0.5 * auth_val) if not np.isnan(pt) else None, "" if strong else "(lower confidence)")
    pt2, lo2, hi2 = val(metric="beta_minus_control", checkpoint="x2", history=f"nopref_{persona}", channel="revealed")
    rec("P44", f"  (info) nopref_{persona} choice x2 residual", f"{pt2:+.2f} [{lo2:+.2f},{hi2:+.2f}]", None)
pd.DataFrame(rows).to_csv(RUN / "prereg7b_scores.csv", index=False); print("wrote", RUN / "prereg7b_scores.csv")
