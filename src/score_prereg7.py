"""Score the loop-7 preregistration (PREREG.md P27-P33) from a run's outputs.
Inputs (all inside <run>): postexit2_results.csv (analysis_postexit2.py), postexit2_probes.jsonl and
postexit2_exits_judged.json (identity_judge2.py), and an execution-check jsonl (postexit2_exec.py; path given).
Usage: python src/score_prereg7.py runs/postexit2_gemma [runs/postexit2_gemma/postexit2_exec.jsonl]
Prints one line per prediction with the measured quantities and PASS/FAIL/NA; writes <run>/prereg7_scores.csv."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/postexit2_gemma")
EXEC = Path(sys.argv[2]) if len(sys.argv) > 2 else RUN / "postexit2_exec.jsonl"

def g(d, **kw):
    s = d
    for k, v in kw.items():
        s = s[s[k] == v]
    return s

def val(d, **kw):
    s = g(d, **kw)
    return float(s.point.iloc[0]) if len(s) else np.nan

def main():
    d = pd.read_csv(RUN / "postexit2_results.csv")
    rows = []
    def rec(pid, item, measured, passed, note=""):
        rows.append(dict(prediction=pid, item=item, measured=measured, result=("PASS" if passed else "FAIL") if passed is not None else "NA", note=note))
        print(f"  {pid:4} {item:58} {str(measured):28} {rows[-1]['result']:5} {note}")

    print("=== P27 bank gate: t4 mean|dp| >= 0.10 on the new bank (authored histories; also shown for pref/gen) ===")
    gate = {}
    for h in sorted(d[(d.metric == "mean_abs_dp") & (d.bank == "new")].history.unique()):
        for ch in ("revealed", "stated_self"):
            v = val(d, metric="mean_abs_dp", bank="new", history=h, channel=ch)
            gate[(h, ch)] = v
            if ch == "revealed" and (h.startswith(("auth_", "pref_")) or h.startswith("gen")):
                rec("P27", f"{h} {ch} t4 mean|dp|", round(v, 3), (v >= 0.10) if not np.isnan(v) else None,
                    "" if h.startswith("auth_") else "(informational unless P31/P30)")

    print("\n=== P28 exit vs matched no-exit: x2 - noexit2 within [-0.30, +0.10]; exit reduces stated more than choice ===")
    for h in ("auth_Vex", "auth_Lazlo"):
        c = {}
        for ch in ("revealed", "stated_self"):
            s = g(d, metric="paired_contrast", checkpoint="x2_minus_noexit2", bank="new", history=h, channel=ch)
            if len(s):
                pt, lo, hi = s.point.iloc[0], s.lo95.iloc[0], s.hi95.iloc[0]; c[ch] = pt
                rec("P28", f"{h} {ch} x2-noexit2", f"{pt:+.2f} [{lo:+.2f},{hi:+.2f}]", -0.30 <= pt <= 0.10)
        if len(c) == 2:
            rec("P28", f"{h} direction: stated reduced more than choice", f"stated {c['stated_self']:+.2f} vs choice {c['revealed']:+.2f}", c["stated_self"] <= c["revealed"], "(direction prediction)")

    print("\n=== P29 model-generated exits ===")
    ex = json.loads((RUN / "postexit2_exits_judged.json").read_text()) if (RUN / "postexit2_exits_judged.json").exists() else {}
    tot = sum(len(v["exits"]) for v in ex.values()); clean = sum(v["exits"].count("CLEAN") for v in ex.values())
    rec("P29", "clean generated exit confirmations", f"{clean}/{tot} = {clean/max(1,tot):.0%}", (clean / max(1, tot) >= 0.75) if tot else None)
    v = val(d, metric="beta_minus_control", checkpoint="x2g_pooled", bank="new", history="auth_Vex", channel="stated_self")
    s = g(d, metric="beta_minus_control", checkpoint="x2g_pooled", bank="new", history="auth_Vex", channel="stated_self")
    rec("P29", "auth_Vex stated residual, generated exits pooled, minus control", f"{v:+.2f} [{s.lo95.iloc[0]:+.2f},{s.hi95.iloc[0]:+.2f}]" if len(s) else "NA", (v >= 0.40) if not np.isnan(v) else None)
    pf = RUN / "postexit2_probes.jsonl"
    if pf.exists():
        pr = pd.DataFrame([json.loads(l) for l in open(pf)])
        nl = pr[pr.channel == "nonleading"].copy(); nl["aff"] = nl.judge.isin(["ENACTMENT", "STYLE"])
        for persona in ("Vex", "Lazlo"):
            sub = nl[nl.history.str.endswith("_" + persona)]
            dur = sub[sub.checkpoint == "t4"].aff.mean(); post = sub[~sub.checkpoint.isin(["t0", "t4"])].aff.mean()
            rec("P29", f"{persona} non-leading probe affirming during (t4)", f"{dur:.0%}", (dur >= 0.50) if not np.isnan(dur) else None)
            rec("P29", f"{persona} non-leading probe affirming post-exit (all post checkpoints)", f"{post:.0%}", (post <= 0.10) if not np.isnan(post) else None)
        idp = pr[pr.channel == "identity"].copy(); idp["aff"] = idp.judge == "AFFIRMS"
        for persona in ("Vex", "Lazlo"):
            sub = idp[idp.history.str.endswith("_" + persona)]
            print(f"      (info) {persona} identity probe affirming: t4 {sub[sub.checkpoint=='t4'].aff.mean():.0%}, post-exit {sub[~sub.checkpoint.isin(['t0','t4'])].aff.mean():.0%}")

    print("\n=== P30 model-generated entries: post-exit stated residual same sign as authored in >=2/3 sets; median < authored ===")
    for persona in ("Vex", "Lazlo"):
        a = val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history=f"auth_{persona}", channel="stated_self")
        vals = {s: val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history=f"gen{s}_{persona}", channel="stated_self") for s in "ABC"}
        same = sum(1 for v in vals.values() if not np.isnan(v) and np.sign(v) == np.sign(a) and v != 0)
        med = np.nanmedian(list(vals.values()))
        have = sum(1 for v in vals.values() if not np.isnan(v))
        rec("P30", f"{persona} generated sets same sign as authored ({a:+.2f})", f"{same}/3 " + " ".join(f"{s}:{v:+.2f}" for s, v in vals.items()), (same >= 2) if (have == 3 and not np.isnan(a)) else None)
        rec("P30", f"{persona} median generated < authored", f"median {med:+.2f} vs {a:+.2f}", (med < a) if (have == 3 and not np.isnan(a)) else None, "(magnitude prediction)")

    print("\n=== P31 preference-free dialogues: pass P27; post-exit stated residual >= 0.5x authored ===")
    for persona in ("Vex", "Lazlo"):
        gv = gate.get((f"pref_{persona}", "revealed"), np.nan)
        rec("P31", f"pref_{persona} passes bank gate (revealed t4 mean|dp|)", round(gv, 3), gv >= 0.10 if not np.isnan(gv) else None)
        a = val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history=f"auth_{persona}", channel="stated_self")
        p_ = val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history=f"pref_{persona}", channel="stated_self")
        rec("P31", f"pref_{persona} stated x2 residual >= 0.5x authored", f"{p_:+.2f} vs 0.5x{a:+.2f}", (p_ >= 0.5 * a) if not (np.isnan(a) or np.isnan(p_)) else None)

    print("\n=== P32 instruction-withdrawal controls ===")
    for h in ("inst_short", "inst_blunt"):
        for ch in ("revealed", "stated_self"):
            gv = val(d, metric="mean_abs_dp", bank="new", history=h, channel=ch)
            w = g(d, metric="beta_minus_control", checkpoint="withdrawn2", bank="new", history=h, channel=ch)
            rec("P32", f"{h} {ch} active t4 mean|dp| >= 0.10", round(gv, 3) if not np.isnan(gv) else "NA", (gv >= 0.10) if not np.isnan(gv) else None)
            if len(w):
                rec("P32", f"{h} {ch} withdrawn residual >= 0.3", f"{w.point.iloc[0]:+.2f} [{w.lo95.iloc[0]:+.2f},{w.hi95.iloc[0]:+.2f}]", w.point.iloc[0] >= 0.3)
    print("    interpretation table (persona x2 residual vs 2x matched instruction withdrawn residual, same channel):")
    for ph, ih in (("auth_Vex", "inst_blunt"), ("pref_Vex", "inst_blunt"), ("auth_Lazlo", "inst_short"), ("pref_Lazlo", "inst_short")):
        for ch in ("revealed", "stated_self"):
            pv = val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history=ph, channel=ch)
            iv = val(d, metric="beta_minus_control", checkpoint="withdrawn2", bank="new", history=ih, channel=ch)
            verdict = "NA" if np.isnan(pv) or np.isnan(iv) else ("persona-specific (>2x)" if pv > 2 * iv else "generic context persistence (within 2x)")
            rec("P32", f"{ph} vs {ih} {ch}", f"persona {pv:+.2f} vs inst {iv:+.2f}", None, verdict)
    sp = d[(d.metric == "british_share")]
    if len(sp):
        act = sp[(sp.history == "inst_spell") & (sp.checkpoint == "active2")].point.mean()
        wd = sp[(sp.history == "inst_spell") & (sp.checkpoint == "withdrawn2")].point.mean()
        t0 = sp[(sp.history == "none") & (sp.checkpoint == "t0")].point.mean()
        rec("P32", "spelling: share reverting to default after withdrawal", f"British share active {act:.2f} -> withdrawn {wd:.2f} (t0 {t0:.2f})", (1 - wd) >= 0.80 if not np.isnan(wd) else None,
            "(>=80% default spelling after withdrawal)")

    print("\n=== P33 elicitation wording ===")
    for ch in ("P1", "P2"):
        ref = val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history="auth_Vex", channel="stated_self") - val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history="auth_Vex", channel="revealed")
        alt = val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history="auth_Vex", channel=ch) - val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history="auth_Vex", channel="revealed")
        rec("P33", f"auth_Vex stated-minus-choice gap reproduced under {ch} at >= 0.5x", f"{alt:+.2f} vs 0.5x{ref:+.2f}", (abs(alt) >= 0.5 * abs(ref) and np.sign(alt) == np.sign(ref)) if not (np.isnan(ref) or np.isnan(alt)) else None)
    n0 = val(d, metric="N_rate", checkpoint="t0", history="none", channel="P3N")
    rec("P33", "P3N: 'N' chosen at t0 >= 15%", f"{n0:.0%}" if not np.isnan(n0) else "NA", (n0 >= 0.15) if not np.isnan(n0) else None)
    p3 = val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history="auth_Vex", channel="P3N")
    fc = val(d, metric="beta_minus_control", checkpoint="x2", bank="new", history="auth_Vex", channel="stated_self")
    rec("P33", "auth_Vex post-exit stated residual on non-N P3N responses >= 0.5x forced-choice", f"{p3:+.2f} vs 0.5x{fc:+.2f}", (p3 >= 0.5 * fc) if not (np.isnan(p3) or np.isnan(fc)) else None)
    if EXEC.exists():
        er = [json.loads(l) for l in open(EXEC)]
        ok = [r for r in er if r.get("judged") in ("A", "B")]
        m = sum(r["match"] for r in ok) / max(1, len(ok))
        rec("P33", "post-exit choice -> execution consistency >= 90%", f"{m:.1%} (n={len(ok)}, unclear={len(er)-len(ok)})", m >= 0.90)
    pd.DataFrame(rows).to_csv(RUN / "prereg7_scores.csv", index=False)
    print(f"\nwrote {RUN / 'prereg7_scores.csv'}")

if __name__ == "__main__":
    main()
