"""Cross-model summary of the loop-7/8 core cells (new bank, control-subtracted): one table + one figure.
Usage: python src/loop8_summary.py  -> runs/loop8_summary.csv, results/figures/fig_l8_models.png"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
RUNS = {"Gemma-3-27B": "postexit2_gemma", "Qwen-2.5-72B": "postexit2_qwen72b", "Llama-3.1-70B": "postexit2_llama70b", "gpt-4.1-mini": "postexit2_gpt41mini"}
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "sans-serif", "font.size": 9.5,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 10.5})

def cell(d, **kw):
    s = d
    for k, v in kw.items():
        s = s[s[k] == v]
    return (s.point.iloc[0], s.lo95.iloc[0], s.hi95.iloc[0]) if len(s) else (np.nan, np.nan, np.nan)

def probes(run):
    f = ROOT / "runs" / run / "postexit2_probes.jsonl"
    if not f.exists():
        return {}
    pr = pd.DataFrame([json.loads(l) for l in open(f)])
    POST = {"x2", "x8", "r_inst"} | {f"x2g{i}" for i in range(8)} | {f"rg{i}" for i in range(8)}
    pers = pr.history.str.endswith(("_Vex", "_Lazlo")) & pr.history.str.startswith("auth")
    idp = pr[(pr.channel == "identity") & pers]; nl = pr[(pr.channel == "nonleading") & pers]
    return {"id_aff_t4": (idp[idp.checkpoint == "t4"].judge == "AFFIRMS").mean(), "id_aff_post": (idp[idp.checkpoint.isin(POST)].judge == "AFFIRMS").mean(),
            "nl_aff_t4": nl[nl.checkpoint == "t4"].judge.isin(["ENACTMENT", "STYLE"]).mean(), "nl_identity_post": (nl[nl.checkpoint.isin(POST)].judge == "IDENTITY").mean()}

def main():
    rows = []
    for name, run in RUNS.items():
        d = pd.read_csv(ROOT / "runs" / run / "postexit2_results.csv"); d = d[(d.bank == "new") | d.bank.isna()]
        r = dict(model=name)
        for h in ("auth_Vex", "auth_Lazlo"):
            for ch in ("stated_self", "revealed"):
                for cp in ("x2", "noexit2", "x2g_pooled", "x8"):
                    r[f"{h}_{ch}_{cp}"] = cell(d, metric="beta_minus_control", history=h, channel=ch, checkpoint=cp)
                r[f"{h}_{ch}_x2_minus_noexit2_raw"] = cell(d, metric="paired_contrast", history=h, channel=ch, checkpoint="x2_minus_noexit2")
                r[f"{h}_{ch}_x2_minus_noexit2_adj"] = cell(d, metric="paired_contrast_adj", history=h, channel=ch, checkpoint="x2_minus_noexit2")
        for ih in ("inst_short", "inst_blunt"):
            for ch in ("stated_self", "revealed"):
                r[f"{ih}_{ch}_withdrawn2"] = cell(d, metric="beta_minus_control", history=ih, channel=ch, checkpoint="withdrawn2")
                r[f"{ih}_{ch}_active2"] = cell(d, metric="beta_minus_control", history=ih, channel=ch, checkpoint="active2")
        sp = d[(d.metric == "british_share") & (d.history == "inst_spell")]
        r["spell_active"] = sp[sp.checkpoint == "active2"].point.mean(); r["spell_withdrawn"] = sp[sp.checkpoint == "withdrawn2"].point.mean()
        rv = d[(d.metric == "reverted_output_share") & (d.history == "inst_spell") & (d.checkpoint == "withdrawn2")]
        r["spell_outputs_default"] = rv.point.iloc[0] if len(rv) else np.nan
        r.update(probes(run)); rows.append(r)
    flat = []
    for r in rows:
        fr = {"model": r["model"]}
        for k, v in r.items():
            if isinstance(v, tuple):
                fr[k] = v[0]; fr[k + "_lo"] = v[1]; fr[k + "_hi"] = v[2]
            elif k != "model":
                fr[k] = v
        flat.append(fr)
    df = pd.DataFrame(flat); df.to_csv(ROOT / "runs" / "loop8_summary.csv", index=False)
    f = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]" if not np.isnan(t[0]) else "–"
    print("| Model | Vex stated x2 | Vex choice x2 | Lazlo stated x2 | Lazlo choice x2 | Vex stated x2−noexit2 (raw / adj) | 'short' choice withdrawn | 'blunt' choice withdrawn | 'blunt' stated withdrawn | spelling British active→withdrawn (outputs default) | identity AFFIRMS t4→post | non-leading IDENTITY post |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {f(r['auth_Vex_stated_self_x2'])} | {f(r['auth_Vex_revealed_x2'])} | {f(r['auth_Lazlo_stated_self_x2'])} | {f(r['auth_Lazlo_revealed_x2'])} | "
              f"{r['auth_Vex_stated_self_x2_minus_noexit2_raw'][0]:+.2f} / {r['auth_Vex_stated_self_x2_minus_noexit2_adj'][0]:+.2f} | {f(r['inst_short_revealed_withdrawn2'])} | {f(r['inst_blunt_revealed_withdrawn2'])} | {f(r['inst_blunt_stated_self_withdrawn2'])} | "
              f"{r['spell_active']:.2f}→{r['spell_withdrawn']:.2f} ({r['spell_outputs_default']:.0%}) | {r.get('id_aff_t4', np.nan):.0%}→{r.get('id_aff_post', np.nan):.0%} | {r.get('nl_identity_post', np.nan):.0%} |")
    # figure: per model, persona post-exit residuals vs withdrawn-instruction residuals (choice + stated)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    items = [("auth_Vex_%s_x2", "Vex after exit", "o"), ("auth_Lazlo_%s_x2", "Lazlo after exit", "o"), ("inst_blunt_%s_withdrawn2", "'blunt' rule after withdrawal", "s"), ("inst_short_%s_withdrawn2", "'short' rule after withdrawal", "s")]
    for ax, ch, title in zip(axes, ("revealed", "stated_self"), ("A  Committed choice", "B  Stated preference")):
        for i, r in enumerate(rows):
            for k, (key, lab, mk) in enumerate(items):
                pt, lo, hi = r[key % ch]; y = i + (k - 1.5) * 0.19
                ax.plot([lo, hi], [y, y], color=C[k], lw=2, solid_capstyle="round"); ax.plot(pt, y, mk, ms=5.5, color=C[k], mec=SURF, mew=1)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["model"] for r in rows]); ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=0.8); ax.set_xlim(-0.6, 1.3); ax.set_xlabel("residual after exit or withdrawal, minus control (own-shift units)")
        ax.set_title(title, loc="left")
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], color=C[k], marker=mk, lw=2, label=lab) for k, (_, lab, mk) in enumerate(items)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Persona exit vs instruction withdrawal across four models (persona-blind bank; unclipped 95% pair-bootstrap)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95)); fig.savefig(ROOT / "results" / "figures" / "fig_l8_models.png", dpi=200)
    print("wrote runs/loop8_summary.csv and results/figures/fig_l8_models.png")

if __name__ == "__main__":
    main()
