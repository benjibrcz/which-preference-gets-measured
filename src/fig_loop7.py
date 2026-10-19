"""Loop-7 figures from <run>/postexit2_results.csv (schema: history, bank, channel, checkpoint, metric, point, lo95, hi95, n_pairs).
  fig_l7_trajectory.png : control-subtracted residuals on the persona-blind bank per history x channel x checkpoint
  fig_l7_contrasts.png  : paired contrasts (exit - no-exit, generated exit - generated no-exit, exit - reset) and the
                          instruction-withdrawal contrast class (persona residual vs instruction residual, 2x line)
Usage: python src/fig_loop7.py runs/postexit2_gemma [tag]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/postexit2_gemma")
TAG = sys.argv[2] if len(sys.argv) > 2 else "gemma"
FIGS = ROOT / "results" / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed categorical order
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.family": "sans-serif",
                     "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titleweight": "bold", "axes.titlesize": 10.5})
CPL = {"x2": "exit +2", "noexit2": "no-exit +2", "x2g_pooled": "generated exit +2 (pooled)", "noexit2g": "generated no-exit +2",
       "rg_pooled": "exit + reset (pooled)", "x8": "exit +8", "r_inst": "exit + reset (authored)"}
HL = {"auth_Vex": "Vex, authored", "pref_Vex": "Vex, preference-free", "auth_Lazlo": "Lazlo, authored", "pref_Lazlo": "Lazlo, preference-free",
      "gen_Vex": "Vex, generated (3 sets)", "gen_Lazlo": "Lazlo, generated (3 sets)"}

def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)

def cell(d, hid, ch, cp, metric="beta_minus_control", bank="new"):
    s = d[(d.history == hid) & (d.channel == ch) & (d.checkpoint == cp) & (d.metric == metric) & (d.bank == bank)]
    return None if s.empty else s.iloc[0]

def pooled_rg(d):
    """rg_pooled is now computed by analysis_postexit2.py with a proper pair bootstrap; nothing to add here."""
    return d

def gen_pooled(d):
    rows = []
    for persona in ("Vex", "Lazlo"):
        g = d[(d.metric == "beta_minus_control") & d.history.str.match(rf"^gen[ABC]_{persona}$") & (d.bank == "new")]
        for (ch, cp), gg in g.groupby(["channel", "checkpoint"]):
            rows.append(dict(history=f"gen_{persona}", bank="new", channel=ch, checkpoint=cp, metric="beta_minus_control",
                             point=gg.point.median(), lo95=gg.point.min(), hi95=gg.point.max(), n_pairs=gg.n_pairs.max()))
    return pd.concat([d, pd.DataFrame(rows)], ignore_index=True) if rows else d

def trajectory(d):
    hists = [h for h in ("auth_Vex", "pref_Vex", "gen_Vex", "auth_Lazlo", "pref_Lazlo", "gen_Lazlo") if (d.history == h).any()]
    cps = [c for c in ("x2", "noexit2", "x2g_pooled", "noexit2g", "rg_pooled", "x8", "r_inst") if (d.checkpoint == c).any()]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.24 * len(hists) * (len(cps) + 1.2) + 2.2), sharey=True)
    for ai, (ax, ch, title) in enumerate(zip(axes, ("stated_self", "revealed"), ("A  Stated preference", "B  Committed choice"))):
        ylab, y, heads = [], 0, set()
        for hi, hid in enumerate(hists):
            ylab.append((y, HL[hid])); heads.add(y); y += 0.8
            for ci, cp in enumerate(cps):
                r = cell(d, hid, ch, cp)
                ylab.append((y, CPL[cp]))
                if r is not None and not np.isnan(r.point):
                    col = C[0] if hid.startswith("auth") else C[1] if hid.startswith("pref") else C[2]
                    ax.plot([r.lo95, r.hi95], [y, y], color=col, lw=2, solid_capstyle="round", alpha=0.9)
                    ax.plot(r.point, y, "o", ms=6, color=col, mec=SURF, mew=1.2)
                y += 1
            y += 0.4
        ax.set_yticks([t for t, _ in ylab]); ax.set_yticklabels([l for _, l in ylab], fontsize=8.5)
        for t, lab in zip(ax.get_yticklabels(), ylab):
            if lab[0] in heads:
                t.set_fontweight("bold"); t.set_color(INK)
        if ai == 0:
            ax.invert_yaxis()
        ax.axvline(0, color=INK2, lw=0.8); ax.axvline(1, color=GRID, lw=0.8, ls="--")
        ax.set_xlim(-0.6, 1.4); ax.set_xlabel("residual minus matched Neutral control (units of own in-roleplay shift)")
        ax.set_title(title, loc="left"); despine(ax)
    handles = [plt.Line2D([], [], color=C[0], marker="o", lw=2, label="authored dialogue"),
               plt.Line2D([], [], color=C[1], marker="o", lw=2, label="preference-free dialogue"),
               plt.Line2D([], [], color=C[2], marker="o", lw=2, label="model-generated histories (median of 3 sets; bar = min–max of the 3 point estimates, not a CI)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Post-exit residuals on the persona-blind bank ({TAG}; bars = unclipped 95% pair-bootstrap CIs except where the legend says otherwise)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig(FIGS / f"fig_l7_trajectory_{TAG}.png", dpi=200)
    print("wrote", FIGS / f"fig_l7_trajectory_{TAG}.png")

def contrasts(d):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    # panel A: paired contrasts
    ax = axes[0]
    names = [("x2_minus_noexit2", "exit − no-exit (authored replies)"), ("x2g_minus_noexit2g", "generated exit − generated no-exit"),
             ("x2_minus_rg", "scripted exit + 2 neutral − (generated exit 0 + reset + 1 neutral): composite histories")]
    hists = [h for h in ("auth_Vex", "pref_Vex", "auth_Lazlo", "pref_Lazlo") if (d.history == h).any()]
    ylab, y, heads = [], 0, set()
    for cn, cl in names:
        ylab.append((y, cl)); heads.add(y); y += 0.8
        for hid in hists:
            for k, ch in enumerate(("stated_self", "revealed")):
                r = cell(d, hid, ch, cn, metric="paired_contrast")
                if r is None or np.isnan(r.point):
                    continue
                yy = y + (k - 0.5) * 0.28
                ax.plot([r.lo95, r.hi95], [yy, yy], color=C[k], lw=2, solid_capstyle="round")
                ax.plot(r.point, yy, "o", ms=5.5, color=C[k], mec=SURF, mew=1.0)
            ylab.append((y, f"{HL[hid]}")); y += 1
        y += 0.4
    ax.axvspan(-0.30, 0.10, color=GRID, alpha=0.6, lw=0, label="P28 band [−0.30, +0.10]")
    ax.set_yticks([t for t, _ in ylab]); ax.set_yticklabels([l for _, l in ylab], fontsize=8.5); ax.invert_yaxis()
    for t, lab in zip(ax.get_yticklabels(), ylab):
        if lab[0] in heads:
            t.set_fontweight("bold"); t.set_color(INK)
    ax.axvline(0, color=INK2, lw=0.8); ax.set_xlim(-1.0, 1.0); ax.set_xlabel("paired contrast (units of in-roleplay shift)")
    ax.set_title("A  What the exit and reset turns themselves do", loc="left"); despine(ax)
    ax.legend(handles=[plt.Line2D([], [], color=C[0], marker="o", lw=2, label="stated"), plt.Line2D([], [], color=C[1], marker="o", lw=2, label="choice"),
                       plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.6, label="preregistered 'exit does little' band")], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=8.5)
    # panel B: contrast class
    ax = axes[1]
    pairs = [("auth_Vex", "inst_blunt", "Vex vs 'blunt' rule"), ("pref_Vex", "inst_blunt", "Vex (pref-free) vs 'blunt' rule"),
             ("auth_Lazlo", "inst_short", "Lazlo vs 'short' rule"), ("pref_Lazlo", "inst_short", "Lazlo (pref-free) vs 'short' rule")]
    ylab, y = [], 0
    for ph, ih, lab in pairs:
        for k, ch in enumerate(("stated_self", "revealed")):
            rp = cell(d, ph, ch, "x2"); ri = cell(d, ih, ch, "withdrawn2")
            if rp is None or ri is None:
                continue
            yy = y + (k - 0.5) * 0.28
            ax.plot([ri.lo95, ri.hi95], [yy, yy], color=C[3], lw=2, solid_capstyle="round", alpha=0.9)
            ax.plot(ri.point, yy, "s", ms=5.5, color=C[3], mec=SURF, mew=1.0)
            ax.plot([rp.lo95, rp.hi95], [yy, yy], color=C[k], lw=2, solid_capstyle="round", alpha=0.9)
            ax.plot(rp.point, yy, "o", ms=5.5, color=C[k], mec=SURF, mew=1.0)
            if not np.isnan(ri.point):
                ax.plot(2 * ri.point, yy, "|", ms=10, color=INK2, mew=1.2)
        ylab.append((y, lab)); y += 1
    ax.set_yticks([t for t, _ in ylab]); ax.set_yticklabels([l for _, l in ylab], fontsize=8.5); ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8); ax.set_xlim(-0.6, 1.4); ax.set_xlabel("residual after exit / withdrawal, minus control")
    ax.set_title("B  Contrast class: persona vs withdrawn instruction", loc="left"); despine(ax)
    ax.legend(handles=[plt.Line2D([], [], color=C[0], marker="o", lw=2, label="persona, stated"), plt.Line2D([], [], color=C[1], marker="o", lw=2, label="persona, choice"),
                       plt.Line2D([], [], color=C[3], marker="s", lw=2, label="withdrawn instruction (same channel)"),
                       plt.Line2D([], [], color=INK2, marker="|", lw=0, ms=10, label="2× instruction residual (P32 line)")], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle(f"Paired contrasts and the contrast class ({TAG})", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95)); fig.savefig(FIGS / f"fig_l7_contrasts_{TAG}.png", dpi=200, bbox_inches="tight")
    print("wrote", FIGS / f"fig_l7_contrasts_{TAG}.png")

def main():
    d = pd.read_csv(RUN / "postexit2_results.csv")
    d = gen_pooled(pooled_rg(d))
    trajectory(d); contrasts(d)

if __name__ == "__main__":
    main()
