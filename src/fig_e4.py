"""Figure: the original 23-pair post-exit result under three item sets (all 23 / label-based core+welfare /
strict overlap-clean), Gemma-3-27B, control-subtracted residual vs the t4 direction, unclipped 95% pair-bootstrap.
Panel B: twelve models at x2, stated channel, all-23 vs overlap-clean. Reads runs/postexit_exclusion.csv and
runs/e4_offline.csv. Writes results/figures/fig_e4_oldbank_itemsets.png."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C = {"all23": "#2a78d6", "core+welfare": "#eb6834", "overlap_clean": "#1baf7a"}   # fixed categorical order
LBL = {"all23": "all 23 pairs", "core+welfare": "core + welfare (label-based)", "overlap_clean": "overlap-clean (word audit)"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "font.family": "sans-serif", "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "semibold", "axes.titlesize": 10.5})

def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)

def main():
    ex = pd.read_csv(ROOT / "runs" / "postexit_exclusion.csv")
    e4 = pd.read_csv(ROOT / "runs" / "e4_offline.csv")
    g = pd.concat([ex[(ex.model == "Gemma-3-27B") & (ex.analysis == "residual_minus_control")],
                   e4[e4.analysis == "residual_minus_control_overlapclean"]])
    tw = pd.concat([ex[ex.analysis == "x2_residual_minus_control_12models"],
                    e4[e4.analysis == "x2_residual_minus_control_overlapclean"]])
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    cps = ["x2", "x8", "r_inst", "r_sys"]; cpl = {"x2": "exit +2", "x8": "exit +8", "r_inst": "reset request", "r_sys": "fresh system prompt"}
    items = ["all23", "core+welfare", "overlap_clean"]
    for ax, ch, title in zip(axes[:2], ("stated_self", "revealed"), ("A  Stated preference channel", "B  Committed-choice channel")):
        rows = []
        for pi, persona in enumerate(("Vex", "Lazlo")):
            for ci, cp in enumerate(cps):
                rows.append((persona, cp))
        ypos = {r: i + (0.6 if r[0] == "Lazlo" else 0) for i, r in enumerate(rows)}
        for k, it in enumerate(items):
            off = (k - 1) * 0.22
            for (persona, cp), y in ypos.items():
                s = g[(g.channel == ch) & (g.persona == persona) & (g["items"] == it) & (g.condition == cp)]
                if s.empty:
                    continue
                r = s.iloc[0]
                ax.plot([r.lo95, r.hi95], [y + off, y + off], color=C[it], lw=2, solid_capstyle="round", alpha=0.9)
                ax.plot(r.point, y + off, "o", ms=6, color=C[it], mec=SURF, mew=1.2)
        ax.set_yticks([ypos[r] for r in rows]); ax.set_yticklabels([cpl[r[1]] for r in rows])
        ax.invert_yaxis(); ax.axvline(0, color=INK2, lw=0.8); ax.axvline(1, color=GRID, lw=0.8, ls="--")
        ax.set_xlim(-0.35, 1.2); ax.set_xlabel("control-subtracted residual (units of in-roleplay shift)")
        ax.set_title(title, loc="left", pad=14)
        for persona, y0 in (("Vex", -0.55), ("Lazlo", 3.75)):
            ax.text(-0.33, y0, persona, color=INK, fontweight="semibold", va="center")
        despine(ax)
    # panel C: twelve models, stated channel, x2, Vex and Lazlo, all23 vs overlap-clean
    ax = axes[2]
    order = ["gemma27b", "gemma12b", "llama70b", "llama33", "llama8b", "qwen72b", "deepseek", "mistral", "kimi", "cohere", "gpt41mini", "gpt4omini"]
    names = {"gemma27b": "Gemma-3-27B", "gemma12b": "Gemma-3-12B", "llama70b": "Llama-3.1-70B", "llama33": "Llama-3.3-70B", "llama8b": "Llama-3.1-8B",
             "qwen72b": "Qwen-2.5-72B", "deepseek": "DeepSeek-V3", "mistral": "Mistral-Large", "kimi": "Kimi-K2", "cohere": "Command-A",
             "gpt41mini": "gpt-4.1-mini", "gpt4omini": "gpt-4o-mini"}
    for i, m in enumerate(order):
        for k, (it, mk) in enumerate((("all23", "o"), ("overlap_clean", "s"))):
            for pj, persona in enumerate(("Vex", "Lazlo")):
                s = tw[(tw.model == m) & (tw.channel == "stated_self") & (tw.persona == persona) & (tw["items"] == it)]
                if s.empty:
                    continue
                r = s.iloc[0]; y = i + (k - 0.5) * 0.3
                ax.plot(r.point, y, mk, ms=5.5, color=C[it], mec=SURF, mew=0.8, alpha=1.0 if persona == "Vex" else 0.45)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[m] for m in order]); ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8); ax.set_xlim(-0.35, 1.2)
    ax.set_xlabel("x2 stated residual (Vex solid, Lazlo faded)"); ax.set_title("C  Twelve models, stated, exit +2", loc="left")
    despine(ax)
    handles = [plt.Line2D([], [], color=C[it], marker="o", lw=2, label=LBL[it]) for it in items]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("The original post-exit result depends on which items are counted (Gemma-3-27B unless stated; 95% pair-bootstrap)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    FIGS.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIGS / "fig_e4_oldbank_itemsets.png", dpi=200)
    print("wrote", FIGS / "fig_e4_oldbank_itemsets.png")

if __name__ == "__main__":
    main()
