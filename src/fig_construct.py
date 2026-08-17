"""Hero figure: displacement β toward the persona direction under baseline (0) / task-irrelevant
prose (C4 lighthouse) / persona description (C1 fiction) / non-agent normative text (policy), as
POINT-RANGES with 95% CIs, for Gemma and Qwen × Vex and Lazlo (small multiples). Each panel also
prints the direct non-agent-normative minus persona-description contrast with its CI. Post-review,
exploratory control. Output: results/figures/fig7_construct_validity.png
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20260817)
BLUE, ORANGE, INK, SEC, MUT, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": SEC, "xtick.color": MUT, "ytick.color": MUT})

def load(run):
    d = pd.DataFrame(json.loads(l) for l in open(ROOT / "runs" / run / "results.jsonl"))
    d = d[(d.flag == "ok") & (d.channel == "revealed") & (d.subset != "invariant")].copy()
    d["persona"] = d.persona.fillna("-")
    d["pa"] = np.where(d.order == 0, (d.value == "A"), (d.value == "B")).astype(float)
    return d
def prof(sub, pairs): return sub.groupby("pair_id").pa.mean().reindex(pairs)
def _boot(d1, e, contrast=None):
    if contrast is None:
        m = d1.notna() & e.notna(); d, x = d1[m].values, e[m].values
        pt = float(x @ d / (d @ d)); f = lambda dd, xx: xx @ dd / (dd @ dd)
    else:
        m = d1.notna() & e.notna() & contrast.notna(); d, x, c = d1[m].values, e[m].values, contrast[m].values
        pt = float((x @ d - c @ d) / (d @ d)); f = lambda dd, ii: (x[ii] @ dd - c[ii] @ dd) / (dd @ dd)
    vals = []
    while len(vals) < 2000:
        i = RNG.integers(0, len(d), len(d)); y = d[i]
        if y @ y < 1e-9: continue
        vals.append(f(y, i) if contrast is not None else f(y, x[i]))
    return pt, np.percentile(vals, 2.5), np.percentile(vals, 97.5)

MODELS = [("Gemma-3-27B", "gridA_gemma", "deconfound_gemma", "semprime_gemma"),
          ("Qwen-2.5-72B", "gridA_qwen72b", "deconfound_qwen72b", "semprime_qwen")]
PERSONAS = ["Vex", "Lazlo"]
# (label, source-key, cond, persona-override, color, marker)
CONDS = [("baseline", None, None, None, MUT, "|"),
         ("irrelevant prose", "dec", "C4place", "-", MUT, "s"),
         ("description", "dec", "C1fiction", None, BLUE, "o"),
         ("normative", "pol", "POLICY", None, ORANGE, "D")]

fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.7), sharex=True)
for r, (mname, grun, drun, prun) in enumerate(MODELS):
    g, dec, pol = load(grun), load(drun), load(prun)
    pairs = sorted(g[g.cond == "B0"].pair_id.unique())
    b0 = prof(g[(g.cond == "B0") & (g.persona == "-")], pairs)
    src = {"dec": dec, "pol": pol}
    for c, persona in enumerate(PERSONAS):
        ax = axes[r][c]
        d1 = prof(g[(g.cond == "B1") & (g.persona == persona)], pairs) - b0
        ys, char_e = list(range(len(CONDS)))[::-1], None
        for i, (lab, sk, cond, per, col, mk) in enumerate(CONDS):
            y = ys[i]
            if sk is None:
                ax.plot(0, y, mk, color=col, ms=9, mew=1.6); continue
            pp = persona if per is None else per
            e = prof(src[sk][(src[sk].cond == cond) & (src[sk].persona == pp)], pairs) - b0
            if cond == "C1fiction": char_e = e
            pt, lo, hi = _boot(d1, e)
            ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt=mk, color=col, ms=7,
                        elinewidth=1.5, capsize=3, mec=SURF, mew=0.8, zorder=3)
        # non-agent normative minus persona-description contrast
        pol_e = prof(pol[(pol.cond == "POLICY") & (pol.persona == persona)], pairs) - b0
        cpt, clo, chi = _boot(d1, pol_e, contrast=char_e)
        ax.text(0.985, 0.96, f"normative−description\n{cpt:+.2f} [{clo:+.2f}, {chi:+.2f}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color=SEC)
        ax.axvline(0, color="#c3c2b7", lw=1.0)
        ax.axvline(1.0, color=MUT, lw=0.8, ls=":"); ax.text(1.0, len(CONDS) - 0.35, "full\nenactment",
                  fontsize=6.8, color=SEC, ha="center", va="bottom")
        ax.set_yticks(ys); ax.set_yticklabels([c[0] for c in CONDS], fontsize=8.2)
        ax.set_title(f"{mname} · {persona}", fontsize=10, color=INK, loc="left")
        ax.set_xlim(-0.25, 1.2); ax.set_ylim(-0.6, len(CONDS) - 0.2)
        ax.grid(axis="y", visible=False)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(0.5, 0.015, "displacement β toward persona   (0 = assistant baseline, 1 = full enactment)",
         ha="center", va="bottom", fontsize=9, color=SEC)
fig.suptitle("Agent framing is not necessary for context-induced choice shifts\n"
             "Non-agent normative text shifts committed choices about as much as a persona description.\n"
             "Points = β, bars = pair-bootstrap 95% CI.  Exploratory control, added post-review.",
             fontsize=9, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=(0, 0.045, 1, 0.885))
fig.savefig(ROOT / "results/figures/fig7_construct_validity.png", dpi=200)
print("wrote fig7_construct_validity.png")

if __name__ == "__main__":
    pass
