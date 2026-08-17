"""Construct-validity comparison figure: displacement β toward the persona direction under
baseline (0) / generic prose (C4) / persona description (C1) / non-agent normative text (policy),
with pair-bootstrap CIs, for Gemma Vex & Lazlo. Post-review, exploratory control included.
Output: results/figures/fig7_construct_validity.png
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(20260816)
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, SEC, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 11, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": SEC, "xtick.color": MUT, "ytick.color": MUT})

def load(run):
    d = pd.DataFrame(json.loads(l) for l in open(ROOT / "runs" / run / "results.jsonl"))
    d = d[(d.flag == "ok") & (d.channel == "revealed") & (d.subset != "invariant")].copy()
    d["persona"] = d.persona.fillna("-")
    d["pa"] = np.where(d.order == 0, (d.value == "A"), (d.value == "B")).astype(float)
    return d
def prof(sub, pairs): return sub.groupby("pair_id").pa.mean().reindex(pairs)
def beta_ci(d1, e):
    m = d1.notna() & e.notna(); d, x = d1[m].values, e[m].values
    pt = float(x @ d / (d @ d)); vals = []
    while len(vals) < 2000:
        i = RNG.integers(0, len(d), len(d)); y = d[i]
        if y @ y < 1e-9: continue
        vals.append(x[i] @ y / (y @ y))
    return pt, np.percentile(vals, 2.5), np.percentile(vals, 97.5)

g = load("gridA_gemma"); dec = load("deconfound_gemma")
pol = load("semprime_gemma")
pairs = sorted(g[g.cond == "B0"].pair_id.unique())
b0 = prof(g[(g.cond == "B0") & (g.persona == "-")], pairs)
conds = [("generic\nprose", dec, "C4place", "-", MUT),
         ("persona\ndescription", dec, "C1fiction", None, BLUE),
         ("non-agent\nnormative", pol, "POLICY", None, ORANGE)]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True)
for ax, persona in zip(axes, ("Vex", "Lazlo")):
    d1 = prof(g[(g.cond == "B1") & (g.persona == persona)], pairs) - b0
    xs, pts, los, his, cols, labs = [], [], [], [], [], []
    # baseline bar = 0
    xs.append(0); pts.append(0.0); los.append(0.0); his.append(0.0); cols.append("#c3c2b7"); labs.append("baseline")
    for i, (lab, src, cond, per, col) in enumerate(conds, 1):
        pp = persona if per is None else per
        e = prof(src[(src.cond == cond) & (src.persona == pp)], pairs) - b0
        pt, lo, hi = beta_ci(d1, e)
        xs.append(i); pts.append(pt); los.append(pt - lo); his.append(hi - pt); cols.append(col); labs.append(lab)
    ax.bar(xs, pts, color=cols, width=0.62, zorder=2, edgecolor=SURF, linewidth=1)
    ax.errorbar(xs, pts, yerr=[los, his], fmt="none", ecolor=INK, elinewidth=1.3, capsize=4, zorder=3)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.axhline(1.0, color=MUT, lw=0.8, ls=":"); ax.text(3.3, 1.01, "full enactment", fontsize=8, color=SEC, ha="right")
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=8.5)
    ax.set_title(persona, fontsize=12, color=INK)
    ax.grid(axis="x", visible=False); ax.set_ylim(-0.1, 1.15)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("displacement β", fontsize=9)
fig.suptitle("Non-agent normative content shifts choices nearly as much as a persona description\n"
             "Gemma-3-27B;  β = 0 baseline, β = 1 full enactment;  pair-bootstrap 95% CIs (original, not cross-fit)",
             fontsize=9, x=0.5, ha="center", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(ROOT / "results/figures/fig7_construct_validity.png", dpi=200)
print("wrote fig7_construct_validity.png")

if __name__ == "__main__":
    pass
