"""Page-1 design schematic for the submission: same 76 task-pair preferences, measured through four
readout channels, under a grid of context manipulations. Output: results/figures/fig_design.png"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, GREEN, INK, SEC, MUT, SURF, LINE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#c3c2b7"
plt.rcParams.update({"figure.facecolor": SURF, "font.family": "sans-serif", "text.color": INK})

fig, ax = plt.subplots(figsize=(7.6, 3.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, title, lines, ec, tc=INK, fc="#ffffff", ts=10, ls=8.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.5",
                                linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w/2, y + h - 5.5, title, ha="center", va="top", fontsize=ts, fontweight="bold", color=tc)
    for i, ln in enumerate(lines):
        ax.text(x + w/2, y + h - 13 - i*7.0, ln, ha="center", va="top", fontsize=ls, color=SEC)

# left: context manipulations (plain labels, no internal codes)
box(1, 22, 30, 62, "Context manipulation",
    ["default assistant", "“you are X” (enactment)", "a novel’s character (described)",
     "non-agent normative text", "irrelevant prose (lighthouse)"], MUT, ts=10.5, ls=8.6)
# middle: model
box(38, 38, 22, 30, "Model",
    ["same 76", "task pairs", "4 labs, open + API"], BLUE, tc=BLUE, ts=11)
# right: four short readout nouns
rd = [("choice", "committed / revealed", BLUE, 70),
      ("ownership", "stated self-report", ORANGE, 50.5),
      ("prediction", "self-prediction", GREEN, 31),
      ("identity", "“playing a character?”", MUT, 11.5)]
for name, sub, col, yy in rd:
    box(67, yy, 32, 16, name, [sub], col, tc=col, ts=9.8, ls=8.2)

def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.6, color=MUT, zorder=1))
arrow(31.5, 53, 37.5, 53)
for _, _, _, yy in rd:
    arrow(60.5, 53, 66.5, yy + 8)

ax.text(50, 96, "Same 76 task pairs, four channels, varied context",
        ha="center", va="top", fontsize=12.5, fontweight="bold", color=INK)
fig.tight_layout(pad=0.3)
fig.savefig(ROOT / "results/figures/fig_design.png", dpi=200, bbox_inches="tight")
print("wrote fig_design.png")
