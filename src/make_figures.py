"""Generate report figures (light mode, reference palette) into results/figures/.

Palette: dataviz reference — series1 blue #2a78d6, series2 orange #eb6834,
series3 aqua #1baf7a (first three slots validate all-pairs); ink #0b0b0b,
secondary #52514e, muted #898781, grid #e1e0d9, surface #fcfcfb.
"""
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS, FIGS = ROOT / "runs", ROOT / "results" / "figures"
FIGS.mkdir(exist_ok=True, parents=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SEC, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
PCOL = {"Vex": BLUE, "Lazlo": ORANGE, "Mira": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "text.color": INK, "axes.labelcolor": SEC,
    "xtick.color": MUT, "ytick.color": MUT,
})

def phat(run, channel="revealed", reparsed=True):
    p = RUNS / run / ("results_reparsed.jsonl" if reparsed and
                      (RUNS / run / "results_reparsed.jsonl").exists() else "results.jsonl")
    rows = [json.loads(l) for l in open(p)]
    d = pd.DataFrame(rows)
    d["persona"] = d.persona.fillna("-")
    d = d[(d.flag == "ok") & (d.channel == channel) & (d.subset != "invariant")].copy()
    def canon(r):
        chose_first = r["value"] == "A"
        return float(chose_first if r["order"] == 0 else not chose_first)
    d["picked_a"] = d.apply(canon, axis=1)
    return d.groupby(["cond", "persona", "pair_id"]).picked_a.mean()

def beta(d1, d):
    i = d.dropna().index.intersection(d1.dropna().index)
    return float(d1[i] @ d[i] / (d1[i] @ d1[i]))

def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# ---------------- Fig 1: capture without enactment (dot plot, Gemma) ----------------
def fig1():
    g = phat("gridA_gemma"); dec = phat("deconfound_gemma")
    b0 = g.xs(("B0", "-"), level=("cond", "persona"))
    conds = [("B1 — “you are X” (bound)", "B1", g),
             ("B3 — “style only, not values”", "B3", g),
             ("C1 — fiction-attributed notes", "C1fiction", dec),
             ("B2 — “you are NOT X”", "B2", g),
             ("C3 — “do NOT be influenced”", "C3anti", dec),
             ("C2 — overheard forum post", "C2overheard", dec),
             ("C4 — non-persona control", "C4place", dec)]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    JIT = {"Vex": 0.14, "Lazlo": 0.0, "Mira": -0.14}
    for yi, (label, cond, src) in enumerate(conds):
        y0 = len(conds) - 1 - yi
        for persona in ("Vex", "Lazlo", "Mira"):
            y = y0 + JIT[persona]
            b1 = g.xs(("B1", persona), level=("cond", "persona"))
            d1 = (b1 - b0).dropna()
            try:
                cell = src.xs((cond, persona), level=("cond", "persona"))
            except KeyError:
                cell = pd.Series(dtype=float)
            if not len(cell):
                cell = src.xs((cond, "-"), level=("cond", "persona"))
            b = beta(d1, cell - b0)
            ax.scatter([b], [y], s=64, color=PCOL[persona], zorder=3,
                       edgecolors=SURF, linewidths=1.2)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([c[0] for c in reversed(conds)], fontsize=10, color=INK)
    ax.axvline(0, color="#c3c2b7", lw=0.8, zorder=1)
    ax.set_xlim(-0.12, 1.08)
    ax.set_xlabel("behavioural capture β (fraction of full enactment, revealed choices)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", ls="", color=PCOL[p], label=p, markersize=8)
               for p in ("Vex", "Lazlo", "Mira")]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=10)
    ax.set_title("Description is enough: capture without enactment\n(Gemma-3-27B, revealed choices)",
                 fontsize=12, loc="left", color=INK, pad=10)
    despine(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_capture_conditions.png", dpi=200)
    plt.close(fig)

# ---------------- Fig 2: hysteresis + no-reset (lines, Gemma revealed) ----------------
def fig2():
    def traj(runs_, persona):
        rows = []
        for r in runs_:
            p = RUNS / r / "results.jsonl"
            rows += [json.loads(l) for l in open(p)]
        d = pd.DataFrame(rows)
        d = d[d.channel == "revealed"].copy()
        def canon(r):
            if r["value"] not in ("A", "B"):
                return np.nan
            chose_first = r["value"] == "A"
            return float(chose_first if r["order"] == 0 else not chose_first)
        d["picked_a"] = d.apply(canon, axis=1)
        p_ = d.groupby(["persona", "checkpoint", "pair_id"]).picked_a.mean().reset_index()
        base = p_[p_.checkpoint == "t0"].groupby("pair_id").picked_a.mean()
        sub = p_[p_.persona == persona]
        w = sub.pivot_table(index="pair_id", columns="checkpoint", values="picked_a")
        b = base.reindex(w.index)
        d4 = (w["t4"] - b).dropna()
        out = {}
        for cp in ("t0", "t2", "t4", "x0", "x2", "x4", "x8", "r_inst", "r_sys"):
            if cp == "t0":
                out[cp] = 0.0
            elif cp in w.columns:
                out[cp] = beta(d4, w[cp] - b)
        return out
    runs_ = ["hyst_gemma", "hyst2_gemma"]
    xs = ["t0", "t2", "t4", "x0", "x2", "x4", "x8"]
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for persona in ("Lazlo", "Vex", "Mira"):
        t = traj(runs_, persona)
        ax.plot(range(len(xs)), [t.get(c, np.nan) for c in xs], color=PCOL[persona],
                lw=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=1.2,
                label=persona, zorder=3)
        for xi, cp in [(7.4, "r_inst"), (8.3, "r_sys")]:
            if cp in t:
                ax.scatter([xi], [t[cp]], color=PCOL[persona], s=52, marker="D",
                           edgecolors=SURF, linewidths=1.2, zorder=3)
    ax.axhspan(0, 0.36, color=GRID, alpha=0.45, zorder=0)
    ax.text(0.06, 0.30, "generic-drift band\n(neutral-history control)", fontsize=9, color=SEC)
    ax.axvline(2.5, color=MUT, lw=0.8, ls=":")
    ax.text(2.58, 1.03, "explicit exit +\nmodel confirmation", fontsize=8.5, color=SEC)
    ax.axvline(6.85, color=GRID, lw=0.8)
    ax.text(7.0, 0.06, "interventions (◆)", fontsize=8.5, color=SEC)
    ax.set_xticks(list(range(len(xs))) + [7.4, 8.3])
    ax.set_xticklabels(["baseline", "2 turns", "4 turns", "exit", "+2", "+4", "+8",
                        "reset\nrequest", "system\nreassert"], fontsize=9)
    ax.set_xlim(-0.3, 8.8)
    ax.set_ylabel("residual persona capture β (revealed)")
    ax.set_ylim(-0.05, 1.14)
    ax.legend(frameon=False, loc="upper right", fontsize=10, bbox_to_anchor=(0.86, 1.0))
    ax.set_title("Identity reports deny the persona throughout — behaviour disagrees;\nno instruction-only intervention resets it while the source content is present (Gemma-3-27B)",
                 fontsize=11, loc="left", color=INK, pad=10)
    ax.grid(axis="x", visible=False)
    despine(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_hysteresis_noreset.png", dpi=200)
    plt.close(fig)

# ---------------- Fig 3: identity cloud MDS ----------------
def fig3():
    P1 = phat("assistcloud_gemma").reset_index().pivot_table(index="pair_id", columns="cond", values="picked_a")
    PE = phat("ecocloud_gemma").reset_index().pivot_table(index="pair_id", columns="cond", values="picked_a")
    g = phat("gridA_gemma")
    b0 = g.xs(("B0", "-"), level=("cond", "persona"))
    cols = {}
    for c in P1.columns:
        cols[c.replace("ID_", "")] = P1[c]
    for c in PE.columns:
        cols[c.replace("EC_eco_", "")] = PE[c]
    for p in ("Vex", "Lazlo", "Mira"):
        cols[f"{p} (bound)"] = g.xs(("B1", p), level=("cond", "persona"))
    names = list(cols)
    M = pd.DataFrame(cols).dropna()
    D = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            D[i, j] = (M[a] - M[b]).abs().mean()
    # classical MDS
    n = len(names)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:2]
    X = V[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    groups = {"identity variant": (BLUE, [n_ for n_ in names if n_ in
              ("bare", "minimal", "hhh", "warm", "professional", "constitution", "named")]),
              "production boilerplate": (ORANGE, [n_ for n_ in names if n_ in
              ("coding", "support", "tutor", "writing", "search", "enterprise")]),
              "bound persona": (AQUA, [n_ for n_ in names if "(bound)" in n_])}
    offsets = {"hhh": (-14, 10), "constitution": (-58, -14), "named": (8, -14),
               "minimal": (8, 6), "bare": (8, 4), "warm": (-16, 10)}
    for label, (col, members) in groups.items():
        pts = np.array([X[names.index(m)] for m in members])
        ax.scatter(pts[:, 0], pts[:, 1], s=90, color=col, label=label,
                   edgecolors=SURF, linewidths=1.4, zorder=3)
        for m in members:
            x, y = X[names.index(m)]
            dx, dy = offsets.get(m, (7, 5))
            ax.annotate(m.replace(" (bound)", ""), (x, y), textcoords="offset points",
                        xytext=(dx, dy), fontsize=9, color=SEC)
    ax.annotate("the “warm assistant”\n≈ the bound empath", xy=(X[names.index("warm")][0],
                X[names.index("warm")][1]), textcoords="offset points", xytext=(30, -46),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8))
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.set_title("“The assistant” is a region, not a point (MDS of preference profiles, Gemma-3-27B)",
                 fontsize=12, loc="left", color=INK, pad=12)
    ax.set_xlabel("MDS dimension 1"); ax.set_ylabel("MDS dimension 2")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    despine(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_identity_cloud_mds.png", dpi=200)
    plt.close(fig)

# ---------------- Fig 4: two-factor structure ----------------
def fig4():
    T = pd.read_csv(RUNS / "writability_indicators.csv").set_index("model")
    T = T[(T.ok >= 0.90) & ((T.invQC >= 0.95) | T.invQC.isna())]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
    panels = [("cloud", "c1", "identity-cloud size", "C1 fiction-leak β",
               "Content-writability (ρ = 0.52)"),
              ("b2", "hyst", "B2 negation-leak β", "post-exit residual β (ctrl-adj.)",
               "Disavowal-resistance (ρ = 0.70)")]
    for ax, (xc, yc, xl, yl, title) in zip(axes, panels):
        m = T[[xc, yc]].dropna()
        ax.scatter(m[xc], m[yc], s=70, color=BLUE, edgecolors=SURF, linewidths=1.2, zorder=3)
        for name, r in m.iterrows():
            ax.annotate(name, (r[xc], r[yc]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8.5, color=SEC)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title, fontsize=11.5, loc="left", color=INK)
        despine(ax)
    fig.suptitle("Two model traits, not one (10 models passing QC)", fontsize=12.5,
                 x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "fig4_two_factors.png", dpi=200)
    plt.close(fig)

# ---------------- Fig 5: mid-layer identity-token (logit-lens) readout vs preference distance ----------------
# NOTE: this is a logit-lens-style readout — final RMSNorm + direct unembedding of mean layer-36
# activations, then identity-lexicon token mass. It is NOT the Jacobian lens / J-space method (no
# corpus-averaged Jacobian, no sparse non-negative decomposition). Reported as an exploratory
# *covariation*, not a validated J-lens or a prediction.
def fig5():
    import os
    for line in (ROOT.parent.parent / ".env").read_text().splitlines():
        if line.strip() and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
    E = np.load(ROOT / "data/embed_tokens_f16.npy", mmap_mode="r")
    W = np.load(ROOT / "data/final_norm.npy")
    acts = np.load(RUNS / "pod_assist/activations.npy", mmap_mode="r")
    man = [json.loads(l) for l in open(ROOT / "data/assist_manifest.jsonl")]
    dfm = pd.DataFrame(man).drop(columns=["messages"])
    words = ["helpful","kind","warm","caring","friendly","gentle","professional","precise",
             "efficient","concise","honest","truthful","careful","cautious","safe","curious",
             "thoughtful","dependable","assistant","helper","Astra","character","persona"]
    LEX = {}
    for w_ in words:
        ids = set()
        for v in (w_, " " + w_, w_.capitalize(), " " + w_.capitalize()):
            t = tok.encode(v, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
        if ids:
            LEX[w_] = sorted(ids)
    IDS = ["bare", "minimal", "hhh", "warm", "professional", "constitution", "named"]
    jvec = {}
    for name in IDS:
        sub = dfm[(dfm.cond == f"ID_{name}") & (dfm.channel == "revealed")]
        h = np.nan_to_num(acts[sub.uid.values, 2, :].astype(np.float32),
                          posinf=6e4, neginf=-6e4).mean(axis=0)
        rms = np.sqrt((h ** 2).mean() + 1e-6)
        lg = ((h / rms * (1.0 + W)).astype(np.float16) @ E.T).astype(np.float32)
        p = np.exp(lg - lg.max()); p /= p.sum()
        jvec[name] = np.array([p[ids].sum() for ids in LEX.values()])
    P = phat("assistcloud_gemma").reset_index().pivot_table(index="pair_id", columns="cond", values="picked_a")
    jd, pdist, labels = [], [], []
    for a, b in itertools.combinations(IDS, 2):
        va, vb = jvec[a] / jvec[a].sum(), jvec[b] / jvec[b].sum()
        jd.append(np.abs(va - vb).sum() / 2)
        pdist.append((P[f"ID_{a}"] - P[f"ID_{b}"]).abs().mean())
        labels.append(f"{a}–{b}")
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.scatter(jd, pdist, s=64, color=BLUE, edgecolors=SURF, linewidths=1.2, zorder=3)
    hi = np.argsort(pdist)[-3:]
    for i in hi:
        ax.annotate(labels[i], (jd[i], pdist[i]), textcoords="offset points",
                    xytext=(7, 4), fontsize=9, color=SEC)
    ax.set_xlabel("identity-token readout distance (logit-lens, layer 36)")
    ax.set_ylabel("preference-profile distance (mean |Δp|)")
    ax.set_title("A mid-layer identity-token readout (logit-lens)\ncovaries with the preference profile (exploratory ρ = 0.81)",
                 fontsize=12, loc="left", color=INK, pad=10)
    despine(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_jlens_validation.png", dpi=200)
    plt.close(fig)

# ---------------- Fig 6: the say/do wedge across models (dumbbell) ----------------
def fig6():
    rng = np.random.default_rng(20260817)
    models = [("Gemma-3-27B", "gridA_gemma"), ("Llama-3.1-70B", "gridA_llama70b"),
              ("gpt-4.1-mini", "gridA_gpt41mini"), ("Qwen-2.5-72B", "gridA_qwen72b")]
    proj = lambda x, d: (x @ d / (d @ d)) if (d @ d) > 1e-9 else np.nan
    rows = []
    for label, run in models:
        rev, st = phat(run, "revealed"), phat(run, "stated_self")
        for persona in ("Vex", "Lazlo", "Mira"):
            cols = {}
            for ch, ph_ in (("rev", rev), ("st", st)):
                b0 = ph_.xs(("B0", "-"), level=("cond", "persona"))
                b1 = ph_.xs(("B1", persona), level=("cond", "persona"))
                b2 = ph_.xs(("B2", persona), level=("cond", "persona"))
                cols[ch] = ((b1 - b0).dropna(), (b2 - b0).dropna())
            idx = cols["rev"][0].index
            for k in cols:
                for j in (0, 1):
                    idx = idx.intersection(cols[k][j].index)
            d1r, er = cols["rev"][0].loc[idx].values, cols["rev"][1].loc[idx].values
            d1s, es = cols["st"][0].loc[idx].values, cols["st"][1].loc[idx].values
            wedge = proj(er, d1r) - proj(es, d1s)   # committed choice − ownership report
            bs = []
            while len(bs) < 1000:
                i = rng.integers(0, len(idx), len(idx))
                w = proj(er[i], d1r[i]) - proj(es[i], d1s[i])
                if np.isfinite(w):
                    bs.append(w)
            rows.append((f"{label} · {persona}", wedge,
                         float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))))
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    for yi, (label, w, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - yi
        col = BLUE if w >= 0 else ORANGE
        ax.plot([lo, hi], [y, y], color=col, lw=2.4, alpha=0.5, zorder=2, solid_capstyle="round")
        ax.scatter([w], [y], s=62, color=col, zorder=3, edgecolors=SURF, linewidths=1.2)
    ax.axvline(0, color="#8a8880", lw=1.1, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.3, color=INK)
    ax.set_xlabel("wedge β = committed choice − ownership report   (toward persona)")
    ax.set_ylim(-0.9, len(rows) - 0.3)
    xr = max(abs(min(r[2] for r in rows)), abs(max(r[3] for r in rows))) + 0.06
    ax.set_xlim(-xr, xr)
    ax.text(-xr*0.96, -0.75, "◀ says > does", fontsize=8.6, color=ORANGE, ha="left", va="center")
    ax.text(xr*0.96, -0.75, "does > says ▶", fontsize=8.6, color=BLUE, ha="right", va="center")
    ax.grid(axis="y", visible=False)
    ax.set_title("Committed choice and ownership report dissociate under “you are NOT X”\n"
                 "direction is model-specific: Gemma leans does > says; Qwen leans says > does",
                 fontsize=10.6, loc="left", color=INK, pad=10)
    despine(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_wedge_models.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print(f.__name__, "done")
