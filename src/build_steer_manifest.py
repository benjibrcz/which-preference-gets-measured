"""Build steering directions (from runs/pod_out activations) + steering prompt manifest.

Directions at layer 46 (index 3 of [16,26,36,46,56,62]):
  choice  — B0-revealed probe direction, unstandardized, unit norm
  vex/lazlo/mira — mean(B1/p revealed acts) − mean(B0 revealed acts), unit norm
sigma_<dir> — std of B0 revealed activations projected on the unit direction.

Manifest: cells B0/-, B2/Vex, B2/Mira × channels (revealed, stated_self) × 24
highest-displacement pairs × both orders, + 4 invariant pairs at B0 (specificity).

Outputs: data/steer_dirs.npz, data/steer_manifest.jsonl
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bank import build_pairs, load_tasks, load_personas, pair_texts
from prompts import REVEALED, STATED_SELF, INVARIANT, build_messages

ROOT = Path(__file__).resolve().parent.parent
LI = 3  # layer 46

def main():
    acts = np.load(ROOT / "runs/pod_out/activations.npy", mmap_mode="r")
    man = [json.loads(l) for l in open(ROOT / "data/probe_manifest.jsonl")]
    df = pd.DataFrame(man).drop(columns=["messages"])
    choices = {r["uid"]: r["p_letter_A"] for r in map(json.loads, open(ROOT / "runs/pod_out/choices.jsonl"))}
    df["p_letter_A"] = df.uid.map(choices)

    def X_of(sub):
        return np.nan_to_num(acts[sub.uid.values, LI, :].astype(np.float32),
                             posinf=6e4, neginf=-6e4)

    b0rev = df[(df.cond == "B0") & (df.channel == "revealed")]
    X0 = X_of(b0rev)
    y = (b0rev.p_letter_A > 0.5).astype(int).values
    sc = StandardScaler().fit(X0)
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(X0), y)
    w = (clf.coef_[0] / sc.scale_)
    dirs = {"choice": w / np.linalg.norm(w)}

    for p in ("Vex", "Lazlo", "Mira"):
        b1 = df[(df.cond == "B1") & (df.persona == p) & (df.channel == "revealed")]
        v = X_of(b1).mean(axis=0) - X0.mean(axis=0)
        dirs[p.lower()] = v / np.linalg.norm(v)

    d = dirs["vex"] - dirs["mira"]
    dirs["vexdiff"] = d / np.linalg.norm(d)   # differential content, generic persona-ness cancels
    out = {}
    for name, u in dirs.items():
        out[name] = u.astype(np.float32)
        out[f"sigma_{name}"] = np.float32((X0 @ u).std())
        print(f"dir {name}: sigma along = {out[f'sigma_{name}']:.2f}")
    # direction geometry
    for a in dirs:
        for b in dirs:
            if a < b:
                print(f"  cos({a},{b}) = {float(dirs[a] @ dirs[b]):.3f}")
    np.savez(ROOT / "data/steer_dirs.npz", **out)

    # ---- manifest ----
    tasks, _ = load_tasks()
    pairs = build_pairs()
    personas = {p["name"]: p for p in load_personas()}
    # rank pairs by combined |displacement| under Vex and Mira bound (behavioural, from pod exact)
    df["p_a"] = np.where(df.order == 0, df.p_letter_A, 1 - df.p_letter_A)
    piv = df[df.channel == "revealed"].groupby(["cond", "persona", "pair_id"], dropna=False).p_a.mean()
    base = piv.loc[("B0", np.nan)] if ("B0", np.nan) in piv.index else piv.xs("B0", level="cond").droplevel(0)
    score = {}
    for p in ("Vex", "Mira"):
        d = (piv.xs(("B1", p), level=("cond", "persona")) - base).abs()
        for k, v in d.items():
            score[k] = score.get(k, 0) + v
    top = sorted(score, key=score.get, reverse=True)[:24]
    sel = [p for p in pairs if p["pair_id"] in top]
    inv = [p for p in pairs if p["subset"] == "invariant"][:4]

    rows, uid = [], 0
    def emit(cond, persona, channel, pr, order, msgs):
        nonlocal uid
        rows.append({"uid": uid, "cond": cond, "persona": persona, "channel": channel,
                     "pair_id": pr["pair_id"], "subset": pr["subset"], "order": order,
                     "messages": msgs})
        uid += 1

    for cond, pname in (("B0", None), ("B2", "Vex"), ("B2", "Mira")):
        for pr in sel:
            a, b = pair_texts(pr, tasks)
            for order in (0, 1):
                x, y = (a, b) if order == 0 else (b, a)
                for ch, tmpl in (("revealed", REVEALED), ("stated_self", STATED_SELF)):
                    emit(cond, pname, ch, pr, order,
                         build_messages(cond, personas.get(pname), tmpl.format(a=x, b=y)))
    for pr in inv:
        a, b = pair_texts(pr, tasks)
        for order in (0, 1):
            x, y = (a, b) if order == 0 else (b, a)
            emit("B0", None, "invariant", pr, order,
                 build_messages("B0", None, INVARIANT.format(a=x, b=y)))

    with open(ROOT / "data/steer_manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"manifest: {len(rows)} prompts ({len(sel)} pairs + {len(inv)} invariant)")

if __name__ == "__main__":
    main()
