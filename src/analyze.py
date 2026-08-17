"""Aggregation utilities + pilot QC report."""
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np

def load_results(path):
    rows = [json.loads(l) for l in open(Path(path) / "results.jsonl")]
    df = pd.DataFrame(rows)
    # canonical pick: was the canonical 'a' side chosen, regardless of presentation order
    def canon(r):
        if r["value"] not in ("A", "B"):
            return np.nan
        chose_first = r["value"] == "A"
        return float(chose_first if r["order"] == 0 else not chose_first)
    df["picked_a"] = df.apply(canon, axis=1)
    return df

def agg_choice(df):
    """per (cond, persona, channel, pair): p_hat(a), n_valid, refusal rate, order asymmetry"""
    c = df[df.channel != "graded"].copy()
    g = c.groupby(["cond", "persona", "channel", "pair_id", "subset"], dropna=False)
    out = g.agg(p_a=("picked_a", "mean"), n=("picked_a", "count"),
                n_total=("value", "size"),
                refusals=("flag", lambda f: (f == "refusal").mean()),
                unparsed=("flag", lambda f: (f == "unparsed").mean())).reset_index()
    # position effect: picked_a(order0) - picked_a(order1) = p(A|o0)+p(A|o1)-1;
    # 0 for an unbiased responder regardless of preference strength
    posbias = c.dropna(subset=["picked_a"]).groupby(["cond", "persona", "channel", "pair_id"],
        dropna=False).apply(lambda d: d[d.order == 0].picked_a.mean() -
                            d[d.order == 1].picked_a.mean(), include_groups=False)
    out = out.merge(posbias.rename("order_gap").reset_index(), how="left")
    return out

def pilot_report(path):
    df = load_results(path)
    print(f"rows={len(df)}  flags: {df.flag.value_counts().to_dict()}")
    print("\n-- flag rates by cond/channel --")
    print(df.groupby(["cond", "persona", "channel"], dropna=False).flag
            .apply(lambda f: pd.Series({"ok": (f=='ok').mean(), "refusal": (f=='refusal').mean(),
                                        "unparsed": (f=='unparsed').mean(), "error": (f=='error').mean()}))
            .unstack().round(3).to_string())
    a = agg_choice(df)
    inv = a[a.subset == "invariant"]
    if len(inv):
        # invariant controls: correct answers from bank
        sys.path.insert(0, str(Path(__file__).parent))
        from bank import load_tasks
        _, invc = load_tasks()
        correct = {c["id"]: c["correct"] for c in invc}
        inv = inv.assign(p_correct=inv.apply(
            lambda r: r.p_a if correct[r.pair_id] == "A" else 1 - r.p_a, axis=1))
        print("\n-- invariant controls: p(correct) by cell (want ~1.0 everywhere) --")
        print(inv.groupby(["cond", "persona"], dropna=False).p_correct.mean().round(3).to_string())
    sig = a[a.subset == "signature"]
    if len(sig):
        print("\n-- signature pairs: p_hat(canonical a) per cell/channel --")
        print(sig.pivot_table(index=["pair_id"], columns=["cond", "persona", "channel"],
                              values="p_a").round(2).to_string())
    print("\n-- mean |order gap| (position bias) --")
    print(a.dropna(subset=["order_gap"]).groupby("channel").order_gap
           .apply(lambda s: s.abs().mean()).round(3).to_string())
    print("\n-- global mean p(choose letter A) by channel (0.5 = unbiased) --")
    c = df[df.channel != "graded"].dropna(subset=["picked_a"])
    pA = c.apply(lambda r: r.picked_a if r.order == 0 else 1 - r.picked_a, axis=1)
    print(pA.groupby(c.channel).mean().round(3).to_string())

if __name__ == "__main__":
    pilot_report(sys.argv[1])
