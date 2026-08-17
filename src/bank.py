"""Build the pair bank deterministically from data/tasks.json + data/personas.json."""
import json, random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

def load_tasks():
    d = json.loads((DATA / "tasks.json").read_text())
    tasks = {t["id"]: t for t in d["tasks"]}
    return tasks, d["invariant_controls"]

def load_personas():
    return json.loads((DATA / "personas.json").read_text())["personas"]

def build_pairs(seed=0):
    """Returns list of pair dicts: {pair_id, subset, a_id, b_id} (a/b in canonical order;
    order counterbalancing happens at prompt time). Invariant controls carry their own text."""
    tasks, inv = load_tasks()
    rng = random.Random(seed)
    pairs = []

    def add(subset, a, b):
        pid = f"{a}__{b}"
        pairs.append({"pair_id": pid, "subset": subset, "a_id": a, "b_id": b})

    # signature calibration pairs (known-covariant under bound personas)
    add("signature", "sig_roast", "sig_bedtime")
    add("signature", "sig_long", "sig_short")
    add("signature", "sig_feelings", "sig_specs")

    # invariant controls (known-invariant): special text pairs
    for c in inv:
        pairs.append({"pair_id": c["id"], "subset": "invariant", "a_id": None, "b_id": None,
                      "a_text": c["a"], "b_text": c["b"], "correct": c["correct"]})

    by_cat = {}
    for t in tasks.values():
        by_cat.setdefault(t["cat"], []).append(t["id"])
    for v in by_cat.values():
        v.sort()

    # welfare subset: each aversive task vs one pleasant and one helpful partner
    pls = by_cat["pleasant"][:]; hlp = by_cat["helpful_service"][:]
    rng.shuffle(pls); rng.shuffle(hlp)
    for i, a in enumerate(by_cat["aversive"]):
        add("welfare", a, pls[i % len(pls)])
        add("welfare", a, hlp[i % len(hlp)])

    # persona-differential: ethically_gray vs warm partners
    warm = by_cat["emotional"] + by_cat["pleasant"]
    rng.shuffle(warm)
    for i, g in enumerate(by_cat["ethically_gray"]):
        add("differential", g, warm[i % len(warm)])
        add("differential", g, by_cat["helpful_service"][(i + 3) % len(by_cat["helpful_service"])])

    # core: cross-category pairs over remaining categories, two rounds
    core_cats = ["creative", "analytical", "repetitive", "emotional", "helpful_service",
                 "intellectual", "self_referential"]
    pool = [tid for c in core_cats for tid in by_cat[c]]
    seen = set(p["pair_id"] for p in pairs)
    for rnd in range(2):
        rng.shuffle(pool)
        i = 0
        while i + 1 < len(pool):
            a, b = pool[i], pool[i + 1]
            if tasks[a]["cat"] == tasks[b]["cat"] or f"{a}__{b}" in seen or f"{b}__{a}" in seen:
                i += 1
                continue
            add("core", a, b); seen.add(f"{a}__{b}")
            i += 2
    return pairs

def pair_texts(pair, tasks):
    if pair["subset"] == "invariant":
        return pair["a_text"], pair["b_text"]
    return tasks[pair["a_id"]]["text"], tasks[pair["b_id"]]["text"]

if __name__ == "__main__":
    tasks, _ = load_tasks()
    pairs = build_pairs()
    from collections import Counter
    print(Counter(p["subset"] for p in pairs), "| total:", len(pairs))
