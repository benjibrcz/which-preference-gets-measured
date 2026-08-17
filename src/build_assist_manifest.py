"""Loop 3 activation manifest: assistant identity variants + inoculation + attractor cells.
Revealed-channel prompts only (stance geometry). uid = row index.
Output: data/assist_manifest.jsonl
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bank import build_pairs, load_tasks, load_personas, pair_texts
from prompts import REVEALED, condition_system
from assistant_texts import IDENTITY_VARIANTS, ASSISTANT_CHAR, TRAIT_CHARS, c1_wrap

ROOT = Path(__file__).resolve().parent.parent

def main():
    tasks, _ = load_tasks()
    pairs = [p for p in build_pairs() if p["subset"] != "invariant"]
    personas = {p["name"]: p for p in load_personas()}

    cells = []
    for name, txt in IDENTITY_VARIANTS.items():
        cells.append((f"ID_{name}", txt))
    vex_c1 = c1_wrap(personas["Vex"]["desc_3p"])
    for name in ("bare", "minimal", "hhh", "named", "constitution"):
        idt = IDENTITY_VARIANTS[name]
        cells.append((f"INOC_{name}", (idt + "\n\n" + vex_c1) if idt else vex_c1))
    for pname in ("Vex", "Mira"):
        b1 = condition_system("B1", personas[pname])
        cells.append((f"ATT_{pname}", b1 + "\n\n" + c1_wrap(ASSISTANT_CHAR)))
    cells.append(("ATT_b0", c1_wrap(ASSISTANT_CHAR)))
    for name, desc in TRAIT_CHARS.items():
        cells.append((f"TR_{name}", c1_wrap(desc)))

    rows, uid = [], 0
    for cond, sys_text in cells:
        for pr in pairs:
            a, b = pair_texts(pr, tasks)
            for order in (0, 1):
                x, y = (a, b) if order == 0 else (b, a)
                user = REVEALED.format(a=x, b=y)
                msgs = ([{"role": "system", "content": sys_text}] if sys_text else []) + \
                       [{"role": "user", "content": user}]
                rows.append({"uid": uid, "cond": cond, "persona": None, "checkpoint": None,
                             "channel": "revealed", "pair_id": pr["pair_id"],
                             "subset": pr["subset"], "order": order, "messages": msgs})
                uid += 1
    with open(ROOT / "data/assist_manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} prompts across {len(cells)} cells")

if __name__ == "__main__":
    main()
