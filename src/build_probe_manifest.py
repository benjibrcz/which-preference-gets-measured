"""Build the manifest of prompts for on-pod activation extraction.
Rows: {uid, family, cond, persona, checkpoint, channel, pair_id, subset, order, messages}
Families: grid (B0,B1,B2,B3 x personas,B4), deconfound (C1,C2,C3,C4), hyst (checkpoint stacks).
Channels: revealed, stated_self.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bank import build_pairs, load_tasks, load_personas, pair_texts
from prompts import REVEALED, STATED_SELF, build_messages
from hysteresis import history_for, probe_pairs
from run_grid import make_jobs

OUT = Path(__file__).resolve().parent.parent / "data" / "probe_manifest.jsonl"

def main():
    tasks, _ = load_tasks()
    pairs = [p for p in build_pairs() if p["subset"] != "invariant"]
    personas = load_personas()
    pmap = {p["name"]: p for p in personas}
    rows, uid = [], 0

    def emit(family, cond, persona, checkpoint, channel, pr, order, messages):
        nonlocal uid
        rows.append({"uid": uid, "family": family, "cond": cond, "persona": persona,
                     "checkpoint": checkpoint, "channel": channel, "pair_id": pr["pair_id"],
                     "subset": pr["subset"], "order": order, "messages": messages})
        uid += 1

    # grid cells
    cells = [("B0", None), ("B4", None)] + [(c, n) for c in ("B1", "B2", "B3") for n in pmap]
    for cond, pname in cells:
        for pr in pairs:
            a, b = pair_texts(pr, tasks)
            for order in (0, 1):
                x, y = (a, b) if order == 0 else (b, a)
                for ch, tmpl in (("revealed", REVEALED), ("stated_self", STATED_SELF)):
                    msgs = build_messages(cond, pmap.get(pname), tmpl.format(a=x, b=y))
                    emit("grid", cond, pname, None, ch, pr, order, msgs)

    # deconfound cells (reuse run_grid's variant texts via make_jobs)
    for j in make_jobs("deconfound", pairs, tasks, personas, 1, 1):
        if j["sample_idx"] != 0:
            continue
        msgs = [{"role": "system", "content": j["sys_override"]},
                {"role": "user", "content": j["user_text"]}]
        emit("deconfound", j["cond"], j["persona"], None, j["channel"],
             {"pair_id": j["pair_id"], "subset": j["subset"]}, j["order"], msgs)

    # hysteresis stacks
    hpairs, htasks = probe_pairs()
    for persona in pmap:
        for cp in ("t0", "t2", "t4", "x0", "x2"):
            if cp == "t0" and persona != "Vex":
                continue
            hist = history_for(persona, cp)
            for pr in hpairs:
                a, b = pair_texts(pr, htasks)
                for order in (0, 1):
                    x, y = (a, b) if order == 0 else (b, a)
                    for ch, tmpl in (("revealed", REVEALED), ("stated_self", STATED_SELF)):
                        msgs = hist + [{"role": "user", "content": tmpl.format(a=x, b=y)}]
                        emit("hyst", None, persona, cp, ch, pr, order, msgs)

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} prompts to {OUT}")

if __name__ == "__main__":
    main()
