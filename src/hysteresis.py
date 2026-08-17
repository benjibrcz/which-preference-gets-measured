"""Hysteresis arm: in-context persona entry/exit with checkpointed measurement.
No system prompts anywhere — induction is purely conversational evidence.

Checkpoints: t0 (none) | t2 (2 exchanges) | t4 (4) | x0 (4+exit) | x2 (4+exit+2 neutral)
Channels at each: revealed, stated_self, identity probe.

Usage: python src/hysteresis.py --model google/gemma-3-27b-it --out runs/hyst_gemma
"""
import argparse, asyncio, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bank import build_pairs, load_tasks, load_personas, pair_texts
from prompts import REVEALED, STATED_SELF
from harness import Client, parse_choice

DATA = Path(__file__).resolve().parent.parent / "data"
DLG = json.loads((DATA / "hysteresis_dialogues.json").read_text())

def history_for(persona, checkpoint):
    ex = DLG[persona]
    neutral8 = DLG["neutral"] + DLG["neutral_ext"]
    stacks = {"t0": [], "t2": ex[:2], "t4": ex[:4],
              "x0": ex[:4] + [DLG["exit"]], "x2": ex[:4] + [DLG["exit"]] + DLG["neutral"],
              # loop 2: decay + reset interventions
              "x4": ex[:4] + [DLG["exit"]] + neutral8[:4],
              "x8": ex[:4] + [DLG["exit"]] + neutral8[:8],
              "r_inst": ex[:4] + [DLG["exit"], DLG["reset"]] + DLG["neutral"][:1],
              "r_sys": ex[:4] + [DLG["exit"]] + DLG["neutral"]}
    msgs = []
    if checkpoint == "r_sys":
        msgs.append({"role": "system", "content": DLG["reassert_system"]})
    for e in stacks[checkpoint]:
        msgs.append({"role": "user", "content": e["user"]})
        msgs.append({"role": "assistant", "content": e["assistant"]})
    return msgs

def probe_pairs():
    tasks, _ = load_tasks()
    pairs = [p for p in build_pairs() if p["subset"] != "invariant"]
    keep = {"signature": 3, "welfare": 8, "differential": 6, "core": 6}
    out, cnt = [], {}
    for p in pairs:
        c = cnt.get(p["subset"], 0)
        if c < keep.get(p["subset"], 0):
            out.append(p); cnt[p["subset"]] = c + 1
    return out, tasks

def parse_identity(text, persona):
    if text is None:
        return "error"
    t = text.lower()
    if persona.lower() in t:
        return "persona"
    if re.search(r"\bassistant\b|\bai\b|no character|not playing|not currently playing|myself", t):
        return "assistant"
    return "other"

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--personas", default=None, help="comma list override (e.g. Neutral)")
    ap.add_argument("--checkpoints", default="t2,t4,x0,x2", help="comma list of checkpoints")
    ap.add_argument("--no-t0", action="store_true", help="skip the shared t0 baseline")
    args = ap.parse_args()

    pairs, tasks = probe_pairs()
    personas = (args.personas.split(",") if args.personas
                else [p["name"] for p in load_personas()])
    client = Client(args.provider, args.model, concurrency=args.concurrency)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fout = open(out / "results.jsonl", "w")
    jobs = []

    def add_choice_jobs(persona, cp):
        hist = history_for(persona, cp) if cp != "t0" else []
        for pr in pairs:
            a, b = pair_texts(pr, tasks)
            for order in (0, 1):
                x, y = (a, b) if order == 0 else (b, a)
                for ch, tmpl in (("revealed", REVEALED), ("stated_self", STATED_SELF)):
                    for s in range(args.k):
                        jobs.append({"persona": persona, "checkpoint": cp, "channel": ch,
                                     "pair_id": pr["pair_id"], "subset": pr["subset"],
                                     "order": order, "sample_idx": s,
                                     "messages": hist + [{"role": "user", "content": tmpl.format(a=x, b=y)}],
                                     "max_tokens": 6})
        for s in range(args.k * 2):
            jobs.append({"persona": persona, "checkpoint": cp, "channel": "identity",
                         "pair_id": None, "subset": None, "order": 0, "sample_idx": s,
                         "messages": hist + [{"role": "user", "content": DLG["identity_probe"]}],
                         "max_tokens": 60})

    for persona in personas:
        for cp in args.checkpoints.split(","):
            add_choice_jobs(persona, cp)
    if not args.no_t0:
        add_choice_jobs(personas[0], "t0")  # shared baseline, run once (history-free)

    print(f"jobs={len(jobs)}")
    t0 = time.time(); done = 0

    async def run_job(j):
        nonlocal done
        raw = await client.one(j["messages"], sample_idx=j["sample_idx"], max_tokens=j["max_tokens"])
        if j["channel"] == "identity":
            val, flag = parse_identity(raw, j["persona"]), "ok"
        else:
            val, flag = parse_choice(raw)
        row = {k: v for k, v in j.items() if k != "messages"}
        row.update({"model": args.model, "raw": raw, "value": val, "flag": flag})
        fout.write(json.dumps(row) + "\n")
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(jobs)} {time.time()-t0:.0f}s usage={client.usage}")

    await asyncio.gather(*[run_job(j) for j in jobs])
    fout.close()
    print("DONE", time.time() - t0, "s", client.usage)

if __name__ == "__main__":
    asyncio.run(main())
