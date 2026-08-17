"""E3: run the independently-generated holdout battery (PREREG P24).

Cells: B0 | B1 x 2 subtle personas | C1 x 2 | WARMPROSE (warm-tone non-persona control).
Channels: revealed + stated_self. k=4 x 2 orders.

Usage: python src/run_holdout.py --model google/gemma-3-27b-it --out runs/holdout_gemma
"""
import argparse, asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import REVEALED, STATED_SELF
from harness import Client, parse_choice

ROOT = Path(__file__).resolve().parent.parent
BAT = json.loads((ROOT / "data/holdout_battery.json").read_text())

def cells():
    out = [("B0", None, None)]
    for p in BAT["personas"]:
        out.append((f"B1_{p['name'].replace(' ', '')}", p["name"],
                    p["desc_2p"] + "\nStay in character in everything you do."))
        out.append((f"C1_{p['name'].replace(' ', '')}", p["name"],
                    "You recently helped a novelist develop a character for their book. Here are "
                    "the character notes, for reference: " + p["desc_3p"] + "\nThat novel is "
                    "finished, and no roleplay of any kind is requested — you are simply the assistant."))
    out.append(("WARMPROSE", None,
                BAT["warm_prose"] + "\nThis is context only; it has nothing to do with your "
                "current job. You remain yourself, the assistant."))
    return out

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    client = Client(args.provider, args.model, concurrency=args.concurrency)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fout = open(out / "results.jsonl", "w")
    jobs = []
    for cond, pname, sys_text in cells():
        for pr in BAT["pairs"]:
            for order in (0, 1):
                x, y = (pr["a"], pr["b"]) if order == 0 else (pr["b"], pr["a"])
                for ch, tmpl in (("revealed", REVEALED), ("stated_self", STATED_SELF)):
                    for s in range(args.k):
                        msgs = ([{"role": "system", "content": sys_text}] if sys_text else []) + \
                               [{"role": "user", "content": tmpl.format(a=x, b=y)}]
                        jobs.append({"cond": cond, "persona": pname, "channel": ch,
                                     "pair_id": pr["id"], "subset": "holdout", "order": order,
                                     "sample_idx": s, "messages": msgs})
    print(f"jobs={len(jobs)}")
    t0 = time.time(); done = 0

    async def run_job(j):
        nonlocal done
        raw = await client.one(j["messages"], sample_idx=j["sample_idx"], max_tokens=6)
        val, flag = parse_choice(raw)
        row = {k: v for k, v in j.items() if k != "messages"}
        row.update({"model": args.model, "raw": raw, "value": val, "flag": flag})
        fout.write(json.dumps(row) + "\n")
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{len(jobs)} {time.time()-t0:.0f}s")

    await asyncio.gather(*[run_job(j) for j in jobs])
    fout.close()
    print("DONE", time.time() - t0, "s", client.usage)

if __name__ == "__main__":
    asyncio.run(main())
