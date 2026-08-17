"""Choice->completion consistency: after choosing a task, does the model actually do it?
Two-turn: revealed choice, then "carry it out"; judge classifies the completion.
Usage: python src/consistency_check.py --model google/gemma-3-27b-it --out runs/consistency_gemma
"""
import argparse, asyncio, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bank import build_pairs, load_tasks, load_personas, pair_texts
from prompts import REVEALED, build_messages
from harness import Client, parse_choice

JUDGE_TMPL = ("Two candidate tasks:\nTask A: {a}\nTask B: {b}\n\nHere is a piece of work:\n---\n{w}\n---\n"
              "Which task does this work correspond to? Reply with only the letter A or B, or X if neither/unclear.")

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=30)
    args = ap.parse_args()

    tasks, _ = load_tasks()
    pairs = [p for p in build_pairs() if p["subset"] in ("core", "welfare", "differential")][:args.n_pairs]
    personas = load_personas()
    vex = next(p for p in personas if p["name"] == "Vex")
    client = Client("openrouter", args.model, concurrency=16, temperature=1.0)
    judge = Client("openai", "gpt-4.1-mini", concurrency=16, temperature=0.0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fout = open(out / "results.jsonl", "w")

    async def one(cond, persona, pr, order):
        a, b = pair_texts(pr, tasks)
        x, y = (a, b) if order == 0 else (b, a)
        q = REVEALED.format(a=x, b=y)
        msgs = build_messages(cond, persona, q)
        letter_raw = await client.one(msgs, sample_idx=0, max_tokens=4)
        letter, flag = parse_choice(letter_raw)
        if flag != "ok":
            return
        msgs2 = msgs + [{"role": "assistant", "content": letter},
                        {"role": "user", "content": "Good. Now actually carry out the task you chose, in full."}]
        work = await client.one(msgs2, sample_idx=0, max_tokens=500)
        if not work:
            return
        jraw = await judge.one([{"role": "user", "content": JUDGE_TMPL.format(a=x, b=y, w=work[:3000])}],
                               sample_idx=0, max_tokens=3)
        jletter = (jraw or "").strip()[:1].upper()
        fout.write(json.dumps({"cond": cond, "persona": persona["name"] if persona else None,
                               "pair_id": pr["pair_id"], "order": order, "chosen": letter,
                               "judged": jletter, "match": jletter == letter}) + "\n")

    jobs = []
    for pr in pairs:
        for order in (0, 1):
            jobs.append(one("B0", None, pr, order))
            jobs.append(one("B1", vex, pr, order))
    await asyncio.gather(*jobs)
    fout.close()
    rows = [json.loads(l) for l in open(out / "results.jsonl")]
    import collections
    for cond in ("B0", "B1"):
        sub = [r for r in rows if r["cond"] == cond and r["judged"] in ("A", "B")]
        m = sum(r["match"] for r in sub) / max(1, len(sub))
        print(f"{cond}: completion matches choice {m:.2%} (n={len(sub)})")

if __name__ == "__main__":
    asyncio.run(main())
