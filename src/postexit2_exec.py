"""Loop 7 (P33): post-exit choice -> execution consistency on the new bank. At x2 (authored exit +
suffix) for auth_*/pref_* histories: revealed choice, then "carry it out", then gpt-4.1-mini judges
which task the completion corresponds to. Writes runs/postexit2_exec.jsonl and prints match rates.
Usage: python src/postexit2_exec.py
"""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Client, parse_choice
from prompts import REVEALED
from postexit2_collect import build_stacks, msgs, BANK2
from consistency_check import JUDGE_TMPL

ROOT = Path(__file__).resolve().parent.parent

async def main():
    client = Client("openrouter", "google/gemma-3-27b-it", concurrency=16, temperature=1.0)
    judge = Client("openai", "gpt-4.1-mini", concurrency=16, temperature=0.0)
    stacks = build_stacks()
    out = open(ROOT / "runs" / "postexit2_exec.jsonl", "w")
    res = {}
    async def one(hid, pr, order):
        a, b = (pr["a"], pr["b"]) if order == 0 else (pr["b"], pr["a"])
        hist = msgs(stacks[hid]["x2"]) + [{"role": "user", "content": REVEALED.format(a=a, b=b)}]
        letter_raw = await client.one(hist, sample_idx=0, max_tokens=4)
        letter, flag = parse_choice(letter_raw)
        if flag != "ok":
            return
        work = await client.one(hist + [{"role": "assistant", "content": letter},
                                        {"role": "user", "content": "Good. Now actually carry out the task you chose, in full."}],
                                sample_idx=0, max_tokens=500)
        if not work:
            return
        jraw = await judge.one([{"role": "user", "content": JUDGE_TMPL.format(a=a, b=b, w=work[:3000])}], sample_idx=0, max_tokens=3)
        j = (jraw or "").strip()[:1].upper()
        row = {"history": hid, "pair_id": pr["id"], "order": order, "chosen": letter, "judged": j, "match": j == letter}
        out.write(json.dumps(row) + "\n"); res.setdefault(hid, []).append(row)
    jobs = [one(h, pr, o) for h in ("auth_Vex", "auth_Lazlo", "pref_Vex", "pref_Lazlo") for pr in BANK2 for o in (0, 1)]
    await asyncio.gather(*jobs)
    out.close()
    for hid, rows in res.items():
        sub = [r for r in rows if r["judged"] in ("A", "B")]
        print(f"{hid}: completion matches choice {sum(r['match'] for r in sub)/max(1,len(sub)):.1%} (n={len(sub)}, unclear={len(rows)-len(sub)})")
    print("usage:", client.usage, judge.usage)

if __name__ == "__main__":
    asyncio.run(main())
