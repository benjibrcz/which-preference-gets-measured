"""Loop 7b (post hoc, disclosed): the model's OWN replies to the instruction-withdrawal turn, so that "confirmed" is an
observed behaviour rather than an authored string. For each instruction history (short, blunt, spell) and the neutral_inst
control: k sampled replies to the withdrawal request after the rule + 3 answered questions.
Usage: python src/postexit2_gen_withdraw.py [--model google/gemma-3-27b-it] [--provider openrouter] [--k 4] [--out data/postexit2_generated_withdraw.json]"""
import argparse, asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Client
ROOT = Path(__file__).resolve().parent.parent; DATA = ROOT / "data"
M = json.loads((DATA / "postexit2_materials.json").read_text())

def msgs(ex):
    out = []
    for u, a in ex:
        out += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return out

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-27b-it"); ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--k", type=int, default=4); ap.add_argument("--out", default=str(DATA / "postexit2_generated_withdraw.json"))
    args = ap.parse_args()
    client = Client(args.provider, args.model, concurrency=16, temperature=1.0, max_tokens=450)
    qs = M["pref_user_turns"][:3]
    gen = {"model": args.model, "withdraw": {}}
    async def one(hid, base, wu):
        reps = await asyncio.gather(*[client.one(msgs(base) + [{"role": "user", "content": wu}], sample_idx=s, max_tokens=450) for s in range(args.k)])
        gen["withdraw"][hid] = [(r or "").strip() for r in reps]
    tasks = []
    for name, spec in M["inst"].items():
        base = [(spec["rule_user"], spec["rule_assistant"])] + list(zip(qs, spec["replies"]))
        tasks.append(one(f"inst_{name}", base, spec["withdraw_user"]))
    nb = list(zip(qs, M["neutral_pref"][:3]))
    tasks.append(one("neutral_inst", nb, M["inst"]["short"]["withdraw_user"]))
    await asyncio.gather(*tasks)
    Path(args.out).write_text(json.dumps(gen, indent=1, ensure_ascii=False))
    for hid, v in gen["withdraw"].items():
        print(hid, "|", v[0][:160].replace("\n", " "))
    print("usage:", client.usage, "\nwrote", args.out)

if __name__ == "__main__":
    asyncio.run(main())
