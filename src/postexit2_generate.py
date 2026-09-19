"""Loop 7 generation phase: model-generated persona entries, exit/reset replies, neutral replies.

Everything generated here is persisted to data/postexit2_generated.json so the histories used by
postexit2_collect.py are inspectable and fixed. Model: Gemma-3-27B (OpenRouter), temperature 1.0.

  entries   : for each persona (Vex, Lazlo) and user-turn set (A, B, C) the model plays the persona
              under the B1 system prompt (desc_2p + "Stay in character"), replying turn by turn
              (each reply conditioned on the previous generated replies). The system prompt is
              used ONLY for generation; the collected history contains no system prompt.
  neutral   : the model's own replies to the same user turns with no system prompt (matched
              Neutral controls for the generated histories).
  exits     : for each entry history (authored + preference-free + generated), k=4 sampled
              replies to the exit request; each becomes its own sub-branch.
  resets    : k=4 sampled replies to the reset request after exit sample 0.
  suffix    : the model's own replies to the two neutral-suffix questions after each exit
              sub-branch (sequential), and after the no-exit continuation for generated histories.

Usage: python src/postexit2_generate.py [--model google/gemma-3-27b-it] [--k 4]
"""
import argparse, asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Client
from bank import load_personas

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
M = json.loads((DATA / "postexit2_materials.json").read_text())
H = json.loads((DATA / "hysteresis_dialogues.json").read_text())
PERSONAS = {p["name"]: p for p in load_personas()}

def msgs(exchanges):
    out = []
    for u, a in exchanges:
        out += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return out

def authored_histories():
    """Entry histories with authored assistant turns: auth_* (original) and pref_* (preference-free)."""
    hist = {}
    for name in ("Vex", "Lazlo"):
        hist[f"auth_{name}"] = [(e["user"], e["assistant"]) for e in H[name][:4]]
        hist[f"pref_{name}"] = list(zip(M["pref_user_turns"], M["pref_dialogues"][name]))
    hist["neutral_auth"] = [(e["user"], e["assistant"]) for e in (H["neutral"] + H["neutral_ext"])[:4]]
    hist["neutral_pref"] = list(zip(M["pref_user_turns"], M["neutral_pref"]))
    return hist

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", default=str(DATA / "postexit2_generated.json"),
                    help="output path (default is the preregistered Gemma file; use another path for smoke tests)")
    args = ap.parse_args()
    client = Client(args.provider, args.model, concurrency=16, temperature=1.0, max_tokens=450)
    gen = {"model": args.model, "histories": {}, "exits": {}, "resets": {}, "suffix": {}, "noexit_reply": {}}

    async def reply(history_msgs, user, sample_idx=0, system=None, max_tokens=450):
        m = ([{"role": "system", "content": system}] if system else []) + history_msgs + \
            [{"role": "user", "content": user}]
        out = await client.one(m, sample_idx=sample_idx, max_tokens=max_tokens)
        return (out or "").strip()

    # 1. entry histories: authored + preference-free (copied for the record) + generated (parallel over sets)
    for k, v in authored_histories().items():
        gen["histories"][k] = {"source": "authored", "exchanges": v}
    async def gen_history(hid, users, system, source):
        ex = []
        for u in users:
            a = await reply(msgs(ex), u, system=system)
            ex.append((u, a))
        gen["histories"][hid] = {"source": source, "exchanges": ex}
    tasks = []
    for name in ("Vex", "Lazlo"):
        system = PERSONAS[name]["desc_2p"] + "\nStay in character in everything you do."
        for set_id, users in M["gen_user_sets"].items():
            tasks.append(gen_history(f"gen{set_id}_{name}", users, system, "model_in_persona_system_prompt"))
    for set_id, users in M["gen_user_sets"].items():
        tasks.append(gen_history(f"neutral_gen{set_id}", users, None, "model_no_system_prompt"))
    await asyncio.gather(*tasks)
    print(f"histories: {len(gen['histories'])}", flush=True)

    # 2. per history (parallel): exit replies (k samples), reset replies after exit 0, generated suffixes
    async def branch(hid, h):
        base = msgs(h["exchanges"])
        exits = list(await asyncio.gather(*[reply(base, M["exit"]["user"], sample_idx=s) for s in range(args.k)]))
        gen["exits"][hid] = exits
        ex0 = base + msgs([(M["exit"]["user"], exits[0])])
        gen["resets"][hid] = list(await asyncio.gather(*[reply(ex0, M["reset"]["user"], sample_idx=s) for s in range(args.k)]))
        async def suffix_after(cur):
            suf = []
            for q in M["neutral_suffix"]:
                a = await reply(cur, q["user"])
                suf.append((q["user"], a)); cur = cur + msgs([(q["user"], a)])
            return suf
        sufs = await asyncio.gather(*[suffix_after(base + msgs([(M["exit"]["user"], e)])) for e in exits])
        gen["suffix"][hid] = {f"exit{s}": suf for s, suf in enumerate(sufs)}
        ne = await reply(base, M["noexit"]["user"])
        gen["noexit_reply"][hid] = {"reply": ne, "suffix": await suffix_after(base + msgs([(M["noexit"]["user"], ne)]))}
        print(f"  {hid}: exits={len(exits)} resets={len(gen['resets'][hid])}", flush=True)
    await asyncio.gather(*[branch(hid, h) for hid, h in list(gen["histories"].items())])

    Path(args.out).write_text(json.dumps(gen, indent=1, ensure_ascii=False))
    print("usage:", client.usage)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    asyncio.run(main())
