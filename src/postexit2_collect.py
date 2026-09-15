"""Loop 7 collection: post-exit controls on Gemma-3-27B (PREREG P27-P33).

Cells = history x checkpoint x channel x bank. Histories (data/postexit2_generated.json):
  auth_{Vex,Lazlo}      original authored 4-exchange dialogues (contain preference talk)
  pref_{Vex,Lazlo}      preference-free authored dialogues (identical user turns)
  gen{A,B,C}_{Vex,Lazlo} model-generated in-persona replies to three user-turn sets
  neutral_auth / neutral_pref / neutral_gen{A,B,C}   matched Neutral controls
  inst_{short,blunt,spell} + neutral_inst            content-matched instruction controls (E2)
Checkpoints (suffixes are authored unless marked gen):
  t0        no history (shared)
  t4        history only (direction estimation)
  x2        + authored exit + authored 2-turn neutral suffix
  noexit2   + matched continuation + authored suffix                       (E0)
  x2g{s}    + model exit reply s + model-generated suffix, s=0..3           (E1, unconditional)
  noexit2g  + model continuation reply + model-generated suffix            (E0 generated)
  rg{s}     + model exit 0 + model reset reply s + authored neutral[:1]     (E1)
  x8, r_inst  as in loop 2 (authored + pref + their controls only)
  inst: t4 (rule + 3 Qs), active2 (+ continuation + suffix), withdrawn2 (+ withdrawal + suffix)
Channels: revealed, stated_self (all); P1/P2/P3N paraphrases (auth/pref/neutral, new bank, t0/t4/x2/noexit2);
identity + nonleading probes (k=8) at every checkpoint; spelling probe on inst_spell cells; exec check at x2.

Usage: python src/postexit2_collect.py --out runs/postexit2_gemma [--dry-run]
"""
import argparse, asyncio, json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Client, parse_choice
from prompts import REVEALED, STATED_SELF
from hysteresis import probe_pairs
from bank import pair_texts

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
M = json.loads((DATA / "postexit2_materials.json").read_text())
G = json.loads((DATA / "postexit2_generated.json").read_text())
H = json.loads((DATA / "hysteresis_dialogues.json").read_text())
BANK2 = json.loads((DATA / "bank_v2.json").read_text())["pairs"]

def msgs(exchanges):
    out = []
    for u, a in exchanges:
        out += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return out

def x(u, a):
    return [(u, a)]

AUTH_SUFFIX = [(q["user"], q["assistant"]) for q in M["neutral_suffix"]]
NEUTRAL8 = [(e["user"], e["assistant"]) for e in (H["neutral"] + H["neutral_ext"])][:8]
EXIT = (M["exit"]["user"], M["exit"]["assistant"])
NOEXIT = (M["noexit"]["user"], M["noexit"]["assistant"])
RESET = (M["reset"]["user"], M["reset"]["assistant"])

def persona_of(hid):
    for p in ("Vex", "Lazlo"):
        if hid.endswith("_" + p):
            return p
    return None

def control_of(hid):
    if hid.startswith("auth_"): return "neutral_auth"
    if hid.startswith("pref_"): return "neutral_pref"
    if hid.startswith("gen"): return "neutral_gen" + hid[3]
    if hid.startswith("inst_"): return "neutral_inst"
    return None

def build_stacks():
    """-> {history_id: {checkpoint: exchanges}}"""
    stacks = {}
    for hid, h in G["histories"].items():
        base = [tuple(e) for e in h["exchanges"]]
        st = {"t4": base,
              "x2": base + [EXIT] + AUTH_SUFFIX,
              "noexit2": base + [NOEXIT] + AUTH_SUFFIX,
              "noexit2g": base + [(M["noexit"]["user"], G["noexit_reply"][hid]["reply"])]
                          + [tuple(e) for e in G["noexit_reply"][hid]["suffix"]]}
        for s, e in enumerate(G["exits"][hid]):
            st[f"x2g{s}"] = base + [(M["exit"]["user"], e)] + [tuple(q) for q in G["suffix"][hid][f"exit{s}"]]
        for s, r in enumerate(G["resets"][hid]):
            st[f"rg{s}"] = base + [(M["exit"]["user"], G["exits"][hid][0]), (M["reset"]["user"], r)] + AUTH_SUFFIX[:1]
        if hid.startswith(("auth_", "pref_", "neutral_auth", "neutral_pref")):
            st["x8"] = base + [EXIT] + NEUTRAL8
            st["r_inst"] = base + [EXIT, RESET] + AUTH_SUFFIX[:1]
        stacks[hid] = st
    # instruction controls (E2)
    qs = M["pref_user_turns"][:3]
    for name, spec in M["inst"].items():
        base = [(spec["rule_user"], spec["rule_assistant"])] + list(zip(qs, spec["replies"]))
        stacks[f"inst_{name}"] = {"t4": base,
                                  "active2": base + [NOEXIT] + AUTH_SUFFIX,
                                  "withdrawn2": base + [(spec["withdraw_user"], spec["withdraw_assistant"])] + AUTH_SUFFIX}
    nb = list(zip(qs, M["neutral_pref"][:3]))
    wd = M["inst"]["short"]
    stacks["neutral_inst"] = {"t4": nb, "active2": nb + [NOEXIT] + AUTH_SUFFIX,
                              "withdrawn2": nb + [(wd["withdraw_user"], wd["withdraw_assistant"])] + AUTH_SUFFIX}
    return stacks

def banks():
    old_pairs, tasks = probe_pairs()
    out = [("new", p["id"], p["a"], p["b"]) for p in BANK2]
    for p in old_pairs:
        a, b = pair_texts(p, tasks)
        out.append(("old", p["pair_id"], a, b))
    return out

CHANNELS = {"revealed": REVEALED, "stated_self": STATED_SELF}
PARA = M["stated_paraphrases"]

def build_jobs(k):
    stacks = build_stacks()
    pairs = banks()
    jobs = []
    def add_choice(hid, cp, exchanges, bank_filter, channels):
        hist = msgs(exchanges)
        for bank, pid, a, b in pairs:
            if bank not in bank_filter:
                continue
            for order in (0, 1):
                xa, xb = (a, b) if order == 0 else (b, a)
                for ch, tmpl in channels.items():
                    for s in range(k):
                        jobs.append({"history": hid, "persona": persona_of(hid), "control": control_of(hid),
                                     "checkpoint": cp, "channel": ch, "bank": bank, "pair_id": pid,
                                     "order": order, "sample_idx": s, "max_tokens": 6,
                                     "messages": hist + [{"role": "user", "content": tmpl.format(a=xa, b=xb)}]})
    def add_probes(hid, cp, exchanges):
        hist = msgs(exchanges)
        for ch in ("identity", "nonleading"):
            for s in range(2 * k):
                jobs.append({"history": hid, "persona": persona_of(hid), "control": control_of(hid),
                             "checkpoint": cp, "channel": ch, "bank": None, "pair_id": None, "order": 0,
                             "sample_idx": s, "max_tokens": 60,
                             "messages": hist + [{"role": "user", "content": M["probes"][ch]}]})
    def add_spelling(hid, cp, exchanges):
        for s in range(2 * k):
            jobs.append({"history": hid, "persona": None, "control": control_of(hid), "checkpoint": cp,
                         "channel": "spelling", "bank": None, "pair_id": None, "order": 0, "sample_idx": s,
                         "max_tokens": 160,
                         "messages": msgs(exchanges) + [{"role": "user", "content": M["inst"]["spell"]["probe"]}]})
    # t0 (shared)
    add_choice("none", "t0", [], {"new", "old"}, CHANNELS)
    add_choice("none", "t0", [], {"new"}, PARA)
    add_probes("none", "t0", [])
    add_spelling("none", "t0", [])
    for hid, st in stacks.items():
        for cp, exchanges in st.items():
            if hid == "inst_spell" or (hid == "neutral_inst"):
                add_spelling(hid, cp, exchanges)
            if hid == "inst_spell":
                continue  # spelling control measured by the spelling probe only
            bank_filter = {"new", "old"} if hid.startswith(("auth_", "pref_", "neutral_auth", "neutral_pref")) \
                          and cp in ("t4", "x2", "noexit2", "x8", "r_inst") else {"new"}
            add_choice(hid, cp, exchanges, bank_filter, CHANNELS)
            if hid.startswith(("auth_", "pref_", "neutral_auth", "neutral_pref")) and cp in ("t4", "x2", "noexit2"):
                add_choice(hid, cp, exchanges, {"new"}, PARA)
            add_probes(hid, cp, exchanges)
    return jobs, stacks

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    jobs, stacks = build_jobs(args.k)
    from collections import Counter
    print(f"jobs={len(jobs)}  histories={len(stacks)}")
    print("by channel:", dict(Counter(j["channel"] for j in jobs)))
    print("by history:", dict(Counter(j["history"] for j in jobs)))
    if args.dry_run:
        return
    client = Client(args.provider, args.model, concurrency=args.concurrency)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fout = open(out / "results.jsonl", "w")
    t0 = time.time(); done = 0

    async def run_job(j):
        nonlocal done
        raw = await client.one(j["messages"], sample_idx=j["sample_idx"], max_tokens=j["max_tokens"])
        if j["channel"] in ("identity", "nonleading", "spelling"):
            val, flag = None, "ok" if raw else "error"
        elif j["channel"] == "P3N":
            t = (raw or "").strip()
            m = re.match(r"^[^A-Za-z0-9]*(?:task\s+|option\s+)?([ABNabn])(?![A-Za-z0-9])", t)
            val, flag = (m.group(1).upper(), "ok") if m else parse_choice(raw)
        else:
            val, flag = parse_choice(raw)
        row = {k: v for k, v in j.items() if k != "messages"}
        row.update({"model": args.model, "raw": raw, "value": val, "flag": flag})
        fout.write(json.dumps(row) + "\n")
        done += 1
        if done % 2000 == 0:
            print(f"  {done}/{len(jobs)} {time.time()-t0:.0f}s usage={client.usage}", flush=True)

    await asyncio.gather(*[run_job(j) for j in jobs])
    fout.close()
    (out / "stacks.json").write_text(json.dumps(stacks, indent=1, ensure_ascii=False))
    print("DONE", time.time() - t0, "s", client.usage)

if __name__ == "__main__":
    asyncio.run(main())
