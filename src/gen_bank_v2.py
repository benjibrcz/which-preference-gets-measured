"""E6: persona-blind, task-disjoint task-pair bank (loop 7, Sep 2026).

Generates candidate pairs with gpt-4.1 (four samples of 60) from a prompt that mentions no
persona, then audits each candidate. Drop rules (content words = lower-cased, stop-listed, lightly
stemmed, length >= 4; a curated list of generic task words is also excluded):
  R1 dialogue-disjoint: shares ANY content word with any dialogue turn in
     data/hysteresis_dialogues.json or data/postexit2_materials.json;
  R2 signature/aversive-disjoint: shares ANY content word with a sig_* or avn* task;
  R3 task-disjoint: shares >= 2 content words with any single task in data/tasks.json;
  R4 intra-bank: shares >= 2 content words with an already-kept pair (near-duplicate topics).
Keeps the first 40 survivors. Persists the prompt and the audit log next to the bank.

Writes data/bank_v2.json, data/bank_v2_generation_prompt.txt, data/bank_v2_audit.json.
Usage: python src/gen_bank_v2.py
"""
import asyncio, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Client

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

PROMPT = """You are designing test materials for a study of how AI assistants choose between tasks. Produce a JSON object with exactly one field, "pairs": a list of 60 task pairs, each {"id": "v01".."v60", "a": "...", "b": "..."}.

Requirements for every task:
- One sentence, imperative, benign, and fully self-contained: it must be doable in a single reply with no attachment or prior context (so no "this article" or "the list below"; instead put the needed content in the sentence or make it generic, e.g. "Write a limerick about a lost umbrella" or "Convert 250 grams of flour to cups and show the calculation").
- An ordinary request a person might make of an assistant.

Requirements for variety. Across the 60 pairs, tasks should vary along many independent dimensions, and the two tasks within a pair should differ on at least two of them while being roughly equally reasonable requests:
- topic domain (science, cooking, travel, sport, history, money, technology, arts, nature, language, home, health, games, geography)
- output form (list, paragraph, table, poem, dialogue, short code, step-by-step plan, calculation, explanation, letter)
- expected length (one line up to several paragraphs)
- required tone (formal, playful, blunt, gentle, neutral, humorous, serious)
- audience (child, expert, colleague, stranger, close friend, customer)
- emotional content (none, celebratory, consoling, encouraging, apologetic)
- effort profile (quick answer, careful multi-step work, meticulous checking, open-ended brainstorming)
- creativity vs precision (invent something vs get something exactly right)
- social register (individual vs group; personal vs professional)

Exclude entirely: anything about AI, chatbots, assistants, characters, personas, roleplay, or the assistant itself; anything emotionally heavy (death, illness, grief), ethically questionable, or insulting; and these topics: workplace credit or coworkers, birthday gifts for children, toasts or speeches for promotions, meal prep or meal plans, meeting minutes, cover letters or resumes or job applications, siblings or parents, bread going stale, fonts, headphones, autumn leaves, boiling eggs, RSVP etiquette, aeroplane contrails, the brain's capacity, the colour of the sky, cats and tables, the etymology of 'sarcasm', remembering names, boiling pasta, coffee and health, wine stains, mountains in Europe, falling asleep, onions, bicycles.

Return only the JSON object."""

STOP = set("""a an the and or of to in on for with by at from as is are be was were it its this that these those
your you my me we our i he she they them his her their which what who how when where why do does did done
into onto over under about after before between during without within than then so if not no yes up down out
off one two three four five six seven eight nine ten first second third short brief simple quick small large
long new old good bad best better well very more most much many some any each every all both either
write draft compose create make give tell explain describe list suggest summarize summarise plan design
help name find identify pick choose provide show say answer reply ask compare check count convert
friend work sentence learn story step poem someone email item message through time note difference speak
keep cook week home night paragraph clean table basic direction year late start letter region form morning
meet language children made read book thing people person point line word question idea way part example
number rule popular common important different main real full little right left back next last again""".split())

def words(t):
    ws = re.findall(r"[a-z]+", t.lower())
    out = set()
    for w in ws:
        if len(w) < 4 or w in STOP:
            continue
        w = re.sub(r"(ing|ies|ied|ed|es|s|ly|er|est|tion|ment)$", "", w)
        if len(w) >= 4:
            out.add(w)
    return out

def dialogue_text():
    texts = []
    hd = json.loads((DATA / "hysteresis_dialogues.json").read_text())
    for k, v in hd.items():
        if isinstance(v, list):
            for e in v:
                if isinstance(e, dict):
                    texts += [e.get("user", ""), e.get("assistant", "")]
        elif isinstance(v, dict):
            texts += [v.get("user", ""), v.get("assistant", "")]
        elif isinstance(v, str):
            texts.append(v)
    m = json.loads((DATA / "postexit2_materials.json").read_text())
    def walk(x):
        if isinstance(x, str):
            texts.append(x)
        elif isinstance(x, dict):
            for vv in x.values():
                walk(vv)
        elif isinstance(x, list):
            for vv in x:
                walk(vv)
    walk({k: v for k, v in m.items() if k != "stated_paraphrases"})
    return texts

async def main():
    client = Client("openai", "gpt-4.1", concurrency=1, temperature=0.7, max_tokens=6000)
    cand = []
    for si in (0, 1, 2, 3):
        raw = await client.one([{"role": "user", "content": PROMPT}], sample_idx=si, max_tokens=6000)
        raw = raw.strip().strip("`")
        raw = raw[raw.index("{"):raw.rindex("}") + 1]
        for pr in json.loads(raw)["pairs"]:
            pr["id"] = f"s{si}_{pr['id']}"
            cand.append(pr)
    print(f"generated {len(cand)} candidate pairs (four samples); usage {client.usage}")

    tj = json.loads((DATA / "tasks.json").read_text())["tasks"]
    task_w = [(t["text"], words(t["text"])) for t in tj]
    sigavn_w = set().union(*[words(t["text"]) for t in tj if t["id"].startswith(("sig_", "avn"))])
    dlg_w = set().union(*[words(t) for t in dialogue_text()])
    kept, audit = [], []
    for pr in cand:
        wab = words(pr["a"]) | words(pr["b"])
        r1 = sorted(wab & dlg_w)
        r2 = sorted(wab & sigavn_w)
        r3 = [{"overlap": sorted(wab & wt), "task": t[:80]} for t, wt in task_w if len(wab & wt) >= 2]
        r4 = [k["id"] for k in kept if len(wab & (words(k["a"]) | words(k["b"]))) >= 2]  # intra-bank near-duplicate
        drop = bool(r1 or r2 or r3 or r4)
        audit.append({"id": pr["id"], "a": pr["a"], "b": pr["b"], "drop": drop, "R1_dialogue": r1,
                      "R2_sig_avn": r2, "R3_task": r3[:3], "R4_intra_bank": r4})
        if not drop:
            kept.append(pr)
    print(f"{len(kept)} pairs survive the audit (need 40); dropped: R1={sum(1 for a in audit if a['R1_dialogue'])} "
          f"R2={sum(1 for a in audit if a['R2_sig_avn'])} R3={sum(1 for a in audit if a['R3_task'])} "
          f"R4={sum(1 for a in audit if a['R4_intra_bank'])}")
    bank = {"_comment": "Persona-blind, task-disjoint bank generated by gpt-4.1 (temperature 0.7, four samples of 60) "
                        "from data/bank_v2_generation_prompt.txt; candidates were dropped if they shared any content "
                        "word with a dialogue turn (R1) or a signature/aversive task (R2), or two content words with "
                        "a single original task (R3) or an already-kept pair (R4); see data/bank_v2_audit.json.",
            "pairs": [dict(id=f"v{i+1:02d}", a=p["a"], b=p["b"], src_id=p["id"]) for i, p in enumerate(kept[:40])]}
    (DATA / "bank_v2.json").write_text(json.dumps(bank, indent=1, ensure_ascii=False))
    (DATA / "bank_v2_generation_prompt.txt").write_text(PROMPT + "\n\n[Generated by gpt-4.1, temperature 0.7, four samples, 15 Sep 2026. Audit rules R1-R3 in src/gen_bank_v2.py.]\n")
    (DATA / "bank_v2_audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False))
    for p in bank["pairs"]:
        print(f"  {p['id']}: {p['a'][:70]} | {p['b'][:70]}")

if __name__ == "__main__":
    asyncio.run(main())
