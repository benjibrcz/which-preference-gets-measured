"""ON-POD script v2: memory-safe activation + choice extraction with resume.
Hooks on selected layers (not output_hidden_states); last-token logits only; token-budget
batching; memmap keyed by uid so restarts resume cleanly.
Run: python pod_extract.py --manifest probe_manifest.jsonl --out /workspace/out
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYERS = [16, 26, 36, 46, 56, 62]
TOKEN_BUDGET = 6000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [json.loads(l) for l in open(args.manifest)]
    N = len(rows)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto", attn_implementation="eager")
    model.eval()
    D = (model.config.text_config.hidden_size if hasattr(model.config, "text_config")
         else model.config.hidden_size)

    # locate decoder layers for hooks
    core = model.model
    while not hasattr(core, "layers"):
        core = core.language_model if hasattr(core, "language_model") else core.model
    blocks = core.layers
    captured = {}
    def mk_hook(li):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[li] = h[:, -1, :].detach().float().cpu()
        return hook
    hooks = [blocks[L - 1].register_forward_hook(mk_hook(li)) for li, L in enumerate(LAYERS)]

    def ids_for(s):
        out = set()
        for v in (s, " " + s, "\n" + s):
            t = tok.encode(v, add_special_tokens=False)
            if len(t) == 1:
                out.add(t[0])
        return sorted(out)
    A_IDS, B_IDS = ids_for("A"), ids_for("B")
    print("A ids:", A_IDS, "B ids:", B_IDS, flush=True)

    texts = []
    for r in rows:
        msgs = r["messages"]
        if msgs and msgs[0]["role"] == "system":
            merged = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}]
            msgs = merged + msgs[2:]
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    lens = [len(tok.encode(t)) for t in texts]

    # resume state
    acts_path = os.path.join(args.out, "activations.npy")
    if not os.path.exists(acts_path):
        np.lib.format.open_memmap(acts_path, mode="w+", dtype=np.float16, shape=(N, len(LAYERS), D))
    acts = np.lib.format.open_memmap(acts_path, mode="r+")
    ch_path = os.path.join(args.out, "choices.jsonl")
    done_uids = set()
    if os.path.exists(ch_path):
        done_uids = {json.loads(l)["uid"] for l in open(ch_path)}
    fout = open(ch_path, "a")
    print(f"resuming: {len(done_uids)}/{N} done", flush=True)

    todo = [i for i in range(N) if rows[i]["uid"] not in done_uids]
    todo.sort(key=lambda i: lens[i])
    batches, cur = [], []
    for i in todo:
        trial = cur + [i]
        if cur and lens[trial[-1]] * len(trial) > TOKEN_BUDGET:
            batches.append(cur); cur = [i]
        else:
            cur = trial
    if cur:
        batches.append(cur)

    with torch.no_grad():
        for bi, idxs in enumerate(batches):
            enc = tok([texts[i] for i in idxs], return_tensors="pt", padding=True,
                      padding_side="left").to(model.device)
            try:
                out = model(**enc, logits_to_keep=1)
            except TypeError:
                out = model(**enc, num_logits_to_keep=1)
            logits = out.logits[:, -1, :].float()
            pa = torch.logsumexp(logits[:, A_IDS], dim=-1)
            pb = torch.logsumexp(logits[:, B_IDS], dim=-1)
            p_choose_A = torch.sigmoid(pa - pb).cpu().numpy()
            for j, i in enumerate(idxs):
                for li in range(len(LAYERS)):
                    acts[i, li] = captured[li][j].numpy().astype(np.float16)
                fout.write(json.dumps({"uid": rows[i]["uid"], "p_letter_A": float(p_choose_A[j])}) + "\n")
            fout.flush()
            del out, logits
            if bi % 40 == 0:
                print(f"batch {bi}/{len(batches)} ({sum(len(b) for b in batches[:bi+1])} prompts)", flush=True)
    for h in hooks:
        h.remove()
    json.dump({"layers": LAYERS, "n": N, "model": args.model}, open(os.path.join(args.out, "meta.json"), "w"))
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
