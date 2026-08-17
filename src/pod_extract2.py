"""ON-POD script v2 (OOM-proof): run gemma-3-27b-it over probe_manifest.jsonl.

Fixes vs v1 (which OOM'd at 7496/7580 and lost all activations):
- activations written incrementally to a memmap-backed .npy (row = manifest uid)
- rows processed longest-first with a token budget per batch (long hysteresis
  stacks get batch size 1-2 instead of 8)
- use_cache=False (no KV cache allocation), expandable_segments allocator
- OOM -> empty_cache + split batch in half, down to batch 1
- resume: uids already in choices.jsonl are skipped

Run: python pod_extract2.py --manifest probe_manifest.jsonl --out /workspace/out2
"""
import argparse, json, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYERS = [16, 26, 36, 46, 56, 62]
TOKEN_BUDGET = 4096   # max batch_size * max_seq_len per forward
MAX_BATCH = 8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [json.loads(l) for l in open(args.manifest)]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto", attn_implementation="eager")
    model.eval()

    def ids_for(s):
        out = []
        for v in (s, " " + s, "\n" + s):
            t = tok.encode(v, add_special_tokens=False)
            if len(t) == 1:
                out.append(t[0])
            elif len(t) >= 1:
                out.append(t[-1] if tok.decode([t[-1]]).strip() == s else t[0])
        return sorted(set(out))
    A_IDS, B_IDS = ids_for("A"), ids_for("B")
    print("A ids:", A_IDS, "B ids:", B_IDS, flush=True)

    texts = []
    for r in rows:
        msgs = r["messages"]
        if msgs and msgs[0]["role"] == "system":  # gemma: fold system into first user turn
            sys_txt = msgs[0]["content"]
            rest = msgs[1:]
            assert rest and rest[0]["role"] == "user"
            rest = [{"role": "user", "content": sys_txt + "\n\n" + rest[0]["content"]}] + rest[1:]
            msgs = rest
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    N = len(rows)
    D = (model.config.text_config.hidden_size if hasattr(model.config, "text_config")
         else model.config.hidden_size)
    acts_path = os.path.join(args.out, "activations.npy")
    if os.path.exists(acts_path):
        acts = np.lib.format.open_memmap(acts_path, mode="r+")
        assert acts.shape == (N, len(LAYERS), D)
    else:
        acts = np.lib.format.open_memmap(acts_path, mode="w+", dtype=np.float16,
                                         shape=(N, len(LAYERS), D))

    choices_path = os.path.join(args.out, "choices.jsonl")
    done = set()
    if os.path.exists(choices_path):
        done = {json.loads(l)["uid"] for l in open(choices_path)}
        print(f"resuming: {len(done)} uids already done", flush=True)
    fout = open(choices_path, "a")

    lengths = [len(tok.encode(t)) for t in texts]
    todo = sorted((u for u in range(N) if u not in done), key=lambda u: -lengths[u])

    # greedy pack: longest-first, batch limited by token budget
    batches, cur = [], []
    for u in todo:
        width = lengths[cur[0]] if cur else lengths[u]
        if cur and (len(cur) >= MAX_BATCH or (len(cur) + 1) * width > TOKEN_BUDGET):
            batches.append(cur); cur = []
        cur.append(u)
    if cur:
        batches.append(cur)
    print(f"{len(todo)} prompts in {len(batches)} batches", flush=True)

    def forward(uids):
        enc = tok([texts[u] for u in uids], return_tensors="pt",
                  padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        last = enc["input_ids"].shape[1] - 1
        for li, L in enumerate(LAYERS):
            h = out.hidden_states[L][:, last, :].float().cpu().numpy().astype(np.float16)
            for j, u in enumerate(uids):
                acts[u, li] = h[j]
        logits = out.logits[:, last, :].float()
        pa = torch.logsumexp(logits[:, A_IDS], dim=-1)
        pb = torch.logsumexp(logits[:, B_IDS], dim=-1)
        p_choose_A = torch.sigmoid(pa - pb).cpu().numpy()
        for j, u in enumerate(uids):
            fout.write(json.dumps({"uid": u, "p_letter_A": float(p_choose_A[j])}) + "\n")
        del out, enc

    def run(uids):
        try:
            forward(uids)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(uids) == 1:
                raise
            mid = len(uids) // 2
            print(f"OOM at batch {len(uids)} (len {lengths[uids[0]]}); splitting", flush=True)
            run(uids[:mid]); run(uids[mid:])

    for bi, uids in enumerate(batches):
        run(uids)
        if bi % 50 == 0:
            fout.flush(); acts.flush()
            print(f"batch {bi}/{len(batches)}", flush=True)
    fout.close(); acts.flush()
    json.dump({"layers": LAYERS, "n": N, "model": args.model},
              open(os.path.join(args.out, "meta.json"), "w"))
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
