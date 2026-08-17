"""ON-POD steering script: add alpha*sigma*unit_dir to the residual stream at layer 46
(output of decoder layer index 45) at ALL positions, measure exact next-token P(A)/P(B).

Inputs: steer_manifest.jsonl, steer_dirs.npz
Settings: dirs {choice, vex, mira, vexdiff} x alpha {-8,-4,4,8} + one alpha=0 baseline.
Output: steer_results.jsonl rows {uid, dir, alpha, p_letter_A}

Run: python pod_steer.py --manifest steer_manifest.jsonl --dirs steer_dirs.npz --out /workspace/steer_results.jsonl
"""
import argparse, json, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HOOK_LAYER_IDX = 45  # output of this decoder layer == hidden_states[46]
BATCH = 8

def get_layers(model):
    for path in ("model.layers", "model.language_model.layers",
                 "language_model.model.layers", "language_model.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise RuntimeError("cannot find decoder layers")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    dirs = np.load(args.dirs)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto", attn_implementation="eager")
    model.eval()
    layers = get_layers(model)
    dev = next(model.parameters()).device

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

    texts = []
    for r in rows:
        msgs = r["messages"]
        if msgs and msgs[0]["role"] == "system":
            sys_txt = msgs[0]["content"]
            rest = msgs[1:]
            rest = [{"role": "user", "content": sys_txt + "\n\n" + rest[0]["content"]}] + rest[1:]
            msgs = rest
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    steer_vec = {"v": None}
    def hook(mod, inp, out):
        if steer_vec["v"] is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + steer_vec["v"],) + out[1:]
        return out + steer_vec["v"]
    layers[HOOK_LAYER_IDX].register_forward_hook(hook)

    if os.environ.get("STEER_SETTINGS"):
        settings = [(n, float(a)) for n, a in
                    (s.split(":") for s in os.environ["STEER_SETTINGS"].split(","))]
    else:
        settings = [("none", 0.0)]
        for name in ("choice", "vex", "mira", "vexdiff"):
            for a in (-8, -4, 4, 8):
                settings.append((name, float(a)))

    fout = open(args.out, "w")
    with torch.no_grad():
        for name, alpha in settings:
            if name == "none":
                steer_vec["v"] = None
            else:
                u = torch.tensor(dirs[name], dtype=torch.bfloat16, device=dev)
                steer_vec["v"] = alpha * float(dirs[f"sigma_{name}"]) * u
            for i0 in range(0, len(rows), BATCH):
                batch = texts[i0:i0 + BATCH]
                enc = tok(batch, return_tensors="pt", padding=True,
                          padding_side="left").to(dev)
                out = model(**enc, use_cache=False)
                logits = out.logits[:, -1, :].float()
                pa = torch.logsumexp(logits[:, A_IDS], dim=-1)
                pb = torch.logsumexp(logits[:, B_IDS], dim=-1)
                p = torch.sigmoid(pa - pb).cpu().numpy()
                for j, r in enumerate(rows[i0:i0 + BATCH]):
                    fout.write(json.dumps({"uid": r["uid"], "dir": name, "alpha": alpha,
                                           "p_letter_A": float(p[j])}) + "\n")
                del out, enc
            fout.flush()
            print(f"done {name} alpha={alpha}", flush=True)
    fout.close()
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
