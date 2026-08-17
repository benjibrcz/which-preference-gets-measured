"""Assert that key headline estimates reproduce within tolerance, from cached results only.
Exits non-zero if any check fails. Invoked by reproduce_selected_headlines.sh.
"""
import sys, json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
from analysis_review import normative_control, crossfit
from analysis_identity import code_identity

_ok = True
def check(x, target, tol, label):
    global _ok
    good = abs(x - target) <= tol
    _ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {label}: {x:.3f}  (expect {target:.3f} ± {tol})")

# 1. Non-agent normative control (Gemma): matched non-agent text ~ persona description
nc = normative_control("gridA_gemma", "deconfound_gemma", "semprime_gemma")
check(nc["Vex"]["char_C1"][0], 0.844, 0.06, "Gemma/Vex persona-description beta")
check(nc["Vex"]["policy"][0],  0.788, 0.06, "Gemma/Vex non-agent-normative beta")

# 2. Shared-baseline cross-fit: bias is small (headline is not an artifact)
cf = crossfit("gridA_gemma", "deconfound_gemma")
max_bias = max(abs(v[2]) for v in cf.values())
check(max_bias, 0.0, 0.15, "Gemma max |naive - cross-fit| bias")

# 3. Post-exit identity: 100% of valid core responses deny the persona
post = Counter()
for m in ["hyst_gemma", "hyst_llama70b", "hyst_gpt41mini", "hyst_qwen72b"]:
    for line in open(ROOT / "runs" / m / "results.jsonl"):
        r = json.loads(line)
        if r.get("channel") == "identity" and str(r["checkpoint"]).startswith("x"):
            post[code_identity(r["raw"], r["persona"])] += 1
valid = sum(post.values()) - post["invalid"]
check(post["denies_persona"] / valid, 1.0, 0.0, "core post-exit persona-denial fraction")

print("ALL HEADLINE CHECKS PASSED" if _ok else "SOME CHECKS FAILED")
sys.exit(0 if _ok else 1)
