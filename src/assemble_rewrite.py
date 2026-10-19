"""Stitch GPT-rewritten parts back into a draft and check that no number was lost or changed.
Usage: python src/assemble_rewrite.py <out.md> <original.md> <part1.md> [<part2.md> ...] [--tail <kept.md>]
Prints the multiset difference of numeric tokens (original vs rewritten) so changed/dropped numbers can be checked by hand."""
import sys, re
from collections import Counter
from pathlib import Path
args = sys.argv[1:]
tail = None
if "--tail" in args:
    i = args.index("--tail"); tail = Path(args[i + 1]).read_text(); args = args[:i] + args[i + 2:]
out, orig, parts = Path(args[0]), Path(args[1]).read_text(), [Path(a).read_text() for a in args[2:]]
new = "\n\n".join(p.strip() for p in parts) + ("\n\n" + tail if tail else "") + "\n"
out.write_text(new)
num = lambda t: Counter(re.findall(r"(?<![\w.])[−-]?\d+(?:[.,]\d+)?%?(?![\w])", t.replace("−", "-")))
a, b = num(orig), num(new)
lost = {k: v - b.get(k, 0) for k, v in a.items() if v > b.get(k, 0)}
added = {k: v - a.get(k, 0) for k, v in b.items() if v > a.get(k, 0)}
print(f"wrote {out}: {len(new.split())} words (original {len(orig.split())})")
print("numeric tokens fewer in rewrite (count lost):", dict(sorted(lost.items(), key=lambda x: -x[1])[:60]))
print("numeric tokens new in rewrite:", dict(sorted(added.items(), key=lambda x: -x[1])[:60]))
