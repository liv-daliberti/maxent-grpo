# Proof cleanup — 2026-09-14

`check_paper_current_contract.py` pins the manuscript's formal blocks against
an archived reference so the mathematics cannot drift accidentally. This
cleanup changes proof *presentation* deliberately, so the reference is re-pinned
to `main.tex` as archived here, and this record states exactly what changed.

Previous reference: `audits/narrative_reorganization_20260911/before/paper/main.tex`
New reference:      `audits/proof_cleanup_20260914/main.tex` (sha256 `c6c19d0467b2064958726980dc3b9b9c3132a7a0e81439cd0ade933536b8b6be`)

## What changed

13 of the 18 original blocks are **byte-identical**. Block count goes 18 -> 20
because one overloaded lemma was split in two.

| Block | Change | Mathematical content |
|---|---|---|
| `lem:maxrl-mean` | Cites `Assumption 1` instead of restating the model in prose | unchanged |
| `lem:sampled-score-bound` | Same | unchanged |
| `thm:grpo-collapse` | Same, plus the closing commentary sentence moved to a `remark` | unchanged. The dropped clause "start from finite logits with $0<P(0)<1$" is now premise (A5) of the assumption block, so the hypothesis is preserved, not weakened |
| `lem:replay-gradient-availability` | Split: keeps only the fresh-group starvation bound | unchanged |
| `lem:replay-ascent` (new) | The replay-ascent formula and its limiting norm, previously the second half of the lemma above | unchanged; the proof is the second half of the original proof, verbatim |

No hypothesis, conclusion, constant, or proof step was altered. The
accompanying `Assumption 1` (A1)-(A5) collects premises that were previously
restated in four different phrasings, and each result now names it.

## Reproducing the comparison

```
python - <<'PY'
import re, pathlib
pat = re.compile(r"\\begin\{(lemma|theorem|corollary|proposition|proof)\}.*?\\end\{\1\}", re.DOTALL)
b = lambda p: {" ".join(m.group(0).split()) for m in pat.finditer(pathlib.Path(p).read_text())}
old = b("paper/audits/narrative_reorganization_20260911/before/paper/main.tex")
new = b("paper/main.tex")
print("unchanged:", len(old & new), "of", len(old))
PY
```
