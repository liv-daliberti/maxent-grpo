# E109 and E105 amendment: repaired Python ReplayDr.GRPO comparators

Frozen on 2026-08-17 before E105 or E109 submission and before any E104/E106
post-update outcome was inspected. PointMaze and every interactive environment
remain excluded.

## Problem corrected

The first E105 Python amendment bound all treatment cells to the E106 snapshot
that accepts the model-native boxed `\lambda n:` spelling, but left comparator
pairing on the historical E78/E79/E80-R1 ReplayDr.GRPO runs. The Python parser
is shared by the semantic estimator and the verified replay bank. Therefore a
Python treatment-minus-historical-ReplayDr contrast would mix two changes:
the v6 semantic advantage and repaired ReplayDr admission. This is a design
confound, not evidence about either method.

This amendment supersedes the earlier statement that comparator pairing was
unchanged, only for the 15 Python treatment cells described below.

## Repaired comparator cohort

E109 contains ReplayDr.GRPO only for Python Factors at all three registered
model scales and all five paired seeds: Qwen2.5-0.5B seeds 43--47, Falcon3-1B
seeds 55--59, and Qwen2.5-3B seeds 70--74, for exactly 15 cells. Each cell
inherits its scale's original E78, E79, or E80-R1 replay configuration,
dataset, seed, optimizer, schedule, checkpoint interval, compute budget, and
hardware class. Its only change from that historical replay cell is the
content-addressed E106 runtime
`e106_python_lambda_b853595e3b158046`. Semantic Shannon coefficient and every
semantic advantage flag are exactly zero; verified replay remains live at its
registered coefficient 0.1.

E109 may be submitted and released only after the complete outcome-blind
E104+E106 mechanism gate passes. For Qwen-3B cells, it follows the already
registered E105 placement partition: any Python seed moved to the A6000 pool
uses that same pool for both E109 and E105; all other seeds retain A100.

## E105 pairing amendment

The 15 E105 Python treatment cells pair only to the corresponding E109 cell.
The 60 non-Python treatment cells retain their historical ReplayDr.GRPO
comparators because the LaTeX-lambda normalization is unreachable for their
verifiers. The registered E105 endpoints, AUCs, five seeds, 3,072-update
horizon, breadth criterion, and correctness criterion do not change. E109 is
an estimand repair: it removes the parser change from the Python paired effect.

The E105 launcher must fail closed if the released E109 ledger is absent,
contains anything other than the 15 Python ReplayDr cells, names another
snapshot, or reports any semantic objective as active.
