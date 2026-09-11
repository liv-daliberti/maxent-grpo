# PointMaze algorithm repair v1: cardinality clarification

**Status: FROZEN WHILE VIABILITY JOB 30204570 WAS RUNNING AND BEFORE ITS
OUTCOME — 2026-07-30**

The admission paragraph in the v1 protocol says “16 unique train maps, eight
held-out maps.” The materializer frozen before model sampling actually
produces eight train maps, four development maps, and four evaluation maps:
16 unique maps total, of which eight are held out from RL training.

This document corrects only that wording. The already admitted row hashes,
families, rotations, programs, seeds, split membership, checker executions,
perturbations, model, sample count, and qualification criteria are unchanged.

