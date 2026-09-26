# PointMaze balanced short warm start v6

**Status: FROZEN AFTER THE TERMINAL V5 DEVELOPMENT RECEIPT AND BEFORE V6
MODEL UPDATE OR V6 MODEL SAMPLING — 2026-07-30**

V5 repaired the inherited orientation bias but its 12-epoch SFT solved 243 of
256 development trajectories, above the unchanged 0.50 trainability ceiling.
All four orientations had prefix success; three were 64/64. V6 therefore
changes only SFT duration.

Use the exact same train-only, 2/2/2/2 orientation-balanced source, 16 route
episodes, 644 public-state decisions, Qwen2.5-0.5B-Instruct base snapshot,
public Markov prompt, and nine-token action support as V5. Train for exactly
three epochs (69 AdamW optimizer steps) with seed `76621`, batch size 4,
gradient accumulation 7, learning rate `2e-5`, seven warmup steps then linear
decay, weight decay `0.01`, gradient cap `1.0`, BF16, and context cap 1536.

Evaluate on the same four development-only v3 rows with 64 trajectories per
prompt, prefix 16, seed `76622`, temperature/top-p 1.0, and no evaluation
rows. Require exactly 256 attempts, at least two prefix-success prompts, at
least two multimode prompts, and aggregate verified rate in the unchanged
inclusive interval [0.02, 0.50]. Failure stops before online training. Pass
authorizes a separately frozen five-seed comparison on untouched evaluation
rows.
