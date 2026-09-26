# PointMaze orientation-balanced warm start v5

**Status: FROZEN AFTER THE TERMINAL V4 DEVELOPMENT RECEIPT AND BEFORE V5
DATA MATERIALIZATION, MODEL UPDATE, OR V5 MODEL SAMPLING — 2026-07-30**

## Diagnosis

The public-Markov-state v3 checkpoint was SFT-trained on eight maps, but all
eight were rotations 0 or 2. The later v3 repair made each train/dev/evaluation
split orientation-balanced while continuing to start from that horizontally
trained checkpoint. Its development receipt then solved the two horizontal
rows 128/128 and the two vertical rows 11/128. Enlarging the horizontal
obstacles in v4 did not repair the asymmetry: the two horizontal rows remained
128/128 while the two vertical rows were 14/128. V4 therefore stopped at
142/256, above the unchanged 0.50 trainability ceiling.

V5 repairs the inherited checkpoint rather than changing the online objective,
the trainability threshold, or another geometry after seeing model outcomes.

## Frozen train-only SFT slate

Use the same four source families and the same two routes per family as the v3
warm start. Replace the duplicated horizontal rotations with:

- `bar7`: rotations 0 and 1;
- `block9`: rotations 2 and 3;
- `bar9`: rotations 0 and 1; and
- `asymmetric_block9`: rotations 2 and 3.

This retains exactly eight maps, 16 route episodes, 644 public-state decisions,
and two examples at every rotation. Every route must replay successfully
against the official process and produce two distinct canonical identities per
map. Only train rows and train certifications may be loaded.

The base model, public Markov prompt, nine-token action support, and SFT
hyperparameters remain those of v3: Qwen2.5-0.5B-Instruct snapshot
`7ae557604adf67be50417f59c2c2f167def9a775`, seed `76601`, 12 epochs,
batch size 4, gradient accumulation 7, exactly 276 AdamW steps, learning rate
`2e-5`, 28 warmup steps, weight decay `0.01`, gradient cap `1.0`, BF16, and
context cap 1536.

## Frozen development qualification

Evaluate the new shared checkpoint on the already admitted
`point_maze_algorithm_repair_v3/dev/multi_answer` four-map,
one-per-orientation slate. Use 64 rollouts per map, prefix 16, seed `76602`,
temperature/top-p 1.0, the unchanged public Markov state, and no evaluation
rows. Require:

- 256 exact attempts;
- at least two prompts with prefix success;
- at least two prompts with two verified canonical modes; and
- an aggregate verified rate in the unchanged inclusive interval [0.02, 0.50].

A failure stops V5. A pass authorizes freezing a compute-matched five-seed
online pair from this exact shared checkpoint. It does not authorize changing
the development slate or loading evaluation rows.
