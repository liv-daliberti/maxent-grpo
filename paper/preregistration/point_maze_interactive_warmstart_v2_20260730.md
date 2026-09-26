# PointMaze compact-public-state warm start v2

**Status: FROZEN AFTER THE TERMINAL V1 DEVELOPMENT RECEIPT AND BEFORE V2 DATA MATERIALIZATION, MODEL UPDATE, OR V2 DEVELOPMENT SAMPLING — 2026-07-30**

## Antecedent

The v1 shared warm start fit its 644 train-only state/action rows with final
restricted-label accuracy 0.90994, then failed the unchanged four-map
development gate with 0/256 verified routes. Across 24,576 online decisions,
the sampled policy selected N or S 24,328 times (0.99015) and almost never
selected the other public actions. This is a terminal v1 result. No evaluation
row has been loaded or sampled.

V2 addresses the observed state-conditioning failure. It does not reinterpret
the v1 receipt, alter the endpoint verifier, relax the viability gate, or
select a different development map.

## Frozen train-only data transformation

Replay the same two independently certified routes on each of the same eight
`point_maze_modebench_v1/train` maps. Require all 16 routes to reproduce their
existing executable canonical identities. Before every action, render the
compact public-state v2 prompt:

- retain the printed public maze through its S/G grid;
- retain the rounded current `(x,y)` position, rounded goal `(x,y)`, coordinate
  convention, and the complete fixed nine-action option table;
- remove the endpoint-only instruction asking for a boxed full program;
- remove recent action history and remaining-horizon text; and
- retain exactly one supervised target option letter per simulator state.

Coordinates are deterministically rounded to three decimals with values below
0.0005 normalized to zero. The transformation may load only the train split
and its train-labeled certification records. It may not load a dev/evaluation
row, a v1 development trajectory, or any verifier/online reward during SFT.

## Frozen shared SFT

- base: exact Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- all parameters updated in BF16;
- seed `75202`, eight epochs, batch size 4, gradient accumulation 7;
- exactly 184 AdamW optimizer steps;
- learning rate `1e-5`, 18 linear-warmup steps then linear decay;
- weight decay `0.01`, gradient norm cap `1.0`, context cap 1536; and
- cross-entropy renormalized over the same nine single-token option labels.

The checkpoint is a shared initializer: if admitted, both online Dr.GRPO and
verified-MaxEnt arms must start from its exact hash.

## Frozen post-SFT development gate

Run the compact v2 policy on exactly the unchanged
`point_maze_modebench_v1/dev/multi_answer` slate:

- four maps, 64 rollouts per map, first 16 as the prefix;
- temperature/top-p `1.0`, seed `75104`;
- at most 96 one-token decisions, each held for five simulator steps;
- at least two prompts with a verified route in the first 16; and
- at least one prompt with two verified canonical route identities among 64.

The action mask remains the fixed nine-token public alphabet. No certified
program, planner output, intermediate verifier feedback, or training target is
shown online. A pass authorizes only a separately frozen, one-seed matched
online-training smoke. A failure stops v2. The evaluation split remains
untouched in either case.
