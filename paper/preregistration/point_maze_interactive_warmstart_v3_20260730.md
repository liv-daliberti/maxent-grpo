# PointMaze public-Markov-state warm start v3

**Status: FROZEN AFTER THE TERMINAL V2 RECEIPT AND BEFORE V3 DATA MATERIALIZATION, MODEL UPDATE, OR V3 DEVELOPMENT SAMPLING — 2026-07-30**

## Antecedent and state-definition diagnosis

V1 fit 644 train-only rows at 0.90994 restricted-label accuracy but sampled
0/256 verified development routes while choosing N/S on 24,328/24,576
decisions. V2 removed action-history and endpoint-program prompt shortcuts,
fit at 0.80124 accuracy, and again sampled 0/256 verified routes; it chose N
on 20,072 decisions and S on 4,445. Both outcomes remain terminal failures.

The executable worker audit found that Gymnasium PointMaze supplies the public
point state as `(x,y,vx,vy)`, and the admitted scripted PD controller uses
`vx,vy`, but the language-policy worker exposed only position and goal. Because
force pulses act on a point mass with momentum, position alone is not Markov;
v1's recent-action text was only an unreliable velocity proxy. V3 corrects
that state omission. It does not expose a planner, route, verifier signal, or
privileged environment state.

## Frozen train-only data and public interface

Replay exactly the same two certified routes for each of the same eight train
maps (16 episodes, 644 decisions), and require the same executable canonical
identities. The worker must expose `velocity_xy=observation[2:4]` at reset and
after every five-step action pulse. The v3 prompt contains only:

- the printed public maze through the S/G grid;
- position, velocity, and goal, each rounded to three decimals;
- the public coordinate convention; and
- the complete fixed nine-action option table.

It omits the boxed endpoint-program instruction, recent action history, and
remaining horizon. The data process may load only train rows and train-labeled
certifications; no dev/evaluation row or v1/v2 trajectory enters SFT.

## Frozen shared SFT

- exact Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- all parameters in BF16, seed `75203`;
- 12 epochs, batch size 4, gradient accumulation 7;
- exactly 276 AdamW optimizer steps;
- learning rate `2e-5`, 28 warmup steps then linear decay;
- weight decay `0.01`, gradient norm cap `1.0`, context cap 1536; and
- cross-entropy renormalized over the same nine single-token action labels.

If admitted, the exact checkpoint is shared by both online arms.

## Frozen post-SFT development gate

Use the unchanged `point_maze_modebench_v1/dev/multi_answer` four-map slate:
64 rollouts per map, first 16 as prefix, temperature/top-p 1.0, seed `75105`,
at most 96 decisions and five simulator steps per decision. Require at least
two prompts with a verified prefix route and at least one prompt with two
verified canonical identities among 64. No planner output, certified program,
intermediate verifier feedback, or evaluation row is exposed. A pass
authorizes only a separately frozen paired online-training smoke; a failure
stops v3.
