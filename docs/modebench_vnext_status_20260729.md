# ModeBench vNext development status — 2026-07-29

This is a development ledger, not a paper-results claim. Earlier failed gates
remain final and are not replaced by later interfaces. E70 Stage A has 40
identity-bound jobs covering the four established domains (two arms, five
seeds); the remaining four rows are still development-gated and have no
confirmatory jobs.

## PantryPlan

The original endpoint interface produced 0 verified completions in 4,096
samples; 4,050 were candidate-grammar failures and no output reached the joint
verifier.

The prospective sequential public-action interface was run by A100 repair job
`30185387`. The sole repair after jobs `30185302` and `30185346` was to require
BF16-capable A100 hardware; those superseded RTX 2080 jobs sampled zero model
tokens. Scientific settings were unchanged.

Result: **pass**.

- 64 development prompts, 64 rollouts each;
- 327/4,096 verified completions;
- 34/64 prompts with a verified completion among the first 16 rollouts; and
- 29/64 prompts with at least two verified canonical keys among 64 rollouts.

Receipt:
`var/artifacts/pantry_plan_interactive_05b_viability_v1_r1.json`
(SHA-256
`9bc747b7086948b3055b6c39d7a5e7a9c6ed95c891ea95f385f68cfb82a78900`).

Interpretation: the untouched 0.5B policy has enough terminal verified mass for
a paired online-training smoke without solver demonstrations. Formatting was
the dominant endpoint-interface failure.

A prospectively frozen, immediately trainable six-bit support policy then
passed more strongly: 62/64 prefix-success prompts, 63/64 multimode prompts,
and 1,496/4,096 verified completions. Its independent audit replayed all 4,096
samples exactly and passed every identity and information-boundary check. The
single permitted 32-update plain-Dr.GRPO plumbing smoke is job `30187473`;
dependency job `30187627` will apply the frozen actor/learner parity audit.
Neither is a confirmatory seed or paper result.

## PointMaze

The original endpoint interface produced 0 verified routes. The corrected
closed-loop public-action interface was run by A100 job `30185388`.

Result: **failed the breadth threshold with nonzero signal**.

- 4 development maps, 64 rollouts each;
- 4/256 verified routes, all on `bar7`;
- 1/4 prompts with a verified completion among the first 16 rollouts; and
- 0/4 prompts with at least two verified canonical keys.

Receipt:
`var/artifacts/point_maze_interactive_05b_viability_v1_r1.json`
(SHA-256
`9372ca9657882fa656beef1ce500f680eb729acfd370bb61a41e7877fec5913d`).

The other three maps ran the full 480 simulator steps without success. Their
minimum final goal distances were 4.022, 0.938, and 3.347. This is no longer a
grammar or actuator failure; it is a local-navigation policy deficit. The next
registered rung is a behavior-cloned warm start from the eight train maps only,
shared exactly by MaxEnt and Dr.GRPO. No planner output or shaping enters
online training, development, or evaluation.

The separate one-shot grid development v1 also failed (0/256); all strings had
valid cardinal syntax, but none was a complete legal goal route. Its
prospectively frozen v2 legal-simple-path diagnostic subsequently passed with
21/256 verified routes, prefix success on 4/4 maps, and two route identities on
2/4 maps. V2 masks wall entry and revisits but does not filter or label paths by
goal or route identity. Its corrected independent audit replayed the official
worker result exactly and passed all source, data, and identity checks; the
original audit wrapper failure is preserved as a non-scientific audit defect.
The v2 diagnostic remains separate from the closed-loop result and cannot
replace it.

Train-only warm-start materialization then passed with 8 train maps, both
certified route modes per map, 16 executable episodes, and 644 exact public
state-to-action examples. The corpus contains no dev/eval identifier and no
model sample. Frozen shared-SFT/development-gate job `30185642` is queued on an
A5000.

## AntMaze

The original v5 fresh-map route gate remains failed. The prospective
three-node audit produced 0/216 verified executions with identical failures on
each node, so v5 is not portable.

Maze-blind v6 waypoint-controller job `30185409` completed normally but failed
its frozen open-plane gate: 65/96 successes (0.677), worst-heading success
0.333, unhealthy termination 0.0625, and median successful duration 211 steps.
The required rates were 0.90 overall and 0.75 for every heading. V6 is
ineligible, so no fresh-maze v6 gate and no Ant language-model sample may run.

Prospective v7 job `30187375` is training from the exact v6 weights with a
reset optimizer, all training waypoints fixed to the operational four-unit
distance, cyclic eight-heading balance, and fresh nonoverlapping evaluation
seeds. It remains maze-blind and cannot launch a route gate unless it passes the
unchanged 0.90-overall/0.75-every-heading criteria.

V7 subsequently completed and remains a failed frozen gate: 86/96 successes
(0.8958), minimum heading success 8/12 (0.667), unhealthy termination 2/96,
and median successful duration 169.5 steps. The thresholds were not relaxed.
V8 job `30187810` subsequently passed its unchanged open-plane gate: 91/96
successes (0.9479), minimum heading success 10/12 (0.833), unhealthy
termination 1/96, and median successful duration 147 steps. The separately
frozen v8 route slate used new map fingerprints, seeds `87300..87311`, and
four-command upper/lower fixtures. Job `30188228` failed on the first frozen
fresh-map upper fixture. No map, seed, fixture, or threshold was substituted,
and no admission-map rerun was made. V8 is therefore ineligible for an AntMaze
language-model sample despite passing open-plane locomotion.

The prospectively frozen v9 continuation addresses the observed local
navigation deficit without training on complete routes. It initializes from
the exact v8 weights, resets the PPO optimizer, and trains only balanced
adjacent-free-cell waypoint episodes on the four designated 7x7 training
maps. Its sealed controller gate uses fresh 8x8 local-evaluation maps and
unchanged overall/per-heading safety thresholds plus a per-map threshold.
Configuration validation passed before submission; training/evaluation job
`30192731` is identity-bound and running or queued. No v9 route slate may be
frozen or executed unless that controller gate passes.

While v9 was still training and before its controller outcome, the complete
conditional downstream staircase was frozen. The route gate contains 12 new
9x9 maps, seeds `97300..97311`, and fixed upper/lower four-token programs. Its
three-node follow-up replays the exact admitted rows and source snapshot 216
times rather than regenerating maps. The subsequent development-only 0.5B
gate uses four development maps, 64 samples per map, a first-16 prefix, seed
`97309`, and thresholds of 2/4 prefix-success maps and 1/4 multimode maps.
Configuration suites pass 8/8 for route admission, 5/5 for cross-node replay,
and 4/4 for viability. None of these conditional gates has executed a v9
route or sampled a language model.

V9 job `30192731` then completed its full 5M-step budget and failed the frozen
controller gate. It passed overall success at 87/96 (0.90625), every-map
success at a minimum of 21/24 (0.875), safety at 0 unhealthy terminations,
and median duration at 184 steps. The northeast heading alone failed balance
at 4/12; every other heading was at least 11/12. The eight northeast failures
were 400-step timeouts. No v9 route job or language-model sample launched.

Prospective v10 was frozen after that terminal receipt and before new
training. It initializes from the exact failed-v9 weights, resets the
optimizer, assigns northeast half of a fixed 2M-transition local-edge
schedule, lowers learning rate to `2e-6`, and evaluates all headings on four
new 10x10 maps using fixed boundary/middle/interior edge indices. Pinned
runtime configuration passed (2 static tests, 1 environment-owned test plus
direct runtime assertions), and identity-bound job `30193111` was released.
The still-unexecuted v9 conditional route/cross-node/viability protocols do
not authorize v10. While v10 was still training and before its outcome, a new
conditional v10 staircase was frozen: 12 fresh 11x11 maps with seeds
`107300..107311`, the exact-slate 216-execution three-node replay, and a
development-only 0.5B gate with seed `107310`. Route configuration passes
8/8, cross-node configuration passes 6/6, and viability configuration passes
6/6. None has executed a v10 route or sampled a model; they remain conditional
on the exact controller and preceding-gate receipts.

## ConstructiveCode

Materialization job `30184636` completed all four CodeContests+ and all four
overlay shards and froze four tasks. Dependency job `30185057` then launched
1,600-replay gate job `30185883`, which failed before candidate execution
because a logical v1 exclusion-ledger path was compared to its relocated
content-addressed snapshot path.

The relocation-only r1 amendment preserved the logical path contract, the
whole-tree ledger hash, every per-task exclusion hash, and all executable
criteria. R1 job `30187501` completed all 1,600 replays with zero wrapper versus
released-checker disagreement, timeout, isolation, output-bound, or latency
failure. V2 nevertheless failed its frozen joint-dual-suite criterion: 1283C
had 0.83 TNR on CodeContests-O while passing Plus-5x, and 1294C had 0.89 TNR on
Plus-5x while passing CodeContests-O.

The separately preregistered v3 applies the extension plan's antecedent
task-level overlay/fallback rule and freezes 12 problem IDs: one task per
witness family in each of disjoint train, development, and evaluation splits.
Both suites and all hard execution checks remain mandatory; only empirical
label-rate/multi-key failure may trigger deterministic Plus-5x fallback. CPU
job `30187935` is materializing the frozen sources and will run 4,800 replays
only if all 12 tasks provide 100 fresh Python-3 correct and incorrect programs.
No model has been sampled and no ConstructiveCode result has been claimed.

## Online-training implementation

The trajectory and compute-matching contract is in
`docs/interactive_verified_maxent_training_design.md`. The initial
length-neutral episodic objective is implemented in
`src/oat_drgrpo/interactive_episode_objective.py`; its tests enforce per-prompt
Dr.GRPO centering, verified-advantage addition outside centering, detached
verified advantages, and equal statistical weight for short and long
episodes.

This objective is not yet wired into an actor/learner rollout pipeline.
Therefore no PantryPlan, PointMaze, or AntMaze paired online smoke or Stage-B
main cell is currently executable merely because an environment gate passes.
