# E49E — Trace-Executed Canonical-Bank MATH Haarnoja (0.5B)

**Status: FROZEN BEFORE ANY E49E JUDGE REQUEST OR TRAINING LAUNCH — 2026-07-24**

This is the sole canonical E49E protocol. The earlier
`e49e_cross_proposal_action_trace_math_haarnoja_05b.md` draft was superseded
before any E49E request or policy launch.

## Why E49D cannot train

E49D's first 100-row materialization passed its programmed double-audit, but
retained only 16 multi-route menus and therefore failed the frozen 20% launch
gate. A fixed supplemental recall was then run without launching training.
Manual integrity spot-checks falsified the sufficiency of the E49D audit:

- a claimed CRT route determined only a residue class and then explicitly
  imported actions absent from its declared combo;
- an indistinguishable-boxes route used ordinary stars and bars for
  distinguishable boxes, while one auditor asserted the false arithmetic
  `C(6,1)=6/5=3`;
- other retained examples used the space diagonal as an inscribed sphere's
  diameter, treated a non-arithmetic sequence as symmetric, used one component
  of a cross product as its magnitude, or applied Binet's formula modulo 4
  without a valid ring argument.

No E49D training job was submitted. E49D remains a preprocessing failure and
its evidence remains immutable.

## Hypothesis

Hard MATH can recover the same mechanism that worked for Countdown and graph
coloring only when its canonical object is mechanically executed. On prompts
with at least two independently trace-certified routes, the current E46
normalized canonical-bank Haarnoja objective should increase audited route
coverage relative to a matched execution-gated Dr.GRPO control without
reducing raw or execution-gated answer quality.

## Exact cohorts and schedule

- Toy: exactly E49B's frozen 50 hard training rows and 50 hard MATH-500
  evaluation rows.
- Full: exactly the OAT 384-row MATH training cohort and exact 500-row
  MATH-500 evaluation cohort.
- Exactly three prompt epochs in both stages.
- Qwen2.5-0.5B-Instruct, seed 45, group size 16, learning rate `2e-7`, one PPO
  epoch, temperature 1, top-p 1, and one A100 per arm.

The original problem and reference answer never change.

## Cross-proposal bank construction

E49E generates no extra toy route merely to pass a support threshold. It
recovers routes from the already completed, answer-blind E49D proposals.
For the conditional full stage, the same fixed E49D direct-plus-two-rescue
proposal procedure runs once per row before E49E certification; it remains
preprocessing only and cannot launch E49D training.

For each proposal, E49E recomputes the E49D individual soundness evidence and
takes only routes in that deterministic retained subset. It rejects any route
whose plan or included action text explicitly references an action ID outside
its declared combo. Exact duplicate single-route menus collapse.

The bounded candidate bank contains at most three routes and twelve actions.
Among feasible subsets it deterministically maximizes, in order:

1. route count;
2. number of distinct source proposals;
3. fewer total actions; and
4. earlier durable proposal order.

This fixes the E49D error of searching for a clique only inside each proposal
instead of canonicalizing the union of plausible solutions for problem `x`.

Before any E49E judge request, a no-network preflight must recompute the bank
from the frozen append-only input for every row. It must observe exactly 100
toy rows (or 884 full rows), no row without a structurally closed candidate,
and write every row ID, candidate count, and candidate-menu hash.

## Action-by-action certification

Each candidate receives two independent temperature-zero, answer-bound
Qwen2.5-72B audits:

1. `literal_action_executor`, seed 492111; and
2. `adversarial_action_checker`, seed 492112.

Each audit must return one execution object for every action ID in the exact
declared order, including the concrete calculation/proof operation, its
output fact, and a local validity decision. It must additionally certify that
the route uses only declared actions, is self-contained without another
strategy, derives the reference answer, and is globally sound. The locally
returned answer is independently checked by the frozen `math_verify` task
validator. Missing, reordered, malformed, vague, invalid, or answer-mismatched
traces veto the candidate.

Completed malformed structured output consumes that fixed audit and fails
closed. Only a transport request that never returns may retry. Every completed
request, including a malformed one, is atomically cached by deterministic
row/menu/role/seed identity, so a later materialization retry cannot resample
it.

## Cross-bank equivalence attack

If two or more candidates survive soundness, the whole surviving bank receives
two independent temperature-zero equivalence attacks:

1. `trace_equivalence_attack_a`, seed 492121; and
2. `trace_equivalence_attack_b`, seed 492122.

Both attacks see the action definitions and the four independent execution
traces. A pair edge exists only if both call the routes self-contained and
distinct, both say the distinction is not reducible by routine algebra, and
both name different route-exclusive decisive operations actually present in
the traces. Any equivalent, ambiguous, malformed, or missing decision removes
the edge. The materializer exposes only the deterministic maximum clique;
singletons remain allowed.

At least 20% of all rows and at least 20% of evaluation rows must retain two
routes. Otherwise the stage does not train.

## Exact policy execution contract

Every answer-positive policy response must begin at character zero with its
exact strategy ID and action combo. It must then contain:

```text
<action_trace>
<action_step id="A1">
actual mathematics executing A1
</action_step>
...
</action_trace>
Therefore \boxed{answer}.
```

The machine parser requires exactly one nonempty block per declared action in
the declared order, with no missing, extra, duplicate, or reordered ID. Two
independently permuted temperature-zero Qwen72 runtime checks then compare the
mathematics inside every block to that action's definition and require the
blocks to connect and derive the boxed answer without importing another
route. Failure produces no canonical key and sets task reward to zero in both
arms.

The canonical outcome is the prompt-local, pre-certified action combo.
Formatting paraphrases therefore merge, while declarations or tags without
execution cannot create novelty.

## Calibration and false-new gate

Before toy training, every retained toy multi-route pair is manually audited
from blinded problem, actions, and execution traces. Any false-new pair is
removed and the maximum clique is recomputed; the complete decision ledger is
frozen. Training requires:

- zero accepted cross-action references;
- zero accepted mathematically invalid routes;
- zero accepted equivalent/paraphrased pairs; and
- all deliberately injected duplicate, reordered, name-drop, hidden-import,
  and invalid-step controls to be rejected.

The toy artifact additionally freezes five natural known-invalid route
controls from the E49D failure and three natural known-equivalent pairs. Every
invalid control must receive both completed soundness audits and fail at least
one. Every equivalent-pair control must have both routes pass soundness, must
receive both completed pair audits, and must not receive a novelty-authorizing
`distinct` decision from both. These gates test both false acceptance and
false-new behavior on the actual finite-menu domain.

All 100 toy rows must have a recomputable trace-certified menu, at least
20/100 overall and at least 10/50 evaluation rows must retain two or more
routes, and every rendered prompt must be at most 2048 tokens. Any failed
condition freezes that artifact as a failed calibration and blocks training.

Before full training, every retained full multi-route pair receives the same
blinded ledger audit. This is intentionally stricter than estimating a
false-new rate from a small sample.

## Frozen implementation identity

Before the first judge request, the launcher copies the exact E49E
materializer, its E49D evidence interpreter, and all imported `src` Python
modules into a content-addressed source snapshot. It also freezes the complete
E49D input JSONL and Qwen72 endpoint record. A durable identity record contains
the snapshot tree hash, input and endpoint hashes, source-data tree hashes,
no-network preflight hash, protocol hash, launcher hash, and Slurm-script hash,
with `new_judge_requests_at_freeze = 0`. The Slurm job runs the snapshotted
materializer and rejects any identity mismatch. Live worktree changes
therefore cannot alter an in-flight or resumed calibration.

One pre-control E49E launch, job `30074100`, was cancelled after 63 seconds
when the missing natural-control gate was detected. It produced seven
completed `literal_action_executor` decisions and no row record. The exact
request function, prompt, seeds, candidate-bank construction, and cache
identity are byte-for-byte unchanged by the control-only successor edits.
Those seven decisions are therefore inherited and never resampled. The final
identity records `inherited_completed_request_count = 7`,
`new_judge_requests_at_freeze = 0`, and hashes the inherited cache tree.

## Matched arms

1. **Matched trace-gated Dr.GRPO:** identical prompts, action traces, two
   runtime audits, task-reward gate, sampling, compute, and passive route
   accounting.
2. **E49E E46 Haarnoja:** initial/minimum alpha 0.10, maximum alpha 0.50,
   normalized entropy target 0.80, log-alpha learning rate 0.003, EMA decay
   0.90, pseudocount 1, surprisal clip 5, and first-discovery bonus beta 0.50.

No other exploration controller is active. Singleton prompts contribute
ordinary task learning but no normalized-entropy dual observation.

## Toy advancement and full success

Raw greedy, raw fixed-seed pass@8, trace-execution-gated greedy/pass@8,
audited correct strategy count, eligible support-at-least-two coverage, alpha,
normalized entropy, bank growth, reward accounting, and judge failure counts
are recorded at epochs 0, 1, 2, and 3.

Full materialization may begin only if both toy jobs complete exactly 150
steps, all four evaluation points exist, epoch-0 metrics match, all
fail-closed accounting/controller invariants hold, and treatment:

- has raw and execution-gated pass@8 at least control;
- has raw greedy within five percentage points of control;
- improves eligible audited support-at-least-two by at least ten percentage
  points; and
- improves eligible mean audited correct strategies by at least 0.20.

The full 384/500 stage completes exactly 1152 steps. It succeeds under the
same quality/invariant gates and requires at least a five-point eligible
support-at-least-two improvement.

Any failed gate blocks advancement. A later change requires a new version;
failed artifacts remain immutable.

## Frozen preprocessing and execution identities

Recorded after the focused tests and before the first E49E trace request:

```text
c93472a8df408202172799d2e5408fc9cd279e94c6975d024731e22ae6ef97a0  src/oat_drgrpo/math_strategy_menu.py
00ac91293c5cb8d56117650cb3135985368b4b18cc98db482c5b53c5f0739c8c  src/oat_drgrpo/math_strategy_canonicalizer.py
8d39d6b89fb58be52f9b15261a302ec4adbf815d15b9322400f7d7471b6b25c6  src/oat_drgrpo/math_grader.py
34ed30ece2bbd1cc69e6ebb2b774c359e701a3778cd278fea6cc34993e4f38e3  ops/math_strategy_calibration/materialize_e49d_strategy_menu_data.py
999767353fcece08b8165a816ed8b3b3ef5da2d1dc64eea742cca50a0680fd37  ops/math_strategy_calibration/materialize_e49e_trace_bank_data.py
d886e7442281554a587b6567a55e390dd66f7a0bd010424280b5d57d7d7f0832  ops/math_strategy_calibration/e49e_known_invalid_controls_toy.json
50131fb803e183280297fe76f675202413a215d252955867a9ce3cfe5ffa828c  ops/math_strategy_calibration/e49e_known_equivalent_controls_toy.json
5b31c368debb1bdabb4cb2de4ed57f82a9feffa28f19122f70e5a66ab6316309  ops/math_strategy_calibration/launch_e49e_trace_bank_job.sh
3aaf6dd52191d6b8b9f1600de278f290fb305895fe95ff54fadeb9f102aa1fb5  ops/slurm/e49e_materialize_trace_banks_node302.slurm
902e1726cd78b3084fc51d219bbcc62bc3f68ec5858e9021c0480b678e5c81bb  tests/test_e49e_trace_bank.py
75046eb9e4fccd0cd4cf411a7532a7fae6110fa3fbfed2fd3e1580343340f5fd  tests/test_math_strategy_menu.py
0279c07646fba30e3c1037c7e91043aaf74da9ef28029828287d17537e67af02  tests/test_math_strategy_canonicalizer.py
a60a288b728020f85c9fd426f40945b6ec7b3da681edeb440234249383e686e7  tests/test_math_strategy_reward_gate.py
```
