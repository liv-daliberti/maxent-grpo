# E49D — Maximal-Certified-Support MATH Haarnoja (0.5B)

**Status: FROZEN BEFORE TRAINING; PREPROCESSING THROUGHPUT, TRANSPORT, SCHEMA, AND FIXED SUPPORT-RECALL AMENDMENTS ONLY — 2026-07-24 11:35 EDT**

## Why this successor exists

E49B established that an unscaffolded 0.5B MATH policy almost never discovers
a second auditable route. E49C then tried to require at least two certified
routes for every problem. Its answer-bound v3 preprocessing correctly exposed
that assumption as unsafe before any E49C training launch:

- complement and inclusion-exclusion for a two-day rain calculation both
  derive the right answer but collapse to the same probability identity;
- repeated doubling, exponential growth, a geometric-sequence term, and
  compound-interest notation collapse to the same fixed operation; and
- forcing another route sometimes produced an invalid geometric argument.

The E49C v3 evidence and canceled jobs remain immutable. E49D changes only the
menu-support assumption: it keeps the largest subset that is actually
certified and permits a singleton when a problem honestly has only one route.
This is the conservative analogue of canonical support in graph coloring and
Countdown—support is discovered, never fabricated.

## Hypothesis

On problems with at least two double-certified menu routes, the current E46
normalized canonical-bank Haarnoja objective will learn broader audited route
coverage than a matched execution-gated Dr.GRPO control, while preserving raw
and execution-gated MATH quality. Singleton-menu problems remain in the exact
same train/eval cohorts and contribute ordinary task learning but no artificial
diversity target.

## Exact data and schedule

- Toy: exactly E49B's frozen 50 hard training rows and 50 hard MATH-500
  evaluation rows.
- Full: exactly the OAT 384-row MATH training cohort and its exact 500-row
  MATH-500 evaluation cohort.
- Both stages use exactly three prompt epochs.
- Both arms use Qwen2.5-0.5B-Instruct, seed 45, group size 16, learning rate
  `2e-7`, one PPO epoch, temperature 1, top-p 1, and one A100 on node302.

The original problem and answer are unchanged. The only policy-facing data
transformation is appending the certified finite action menu and exact response
contract.

## Answer-blind proposals and answer-bound certification

Frozen Qwen2.5-72B first proposes two or three problem-local action sequences.
The generator and route ideator never receive the reference answer. Imported
E49C menus are proposals only and receive no grandfathering.

Each proposal receives two separately prompted, temperature-zero audits:

1. `soundness_execution` independently solves the problem and literally
   executes every proposed sequence against the auditor-only reference answer.
2. `equivalence_attack` tries to collapse every pair to the same decisive
   equation, identity, invariant, construction, or search space under routine
   algebra.

Both use frozen seeds 491711 and 491712. A strategy is eligible only if both
auditors call it sound, assign failure code `none`, and certify that its
actually derived answer matches the reference. An edge between two strategies
exists only if both auditors call the pair distinct and name nonempty,
different route-exclusive decisive operations.

The materializer deterministically selects the maximum clique of individually
eligible strategies in proposal order, prunes unused actions, and renumbers
the retained `A` and `S` IDs. The first materialization allowed one
answer-blind direct proposal and one answer-blind route-ideation rescue
proposal, in addition to any imported proposal that was re-audited. After that
complete preprocessing run retained only 16/100 multi-route menus and the
frozen 20% gate blocked training, the support-recall amendment added exactly
one supplemental answer-blind rescue for every certified singleton. The same
direct-plus-two-rescue procedure is fixed in advance for the conditional full
cohort. A certified multi-route subset is accepted immediately. If none is
found, the first certified singleton is retained. A row with no certified
route blocks the stage.

The durable record contains both raw auditor response IDs, roles, seeds,
finish states, proposal menu, and retained subset. The audit-only preflight
recomputes the maximum certified subset from those bytes and requires its hash
to equal the embedded policy menu. At least 20% of each stage's rows must have
two or more retained routes or that stage does not launch.

Auditor derivations and reference answers are never passed into proposal
feedback or policy data.

The operations-only reduction from two redundant proposals of each kind to
one, and the use of all eight frozen judge sequence slots, are recorded in
`e49d_preprocessing_throughput_amendment_20260724.md`.

The fail-closed action-ID/uniqueness schema tightening is recorded in
`e49d_proposal_schema_amendment_20260724.md`.

The deterministic collapse of identical proposed action combos is recorded
in `e49d_duplicate_combo_collapse_amendment_20260724.md`.

The concise structured-audit bounds that prevent transport truncation are
recorded in `e49d_audit_transport_bound_amendment_20260724.md`.

The seeded temperature-0.2 answer-blind proposal sampling and shorter proposal
caps are recorded in `e49d_proposal_sampling_amendment_20260724.md`; both
certification auditors remain temperature-zero.

The removal of vLLM's unsupported guided-schema `uniqueItems` annotation,
while retaining the identical fail-closed local uniqueness check, is recorded
in `e49d_guided_schema_compatibility_amendment_20260724.md`.

The relocation of audit text-length checks from a crashing xgrammar compiler
to the deterministic durable-evidence validator is recorded in
`e49d_xgrammar_audit_schema_amendment_20260724.md`.

The endpoint-wide switch from the faulty xgrammar compiler to vLLM's
supported guidance structured-output backend is recorded in
`e49d_guidance_backend_amendment_20260724.md`.

Terminal-job recovery through explicit `sacct` states when Slurm has already
purged a job from `squeue` is recorded in
`e49d_slurm_accounting_amendment_20260724.md`.

Isolation from a concurrent E50 edit of the shared live source tree is
recorded in `e49d_source_snapshot_isolation_amendment_20260724.md`; E49D uses
the already frozen, hash-verified source snapshot from its configuration
preflight.

The preprocessing failure at 16/100 multi-route menus and the single fixed
supplemental recall attempt for each certified singleton are recorded in
`e49d_support_recall_amendment_20260724.md`. Transport failures do not consume
that attempt; they remain durable evidence and may be retried only until one
request completes. The 20% launch gate is unchanged.

## Exact execution and reward contract

Every response must begin at character zero with exactly:

```text
<strategy_id>Sj</strategy_id>
<action_combo>Aa>Ab>...</action_combo>
```

The pair must occur in that prompt's retained menu. For an answer-positive
completion, two independently permuted frozen Qwen2.5-72B runtime checks must
both verify mathematical integrity and material execution of every declared
action in order. Missing, unknown, reordered, name-dropped, skipped,
route-switched, malformed, truncated, ambiguous, or disagreeing responses
receive no canonical key.

In both arms, an answer-positive completion keeps its task reward only after
this exact admission. Rejected task reward is set to zero before ordinary
Dr.GRPO centering. The canonical outcome is the audited prompt-local action
combo, so formatting paraphrases merge and a declaration alone cannot create
novelty.

## Matched arms

1. **Matched execution-gated Dr.GRPO:** identical menus, validator, runtime
   audits, task-reward gate, sampling, data, and compute; passive strategy
   tracking only.
2. **E49D E46 Haarnoja:** initial/minimum alpha 0.10, maximum alpha 0.50,
   normalized entropy target 0.80, log-alpha learning rate 0.003, EMA decay
   0.90, pseudocount 1, surprisal clip 5, and first audited discovery bonus
   beta 0.50.

No token-entropy, uncertainty, semantic-Shannon, DIAYN, XDR, curriculum, or
other exploration controller is active.

For singleton support, normalized entropy is ineligible and contributes no
dual update. This is already the frozen E46 controller behavior; only prompts
whose discovered support is at least two influence the normalized Haarnoja
target.

## Evaluation and toy advancement

Raw greedy and fixed-seed pass@8 are recorded at epoch 0, 1, 2, and 3. The
terminal stored traces are re-audited under the exact execution contract for:

- execution-gated greedy and pass@8;
- mean audited correct strategies per prompt;
- all-prompt fraction with at least two audited correct strategies; and
- the same coverage conditioned on prompts whose retained menu has at least
  two strategies.

The full stage may launch only if:

1. all 100 toy rows have a recomputable certified menu and at least 20 have
   multi-route menus;
2. both matched jobs complete 150 steps and all four frozen eval points;
3. all reward-accounting, controller-direction, alpha-bound, and monotone-bank
   invariants pass;
4. both arms have identical epoch-0 raw metrics;
5. treatment terminal raw pass@8 and execution-gated pass@8 are each at least
   the control;
6. treatment raw greedy is within 5 percentage points of the control; and
7. treatment audited support-at-least-two coverage on multi-route-eligible
   prompts is at least control plus 10 percentage points, with treatment mean
   audited correct strategy count at least control plus 0.20.

If the toy fails, the full stage does not launch. Any later change requires a
new protocol/run stamp; failed artifacts remain immutable.

## Full success criterion

The exact 384/500 three-epoch stage succeeds only if all invariants hold,
treatment raw and execution-gated pass@8 are each at least control, treatment
raw greedy is within 5 percentage points, and treatment improves
multi-route-eligible audited support-at-least-two coverage by at least 5
percentage points.

## Frozen implementation identities

Recorded after tests and before the first E49D menu request:

```text
fcfcb1fb403362378a43cb2b2c0aed739f8db1a0725aa048746be4b77e974a59  var/artifacts/source_snapshots/e49d_maximal_support_math_27386bb32552105c2a073a1dfa3e3b3b31839eb78225bec1dbc05daebbdfabcd/src/oat_drgrpo/math_strategy_menu.py
f9855ce3842d30bcfad58f6e1f8c0e151bf8e97b3fc158f7ab2f10bba549bf32  var/artifacts/source_snapshots/e49d_maximal_support_math_27386bb32552105c2a073a1dfa3e3b3b31839eb78225bec1dbc05daebbdfabcd/src/oat_drgrpo/math_strategy_canonicalizer.py
4e984c3cba54dbec50984b3c571c53adbf99ce5606ed8e42efd1f76bdfbbece7  var/artifacts/source_snapshots/e49d_maximal_support_math_27386bb32552105c2a073a1dfa3e3b3b31839eb78225bec1dbc05daebbdfabcd/src/oat_drgrpo/online_canonical_bank.py
d5fed3bbcea20807e72015ae88a2ef00de0af0e5ec5ca6049cb04c07d147ce0e  var/artifacts/source_snapshots/e49d_maximal_support_math_27386bb32552105c2a073a1dfa3e3b3b31839eb78225bec1dbc05daebbdfabcd/src/oat_drgrpo/online_canonical_controller.py
58976324034691fe001ded0c9a84ddb8c76a1f416b2de17106da5c0e99b39a01  var/artifacts/source_snapshots/e49d_maximal_support_math_27386bb32552105c2a073a1dfa3e3b3b31839eb78225bec1dbc05daebbdfabcd/src/oat_drgrpo/learner/grpo.py
58d98b326a07316a2d873b3d9e20ebdb04f44ff519303735ce0b2f10873f2261  var/artifacts/source_snapshots/e49d_maximal_support_math_27386bb32552105c2a073a1dfa3e3b3b31839eb78225bec1dbc05daebbdfabcd/src/oat_drgrpo/args.py
34ed30ece2bbd1cc69e6ebb2b774c359e701a3778cd278fea6cc34993e4f38e3  ops/math_strategy_calibration/materialize_e49d_strategy_menu_data.py
dc6caf4b435748185aebbe5d04ce494170e7e2da8d2b5a12f9f8dfb223eb2a7c  ops/math_strategy_calibration/launch_e49d_menu_job.sh
bd78a593282f684a1c104474df767caef533aaaa522445aa6ce109a07f148cc6  ops/slurm/e49d_materialize_menus_node302.slurm
3668e354361fbb4f4112e453acfde45bdcf25806dc455877c5d83c5448763bb6  ops/slurm/e47_qwen72_node105.slurm
53c0d9471b8830c95657c2bb357d638613823516c9b9a49cd1f96a86f436e516  ops/exp_scaling/launch_e49d_maximal_support_math_05b.sh
adecbad697c9ac6bba397ddf41c36f05f6c969f7338ade474680f0a4cdfe62a6  ops/exp_scaling/analyze_e49d_maximal_support_math.py
85d952097cc650058db2b5cdd5e293531fe70bb0302df33319d961ee666b4a91  ops/exp_scaling/audit_e49d_eval_contract.py
ee20f58f935c2e712533daa1c6d4fb93d4ccb6300b19e6783b002450a4ff12c4  ops/exp_scaling/audit_e49d_prompt_lengths.py
d5726091ad3992def7e70aaa7ef0ba967893d73fa2e7f75cf094d9cc5ea76258  ops/exp_scaling/plot_e49d_maximal_support_math.py
e8e333fb1df1025d15234682dd2999ebc147c62fde4e4e1d2cc130242c45fa7d  ops/exp_scaling/monitor_e49d_maximal_support_math.sh
a80402bd0414049d2442a0bf8bb508482e92d14e8faafb33e0d2d04756120e8d  ops/exp_scaling/babysit_e49d_maximal_support_math.sh
6f292340ecc38eae7ca6be4b9fca0a6f26d707449c3d89e1232e12a53eda7761  ops/exp_scaling/launch_e49d_eval_audit_job.sh
7584a68b465d68720799319740ca15b2069157cc0c9a86b168b44d89ffd09752  ops/slurm/e49d_eval_audit_node302.slurm
f9b8b5ad32a1b1b05d1eb4c054536f12ab2326a91568cfc5c593189d7459ebf0  tests/test_e49c_menu_audit.py
a5bb0609fb1e5b3bae25f8df62eae071a9352c7cc200ca58af72b86cdf84d453  tests/test_e49_math_strategy_contract.py
38a19634f4ecfdd5c246e3ad0b77e77d7ee0d50e9cdeae25ecd5d7e1b2424035  tests/test_math_strategy_menu.py
ef3b80cfcf213e4b85c2ac2b02e38f6251b8b24865d65c9bd0afcf240f866c2c  paper/preregistration/e49d_preprocessing_throughput_amendment_20260724.md
d8bbd48d62df906740c87ce17bc8cd6f4513859ce3297685ae9a65f1db03d0be  paper/preregistration/e49d_proposal_schema_amendment_20260724.md
dd1171e2943364b4974872356ed18a051e0fde2d9fb1e79e20611b02eb368c2d  paper/preregistration/e49d_duplicate_combo_collapse_amendment_20260724.md
cb3d73ee1ea3bc8a24f4d799f2ce71b8cb7d000ae92f836faf227114fc53cb21  paper/preregistration/e49d_audit_transport_bound_amendment_20260724.md
1a919bd1f2fd7c119695c44eba0461202c1d3dd0108806fa2a57454d280ae199  paper/preregistration/e49d_proposal_sampling_amendment_20260724.md
956d3554d360933e01f845cb3e1a0a2fb0c3bf08c535469e83465102f48016e9  paper/preregistration/e49d_guided_schema_compatibility_amendment_20260724.md
c32e273f752fa788bc30aec3c1c577d8decade2ce1d02273aceb2c959decb169  paper/preregistration/e49d_xgrammar_audit_schema_amendment_20260724.md
5aadb2a733840c9caa93eb391ce1f48a7d49e13aca6acad6eada0de23b8ec470  paper/preregistration/e49d_guidance_backend_amendment_20260724.md
f473cbeb268abf8202803d112742603fab1e3cf6069a7a72d0c30ff46ae265e4  paper/preregistration/e49d_slurm_accounting_amendment_20260724.md
64aaaa159ab56fa80b433cb97ae7e84314030faa12bf12533d169a09b912f228  paper/preregistration/e49d_source_snapshot_isolation_amendment_20260724.md
c93493840c448ff7a7603fcbefe1cca70eca497003c475e4092d9a414fbd8031  paper/preregistration/e49d_support_recall_amendment_20260724.md
```
