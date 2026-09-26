# E69 Gate 2 R3: MATH-dev evaluation-split contract repair

Date frozen: 2026-07-29, after the original seed-43 MATH-dev GRPO
training attempt reached step 2304 and before either affected MATH-dev arm
produced any Gate 2 evaluation outcome.

## Observed implementation defect

The sealed MATH route-development evaluation dataset is
`var/data/math12k_384_route_dev128_v1/eval`. Its immutable
`dataset_dict.json` names the sole split `math_dev`. The Gate 2 launchers
instead set `OAT_ZERO_TEST_SPLIT=math`.

`ZeroMathLearner` filters the loaded evaluation dictionary by
`args.test_split`. The mismatch therefore selected an empty evaluation
dictionary. `OAT_ZERO_ALLOW_SPARSE_EVAL=1` governs evaluation cadence; it was
not intended to authorize zero matched evaluation datasets. The training
loop still logged evaluation boundaries and checkpoints, so the defect was
only exposed when the frozen Gate 2 auditor correctly reported all seven
MATH-dev evaluation passes missing.

Job 30159729 completed training through step 2304 with exit code 0 and zero
restarts. Job 30160101 was still training when this repair was frozen. Their
training telemetry was visible, but neither run had an
`eval_mode_coverage_draws.jsonl` file and no MATH-dev Gate 2 evaluation
outcome was available. The repair decision is therefore not based on a
greedy, mean@8, pass@8, or distinct@8 value.

## Prospective repair

The complete two-arm MATH-dev cohort is replaced from initialization:

- Dr.GRPO, seed 43; and
- verified-first global endpoint replay, seed 43.

Both replacements use the same model, sealed train/evaluation data, prompt
template, verifier, seed, six prompt passes, optimizer, learning rate,
sampling budget, replay budget, evaluation seed, evaluation cadence, and
terminal-checkpoint rule. The only configured value changed from the
affected attempts is:

`OAT_ZERO_TEST_SPLIT=math` -> `OAT_ZERO_TEST_SPLIT=math_dev`.

The source contract is repaired minimally:

1. `math_verified_answer` accepts the sealed `math_dev` split as well as the
   historical `math` spelling; and
2. online evaluation fails closed if split selection yields zero datasets.

The first change only permits the split name already frozen in the data. The
second prevents another nominally successful training run from silently
omitting every evaluation. Neither change affects rollout generation,
verification, replay, PPO, optimizer updates, or any gate threshold.

The replacements may not resume either affected attempt. The affected
attempts remain immutable provenance, are excluded from the effective Gate 2
cells, and are never averaged with the replacements. Replacing both arms is
required because both inherited the same defective split setting; selecting
only one arm after seeing training telemetry would break the paired screen.

## Outcome and scope firewall

- No MATH-dev Gate 2 evaluation outcome existed when this repair was frozen.
- Training telemetry existed and did not determine the repair scope.
- No Graph, Countdown, Python, or MathIR cell is replaced.
- The temporal mechanism rule and every outcome/compute gate are unchanged.
- MATH-500 remains sealed.
- Gate 3 remains forbidden until the corrected cells are terminal, all seven
  evaluation passes exist for both arms, Python is terminal with positive
  temporal recurrence, and the complete frozen Gate 2 audit passes with zero
  integrity violations.

This is an implementation-contract repair, not an execution-only retry and
not an outcome-informed whitelist.

## Held-submission audit correction

The first atomic submission attempt created held jobs 30172460 and 30172461.
The launcher then rejected Slurm's `TresPerNode=gres/gpu:a100:1` spelling
because its assertion expected a different normalized spelling. Its cleanup
trap cancelled both jobs at zero runtime before release, and no R3 identity
or outcome existed. Their manifest and accounting are retained as rejected
startup provenance. The resubmission changes only that held-record assertion;
the frozen training source and both arm configurations are unchanged.
