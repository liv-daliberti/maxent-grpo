# E69 verified-route successor protocol

Date frozen: 2026-07-28

## Scientific question

Can a task-reward-first exploration policy improve verified route coverage without
trading away answer quality, and can that improvement transfer across prompts and
across Graph, Countdown, Python, MathIR, and free-form mathematics?

E69 is a successor study, not a retrospective reinterpretation of E66 or E68.
E66/E68 remain the actuator ablation: E68 tests whether a one-proposal
singleton escape increases support when the verifier and route space are exact.
E69 tests whether the useful mechanism survives a stricter, cross-prompt route
identity and a neutral task-first policy.

## Frozen data firewall

- The MATH training population remains exactly source rows 0 through 383 from
  the pinned MATH12K artifact used by E39/E64.
- A new 128-item route-development population is selected only from MATH12K
  source rows 384 through 11,999.
- Development selection is deterministic, unique by normalized problem text,
  and stratified jointly by subject and difficulty level using largest-remainder
  proportional allocation followed by a seeded SHA-256 ordering within each
  stratum.
- Any candidate whose normalized problem overlaps the 384 training problems or
  MATH-500 is ineligible.
- Any candidate whose pinned `qwen_math_route` prompt exceeds 1,024 tokens is
  ineligible.
- The materialized E69 root contains the 384 training rows and the 128
  MATH12K-development rows. It contains no MATH-500 row, answer, prediction, or
  score.
- MATH-500 may be used during development only as a pinned set of normalized
  problem identities for the one-time overlap firewall. Its evaluation results
  remain sealed until the algorithm, hyperparameters, stopping rule, and
  checkpoint-selection rule are frozen.

The materializer is
`ops/route_successor/materialize_e69_math12k_route_dev.py`; its manifest is the
authoritative population record.

## Route identity

Every admitted exploration mode has two identities:

1. **Endpoint identity:** prompt-local, derived only after the ordinary task
   verifier accepts the answer.
2. **Route identity:** derived from a bounded, executable trace whose operations
   and dependencies are independently checked. Surface prose, action labels,
   numeric binding values, and an LLM judge do not define route identity.

For MathIR, the existing exact interpreter is the reference implementation. A
route-family signature is computed from the verified command/dependency
skeleton and an alpha-normalized initial equation. It must be invariant to
prompt-local action relabeling and coefficient-symbol relabeling.

For free-form MATH, the final answer continues to be judged by the established
math verifier. A route signature is admitted only when a separate restricted
trace parses, all referenced inputs are authorized, every transition executes,
and the terminal trace claim agrees with the verified answer. A correct answer
without a valid route trace still receives ordinary task reward but receives no
route-novelty reward.

No Python `eval`, untrusted code execution, prose canonicalization, or
model-based strategy judge is allowed at the admission boundary.

## Objective and policy

The neutral policy is task-reward-first:

- ordinary verified task reward is unchanged;
- invalid or incorrect responses never enter the route bank;
- route novelty is a separate, bounded advantage term;
- counterfactual proposals are eligible only for verified singleton groups;
- proposal PPO rows remain disabled;
- at most one proposal may be admitted per optimizer update;
- proposal generation and its token budget are charged to the experiment's
  compute accounting;
- novelty cannot turn a task-reward failure into a positive task example.

The explorer and learner are reported separately: exploration may add a verified
route to replay, but the policy update is still anchored by ordinary verified
task examples.

## Frozen experimental ladder

### Gate 0: validator contracts

Required before any training:

- all existing verifier tests pass;
- malformed traces fail closed;
- formatting and action-label aliases do not create routes;
- coefficient-symbol relabeling preserves route signatures;
- semantically distinct verified routes remain distinct;
- no trace can introduce an unchecked numeric literal or terminal answer.

### Gate 1: offline coverage

On MathIR train and eval:

- 100% of enumerated admitted programs execute successfully;
- every prompt has exactly the reference finite endpoint support;
- route signatures recur across prompts within every equation family;
- no raw action label or binding value appears in a route signature.

On the sealed MATH12K route-dev split, using a fixed pretraining-model sample
archive:

- report task-verifier acceptance and trace-verifier acceptance separately;
- at least 80% of task-correct sampled responses must have a valid trace before
  route novelty is allowed in MATH training;
- manually inspect a deterministic 50-item accepted-trace sample and report any
  false admission as a hard failure.

Failure at Gate 1 means revise the trace language or abstain on free-form MATH;
it does not authorize inspecting MATH-500 scores.

### Gate 2: compute-matched one-seed screen

Compare the neutral task-first control and exactly one route-aware successor
with seed 43 and six prompt passes on Graph, Countdown, Python, MathIR, and the
sealed MATH12K route-dev split. Match training prompts, sampled tokens, proposal
tokens, optimizer updates, evaluation cadence, and checkpoint choice.

Advance only if:

- no domain loses more than 0.02 absolute greedy verified accuracy;
- no domain loses more than 0.02 absolute pass@8;
- at least three of four executable ModeBench domains improve verified distinct
  support;
- MathIR improves both verified distinct support and pass@8;
- MATH route-dev improves either greedy accuracy or pass@8 without worsening
  the other by more than 0.01;
- all audit and compute-matching checks pass.

There is no hyperparameter sweep after seeing Gate 2 outcomes. A failed gate
returns to mechanism design with a new experiment identifier.

### Gate 3: confirmatory panel

Run the frozen control and successor for seeds 43, 44, and 45, six prompt passes,
on Graph, Countdown, Python, MathIR, and MATH12K route-dev. Report per-seed
curves, paired deltas, bootstrap intervals, verifier health, support growth,
proposal admissions, and charged compute.

Freeze the algorithm, all coefficients, prompt formats, checkpoint-selection
rule, and analysis code before unsealing MATH-500.

### Gate 4: one-time held-out transfer

Evaluate the frozen three-seed checkpoints on MATH-500 once. MATH-500 is the
fifth panel area but must be labeled **held-out MATH-500 transfer**, not as a
ModeBench domain. Report all outcomes regardless of direction. No tuning,
checkpoint reselection, or rerun based on MATH-500 is permitted.

## Primary reporting

For every area report greedy verified accuracy, mean@8, pass@8, verified
distinct support where defined, prompt/token/update accounting, and paired
seed-level uncertainty. The primary success claim requires task-quality
noninferiority in all five areas and a positive route-support result in at least
three executable domains. A result that increases support but misses the
task-quality condition is a mechanism result, not a successful exploration
algorithm.
