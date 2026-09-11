# E68 recovery and five-domain successor plan

Date: 2026-07-28

Status: research planning document, not a preregistration and not authorization
to submit jobs.

## Executive conclusion

The current evidence supports three conclusions.

1. E58 already improves verified exploration on four executable ModeBench
   domains, although the size and seed stability of the effect vary.
2. E68's separated-support actuator has a credible causal signal on MathIR.
   It is not merely increasing a counter: after pre-intervention equivalence,
   its advantage grows late in training and is positive for all three seeds at
   pass 10.
3. Held-out MATH-500 is not currently an exploration domain. All correct
   solutions for one prompt collapse to the single key
   `math_verified_answer:correct`, so the part of the algorithm that discovers
   and balances multiple verified modes has no transferable object to act on.

The missing component is therefore not more entropy pressure. It is a
validator-bound, transferable definition of a reasoning route. The successor
should preserve the existing prompt-local endpoint modes and add a second
route level derived from executable or checkable traces. For free-form math,
final-answer correctness remains the admission gate, while the route
signature supplies the diversity and cross-prompt transfer object.

The other necessary change is conceptual separation between discovery and
optimization. An explorer may search for underrepresented verified routes,
but the serving policy should remain task-reward-first and learn only from
verified exemplars under a conservative replay budget. E68 is already a major
step in this direction: proposal-only support is isolated from on-policy
novelty counts and proposal rows never enter PPO.

## What E66 versus E68 currently says

Only MathIR supplies a complete three-seed E66/E68 comparison beyond
initialization. Graph, Countdown, and Python did not produce a corresponding
E68 surface before the jobs were canceled.

The table below is reconstructed directly from the four fixed K=8 draws in
each seed's retained `eval_mode_coverage_draws.jsonl`. Values are three-seed
means.

| pass | E66 pass@8 | E68 pass@8 | delta | E66 mean@8 | E68 mean@8 | delta | E66 distinct@8 | E68 distinct@8 | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .215 | .215 | .000 | .036 | .036 | .000 | .215 | .215 | .000 |
| 1 | .330 | .339 | +.008 | .073 | .076 | +.003 | .343 | .354 | +.010 |
| 2 | .432 | .447 | +.014 | .115 | .125 | +.010 | .449 | .464 | +.015 |
| 3 | .549 | .561 | +.011 | .316 | .299 | -.017 | .572 | .587 | +.014 |
| 4 | .646 | .678 | +.032 | .449 | .441 | -.008 | .681 | .708 | +.027 |
| 5 | .762 | .742 | -.021 | .565 | .547 | -.018 | .785 | .791 | +.006 |
| 6 | .767 | .795 | +.028 | .607 | .612 | +.004 | .790 | .852 | +.062 |
| 8 | .817 | .839 | +.022 | .681 | .704 | +.023 | .847 | .926 | +.079 |
| 10 | .818 | .880 | +.062 | .692 | .751 | +.059 | .869 | .987 | +.118 |

At pass 10, E68's paired seed deltas in pass@8 are `+.084`, `+.059`, and
`+.043`. The effect is therefore not carried by one seed. The transient
pass-5 loss matters: the actuator has a delayed and non-monotone payoff, so
peak selection would be misleading. Terminal pass 12 and fixed-checkpoint AUC
remain necessary.

The mechanism record also behaves as intended:

- E66 and E68 were equivalent before the first E68 admissions; the frozen
  audit compared 960 paired pre-intervention updates with zero violations.
- First admissions occurred at steps 319, 306, and 338.
- The final retained cumulative admission counters are 103, 107, and 109,
  for 319 proposal-derived verified outcomes across the three seeds.
- No update admitted more than one alternate.
- Proposal rows sent to PPO are exactly zero.
- The summed absolute proposal-to-objective outcome delta is exactly zero.
- At step 4294, mean verified support per prompt is `1.342` for E66 and
  `1.476` for E68.
- Mean cumulative verified discoveries per seed are `513.3` for E66 and
  `565.3` for E68.

This is encouraging causal evidence for the separated-support actuator on
MathIR. It is not yet a cross-domain result and it is not terminal. All six
MathIR jobs stopped at step 4295 because the replicated-group permutation seed
exceeded NumPy's 32-bit `RandomState` limit. Each has a step-4224 checkpoint.

## What held-out MATH-500 says

All six E64 runs reached step 4608 and wrote terminal markers. Directly reading
the frozen primary evaluations gives:

| metric | Dr.GRPO | E58 | E58 - Dr.GRPO |
|---|---:|---:|---:|
| greedy pass@1 | .3487 | .3420 | -.0067 |
| sampled mean@8 | .3200 | .3126 | -.0074 |
| sampled pass@8 | .6067 | .5993 | -.0073 |

The frozen realism gate requires greedy and mean@8 each to be no more than
`.02` below control and at least one to be directionally higher. The direct
terminal values satisfy non-inferiority but fail the directional clause. Once
the reporting audit is repaired and regenerated, this gate should evaluate
`fail`.

That outcome is scientifically coherent with the task contract:

- verified support is structurally at most one per prompt;
- the known-mode balance gradient is structurally zero;
- the replay bank is prompt-local;
- training and evaluation prompts are disjoint; and
- free-form prose is deliberately not treated as a verified reasoning mode.

E58 can replay a model-generated correct answer for a training prompt, but it
cannot currently learn that two correct derivations share a reusable strategy
or that one verified strategy transfers to a different problem. MATH-500 is
therefore testing ordinary generalization from a small MATH12K subset, not the
mechanism that drives the ModeBench improvements.

Prior engineering results reinforce this diagnosis. E62R10 showed that
verified counterfactual transformations can make Python exploration extremely
active: support reached `6.09`, 793 outcomes were admitted in 384 updates, and
terminal pass@8 reached `1.0`. But E63 showed that more support is not
automatically better: the counterfactual Graph arm reached support `5` yet
finished below its control in both pass@8 (`.667` versus `.805`) and
distinct@8 (`1.151` versus `1.570`). E68's singleton gate, one-admission cap,
and separated objective support address this failure mode, but they do not
create a transferable math representation.

## Research target

Build a compute-matched exploration algorithm that:

1. never learns from an unverified outcome;
2. treats task correctness as the primary objective;
3. discovers alternatives through a separate explorer or proposal path;
4. keeps proposal-only support out of on-policy novelty counts and PPO;
5. represents both prompt-local endpoint modes and cross-prompt route modes;
6. replays verified exemplars conservatively, with no evaluation feedback;
7. improves breadth without a material loss in per-draw correctness; and
8. transfers to a held-out math development set before MATH-500 is opened.

The endpoint and route levels should be hierarchical rather than mutually
exclusive:

- **Endpoint signature:** the existing validator-bound ModeBench outcome.
- **Route signature:** a normalized executable/checkable trace describing how
  the endpoint was reached.

For Graph, Countdown, Python, and MathIR, the existing endpoint signature
remains the primary scientific object. Route signatures are secondary and may
improve transfer. For free-form math, answer correctness is the endpoint gate,
while route signatures are the only diversity object.

## Track A: recover the frozen evidence

This track repairs infrastructure and completes already-frozen questions. It
must not tune the algorithm from observed outcomes.

### A0. Preserve state

- Create a deliberate Git checkpoint of code, protocols, tests, reporting
  scripts, and this plan.
- Record the 2026-07-28 cancellation of all queued `xdr_train` jobs.
- Preserve job logs, step-4224 checkpoints, identities, and source snapshots.
- Keep the copied E68-provenance figure as an immutable snapshot.

### A1. Repair deterministic failures

- Reduce the replicated permutation seed into NumPy's valid 32-bit domain,
  preferably with exact modulo arithmetic. Values below the boundary must be
  unchanged, so the patch affects only the previously unreachable region.
- Add boundary tests at steps 4294 and 4295, multi-rank permutation equality,
  checkpoint resume, and seed separation.
- Repair the E64 audit so caught grader-worker traceback diagnostics are not
  classified as training crashes. Slurm exit status, terminal markers, and
  the dedicated verifier-sensitivity audit must remain fail-closed.
- Run the focused replicated-group, E64, E66, E68, cadence, and reporting
  suites before any submission.

### A2. Regenerate the current record

- Ingest all six terminal E64 runs and publish the failed directional realism
  gate without changing it.
- Recompute the E66/E68 pass-10 fixed surface and mechanism audit from raw
  artifacts.
- Restart reporting only after the audit distinguishes scientific failure,
  recoverable infrastructure interruption, and caught verifier diagnostics.

### A3. Decide how much of E68 to finish

There are two defensible choices:

- **Evidence-completion choice:** preregister a source-only recovery, resume
  the six MathIR jobs from step 4224, then relaunch the canceled E61, E66, and
  E68 jobs with new job IDs and their retained checkpoints. This preserves the
  54-run paper estimand.
- **Successor-first choice:** finish only the six paired MathIR recoveries to
  close the strongest causal result, label the rest of E68 incomplete, and
  move compute to the route-aware successor.

The choice should be based on paper value and compute budget, not interim
performance. If the 54-run claim is still desired, all 24 E66/E68 runs and all
24 E61-R1 runs must eventually reach the frozen terminal surface.

## Track B: build the transferable successor

This should receive a new experiment number and protocol. It cannot be called
an E68 repair after terminal MATH-500 values and E68 interim outcomes are
known.

### B1. Establish route-signature feasibility offline

Start with MathIR as the executable reference and MATH12K as the target.

- Define a dual-channel math output: final answer plus a restricted,
  checkable action trace or program.
- Normalize traces into route signatures using executed operations,
  dependency structure, and state transitions—not prose or an LLM label.
- Require the ordinary math verifier to accept the final answer and the trace
  checker to accept every claimed transition.
- Measure parser/checker coverage on reward-positive MATH12K outputs,
  signature stability under harmless formatting changes, route multiplicity,
  and recurrence of route families across disjoint prompts.

Do not launch RL unless:

- at least 80% of reward-positive development outputs receive a stable route
  signature;
- route identity is unchanged by formatting-only perturbations;
- multiple verified routes exist for a meaningful subset of prompts; and
- some route families recur across disjoint prompts, establishing a possible
  transfer channel.

If coverage fails, first train or prompt a structured-output adapter. Do not
compensate by using unverified prose labels.

### B2. Implement hierarchical verified replay

Maintain two separate stores:

- a prompt-local endpoint bank, matching the existing canonical contract; and
- a cross-prompt route library keyed by normalized verified trace signatures.

Each exemplar records neutral/proposal provenance, source prompt, validator
identity, route signature, endpoint signature, model likelihood, and whether
the neutral policy later reproduced it.

The proposal path should retain E68's safety rules:

- activate only after warmup and a run-relative entropy deficit;
- admit at most one alternate per group;
- keep proposal-only outcomes out of PPO and on-policy novelty counts;
- require exact validator success and route-checker success; and
- graduate an exemplar into ordinary on-policy counts only after neutral
  reproduction.

Add a conservative replay-admission rule to prevent the E63 failure:

- require an anchor-relative likelihood or KL trust-region check;
- bound total proposal and replay tokens under the matched compute budget;
- prefer proposals whose route recurs across prompts or later graduates under
  the neutral policy; and
- preserve a task-reward-first neutral update even when the explorer is active.

The key experiment is whether route replay changes performance on other
prompts, not merely whether it increases stored support.

### B3. Run a compute-matched causal pilot

Use a new held-out math development split. MATH-500 remains sealed.

Compare:

1. Dr.GRPO;
2. E66-style E58 with endpoint-only replay;
3. E68 separated-support endpoint proposals; and
4. the hierarchical route-aware successor.

Count proposal generation, neutral rollouts, replay tokens, and verifier
calls in the compute budget. A control must receive the same total sampling
budget; otherwise an exploration gain is confounded with extra inference.

First run one seed for two or three passes on MathIR and the math development
task. Advance only if:

- every admitted route and endpoint is verifier-valid;
- proposal rows in PPO and proposal-to-objective count deltas remain zero;
- cross-prompt route replay is nonzero;
- pass@1 and mean@8 lose no more than `.02`;
- verified route breadth and neutral reproduction both increase; and
- the gain persists over the last two fixed checkpoints.

### B4. Three-seed short-horizon cross-domain test

Freeze the successor and run three seeds through six passes on all four
ModeBench domains plus the separate held-out math development set.

Advance only if:

- no domain loses more than `.03` mean pass@8 against the E66-style control;
- no paired seed loses more than `.10`;
- at least three ModeBench domains improve endpoint breadth or fixed-surface
  AUC;
- Python's worst seed does not collapse;
- math-development greedy and mean@8 are each within `.01` of control and at
  least one is positive;
- route reuse predicts improvement on prompts other than the source prompt;
  and
- all mechanism, compute, cadence, and integrity audits pass.

Use the complete fixed surface, not the best checkpoint.

### B5. Final five-domain confirmation

Only after B4 passes:

- freeze all coefficients, route schemas, compute budgets, seeds, and
  checkpoints;
- run the full 12-pass ModeBench comparison;
- run the matched MATH12K training cohort;
- evaluate MATH-500 once on its frozen schedule without using it for model or
  hyperparameter selection; and
- report failure gates as boundaries rather than retuning after inspection.

## Immediate order of work

1. Commit/archive the current scientific state.
2. Fix and test the permutation-seed boundary.
3. Fix the E64 caught-traceback audit and regenerate terminal MATH-500.
4. Freeze a paired MathIR recovery amendment and recover the six runs from
   step 4224.
5. Decide between full E68 evidence completion and successor-first compute.
6. Prototype the route signature offline on MathIR and MATH12K.
7. Create a separate math development split and keep MATH-500 sealed.
8. Run the compute-matched causal ladder B3 through B5.

## The claim to aim for

The strongest defensible successor claim is not “entropy improves everything.”
It is:

> A task-reward-first policy can use a separate, verifier-bound explorer to
> discover endpoint and route alternatives, replay them conservatively, and
> improve both solution breadth and held-out correctness under matched compute.

E68 supplies evidence that the separated-support actuator can work. The next
research step is to give that actuator a transferable verified route object
and to prove that the added discoveries help prompts other than the ones that
generated them.

## Execution update: 2026-07-28

The recovery and successor work has now established:

- the replicated-group seed overflow is repaired with exact pre-boundary
  schedule preservation and deterministic modulo wrapping;
- all six paired E66/E68 MathIR continuations resumed from step 4224 and crossed
  the original step-4295 failure boundary;
- E64 is terminal and auditable, with its caught `math_verify` diagnostic
  classified narrowly and all other tracebacks still fatal;
- a frozen 128-row MATH12K route-development split now spans all 35
  subject-by-level cells, has zero normalized overlap with train and MATH-500,
  and contains no MATH-500 rows or scores;
- MathIR has a separate alpha-normalized route signature derived from exact
  executed operations, independent of action labels, coefficient names, and
  numeric bindings;
- the full MathIR offline gate passed on 512 prompts and 2,560 public-validator
  replays, with exactly five recurring route signatures per family and zero
  violations; and
- free-form math has a first fail-closed executable numeric trace language with
  problem-grounded leaves, computed rather than declared intermediate values,
  full dependency-use checks, and independent agreement with the ordinary
  boxed-answer verifier.

The paired MathIR recovery is now terminal. All six registered continuations
completed with exit code `0:0` at step 4608 (12 passes), and each seed has all
49 fixed primary evaluation points. At the terminal checkpoint, E68 minus E66
in the three-seed mean is `+.0260` greedy, `+.0301` mean@8, `+.0267` pass@8,
and `+.1367` distinct@8. Greedy, pass@8, and distinct@8 are positive in every
paired seed; mean@8 is positive in two of three. Mean tracked outcomes rise by
48 and mean support per prompt by `.1265`. The exact scheduler, audit, curve,
figure, and hash evidence is sealed by
`var/artifacts/e66_e68_mathir_terminal_recovery_summary.json`.

The explicit compute-budget choice is **successor first**. The remaining
non-Math E66/E68 cells stay labeled incomplete; they are not silently promoted
to a cross-domain result. This choice closes the strongest causal result while
avoiding a 54-run completion whose endpoint-only mechanism already failed the
frozen MATH-500 directional-transfer clause. It does not retune E68 or alter
any observed E66/E68 arm.

The next hard gate is empirical trace coverage on fixed base-model samples from
the sealed MATH12K route-development split. The 80% threshold remains binding:
if the model cannot reliably emit the restricted trace, MATH route novelty does
not launch, even though ordinary task reward remains available.
