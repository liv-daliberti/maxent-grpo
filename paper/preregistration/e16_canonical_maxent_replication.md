# E16 E15-derived canonical MaxEnt replication

**Status: FROZEN V3 PASS@1-EVALUATOR RECOVERY AMENDMENT AFTER THE CANCELLED
STAGE-R V2 COHORT, BEFORE V3 SUBMISSION OR OUTCOMES (2026-07-19).**

E16 is the requested 0.5B, two-environment, three-method, three-seed
extension of E15. It retains E15's scientific object: a finite, fixed-horizon
canonical action policy trained with direct on-policy maximum entropy. It does
not use E13's free-form completion policy, response-length controller, or
entropy target. An earlier unsubmitted E13-derived E16 draft and launcher were
retired before configuration because they changed the policy and objective.

This document was initially frozen after the canonical Countdown
implementation, exact online entropy measurement, endpoint audits,
frozen-snapshot import closure, and both configuration-only validators passed
review. Stage-S v1 then exposed the terminal-boundary integration failure
recorded below. Stage-S v2 passed and authorized Stage-R v2, but the first
analytical evaluations exposed the Countdown-only greedy-evaluator defect
recorded below. At the user's request, all eighteen Stage-R v2 jobs were
cancelled and are ineligible for analysis. The source and execution identities
here bind the prospective v3 recovery before any v3 job or outcome. The
standard monitor must display the requested 0.5B analytical cells as `DESIGN`
until the v3 analytical jobs actually exist and must report the fresh six-cell
engineering smoke separately.

The identities below bind the source and complete execution surface. The
launcher rejects source, tooling, dataset, runtime, or stage-config drift
rather than treating the `FROZEN` label as authorization by itself.

| Frozen identity | SHA-256 |
|---|---|
| Python `src/**/*.py` | `0e9929f09bac51ddcd85a6d5a506bf0613279a6fd983e0147da2f39a7aa320ec` |
| Shell/tooling execution surface | `b7f01de3e9a9247afe6d95f984ce77c0e0615e4aa067a7fa77c701da15bc9597` |
| Dataset/codec identity | `efc23631bd96c631e04e5a8161533cfdfc88e8b737f4e77d9d78a547b66fefce` |
| Runtime/tokenizer identity | `40bafc191219c2612ad6b74aa07aa878eef06644323aec5cfb6a6dd18d7a3e66` |
| Stage-S config | `7c91976613e1f23ff965256c7ff4abd49ce73b7f1f32122b6f9e60820a72b761` |
| Stage-R config | `10f9183ae4155b28e46addafb49f38800ab6fd39c3f054a81a828326c64d43aa` |

## Antecedent evidence and scope

E15 selected fixed `alpha=0.10` on the 27-action graph-coloring policy at seed
9005. Its exact endpoint action entropy was
`2.7974740052946436` nats, exact valid probability was
`0.261672194741692`, valid-mode effective support was
`4.437708099723962`, and valid-probability retention relative to E14 C0 was
`0.833154545824017`. Both `alpha=0.075` and `alpha=0.10` were viable; E14 had
also established `alpha=0.05` as behaviorally safe.

E16 is limited to Qwen2.5-0.5B-Instruct, graph coloring and Countdown, methods
fixed/proportional/Haarnoja dual, and paired seeds 43/44/45. A separate
engineering smoke uses seed 9006 and is excluded from analysis. E16 does not
authorize 3B/7B runs or a free-text comparison.

The existing Dr.GRPO curves use a different free-text policy and prompt. They
may be shown only as explicitly labeled historical context; they are not a
canonical-policy control and cannot support an E16 treatment contrast. E16's
primary comparisons are among its three randomized methods, with fixed
canonical MaxEnt as the reference.

## Finite canonical policies

Both policies emit exactly three single-token digits, terminate
deterministically, and use learner-side fixed-shape on-policy sampling. Their
entropy is the entropy of the complete finite action distribution, not token
entropy over the model vocabulary and not entropy over free-form text.

### Graph coloring

E16 retains E15 unchanged: each position chooses a color from digits
`{1,2,3}`. The complete support contains `3 x 3 x 3 = 27` colorings and has
maximum entropy `log(27) = 3.295836866004329` nats.

### Countdown

For public ordered input numbers `(n1,n2,n3)`, a three-digit code selects one
of exactly 108 canonical symbolic expression trees:

1. The first digit selects one of six root forms:
   `pair+s`, `pair*s`, `pair-s`, `s-pair`, `pair/s`, or `s/pair`.
2. The second digit in `{1,2,3}` selects singleton `s`; the other two numbers,
   in input order, are `(a,b)`.
3. The third digit selects one of six inner-pair forms:
   `a+b`, `a*b`, `a-b`, `b-a`, `a/b`, or `b/a`.

The position supports are therefore `(6,3,6)`, the complete support has
`6 x 3 x 6 = 108` leaves, and maximum entropy is
`log(108) = 4.68213122712422` nats. Digits 1–6 are distinct one-token actions
under the frozen tokenizer. The decoder uses only the numbers and grammar;
it never reads the target or solution set.

An exhaustive preflight over all 384 train and 128 evaluation rows in
`exact_countdown_easy3_probe` must reproduce exactly 108 distinct canonical
grader keys per prompt. Every declared valid answer mode must be represented
by exactly one code and no other code; the audited valid support is 2–8 modes
per prompt. Any alias, omission, target access, tokenizer drift, or variable
horizon invalidates the protocol.

## Objective and entropy observation

For canonical action `a`, every arm optimizes

\[
J(\theta)=J_{\mathrm{Dr.GRPO}}(\theta)
 + \alpha H\!\left(\pi_\theta(a\mid x)\right).
\]

The actor-gradient estimator is E15's exclusive-prefix, direct on-policy
entropy surrogate. Entropy receives no action-length division; the combined
update retains exactly one shared outer Dr.GRPO normalization by
`T_max=192`. There is no response-length target, EOS controller, token-entropy
bonus, xDr weighting, candidate projection, or KL-to-old-policy objective.

Adaptive control must not observe the noisy 16-rollout entropy estimate. After
each optimizer update, the learner exactly traverses the current training
prompt's finite policy tree—13 prefixes for graph coloring and 25 for
Countdown—and computes complete-action entropy by the chain rule. This
detached, label-free quantity controls the next update's coefficient and is
logged for all three methods. Controller checkpoints use a distinct exact
canonical-action entropy unit and must reject old sampled-sequence controller
states.

## Three methods

E15's selected entropy fraction is

\[
\rho=2.7974740052946436/\log(27)=0.8487901916960259.
\]

The exact targets are therefore `2.7974740052946436` nats for graph coloring
and `rho*log(108) = 3.9741470618167156` nats for Countdown. The same alpha
units transfer because alpha is reward per canonical-action nat; Countdown
safety does not transfer and is separately gated.

| Method | Frozen candidate rule |
|---|---|
| Standard MaxEnt fixed | `alpha=0.10` on every update |
| Standard MaxEnt proportional | base `0.075`, max `0.10`, explicit domain target above, EMA `0.9`, gain `4`, immediate control |
| Standard MaxEnt Haarnoja dual | base `0.075`, min `0.05`, max `0.10`, explicit domain target above, log-alpha Adam LR `0.005`, betas `0.9/0.999`, immediate control |

The bounds contain only coefficients already behaviorally safe in E14/E15.
No adaptive arm may exceed `0.10`, because E15 M10 retained only 83.32% of
C0 valid probability against an 80% gate. The proportional and dual arms use
the same target but remain distinct update rules.

## Stage-S v1 failure and reviewed v2 recovery

The jointly released v1 smoke used jobs 30013125--30013130. The three graph
cells entered canonical training normally. During the first Countdown
evaluation, vLLM V0 invoked the position-dependent logits processor once with
three generated tokens before enforcing `max_tokens=3`. The v1 processor
treated that terminal bookkeeping callback as an attempted fourth action and
raised `canonical generation exceeded its fixed horizon`. No fourth token was
sampled and no E16 analytical job was submitted. The failure is an
actor/runtime integration defect, not a treatment outcome.

The sole source repair makes the callback at exactly `position == horizon` a
no-op; callbacks beyond the horizon still fail closed, and the independent
post-generation validator still rejects every overlong response. A regression
test reproduces both the accepted terminal callback and the rejected true
overrun. Because this changes the source identity, no v1 cell is eligible for
the approval. Recovery therefore relaunches all six cells together under the
fresh `gce16_canonical_maxent_joint_smoke_v2` and
`cde16_canonical_maxent_joint_smoke_v2` prefixes. GPU model and host-memory
placement are operational choices and do not change the frozen model, data,
objective, seed, update budget, or gate.

## Stage-R v2 greedy-evaluator defect and reviewed v3 restart

All six Stage-S v2 cells completed and passed their immutable endpoint and
raw-evidence replay gates. Stage-R v2 then launched jobs 30013329--30013346.
Its first inline evaluations showed zero Countdown greedy `pass@1` alongside
nonzero sampled `mean@8`. The inherited OAT greedy evaluator was passing each
raw three-digit canonical Countdown action code, such as `123`, directly to
the arithmetic grader. Training rollouts and sampled coverage already decoded
the same action code into its full expression before grading, and graph
coloring requires no transformation because its code is itself the answer.

The sole source repair overrides canonical greedy evaluation so every action
code is decoded with its prompt reference before oracle grading, while
preserving OAT's sample-major ordering and response/reward shapes. A regression
test verifies the decoded responses, reference alignment, and returned score
matrix. Although the defect was isolated to inline Countdown evaluation and
the saved raw outputs were exactly recoverable, the user requested a complete
clean restart out of concern for training. All eighteen v2 analytical jobs
were therefore cancelled after roughly twenty minutes and no v2 analytical
metric is eligible for E16.

Because the repair changes the frozen source identity, v2 smoke approval
cannot authorize the restart. Recovery reruns all six smoke cells under fresh
`gce16_canonical_maxent_joint_smoke_v3` and
`cde16_canonical_maxent_joint_smoke_v3` prefixes. Only a new all-six approval
may release the eighteen analytical jobs under
`gce16_canonical_maxent_05b_v2` and `cde16_canonical_maxent_05b_v2`. The full
launcher now executes the exact frozen verifier bound by that approval rather
than an equivalent live-path copy. GPU model and host-memory placement remain
operational choices.

## Staged execution

Stage S is a jointly launched six-run engineering smoke: two environments by
three methods, seed 9006, 32 optimizer updates. It must establish immutable
source/data/config identities; exact code-support and tokenizer identities;
normalized behavior/current distributions; finite overlap diagnostics;
post-update exact-entropy identities; one-step controller handoff and exact
recurrence replay; coefficient bounds; deterministic three-token termination;
positive rollout reward; and exact endpoint enumeration. It is an operational
gate, not scientific evidence. Failure of any cell holds all eighteen
replication runs. The smoke is not required to reach its target within 32
updates.

Stage R is launched only from an immutable all-six smoke approval. It contains
both environments, all three methods, and paired seeds 43/44/45: 18 runs.
Each run trains over the complete frozen prompt pool for exactly five epochs,
one PPO epoch per group, group size 16, learning rate `2e-7`, `beta=0`, and
temperature 1. Graph coloring has 960 prompt updates and evaluates every 48;
Countdown has 1,920 prompt updates and evaluates every 96. Automatic resume
and watchdog requeue are prohibited; a failed run is a fresh, visibly linked
attempt under a reviewed recovery rule.

Every terminal checkpoint receives exact full-support evaluation on the
frozen evaluation prompts. Reporting includes all seeds and paired
fixed-versus-adaptive differences for exact action entropy, exact valid
probability, exact valid-mode entropy/effective support, rollout reward,
pass@1, pass@8, and coverage@8. Code entropy and semantic valid-mode entropy
remain separate quantities. With three seeds, results are replication effect
sizes, not asymptotic significance claims.

## Retrospective Stage-R V3 outcome — appended after cohort completion

**This section was written after all eligible Stage-R V3 jobs completed. All
preceding text remains the frozen prospective protocol.**

All eighteen jobs 30013443--30013460 completed the exact five-pass budget with
Slurm state `COMPLETED`, exit code `0:0`, and no alert in the terminal
watchdog record. Training logs contain zero invalid canonical actions. The
registered terminal sampled endpoints are therefore eligible and encouraging:
in the three-seed mean, every method improves every sampled metric over the
common initialization in both domains. This aggregate statement is not
seed-uniform; several individual seed/metric pairs do not improve.

Across seeds 43/44/45, graph-coloring fixed/proportional/dual mean
`pass@8` is 0.913/0.892/0.899 and mean coverage is
0.360/0.338/0.352. On Countdown, where the adaptive comparison separates,
the corresponding `pass@8` values are 0.609/0.651/0.651 and coverage is
0.255/0.287/0.267. Relative to paired fixed MaxEnt, Countdown proportional
control changes `mean@8` by +9.1 points, coverage by +3.3 points, distinct
correct modes by +0.143, and greedy `pass@1` by +5.2 points. The dual changes
the same outcomes by +9.4, +1.2, +0.036, and +3.9 points/modes.

These are three-seed effect sizes from the registered terminal sampled
evaluation, not asymptotic significance claims. The exact full-support
terminal audits promised above have not yet been produced; exact held-out
action entropy, valid probability, and valid-mode effective support therefore
remain pending and are not inferred from sampled coverage. The complete
sampled endpoint table and immutable provenance hashes are in
[`../results/e16_canonical_maxent_replication.md`](../results/e16_canonical_maxent_replication.md)
and its machine-readable JSON companion.
