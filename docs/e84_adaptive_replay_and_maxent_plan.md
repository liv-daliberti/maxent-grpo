# E84 plan: adaptive verified replay and adaptive semantic MaxEnt

**Draft for discussion, 2026-08-09. Nothing here is frozen and nothing is
submitted.** Cohort: Qwen2.5-0.5B and Falcon3-1B, five static domains, paired
against the completed E78/E79 and E81/E82 arms.

## 0. A blocking finding that must be fixed before any adaptive campaign

Semantic MaxEnt is **structurally inert on PantryPlan**, in every cell of both
E81 and E83 (10 of 10 runs, all five seeds each):

| domain | reward-positive fraction | parseable fraction | eligible fraction | mean \|A_sem\| RMS |
|---|---|---|---|---|
| Countdown | 0.561 | 0.971 | 0.561 | 0.0069 |
| **PantryPlan** | **0.544** | **0.000** | **0.000** | **0.0000** |

The policy solves 54% of PantryPlan rollouts and the replay bank fills normally
(mean bank size 7.1 modes/prompt), but the semantic tracker parses **nothing**.

Root cause: the semantic term derives its outcome key from
`extract_normalized_final_answer`, the free-form `\boxed`/text extractor
(`grpo.py:148`), while PantryPlan runs canonical actions
(`canonical_action_task=pantry_support_mask`, 8-token responses) whose outcome
key comes from the canonical-action decoder the bank uses. The two key paths
disagree, so every PantryPlan row is scored unparseable and receives exact zero.

Two consequences:

1. **Semantic MaxEnt is inert on any canonical-action task**, not just this one.
2. **The PantryPlan column of E81, E82, and E83 is null by construction.** E81's
   reported PantryPlan `distinct@8` gain of $+0.076$ cannot be a semantic-MaxEnt
   effect — that arm's applied objective was identical to E78 replay. It must be
   reported as run-to-run variance, not as evidence, and the same applies to
   E83's PantryPlan $+0.000$.

The fix is to route the semantic key through the same validated canonical
outcome key the bank already uses whenever canonical actions are active. That is
a real code change with its own tests, and it should land and be smoke-tested
before an adaptive campaign, or the adaptive arms inherit the same dead domain.

## 1. What the fixed-dose data actually says

At a *fixed* $\eta = 0.10$, the realized semantic advantage is not a fixed
intervention. Mean $|A_{\mathrm{sem}}|$ RMS against task-advantage RMS, E81:

| domain | A_sem RMS | A_task RMS | realized ratio | E81 distinct@8 effect |
|---|---|---|---|---|
| Countdown | 0.0069 | 0.187 | **3.7%** | +0.115 |
| Graph coloring | 0.0068 | 0.415 | 1.6% | −0.022 |
| Python factors | 0.0024 | 0.125 | 1.9% | +0.016 |
| MathIR | 0.0013 | 0.127 | 1.0% | +0.035 |
| PantryPlan | 0.0000 | 0.314 | 0.0% | (inert) |

Across seeds the spread is worse: Python factors' eligible fraction ranges
0.19–0.79 and its A_sem RMS ranges 0.0003–0.0025, an eightfold swing at an
identical nominal coefficient.

**This is the case for adaptation, and it is a different case than "tune the
coefficient harder."** The defect is that a fixed $\eta$ delivers a realized
pressure that varies by more than an order of magnitude across domains and
seeds, because the centered surprisal depends on how many verified modes the
bank happens to hold. The adaptive rule's job is to make the dose *mean the same
thing everywhere*, not to chase a better outcome.

## 2. What the controller history says

Every previous controller in this program failed, and the failures are
informative rather than discouraging:

- **Token-entropy MaxEnt (E4–E13).** Proportional and Haarnoja-dual control
  retained only 17.7% and 11.7% of their frozen entropy target. A fixed
  $\alpha=0.50$ stress arm ended at 168.75 tokens with 14/16 responses lacking
  EOS and zero evaluation accuracy. E12 found *no safe fixed dose* at all. The
  lesson: a controller acting on an unbounded objective produces length blow-up.
  Our semantic term is bounded by construction ($|A_{\mathrm{sem}}| \le \eta$),
  which removes that specific failure mode.
- **E72 B2b.** The coefficient *floor* was the binding parameter: at the 0.005
  default the arm reached 64× its entropy target and pass@8 .000; at a 1e-4
  floor the same cell reached pass@8 .545. The lesson: bounds are not safety
  rails, they are the design.
- **The singleton trap**, already handled in `online_canonical_controller.py`:
  normalized bank entropy is *undefined* for a singleton bank, and treating it
  as zero "would spuriously drive alpha upward before the learner has discovered
  a second valid outcome." Our data shows this is the common case — MathIR and
  Python sit at 1.08 and 1.14 modes per prompt.

Existing implementations to reuse rather than rewrite: `MaxEntInverseController`,
`MaxEntProportionalController`, `MaxEntDualController`,
`OnlineCanonicalPolicyEntropyController`, `OnlineCanonicalDualController`.

## 3. Proposed design

### Principle

Adapt to equalize **realized pressure**, never to chase an outcome. No
controller may observe evaluation results, gold support, a desired mode count,
or a target `distinct@8`. Every controller freezes rather than extrapolates when
its sensor is undefined.

### Arm A — adaptive semantic MaxEnt (RMS-targeted)

$$\eta_{t+1} = \mathrm{clip}\!\left(\eta_t \cdot \left(\frac{\rho\,\widehat{\mathrm{RMS}}_{\text{task}}}{\widehat{\mathrm{RMS}}_{\text{sem}}}\right)^{g},\ \eta_{\min},\ \eta_{\max}\right)$$

- $\rho$ = registered target ratio. Propose **0.05**: fixed $\eta=0.10$ delivered
  1.0–3.7%, and the one domain with a clear benefit (Countdown) sat at the top
  of that range. $\rho$ is chosen from the *mechanism* telemetry, never from
  `distinct@8`.
- $\widehat{\mathrm{RMS}}$ are EMAs over 64 updates; damping $g = 0.5$;
  $\eta \in [0.02, 0.40]$; per-step change capped at $\times 1.1$.
- **Hard no-op** (freeze $\eta$, do not update the EMA) when eligible fraction
  $< 0.05$, or the prompt has $< 2$ distinct verified modes, or
  $\widehat{\mathrm{RMS}}_{\text{task}} < 10^{-3}$. This is the singleton trap
  and the near-zero-denominator instability, both of which I flagged earlier in
  this program as reasons *not* to RMS-normalize naively. The no-op is what makes
  it safe.

### Arm B — adaptive verified replay

Sensor: EMA of the **verified reward-positive fraction** $s_t$ — how often the
policy is currently producing anything the bank can keep.

$$\mu_{t+1} = \mathrm{clip}\!\left(\mu_0 \cdot \left(\frac{s^\star}{\max(s_t,\ s_{\min})}\right)^{g},\ 0.05,\ 0.40\right)$$

Motivation is the strongest new evidence in the campaign: on Python factors the
3B Dr.GRPO control saw reward on **0.2%** of 3,072 updates and never got off the
ground (first success at step 871, final `distinct@8` exactly 0), while the
replay arm caught its first success at step 308 and bootstrapped to the best
`distinct@8` in the campaign. Falcon's Python control is likewise 0.000. Replay
is worth most exactly when successes are rare, and that is a quantity the run
can measure about itself without looking at evaluation.

### Arm C — both adaptive

Only worth running if A or B shows an effect. Running it first would repeat the
original identification failure that E78–E83 were built to escape.

## 4. Identifiability

The existing arms are the comparators, so nothing needs re-running:

| | fixed replay | adaptive replay |
|---|---|---|
| **no semantic** | E78 / E79 `replay` | **Arm B** |
| **fixed semantic** | E81 / E82 | — |
| **adaptive semantic** | **Arm A** | **Arm C** |

Primary estimand for Arm A is the paired difference against **E81/E82** (fixed
semantic), which isolates adaptation of the coefficient. For Arm B it is against
**E78/E79 replay**, which isolates adaptation of the replay dose.

## 5. Staging and cost

| stage | content | cells | gate to proceed |
|---|---|---|---|
| 0 | PantryPlan key fix + controller telemetry + 1-seed smoke, 2 domains, 2 families | 8 | $\eta_t$ not pinned at a bound in >20% of updates; realized ratio converges to $\rho \pm 50\%$ |
| 1 | Arm A, 2 families × 5 domains × 5 seeds | 50 | any domain shows a paired effect vs E81/E82 |
| 2 | Arm B, same shape | 50 | as above vs E78/E79 |
| 3 | Arm C | 50 | only if 1 or 2 passes |

Up to 158 cells. For context the campaign has run 245 cells to date and E80r1
still has ~35 pending on the single A100 node, so Stage 3 is a genuine
commitment, not a rounding error.

**Recommendation: authorize Stage 0 only.** It is 8 cells, it fixes a real bug
that currently voids one domain of three cohorts, and its gate is a mechanism
check rather than an outcome check — so passing it does not bias anything
downstream.

## 6. Pre-registered stopping conditions

Fixed now, before Stage 0 runs:

1. $\eta_t$ or $\mu_t$ pinned at a bound for >20% of updates → the controller is
   mis-specified; stop and re-derive the bounds, do not widen them post hoc.
2. Realized RMS ratio fails to reach $\rho \pm 50\%$ within 2 passes → the sensor
   does not control the actuator; stop.
3. Mean response length rises >25% above the fixed-dose arm, or no-EOS fraction
   exceeds 0.05 → the E4–E13 failure mode has returned; stop.
4. Any arm's applied replay gradient is nonzero where the protocol says zero, or
   any controller observes an evaluation quantity → integrity failure; discard
   the cohort.

## 7. Open questions for tomorrow

- Is $\rho = 0.05$ the right target, or should Stage 0 measure the realized ratio
  under a *fixed* $\eta$ on the repaired PantryPlan first and set $\rho$ from the
  five-domain median?
- Should Arm B's sensor be reward-positive fraction or banked-mode survival? The
  former is better motivated by the cold-start evidence; the latter is closer to
  what replay actually protects.
- PointMaze stays excluded from all arms: it is warm-started rather than
  base-initialized, its arms move by <0.05 modes over eight passes, and its
  interactive trainer has no semantic-advantage path.
