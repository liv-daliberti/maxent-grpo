# E72 B1a confirmation: does verified replay alone reproduce xGRPO?

**Status: FROZEN BEFORE THE CONFIRMATION COHORT IS SUBMITTED — 2026-08-01**

## Provenance, stated first

This protocol exists because the arm it concerns was launched without one.

B1a — the frozen treatment with open-set discovery credit removed and verified
replay untouched — was submitted on 2026-07-31 on seeds 43–47 without a frozen
protocol, unlike B3a. Its design was inherited from the B3a cohort (same
domains, seeds, schedule, evaluation contract, GPU-model pinning, and
inheritance checks), but no interpretations were registered in advance, and its
partial results were observed before this document was written.

Those five seeds are therefore a **discovery sample**. They cannot be reported
as a confirmatory test of a hypothesis they generated, and nothing in this
document is a claim about them. Their role is to have raised the question that
the cohort registered below is designed to answer.

The alternative — reporting the existing seeds as "exploratory but same design"
and stopping — was rejected. The observation at stake reverses the emphasis of
Section~4 of the manuscript, and an emphasis reversal supported only by a
sample that was inspected before its analysis was fixed is not evidence we are
willing to publish.

## What was observed in the discovery sample

Recorded here so that the confirmation cannot be quietly re-scoped later. On the
seeds and domains terminal at the time of writing, B1a's `distinct@8` was close
to xGRPO's and far above matched Dr.GRPO's: Graph coloring 2.505 against 2.406
and 0.325 (5/5 seeds); Countdown 1.846 against 1.893 and 0.627 (5/5);
PantryPlan 2.246 against 2.143 and 0.659 (4/5); MathIR 0.986 against 0.911 and
0.656 (2/5). Differences from xGRPO were within roughly ±0.10, comparable to
the seed spread within either arm.

## Hypothesis

**H.** Verified replay alone reproduces xGRPO's retained support: removing
open-set discovery credit does not reduce `distinct@8` at the terminal
checkpoint.

This is a claim of **equivalence**, not superiority. The discovery sample gives
no reason to believe B1a exceeds xGRPO, and any such reading would be an
artifact of seed noise.

## Cohort

- Arms: **B1a** (`verified_first_replay_only_ablation`) and **xGRPO**
  (`verified_first_global_replay_canonical`), both trained from the pinned
  Qwen2.5-0.5B-Instruct initialization.
- **Seeds: 48, 49, 50, 51, 52** — disjoint from the 43–47 used by every
  existing cohort. Both arms are re-run on them; the published xGRPO seeds
  cannot serve as the comparator because they are not paired with these seeds.
- Domains: all five ModeBench domains.
- Size: 2 arms × 5 seeds × 5 domains = **50 runs**, ~560 GPU-hours.
- Everything else matches the common design: 12 passes, 4,608 optimizer
  updates, $G = 16$, $\beta_{KL} = 0$, rollout temperature 1, the same fixed
  128-prompt evaluation split, greedy decoding plus four deterministic
  temperature-one $K = 8$ replicates, and per-domain data, template, response
  budget, and draw seeds inherited from the source runs.
- **Placement: both arms of a seed run on the same GPU model.** Because these
  seeds are new, the model is chosen freely per domain and recorded before
  submission; the constraint is that the pair matches, not which model it is.

  *Recorded before submission, 2026-08-01:* **all 50 runs on a5000**, submitted
  to `node105` under the `mltheory` partition. That model was chosen because the
  a5000 pool is the one whose cross-node bit-exactness was verified during the
  decoding sweep, and because `mltheory` is not preemptible, so no run competes
  with the checkpoint-resume path. The single-node restriction costs wall clock
  (ten GPUs, roughly five waves) and buys uniformity; it is a scheduling choice
  and carries no scientific content beyond keeping every pair matched.

If compute forces a reduction, domains are dropped in this order — MathIR,
Countdown, Python factors — and the reason is recorded. Graph coloring and
PantryPlan are retained in every reduction because they carry the largest
xGRPO effects and therefore the most power to detect a shortfall.

## Analysis, fixed in advance

Primary quantity: `distinct@8` at terminal pass 12, per domain, five paired
seeds, no pooling across domains.

**Equivalence margin.** $\delta = 0.15$ `distinct@8`, the same margin the B3a
protocol used to define "close to". It was chosen before this cohort and is not
adjusted after seeing it.

Per domain, let $d$ be the mean paired difference (B1a minus xGRPO) with a
seed-level paired bootstrap interval (10,000 resamples):

- **E (equivalent).** The interval lies entirely within $(-\delta, +\delta)$.
- **W (worse).** The interval lies entirely below $+\delta$ and excludes zero
  from below — B1a loses support relative to xGRPO.
- **B (better).** The interval lies entirely above zero and above $+\delta$.
- **U (undetermined).** Anything else, including intervals that straddle a
  boundary. Reported as undetermined; not rounded toward the favourable side.

## Registered interpretations

- **C1.** Verdict E in at least four of five domains → discovery credit is not
  necessary for retained support in ModeBench. Section~4.2's rare-mode and
  first-discovery advantages are demoted from core mechanism to optional
  component, the contribution is restated around verified replay, and
  `tab:component-ablations` gains a discovery row reading "removal costs
  nothing measurable."
- **C2.** Verdict W in two or more domains → discovery credit contributes, the
  discovery sample was misleading, and the manuscript's current framing stands
  with the per-domain magnitudes reported.
- **C3.** A mixture of E and W → both are reported per domain, and the method
  section states plainly that discovery credit matters in some domains and not
  others, naming which.
- **C4.** Verdict B anywhere → reported as-is. We do not expect it and will not
  claim it from a single domain.

In every case, the discovery-sample seeds 43–47 are reported separately and
labelled as the sample that generated the hypothesis, never merged into the
confirmation estimate.

## Failure policy

As in the B3a protocol: CUDA OOM, non-finite loss or coefficient, traceback,
malformed checkpoint, identity mismatch, missing seed, or failure to reach the
fixed terminal budget is a run failure; a failed run may resume only from its
own source-bound checkpoint with the same arm, seed, data, and protocol. A
domain missing any of its five terminal seeds in either arm is unreported.

Arm-specific integrity, checked on every logged update rather than assumed:

- B1a must report `semantic_shannon_coef` $= 0$, novelty credit $= 0$, and the
  separate-advantage, success-conditioned, and open-set adaptation switches
  off, while `online_canonical_replay_compute_only` is $0$ and the applied
  replay score gradient is nonzero.
- xGRPO must report the frozen doses and a nonzero applied replay gradient.

A run violating its arm's conditions is discarded, not reinterpreted. The first
B1a launch attempt failed argument validation at startup on exactly this
surface and was discarded rather than repaired in place; its failure logs are
retained.
