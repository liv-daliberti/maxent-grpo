# Pre-registration: Group-level exploration as a learning investment — does the advantage scale with training compute, and does it survive as a downstream policy?

**Status:** DRAFT for review (not yet committed). The git commit timestamp of
the final version is authoritative and must precede any run listed below.

**Registered before:** any extended-budget (> 1 standard epoch) or two-stage
graph-coloring run at either scale. The existing single-epoch runs
(`gccomp1_stable` at 0.5B, `gc3b` at 3B) are pre-observed and are reused only
as the leftmost (1-epoch) point of the compute axis, disclosed as such.

## Motivation (known at registration)

At the standard one-epoch budget, xDr.GRPO (tau = 0.05) already beats Dr.GRPO
on answer-mode coverage at flat per-sample accuracy: 3B graph coloring
pass@8 .656 -> .829 (+17.4), coverage@8 +10.5, conditional distinct modes per
solved prompt 1.32 -> 1.77, mean@8 and greedy flat (paper Sec. 7). The paper's
mechanism claim is that uniform Dr.GRPO aggregation asymmetrically prefers a
single mode and therefore **collapses the within-prompt candidate distribution
as training proceeds**, while xDr.GRPO preserves it. That claim is currently
supported only by a static one-epoch comparison and by training-time weight
diagnostics; it has never been tested as a *dynamic* prediction about how the
gap evolves with compute, nor cashed out as a *downstream learning* benefit.
This registration fixes two such tests (E1, E2) and one rival-explanation
control (E3) before running them.

## Common recipe (fixed; identical to the paper's matched protocol)

Qwen2.5-0.5B-Instruct (pilot) and Qwen2.5-3B-Instruct (confirmatory);
graph-coloring completion pool (`exact_answer_mode_probe`, 192 train / 96+96
eval at 0.5B; `exact_gc_large_probe`, 1024 train / 256+256 eval at 3B);
critic_type=drgrpo; beta = lambda = 0; num_ppo_epochs = 1; lr 2e-7 constant;
G = 16 (0.5B) / 32 (3B); temperature 1.0, top-p 1.0; T_max = 192; bf16;
fused-AdamW shim. Training seeds 43, 44, 45 per arm. Evaluation: eval seeds
1001, 1002, 1003; K = 8 samples at temperature 1.0 plus one greedy pass; the
standard coverage harness (`ops/eval_exact_answer_mode_coverage.py`). Arms
differ only in the aggregation weight / rollout temperature knobs named below;
every other knob is pinned via `--export` as in the paper's submit script.

Metrics at every evaluated checkpoint: pass@8, coverage@8, distinct@8, mean@8,
greedy pass@1, and conditional distinct-modes-per-solved-prompt.

---

## E1 — Compute-scaling divergence (primary new result)

**Hypothesis.** The xDr.GRPO minus Dr.GRPO advantage on multi-answer coverage
*grows with training compute*, because the Dr.GRPO baseline's within-prompt
candidate distribution collapses as training continues while xDr.GRPO's does
not.

**Design.**
- Arms: `grpo` (Dr.GRPO, tau = inf) and `xdr_tau0p05` (xDr.GRPO, tau = 0.05).
  No other arms in E1.
- Compute axis: extend `MAX_TRAIN` from the standard 1 epoch (0.5B 9,216 /
  3B 32,768 prompt-consumptions) to **8 epochs** (0.5B 73,728 / 3B 262,144),
  the pool cycling. Save a checkpoint every 512 optimizer steps (~16 points
  on the axis); **all** intermediate checkpoints retained and evaluated
  (`OAT_ZERO_MAX_SAVE_NUM` raised; rolling eval-then-archive if disk binds).
- The 1-epoch endpoint reuses the pre-observed `gccomp1_stable` / `gc3b`
  checkpoints where available; all longer points are fresh.
- **Budget-calibration rule (fixed before 3B):** the 3B horizon is 8 epochs
  *unless* the 0.5B pilot shows the baseline's coverage@8 still rising at
  8 epochs, in which case both scales extend to 16 epochs. The horizon is
  chosen from the pilot and frozen before the 3B launch; no post-hoc
  extension of a running 3B arm will be reported as confirmatory.

**Primary outcome (fixed).** The **arm x log(compute) interaction on
coverage@8**, estimated by a prompt-clustered (CR1) linear model over all
evaluated checkpoints of both arms:
`coverage ~ arm + log_step + arm:log_step + train_seed FE + eval_seed FE`.
The primary coefficient is `arm:log_step` (the treatment effect's slope in
log-compute); positive with a 95% CI excluding 0 confirms the hypothesis.
Reported with CI and raw p-value (single primary, no multiplicity adjustment).

**Secondary (Holm-adjusted within family).**
- The same interaction on pass@8 and on conditional distinct-modes-per-solved.
- Each arm's *own* coverage@8 slope over the second half of the axis (predict
  Dr.GRPO slope <= 0 = plateau/collapse; xDr slope > 0).
- Baseline conditional distinct-modes-per-solved at first vs last checkpoint
  (predict a decrease toward 1.0).

**Guardrail (Holm-adjusted).** mean@8 and greedy pass@1 at the final long
checkpoint: xDr must be equivalent to Dr.GRPO within +/-3 points (TOST) or
better. A significant per-sample accuracy *deficit* at the long horizon is
reported as a cost of the extended schedule.

**Interpretation rules (fixed before launch).**
- Positive primary interaction + guardrail met: the headline result becomes
  "group-level exploration converts additional RL compute into answer-set
  coverage that the baseline cannot, and the advantage is a scaling effect,
  not a one-epoch artifact." This replaces the static single-point framing.
- Null interaction but positive, compute-stable level effect: reported as
  "the advantage is real but does not grow with compute" — weaker, honest.
- Baseline coverage@8 does **not** plateau/collapse (keeps rising in parallel
  with xDr): the divergence hypothesis is **not supported**; reported as such,
  and the paper retains only the level effect. No re-specification of the
  primary to rescue a null.

---

## E2 — Two-stage exploration -> exploitation (downstream-learning capstone)

**Hypothesis.** A policy that *explores* with xDr.GRPO reaches a higher final
performance under a subsequent phase of *standard* Dr.GRPO optimization than a
policy that explored with Dr.GRPO, because the candidate-distribution collapse
incurred during a Dr.GRPO exploration phase is not recovered by further
Dr.GRPO training — the later update has less within-group structure to learn
from. This is the direct instantiation of the paper's thesis sentence
("exploration ... affects how informative the resulting group-relative update
can be").

**Design.**
- Phase A (explore), budget B_A = 1 standard epoch: two arms, `xdr_tau0p05`
  and `grpo`.
- Phase B (exploit), budget B_B = 1 standard epoch: **both** Phase-A
  checkpoints continued with **plain Dr.GRPO (tau = inf)**, byte-identical
  recipe and seed handling; the only difference between the two Phase-B runs
  is which Phase-A checkpoint initialized them.
- Reference arms (already produced by E1, not re-run): "xDr all the way" and
  "Dr.GRPO all the way" for total budget B_A + B_B.
- Seeds 43, 44, 45; each Phase-A seed continues into the same Phase-B seed.

**Primary outcome (fixed).** Final (end of Phase B) **pass@8**, xDr-initialized
minus Dr.GRPO-initialized, prompt-clustered LPM with train-seed and eval-seed
FEs (2 x 3 x 3 x 256 = 4,608 obs, 256 clusters at 3B). CI + raw p-value.

**Secondary (Holm-adjusted).** Final greedy pass@1 and mean@8 (does the
exploration investment become a *hard per-sample accuracy* win after
exploitation?), final coverage@8, final conditional diversity.

**Interpretation rules (fixed).**
- Positive primary: an xDr exploration phase is a durable learning investment
  that standard RL cashes in — the strongest form of the "exploration helps
  learning" claim, and the intended new abstract headline.
- If greedy/mean@8 secondary is also positive: report the hard-accuracy win
  explicitly.
- Null primary: the Phase-A distribution advantage does not survive a Dr.GRPO
  exploitation phase; reported honestly as a boundary (exploration helps
  concurrently but is erased by greedy exploitation), which is itself an
  informative result about when the ordering of exploration matters.

---

## E3 — Raised-rollout-temperature control (rival-explanation rebuttal)

**Hypothesis (to be refuted).** xDr.GRPO's coverage gain is reproducible by
simply sampling rollouts hotter under an unchanged Dr.GRPO update.

**Design.** A Dr.GRPO arm with raised rollout temperature (T = 1.2 via
`OAT_ZERO_ZERO_TEMPERATURE`/rollout-temperature knob), matched budget and pool,
seeds 43-45, run at the standard 1 epoch and (if E1 confirms) at the long
horizon. Evaluation identical.

**Outcome / interpretation.** Reported alongside xDr at matched budget.
Predicted: raising rollout temperature increases surface variation but does not
reproduce xDr's coverage@8 / pass@8 gain at flat accuracy — it either fails to
move coverage as much or pays a mean@8 cost — because it perturbs the sampling
distribution rather than the credit assignment. If the temperature control
*does* reproduce the full xDr signature at flat accuracy, that is reported as a
serious threat to the mechanism claim and the paper is revised accordingly.
This control is a rebuttal, not a confirmatory arm; it carries no primary.

---

## Sequencing (fixed)

1. 0.5B pilot of E1 and E2 (hypothesis-generating; locates the collapse
   timescale and fixes the 3B horizon per the calibration rule above).
2. Commit this registration (final horizon filled in from the pilot).
3. 3B confirmatory E1, E2, E3.
4. Downstream-payoff evaluation on the final checkpoints (best-of-N with the
   exact verifier; self-consistency / majority-vote pass@1) — descriptive,
   reported to cash coverage into a selection metric; not a registered primary.

## Analysis conventions

Identical to the paper: `ops/analyze_countdown_comparative.py`, pooled and
per-domain, prompt-clustered CR1 SEs, run-clustered robustness pass, Holm
within outcome family for non-primary arms, final-checkpoint rule replaced here
by the full-axis evaluation for E1 (every saved checkpoint) and the
end-of-Phase-B checkpoint for E2.
