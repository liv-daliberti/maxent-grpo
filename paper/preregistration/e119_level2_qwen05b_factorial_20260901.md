# E119: matched Level-2 Qwen-0.5B Dr.GRPO × MaxRL × replay factorial

Date frozen: 2026-09-01, after the second independent frozen-base admission
run and before any E119 training submission or treatment outcome.

## Question

On the admitted matched Level-2 benchmark, what are the separate and joint
effects of the finite-rollout binary MaxRL estimator and verified canonical
replay on correctness and retained semantic support?

## Design

Train Qwen2.5-0.5B-Instruct on all five Level-2 r5 domains under a complete
two-by-two factorial:

- `drgrpo`: ordinary Dr.GRPO task estimator; compute-matched replay traversal
  with an exactly zero replay derivative.
- `replay_drgrpo`: ordinary Dr.GRPO plus the established
  `verified_likelihood_per_rollout` derivative at alpha 0.10.
- `maxrl`: binary MaxRL task estimator; compute-matched replay traversal with
  an exactly zero replay derivative.
- `replay_maxrl`: binary MaxRL plus the same verified replay derivative.

The cohort is 5 domains × 4 arms × 5 paired seeds (43--47) = 100 new runs.
No Level-1 endpoint is reused as a Level-2 treatment comparator.

## Frozen benchmark and interface

- Data root: `var/data/modebench_harder_v2_matched_r5`.
- Identity SHA-256:
  `2af80f4d31a44482574b314cc37ef84bde48c53ecc2ff78dadf97571f0d73fb2`.
- Training uses each domain's 384-row `train` split; evaluation uses its
  untouched 128-row `eval` split. Development rows are never loaded by E119.
- Submission requires the final and independent-repeat admission reports to
  admit every domain under both frozen base models.
- Qwen prompt profiles and target-blind legal syntax match the admitted
  Qwen-0.5B interface: boxed-direct/no grammar for Graph Coloring;
  hybrid/domain-legal-v1 for Python Factors, MathIR, and PantryPlan; and
  hybrid/countdown-legal-v3 for Countdown.
- The Countdown grammar depends only on the supplied operands, never the
  target or verifier outcome. Other grammars constrain syntax, not correctness.
- The identical prompt, grammar, response budget, and verifier apply to every
  arm within a domain and seed.

## E78/E118-matched schedule

- Frozen Qwen2.5-0.5B-Instruct checkpoint used by E78/E118.
- 384 training prompts, exactly 8 passes, 3,072 optimizer updates.
- Group size 16, one PPO epoch, learning rate 2e-7, beta_KL 0,
  temperature 1, top-p 1.
- Evaluation and resumable checkpoints every 192 updates, yielding the fixed
  pass grid 0, 0.5, 1.0, ..., 8.0.
- Sampled evaluation uses pass@8/mode coverage with four registered draws.
- Replay capacity 16 and one deterministic global replay group per update.
- Semantic-MaxEnt, canonical-bank entropy, token entropy, xDr, DAPO, RLEP,
  DIAYN, and counterfactual proposal objectives are disabled in all arms.

## Estimands

Within each domain and seed:

1. `replay_drgrpo - drgrpo`: replay effect under Dr.GRPO.
2. `maxrl - drgrpo`: task-estimator effect without replay.
3. `replay_maxrl - maxrl`: replay effect under MaxRL.
4. Difference in differences: replay/MaxRL interaction.

Primary terminal endpoints are sampled pass@8 and distinct correct modes@8.
Secondary endpoints are greedy accuracy, sampled mean correctness@8, excess
multiplicity, retained discovered support, and complete pass-0--8 trajectory
AUC. Report all five paired seeds per domain; do not select checkpoints or
pool domains into one confirmatory effect.

## Integrity and stopping

All 100 jobs are submitted held from one hash-bound source/ops snapshot. They
are released only after exact scheduler-environment audit and immutable-ledger
publication. Existing run directories, malformed data identities, non-admitted
reports, objective leakage, missing syntax profiles, non-finite losses, or
missing pass-8 endpoints fail closed. Infrastructure interruption may resume
only the same scientific cell from its exact checkpoint and replay state.

