# PantryPlan support-retention repair v1

**Status: FROZEN BEFORE DATA AUDIT AND BEFORE MODEL SAMPLING — 2026-07-30**

## Scope

This is a secondary repair campaign motivated by the already-observed decline
in distinct@8 in the original PantryPlan five-seed cohort. It cannot alter the
original run identities, audit, graph, or terminal interpretation.

## Fresh information boundary

Materialize PantryPlan with generator seed 74103, 96 train, 16 development,
and 32 evaluation instances per family. The resulting train, development, and
evaluation fingerprints must be pairwise disjoint and must also be disjoint
from every PantryPlan v2 split. Hyperparameter selection may inspect only the
new development split. The new evaluation split remains sealed until this
protocol's final dose and five-seed launcher are frozen.

## Development pair

- Initial model: Qwen2.5-0.5B-Instruct.
- Arms: compute-matched Dr.GRPO and verified-first global replay MaxEnt.
- Seed: 76401; development-only and not a final seed.
- Training: the first 32 train prompts, three complete passes, exactly 96
  optimizer updates, 16 rollouts per prompt.
- Evaluation: the 64-instance fresh development split at pass 0 and every
  eight updates, four deterministic temperature-one K=8 draws.
- Optimizer learning rate: 2e-7 in both arms.
- Treatment: novelty beta 0.50 and split mass/balance replay alpha 0.20.
- Control: identical bank construction, scoring, replay scheduling, forward
  passes, and backward graph, with exploration and replay derivatives set to
  exact zero.
- No token entropy, gold support, target mode count, coefficient projection,
  checkpoint selection, resume, or evaluation feedback to training.

## Qualification rule

The pair must complete the exact schedule with matched compute traversal,
finite telemetry, exact-zero control derivatives, and nonzero treatment
novelty/replay derivatives. At the terminal development coordinate:

1. treatment distinct@8 must exceed control distinct@8;
2. treatment pass@8 must be no lower than control pass@8;
3. treatment mean@8 must be at least 90% of control mean@8; and
4. treatment distinct@8 must retain at least 60% of its own pass-0 value.

Failure stops this dose. No result-dependent extension or evaluation-set query
is authorized.

