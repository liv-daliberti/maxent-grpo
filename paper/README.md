# Mode Collapse under GRPO

[`main.tex`](main.tex) is the ICLR 2026-format source for **Mode Collapse under
GRPO: ModeBench and x-Mode GRPO**. [`main.pdf`](main.pdf) is the built
manuscript.

## Paper in one paragraph

A binary verifier distinguishes correct from incorrect responses but does not
distinguish redundant correct responses from genuinely different correct
outcomes. Under Dr.GRPO, frequent correct outcomes receive more on-policy
updates while a correct outcome that disappears from the sampled group receives
none. ModeBench makes this loss of verified outcome support measurable with
validators that both execute a response and assign a canonical outcome key.
x-Mode GRPO augments Dr.GRPO with verified replay: it stores one policy-generated,
validator-positive exemplar for each observed key and revisits the bank with a
deterministic recurrent schedule and uniform teacher-forced likelihood.

## What is measured

For prompt `x`, validator-positive response `y` receives an executable outcome
key `kappa_x(y)`. The target distribution is the policy distribution over keys
conditioned on verifier success, not token strings.

- `pass@1`: greedy success probability.
- `pass@8`: probability that eight samples contain at least one correct answer.
- `distinct@8`: expected number of distinct correct executable keys in eight
  samples.

The primary collapse diagnosis compares `distinct@8` with `pass@8`: near
equality means successful eight-sample sets almost never contain a second
correct mode. A falling `distinct@8` alongside stable correctness shows support
contraction. Token entropy alone is not evidence of outcome diversity because
formatting aliases can map to the same executed key.

## Theory guarantee

Appendix A proves that expected binary-reward GRPO and Dr.GRPO flow is
generically winner-take-all over equally rewarded correct modes. It then shows
that every banked mode receiving a recurrent positive replay dose is protected
from extinction. Once the verified bank contains the full correct support,
uniform replay has the uniform correct-mode distribution as its replay optimum.
The guarantee is therefore explicitly post-discovery and applies only to modes
the policy has produced, the validator has accepted, and the bank has retained.

## ModeBench

The five benchmark domains are:

- Graph coloring: the complete executed coloring assignment.
- Countdown: the normalized, executed arithmetic tree.
- Executable Python factors: the function's return vector on hidden cases.
- MathIR: the exact equation-state trajectory produced by an executed action
  menu.
- PantryPlan: the ingredient support of a constraint-feasible plan.

Every admitted key is both correct and executable. There is no gold catalogue
of modes and no clustering of free-form text.

## Evidence map

| Evidence block | Status | Claim licensed |
|---|---|---|
| Qwen2.5-0.5B x-Mode GRPO versus matched Dr.GRPO | Interim; running | Dated descriptive trajectories only; final pass-8 effect withheld |
| Falcon3-1B and Qwen2.5-3B aligned comparisons | Interim; running | Currently paired within-model trajectories only; no pooling or cross-model effect |
| PointMaze extensions | Interim; running | Separate interactive stratum; no pooling with the five static domains |
| Earlier broad comparison | Historical multi-component treatment | Motivation and provenance only; not an estimate of x-Mode GRPO |
| Earlier rehearsal comparison | Historical component evidence | Motivation for replay; not evidence for any coefficient rule |

The clean comparison uses Qwen2.5-0.5B-Instruct, seeds 43--47, exactly eight
passes, and checkpoints every half pass. Both arms perform the same bank
maintenance, recurrent traversal, exemplar scoring, and backward pass. The
control applies an exact-zero replay derivative; the x-Mode GRPO arm applies the
fixed replay weight `.10`. Pass 8 is the sole primary endpoint, and intermediate
checkpoints are trajectory measurements rather than model-selection candidates.
The aligned Falcon3-1B and Qwen2.5-3B comparisons preserve the same replay
intervention under model-specific optimizer recipes fixed before replay-arm
outcomes. The dated August 6 snapshot is frozen: missing
cells are not imputed, paired means use only seeds available in both arms at a
checkpoint, and the snapshot changes no stopping rule or endpoint. Training
continues after that cutoff.

## Reproduce the paper

From the repository root:

```bash
python ops/plot_paper_modebench_examples.py
python ops/plot_paper_verified_replay_mechanism.py
python ops/plot_paper_modecollapse.py
python ops/plot_paper_collapse_toy.py
make -C paper
```

The plot scripts regenerate:

- [`figures/verified_replay_mechanism.pdf`](figures/verified_replay_mechanism.pdf):
  the verify, store, recurrent revisit, and uniform replay mechanism;
- [`figures/modebench_examples.pdf`](figures/modebench_examples.pdf): one
  execution-checked response/verify/key row per benchmark domain;
- [`figures/modecollapse_story.pdf`](figures/modecollapse_story.pdf): the
  model-backed Graph Coloring collapse example from the historical multi-component treatment;
- [`figures/modecollapse_training_compact.pdf`](figures/modecollapse_training_compact.pdf):
  the compact historical training surface retained for provenance;
- [`figures/figure4_interim_20260806.pdf`](figures/figure4_interim_20260806.pdf):
  the frozen, incomplete three-model trajectory snapshot used by the paper;
- the standalone training and decoding figures described in the manuscript
  appendix.

Current clean endpoints will replace the historical treatment surface after all
registered cells reach pass 8.

## Source-of-truth artifacts

- Clean x-Mode GRPO protocol:
  [`preregistration/e78_verified_replay_only_05b_20260804.md`](preregistration/e78_verified_replay_only_05b_20260804.md)
- Submitted clean-comparison jobs:
  [`../var/artifacts/e78_verified_replay_only_05b_jobs.json`](../var/artifacts/e78_verified_replay_only_05b_jobs.json)
- Frozen interim Figure 4 provenance:
  [`figures/figure4_interim_20260806.json`](figures/figure4_interim_20260806.json)
- Frozen interim four-metric table provenance:
  [`results/figure4_interim_20260806_table.json`](results/figure4_interim_20260806_table.json),
  generated by [`../ops/exp_scaling/build_figure4_interim_table.py`](../ops/exp_scaling/build_figure4_interim_table.py)
- Live figure generator: [`../ops/exp_scaling/plot_figure4_with_falcon_preview.py`](../ops/exp_scaling/plot_figure4_with_falcon_preview.py)
- Opening model trajectory:
  [`../var/artifacts/paper_graph_collapse_toy.json`](../var/artifacts/paper_graph_collapse_toy.json)
- Historical replay protocol retained for provenance:
  [`preregistration/e58_global_verified_replay_canonical_05b.md`](preregistration/e58_global_verified_replay_canonical_05b.md)

## Implementation map

- Environments and executable keys:
  [`../src/oat_drgrpo/canonical_actions.py`](../src/oat_drgrpo/canonical_actions.py),
  [`../src/oat_drgrpo/python_modebench.py`](../src/oat_drgrpo/python_modebench.py),
  [`../src/oat_drgrpo/mathir.py`](../src/oat_drgrpo/mathir.py), and
  [`../src/oat_drgrpo/pantry_plan.py`](../src/oat_drgrpo/pantry_plan.py)
- External Python execution boundary:
  [`../src/oat_drgrpo/python_modebench_process.py`](../src/oat_drgrpo/python_modebench_process.py)
- Verified replay loss and online bank:
  [`../src/oat_drgrpo/canonical_replay.py`](../src/oat_drgrpo/canonical_replay.py)
  and [`../src/oat_drgrpo/online_canonical_bank.py`](../src/oat_drgrpo/online_canonical_bank.py)
- Clean experiment launcher:
  [`../ops/exp_scaling/launch_e78_verified_replay_only_05b.py`](../ops/exp_scaling/launch_e78_verified_replay_only_05b.py)

## Claim boundary

Verified replay is a retention mechanism, not a discovery oracle: it cannot
protect a mode before the policy produces it and the validator accepts it. The
running comparison is the first five-domain x-Mode GRPO estimate. Historical
results from broader objectives remain auditable but are not attributed to x-Mode GRPO.
