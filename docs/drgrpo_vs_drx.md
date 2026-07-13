# Dr.GRPO vs Dr.X-GRPO

This is the clean comparison surface for the repo.

## Methods

Dr.GRPO baseline:

- objective: `grpo`
- launcher: `ops/run_oat_zero_exact_1p5b_upstream.sh`
- node302 Slurm:
  `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm`

Dr.X-GRPO:

- objective: `maxent_listwise`
- primary update: token-level Dr.GRPO PPO ratio with centered Dr.X rewards
- token advantage: `A_i = R_i - mean_j R_j`
- Dr.X reward:
  `R_i = correctness(q, o_i) + lambda * (1 - p_correct) * h_i`
- entropy gate:
  `1 - p_correct`, so output-entropy exploration turns off for all-correct
  groups
- output entropy score: `h_i` is the sampled output's per-token old-policy
  self-information, centered within the prompt group and normalized to `[-1, 1]`
- token normalization: OAT Dr.GRPO constant max-length normalizer
- token loss denominator: all valid rows in the minibatch, matching OAT Dr.GRPO;
  zero-advantage rows contribute zero numerator but remain in the mean
- zero-signal cadence: valid zero-advantage batches still run a zero-loss
  backward/optimizer step, matching Dr.GRPO optimizer cadence
- extra listwise clip objective: off
- Dr.X projection/sequence auxiliary objective: on in the fixed MaxEnt-test
  recipe with `maxent_sequence_aux_coef=0.05`; off only for the legacy control
- semantic remix: `anchor_rare`
- explicit length penalties: removed from the Dr.X utility
- launcher: `ops/run_oat_zero_exact_drx_1p5b_upstream.sh`
- node302 Slurm:
  `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_exact_drx_node302.slurm`

## Launch

Baseline:

```bash
sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm
```

Dr.X-GRPO:

```bash
sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_exact_drx_node302.slurm
```

The Dr.X-GRPO launcher should resolve these values in the startup log:

```text
maxent_drgrpo_token_primary=1
maxent_drgrpo_token_advantage_source=utility_centered
maxent_drgrpo_token_length_normalizer=max_length
maxent_exact_drx_weight_source=sequence_clipped
maxent_use_clip_objective=0
maxent_sequence_aux_coef=0.05
semantic_entropy_lambda=0.10
maxent_semantic_remix_mode=anchor_rare
```

## Launcher Contract

The public launchers are:

- `ops/run_oat_zero_exact_1p5b_upstream.sh`
- `ops/run_oat_zero_exact_drx_1p5b_upstream.sh`

Both resolve the trainer from `OAT_ZERO_TRAINER_MODULE`, defaulting to
`oat_drgrpo.train_zero_math`, and the source tree from `OAT_ZERO_SOURCE_ROOT`,
defaulting to `src/`. They validate the canonical `paper310` runtime before
launching unless you explicitly point `OAT_ZERO_PYTHON` and
`OAT_ZERO_PYTHON_LIB_DIR` at another compatible environment.

Required data defaults (ModeBench; both datasets are generated
deterministically on first use, and `OAT_ZERO_TASK` selects the domain):

```text
OAT_ZERO_TASK=countdown            # or graph_coloring
OAT_ZERO_PROMPT_DATA=var/data/exact_countdown_mode_probe/train   # derived from OAT_ZERO_TASK
OAT_ZERO_EVAL_DATA=var/data/exact_countdown_mode_probe/eval      # derived from OAT_ZERO_TASK
OAT_ZERO_INPUT_KEY=problem
OAT_ZERO_OUTPUT_KEY=answer
```

Useful runtime overrides:

```text
SAVE_PATH
OAT_ZERO_WB_PROJECT
OAT_ZERO_WB_RUN_NAME
OAT_ZERO_USE_WB
OAT_ZERO_LOCAL_ROOT
OAT_ZERO_RESUME_DIR
OAT_ZERO_RESUME_TAG
```

## Semantics

Dr.X-GRPO keeps a pure MaxEnt-style output entropy signal without a separate
length-tempering layer. For each sampled output:

```text
raw_entropy_i = -mean_t log pi_old(o_{i,t} | q, o_{i,<t})
h_i = normalize_j(raw_entropy_i - mean_j raw_entropy_j)
```

The entropy term is gated by the group's correctness rate:

- mostly wrong groups: `g(p_correct) > 0`, so above-average output entropy
  receives positive utility
- mixed groups: correctness still dominates while above-average output entropy
  receives a small exploration tilt
- all-correct groups: `g(p_correct) = 0`, so the update stops rewarding
  exploration once the prompt group is solved

The centered reward expands linearly:

```text
A_i = R_i - mean_j R_j
    = correctness_i - mean_j correctness_j
      + lambda * (1 - p_correct) *
        (h_i - mean_j h_j)
```

The public Dr.X launcher disables the extra listwise clip objective but enables
the Dr.X projection/sequence auxiliary objective at a small coefficient. In the
fixed recipe, that auxiliary objective projects the active MaxEnt candidate
target for informative groups and uses the neutral projection target only for
neutral groups. The primary token update still uses the same constant max-length
normalization as OAT Dr.GRPO; any length behavior should come from the
data/rewards, not an accidental response-length rescaling or explicit utility
penalty. When centered Dr.X rewards produce zero token signal but valid
denominator rows, Dr.X still performs a zero-loss backward/optimizer step so the
optimizer cadence matches Dr.GRPO.

To reproduce the legacy failed recipe, override:

```bash
OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF=0.0 \
  sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_exact_drx_node302.slurm
```

## Evaluation

Evaluate trained checkpoints with the answer-mode coverage tooling:

```bash
RUN_STAMP_PREFIX=<stamp> ops/run_countdown_comparative_eval.sh   # on a GPU node
var/seed_paper_eval/paper310/bin/python ops/analyze_countdown_comparative.py \
  --output-root var/artifacts --stamp-prefix <stamp>
```

This reports pass@K, mean@K, distinct@K, and coverage@K per arm, plus the
prompt-clustered regression of each treatment arm against the Dr.GRPO
baseline.

## W&B Metrics

W&B keeps the public comparison view focused on loss/reward, eval, tau/beta,
clip, semantic cluster, and Dr.X utility metrics. Legacy objective metrics and
high-cardinality debug diagnostics are filtered from W&B but still appear in
local logs. Set `OAT_ZERO_WANDB_LOG_DEBUG_METRICS=1` to send the full debug
metric set to W&B for a diagnostic run.
