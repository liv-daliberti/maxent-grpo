# Ops Scripts

This directory keeps the active Dr.GRPO vs Dr.X-GRPO launch and evaluation
surface small.

## Active Training

- `run_oat_zero_exact_1p5b_upstream.sh`
- `run_oat_zero_exact_drx_1p5b_upstream.sh`
- `run_oat_zero_tiny_probe.sh`
- `slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm`
- `slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_exact_drx_node302.slurm`
- `slurm/train_tiny_probe_node302.slurm`

## Tiny Probe

`run_oat_zero_tiny_probe.sh` runs the same OAT trainer on a tiny arithmetic
probe dataset under `var/data/tiny_math_probe`. It defaults to
`Qwen/Qwen2.5-0.5B-Instruct` with one local candidate group per SGD update.

Useful variants:

```bash
OAT_ZERO_TINY_VARIANT=grpo ops/run_oat_zero_tiny_probe.sh
OAT_ZERO_TINY_VARIANT=current_drx ops/run_oat_zero_tiny_probe.sh
OAT_ZERO_TINY_VARIANT=drx_aux ops/run_oat_zero_tiny_probe.sh
OAT_ZERO_TINY_VARIANT=drx_weighted ops/run_oat_zero_tiny_probe.sh
```

For an even smaller mechanics-only run:

```bash
OAT_ZERO_TINY_MODEL=smollm2-135m OAT_ZERO_TINY_VARIANT=current_drx ops/run_oat_zero_tiny_probe.sh
```

ModeBench uses verifier specs in the `answer` column instead of enumerating all
valid outputs. By default it builds calibrated synthetic graph-coloring and
Countdown rows: graph rows ask for the missing color digits in a partial
coloring, and Countdown rows require expressions that use each number exactly
once. The generators are deterministic given the seed, and the launchers
rebuild a data root automatically whenever its `dataset_dict.json` is missing.

## Active Evaluation

- `eval_exact_answer_mode_coverage.py`
- `eval_exact_answer_mode_pareto.py`
- `slurm/eval_answer_mode_coverage_node302.slurm`
- `slurm/eval_answer_mode_pareto_node302.slurm`

## Comparative (paper protocol)

- `submit_countdown_comparative.sh`: submits the full matched quartet —
  `grpo` (Dr.GRPO baseline, the xDr.GRPO tau=inf endpoint), `xdr_tau*`
  (xDr.GRPO candidate-level tempered aggregation, one arm per tau in
  `OAT_ZERO_XDR_TAUS`), `grpo_entropy` (Token-MaxEnt control), and `seed`
  (SEED-Dr.GRPO per-prompt semantic-entropy scaling,
  `OAT_ZERO_SEED_ENTROPY_ALPHA`) — with matched data, model, G, and budget.
  `OAT_ZERO_COMPARATIVE_TASK` selects countdown (default) or graph_coloring.
- `run_countdown_comparative_eval.sh`: per evaluation seed, K sampled
  completions per prompt plus one greedy pass over every arm's final
  checkpoint (run on a GPU node).
- `analyze_countdown_comparative.py`: builds prompt-level outcomes
  (pass@K, mean@K, distinct@K, coverage@K, greedy pass@1) and estimates the
  paper's linear probability model per treatment arm vs the baseline with
  train-seed and eval-seed fixed effects and prompt-clustered (CR1) standard
  errors.
