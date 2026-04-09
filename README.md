# Minimal OAT 1.5B Training Tree

This repository is intentionally cut down to one thing:

- 1.5B OAT Dr.GRPO
- 1.5B OAT Dr.GRPO-Explorer

The live code is under `src/oat_drgrpo/`:

- `train_zero_math.py`: the patched OAT learner
- `listwise.py`: the Explorer / DrX prompt-group helpers
- `math_grader.py`: the verifiable-math reward/grader

The only retained launch surface is:

- `ops/run_oat_zero_exact_1p5b_upstream.sh`
- `ops/run_oat_zero_explorer_1p5b_upstream.sh`
- `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm`
- `ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm`

Datasets used by those scripts live under:

- `datasets/train/math_12k`
- `datasets/evaluation_suite`

## Environment Setup

There are two supported ways to use this repo.

### 1. Canonical training runtime

The launchers are written around the canonical `paper310` environment:

- Python: `var/seed_paper_eval/paper310/bin/python`
- Source root: `src/`
- Trainer module: `oat_drgrpo.train_zero_math`

The exact launcher validates the key runtime versions before starting training:

- Python `3.10.x`
- `torch==2.6.0`
- `transformers==4.51.3`
- `vllm==0.8.4`
- `oat-llm==0.1.3.post1`
- `deepspeed==0.16.8`
- `math-verify==0.7.0`
- `fire==0.7.0`

The Slurm scripts default to that environment automatically. If you want to use a different interpreter, override:

```bash
export OAT_ZERO_PYTHON=/path/to/python
export OAT_ZERO_PYTHON_LIB_DIR=/path/to/python/lib
```

### 2. Local development environment

If you just want to edit code, import modules, and run smoke checks locally:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -e .
```

This gives you the repo package from `src/oat_drgrpo/`. For real training, you will still want the canonical runtime above unless you intentionally reproduce all training dependencies yourself.

### Repo environment variables

Before running ad hoc commands, source the repo environment helper:

```bash
source ops/repo_env.sh
```

This keeps caches and runtime state under `var/` instead of spilling into home-directory defaults:

- Hugging Face caches under `var/cache/huggingface`
- pip cache under `var/cache/pip`
- W&B state under `var/wandb`
- temporary files under `var/tmp`

## Data Layout

The kept launchers expect:

```text
datasets/
  train/
    math_12k/
  evaluation_suite/
```

The default training prompt column is `problem`, and the default evaluation answer column is `answer`.

## Running Jobs

Baseline Dr.GRPO:

```bash
sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm
```

Explorer / DrX:

```bash
sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm
```

The launcher logs the resolved config at startup, including:

- Python path
- source root
- prompt/eval datasets
- batch geometry
- exact explorer knobs such as `tau`, `candidate_kl_coef`, and `maxent_exact_drx_weight_source`

Logs are written to:

- `var/artifacts/logs/*.out`
- `var/artifacts/logs/*.err`

## Quick Checks

Minimal syntax/import checks:

```bash
source ops/repo_env.sh
python -m py_compile src/oat_drgrpo/train_zero_math.py src/oat_drgrpo/listwise.py src/oat_drgrpo/math_grader.py
bash -n ops/run_oat_zero_exact_1p5b_upstream.sh ops/run_oat_zero_explorer_1p5b_upstream.sh
```

If you want to sanity-check the exact launcher without submitting a long job, inspect the startup banner in the Slurm log. The exact launcher also bootstraps `flash-attn==2.7.4.post1` on-node when needed.
