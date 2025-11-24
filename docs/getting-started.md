Overview
========

MaxEnt‑GRPO is a clean training stack for GRPO with optional maximum‑entropy weighting. It targets practical math training and evaluation while keeping code simple and production‑oriented.

Install
=======

- Python 3.10+ recommended
- GPU with recent CUDA when training
- A working vLLM server if you plan to run generation

Quickstart
==========

1) Create the local environment and launch training:

```bash
make conda-local && conda activate ./var/openr1
pip install -c configs/constraints.txt -e .[dev]
sbatch ops/slurm/train.slurm --model Qwen2.5-1.5B-Instruct --task grpo --config math --accelerator zero3
```

`ops/slurm/train.slurm` provisions the repo-local env via `configs/environment.yml`, configures caches under `./var/`, and dispatches training/inference processes. For a no-Slurm smoke test, use the Hydra console scripts instead:

```bash
# Baseline GRPO with inline overrides
maxent-grpo-baseline command=train-baseline training.output_dir=var/data/out

# MaxEnt-GRPO using a YAML recipe
GRPO_RECIPE=configs/recipes/Qwen2.5-1.5B-Instruct/maxent-grpo/config_math.yaml \
  maxent-grpo-maxent
```

2) Evaluate or generate with your model:

- Training and evaluation: see Guides → Training
- Bulk generation via vLLM: see Guides → Generation

What’s Inside
=============

- `src/maxent_grpo/grpo.py`: Minimal GRPO training entrypoint
- `src/maxent_grpo/config/`: Dataclasses for all runtime configuration
- `src/rewards.py`: Reward functions and registry
- `src/generate.py`: Distilabel pipeline + CLI for batch generation
- `configs/recipes/…`: Example YAML recipes

Quick Links
===========

- [Training Guide](guides/training.html) — launch training, shape rewards, configure datasets.
- [Generation Guide](guides/generation.html) — vLLM + Distilabel for batch inference and dataset creation.
- [Evaluation](guides/evaluation.html) — run LightEval benchmarks with vLLM and Slurm helpers.
- [API Reference](api.html) — browse modules and configuration dataclasses.
