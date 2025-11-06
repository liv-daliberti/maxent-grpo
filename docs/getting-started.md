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
sbatch training.sh
```

The script provisions a local conda env under the repo, installs deps from `environment.yml`, and launches GRPO. It includes an SBATCH header (account/partition/GRES) you may need to adapt for your cluster. For local smoke tests, you can also run `bash training.sh`.

2) Evaluate or generate with your model:

- Training and evaluation: see Guides → Training
- Bulk generation via vLLM: see Guides → Generation

What’s Inside
=============

- `src/grpo.py`: Minimal GRPO training entrypoint
- `src/configs.py`: Dataclasses for all runtime configuration
- `src/rewards.py`: Reward functions and registry
- `src/generate.py`: Distilabel pipeline + CLI for batch generation
- `recipes/…`: Example YAML recipes

Quick Links
===========

:::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} Training Guide
:link: guides/training.html
How to launch training, shape rewards, and configure datasets.
:::

:::{grid-item-card} Generation Guide
:link: guides/generation.html
Use vLLM + Distilabel for batch inference and dataset creation.
:::

:::{grid-item-card} Evaluation
:link: guides/evaluation.html
Run LightEval benchmarks with vLLM and Slurm helpers.
:::

:::{grid-item-card} API Reference
:link: api.html
Browse modules and configuration dataclasses.
:::

:::
