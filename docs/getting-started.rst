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

.. code-block:: bash

   make conda-local && conda activate ./var/openr1
   pip install -c configs/constraints.txt -e .[dev]
   sbatch ops/slurm/train.slurm --model Qwen2.5-1.5B-Instruct --task grpo --config math --accelerator zero3

``ops/slurm/train.slurm`` provisions the repo-local env via ``configs/environment.yml``, configures caches under ``./var/``, and dispatches training/inference processes. For a no-Slurm smoke test, use the Hydra console scripts instead:

.. code-block:: bash

   # Baseline GRPO with inline overrides
   maxent-grpo-baseline command=train-baseline training.output_dir=var/data/out

   # MaxEnt-GRPO using a YAML recipe
   GRPO_RECIPE=configs/recipes/Qwen2.5-1.5B-Instruct/maxent-grpo/config_math.yaml \
     maxent-grpo-maxent

   # InfoSeed (adds seed-conditioned data augment + auxiliary loss)
   GRPO_RECIPE=configs/recipes/Qwen2.5-1.5B-Instruct/infoseed/config_math.yaml \
     maxent-grpo-infoseed

2) Evaluate or generate with your model:

- Training and evaluation: see Guides → Training
- Bulk generation via vLLM: see Guides → Generation

Some MaxEnt/InfoSeed recipes enable the τ/β meta-controller (analytic mode) so weighting parameters track entropy/KL targets automatically. Check the recipe and disable it via ``--controller_meta_enabled false`` (or set the YAML field) when you need fixed hyperparameters for ablations; re-enable and tune with ``controller_meta_method``, ``controller_meta_lr``, and friends when you want the learned controller back. Baseline/paired GRPO recipes keep the controller off unless you override the flag.

What’s Inside
=============

- ``src/maxent_grpo/grpo.py``: Minimal GRPO training entrypoint
- ``src/maxent_grpo/config/``: Dataclasses for all runtime configuration
- ``src/rewards.py``: Reward functions and registry
- ``src/generate.py``: Distilabel pipeline + CLI for batch generation
- ``configs/recipes/…``: Example YAML recipes

Quick Links
===========

- `Training Guide <guides/training>`_ — launch training, shape rewards, configure datasets.
- `Generation Guide <guides/generation>`_ — vLLM + Distilabel for batch inference and dataset creation.
- `Evaluation <guides/evaluation>`_ — run LightEval benchmarks with vLLM and Slurm helpers.
- `API Reference <api>`_ — browse modules and configuration dataclasses.
