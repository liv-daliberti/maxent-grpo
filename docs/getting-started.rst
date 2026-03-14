Overview
========

MaxEnt‑GRPO is a clean training stack for GRPO with optional maximum‑entropy weighting and a GRPO + entropy‑bonus mode. It targets practical math training and evaluation while keeping code simple and production‑oriented.

Install
=======

- Python 3.10+ recommended
- GPU with recent CUDA when training
- A working vLLM server if your training recipe uses server-side rollouts

Quickstart
==========

1) Create the local environment and launch training:

.. code-block:: bash

   make conda-local && conda activate ./var/openr1
   pip install -c configs/constraints.txt -e .[dev]
   sbatch ops/slurm/train_dual_4plus4.slurm --config math --accelerator zero3 --run-only both

``ops/slurm/train_dual_4plus4.slurm`` provisions runtime caches under ``./var/`` and dispatches the experiment-profile stacks: GRPO + entropy-MaxEnt together, or a single stack via ``--run-only grpo|maxent|listwise``. For the full three-way comparison, use ``ops/run_experiment_triplet_single_node.sh``. For a no-Slurm smoke test, use the Hydra console scripts instead:

.. code-block:: bash

   # Baseline GRPO with inline overrides
   maxent-grpo-baseline command=train-baseline training.output_dir=var/data/out

   # MaxEnt-GRPO using a YAML recipe
   maxent-grpo command=train-maxent \
     maxent.recipe=configs/recipes/Qwen2.5-1.5B-Instruct/maxent-grpo/config_math.yaml

2) Inspect training setup and launch details in Guides → Training.

The Hydra convenience MaxEnt presets can enable the τ/β meta-controller
(analytic mode) so weighting parameters track entropy/KL targets
automatically. Check the recipe and disable it via
``--controller_meta_enabled false`` (or set the YAML field) when you need
fixed hyperparameters for ablations; re-enable and tune with
``controller_meta_method``, ``controller_meta_lr``, and friends when you want
the learned controller back. The paired flat GRPO/MaxEnt recipes keep the
controller off unless you override the flag.

What’s Inside
=============

- ``src/maxent_grpo/grpo.py``: Minimal GRPO training entrypoint
- ``src/maxent_grpo/config/``: Dataclasses for all runtime configuration
- ``src/rewards.py``: Reward functions and registry
- ``configs/recipes/…``: Example YAML recipes

Quick Links
===========

- `Training Guide <guides/training>`_ — launch training, shape rewards, configure datasets.
- `Evaluation <guides/evaluation>`_ — run LightEval benchmarks with vLLM and Slurm helpers.
- `API Reference <api>`_ — browse modules and configuration dataclasses.
