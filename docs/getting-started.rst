Overview
========

The canonical training path in this repository is now the upstream OAT
README-flash stack plus a local listwise maxent-explorer overlay.

Active baseline launcher:

- ``ops/run_oat_zero_exact_1p5b_upstream.sh``
- ``ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm``

Active explorer launcher:

- ``ops/run_oat_zero_explorer_1p5b_upstream.sh``
- ``ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm``

Retired TRL/Hydra orchestration and older noncanonical launchers are archived
under ``archive/trl/``.

Canonical Runtime
=================

The working runtime is the repo-local ``paper310`` environment:

- ``python==3.10.20``
- ``torch==2.6.0``
- ``transformers==4.51.3``
- ``vllm==0.8.4``
- ``oat-llm==0.1.3.post1``
- ``deepspeed==0.16.8``
- ``flash-attn==2.7.4.post1`` via the launch-time overlay

Validate it before training:

.. code-block:: bash

   python tools/audit_oat_setup.py

Quickstart
==========

1. Launch the canonical baseline:

.. code-block:: bash

   sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm

2. Launch the listwise maxent-explorer variant on the same stack:

.. code-block:: bash

   sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm

3. For local shell launches instead of Slurm:

.. code-block:: bash

   bash ops/run_oat_zero_exact_1p5b_upstream.sh
   bash ops/run_oat_zero_explorer_1p5b_upstream.sh

What Is Archived
================

- ``archive/trl/ops/`` keeps retired orchestration wrappers and experiment launchers.
- ``archive/trl/ops/slurm/`` keeps retired Slurm entrypoints.
- ``src/maxent_grpo/`` remains in the repo for reference and historical work, but it is not the canonical training front door anymore.

Quick Links
===========

- `OAT Upstream DR.GRPO <guides/oat-upstream-drgrpo>`_ - exact working stack and explorer overlay.
- `Training Guide <guides/training>`_ - canonical launch flow plus archive notes.
- `Runtime <guides/runtime>`_ - pinned runtime and validation checks.
