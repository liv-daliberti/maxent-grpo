Training
========

The canonical training path in this repository is now the upstream OAT
README-flash stack. The only active training launchers under ``ops/`` are the
baseline DR.GRPO path and the listwise maxent-explorer overlay on top of that
same stack.

Canonical Launchers
-------------------

Baseline DR.GRPO:

- ``ops/run_oat_zero_exact_1p5b_upstream.sh``
- ``ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm``

Listwise maxent-explorer:

- ``ops/run_oat_zero_explorer_1p5b_upstream.sh``
- ``ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm``

Use the baseline wrapper for the exact README-flash OAT setup:

.. code-block:: bash

   sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm

Use the explorer wrapper for the listwise overlay on the same runtime:

.. code-block:: bash

   sbatch ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm

Shared Runtime
--------------

Both launchers share the same runtime and bootstrap path:

- upstream checkout: ``understand-r1-zero/``
- local python env: ``var/seed_paper_eval/paper310``
- launch-time flash-attn overlay
- local extension module: ``oat_zero_ext/listwise.py``
- patched learner: ``understand-r1-zero/train_zero_math.py``

Before launching, validate the runtime:

.. code-block:: bash

   python tools/audit_oat_setup.py

Objective Routing
-----------------

The baseline path stays on native OAT DR.GRPO.

- ``objective=grpo``
- ``critic_type=drgrpo``
- ``prompt_template=r1``

The explorer path is opt-in and only changes the learner objective:

- ``objective=maxent_listwise``
- ``beta=0.0`` by default; raise it only if you want the reference-weight term
  active
- ``maxent_tau=0.5``
- ``maxent_q_temperature=2.0``
- ``train_batch_size_per_device=8`` so prompt groups stay intact

The shared launcher validates listwise-only constraints before training starts,
and the learner validates them again inside the OAT process. See
:doc:`oat-upstream-drgrpo` for the full set of guardrails.

What Changed
------------

The active training surface under ``ops/`` has been reduced to the working OAT
stack only. Older experiment orchestration has been retired from the active
surface and moved under ``archive/trl/``.

Archived material includes:

- older TRL/Hydra orchestration wrappers
- retired Slurm launchers for those wrappers
- pre-canonical pure OAT launchers that used older layouts or assumptions

Archive Location
----------------

- ``archive/trl/ops/``
- ``archive/trl/ops/slurm/``

Those files are preserved for reference, but they are no longer the canonical
way to launch training from this repository.
