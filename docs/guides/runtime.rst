Runtime Setup
=============

This page summarizes the canonical runtime for the active OAT training path.

Canonical OAT Runtime
---------------------

The working environment is the repo-local ``paper310`` runtime used by the
README-flash OAT launchers:

- ``python==3.10.20``
- ``torch==2.6.0``
- ``transformers==4.51.3``
- ``vllm==0.8.4``
- ``oat-llm==0.1.3.post1``
- ``deepspeed==0.16.8``
- ``flash-attn==2.7.4.post1`` via the launch-time overlay

The canonical interpreter is:

- ``var/seed_paper_eval/paper310/bin/python``

Validate the runtime with:

.. code-block:: bash

   python tools/audit_oat_setup.py

Why This Matters
----------------

The upstream OAT training path proved sensitive to version drift. This
repository now keeps one canonical runtime for the active launchers so the
baseline DR.GRPO path and the listwise explorer overlay share the same working
stack.

Active Launchers
----------------

- ``ops/run_oat_zero_exact_1p5b_upstream.sh``
- ``ops/run_oat_zero_explorer_1p5b_upstream.sh``
- ``ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_node302.slurm``
- ``ops/slurm/train_understand_r1_zero_qwen2p5_math_1p5b_r1_readme_flash_explorer_node302.slurm``

Environment Notes
-----------------

- Flash attention is installed at launch time into a local overlay, rather than
  being assumed to exist globally.
- The OAT launchers route caches and temporary files into ``var/`` or
  node-local scratch instead of relying on ambient home-directory state.
- The explorer path reuses the same runtime and only switches the learner
  objective to ``maxent_listwise``.

Archived Runtime Surface
------------------------

Older TRL/Hydra launchers and other retired training wrappers are kept under
``archive/trl/``. They are not part of the active runtime contract anymore.
