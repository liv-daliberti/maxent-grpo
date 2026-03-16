Method Identity
===============

This repo now treats method selection as **two explicit axes** instead of one
overloaded knob:

- **Algorithm family**: controlled by ``training.objective`` plus
  ``training.seed_grpo_enabled``.
- **Loss backend**: controlled by ``training.grpo_loss_type``.

That separation matters because ``objective: grpo`` does **not** tell you
whether a run is plain GRPO, BNPO-style GRPO, Dr.GRPO, or SEED-GRPO on top of
Dr.GRPO.

Current 1.5B Math Presets
-------------------------

.. list-table::
   :header-rows: 1

   * - Preset
     - Family
     - ``objective``
     - ``seed_grpo_enabled``
     - ``grpo_loss_type``
     - Canonical label
   * - ``configs/recipes/hydra/grpo_custom_math.yaml``
     - Baseline GRPO
     - ``grpo``
     - ``false``
     - ``dr_grpo``
     - ``Dr.GRPO``
   * - ``configs/recipes/hydra/maxent_entropy_math.yaml``
     - Entropy MaxEnt
     - ``maxent_entropy``
     - ``false``
     - ``dr_grpo``
     - ``Entropy MaxEnt (Dr.GRPO loss)``
   * - ``configs/recipes/hydra/maxent_listwise_math.yaml``
     - Listwise MaxEnt
     - ``maxent_listwise``
     - ``false``
     - ``dr_grpo``
     - ``Listwise MaxEnt (Dr.GRPO loss)``
   * - ``configs/recipes/hydra/seed_grpo_math.yaml``
     - SEED-GRPO
     - ``grpo``
     - ``true``
     - ``dr_grpo``
     - ``SEED-GRPO (Dr.GRPO loss)``

The filename ``grpo_custom_math.yaml`` is historical. In the current 1.5B math
setup, it is the baseline **Dr.GRPO** preset because it pins
``grpo_loss_type: dr_grpo``.

Source of Truth
---------------

- ``src/maxent_grpo/objectives.py``: normalizes the top-level objective family.
- ``src/maxent_grpo/methods.py``: resolves the final method identity from
  family + backend.
- ``src/maxent_grpo/config/grpo.py``: validates and normalizes
  ``grpo_loss_type`` and the family-selection flags.
- ``src/maxent_grpo/training/trl_trainer.py``: logs the resolved method at
  trainer startup.
- ``src/maxent_grpo/training/runtime/logging.py``: writes
  ``run/method_name``, ``run/method_family``, ``run/method_backend``, and
  ``run/method_slug`` into run metadata/W&B config.

Family-Specific Code
--------------------

- **Baseline GRPO / Dr.GRPO backend**:
  ``src/maxent_grpo/training/trl_trainer.py``
- **SEED-GRPO advantage scaling**:
  ``src/maxent_grpo/training/rewards.py``
- **Entropy MaxEnt objective**:
  ``src/maxent_grpo/training/trl_trainer.py``
- **Listwise MaxEnt objective + tau/q/beta weighting**:
  ``src/maxent_grpo/training/trl_trainer.py`` and
  ``src/maxent_grpo/training/weighting/logic.py``

Recommended Convention
----------------------

For reproducibility, always record both axes together:

- family: ``grpo`` / ``seed_grpo`` / ``maxent_entropy`` / ``maxent_listwise``
- backend: ``grpo`` / ``bnpo`` / ``dr_grpo``

In practice, use the runtime metadata fields rather than inferring from
filenames alone:

- ``run/method_name``
- ``run/method_family``
- ``run/method_backend``
- ``run/method_slug``
