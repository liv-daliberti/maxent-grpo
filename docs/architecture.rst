Training Architecture
=====================

The training stack now uses one shared path for both GRPO and MaxEnt runs.
The only intended objective-level divergence is inside
``maxent_grpo.training.trl_trainer``.

Flow Diagram
------------

.. code-block:: text

       CLI / Slurm Entrypoint
       (``src/maxent_grpo/grpo.py``)
                │
                ▼
      Shared Training Pipeline
      (``maxent_grpo.training.baseline``)
                │
                ▼
      Dataset + Prompt Mapping
      (shared ``prompt``/``answer`` transform)
                │
                ▼
      Reward Resolution
      (``maxent_grpo.training.rewards``)
                │
                ▼
      Objective + Loss
      (``maxent_grpo.training.trl_trainer``)
                │
                ▼
      TRL/HF Optimization + Checkpointing


Stage Breakdown
---------------

Entrypoint
    ``src/maxent_grpo/grpo.py`` is the canonical trainer entrypoint used for
    both GRPO and MaxEnt variants.

Shared Pipeline
    :func:`maxent_grpo.training.baseline.run_baseline_training`
    performs all shared setup (dataset loading, prompt mapping, tokenizer/model,
    trainer wiring, train/eval, save/resume).

Rewards
    :mod:`maxent_grpo.training.rewards` resolves reward functions and weights
    with identical logic across both objectives.

Objective Boundary
    :mod:`maxent_grpo.training.trl_trainer` contains the objective-specific
    behavior (GRPO vs MaxEnt). Keep divergence localized here for fair
    comparisons.
