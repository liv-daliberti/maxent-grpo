Training Architecture
=====================

The MaxEnt‑GRPO trainer follows a modular flow so each stage can evolve
independently (e.g., swapping generation backends or integrating new reward
models).  The diagram below shows the high‑level data path; each stage links to
its corresponding module in the codebase.

Flow Diagram
------------

.. code-block:: text

       Prompts / Dataloader
                │
                ▼
      Generation & vLLM bridge
      (``maxent_grpo.training.rollout``)
                │
                ▼
      Reward + Reference Prep
      (``maxent_grpo.training.pipeline.prepare_training_batch``)
                │
                ▼
      Loss / Optimizer / Controllers
      (``maxent_grpo.training.weighting.loss`` + ``maxent_grpo.training.loop``)
         │                    │
         │                    └──► Checkpointing / Controllers
         │                         (``maxent_grpo.training.state``, ``maxent_grpo.training.zero_utils``)
         ▼
      Metrics & Logging
      (``maxent_grpo.training.metrics``)


Stage Breakdown
---------------

Generation
    :mod:`maxent_grpo.training.rollout` and
    :mod:`maxent_grpo.training.rollout.vllm` construct grouped completions,
    handling prompt truncation, tokenizer quirks, and optional multi-round
    vLLM retries.

Reward & Reference Preparation
    :func:`maxent_grpo.training.pipeline.prepare_training_batch`
    orchestrates reward computation (:mod:`maxent_grpo.training.rewards`),
    reference log-prob gathering, weighting, and sequence scoring so downstream
    stages always receive a fully populated :class:`PreparedBatch`.

Loss / Optimizer
    :mod:`maxent_grpo.training.weighting.loss` converts the sequence scores into
    weighted objectives, while :mod:`maxent_grpo.training.loop` handles
    gradient accumulation, learning-rate schedules, adaptive controllers,
    validation, and checkpoint cadence.

Logging & Checkpointing
    :mod:`maxent_grpo.training.metrics` builds structured metric payloads for
    console or W&B via :mod:`maxent_grpo.telemetry.wandb`, while
    :mod:`maxent_grpo.training.state` stores controller state, mirrors HF
    metadata, and manages distributed checkpointing.

This structure makes it clear where to inject new behaviors (e.g., custom
rewards or logging sinks) without modifying the rest of the pipeline.
