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
      (``training.generation``)
                │
                ▼
      Reward + Reference Prep
      (``training.pipeline.prepare_training_batch``)
                │
                ▼
      Loss / Optimizer / Controllers
      (``training.weighting.loss`` + ``training.loop``)
         │                    │
         │                    └──► Checkpointing / Controllers
         │                         (``training.state``, ``training.zero_utils``)
         ▼
      Metrics & Logging
      (``training.metrics``)


Stage Breakdown
---------------

Generation
    :mod:`training.generation` and
    :mod:`~training.generation.vllm` construct grouped completions,
    handling prompt truncation, tokenizer quirks, and optional multi-round
    vLLM retries.

Reward & Reference Preparation
    :func:`training.pipeline.prepare_training_batch`
    orchestrates reward computation (:mod:`training.rewards`), reference
    log-prob gathering, weighting, and sequence scoring so downstream stages
    always receive a fully populated :class:`PreparedBatch`.

Loss / Optimizer
    :mod:`training.weighting.loss` converts the sequence scores into
    weighted objectives, while :mod:`training.loop` handles gradient
    accumulation, learning-rate schedules, adaptive controllers, validation,
    and checkpoint cadence.

Logging & Checkpointing
    :mod:`training.metrics` builds structured metric payloads for console or W&B
    via :mod:`telemetry.wandb`, while :mod:`training.state` stores controller
    state, mirrors HF metadata, and manages distributed checkpointing.

This structure makes it clear where to inject new behaviors (e.g., custom
rewards or logging sinks) without modifying the rest of the pipeline.
