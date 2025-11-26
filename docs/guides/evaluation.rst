Evaluation
==========

This project supports LightEval benchmarks with vLLM decoding and optional Slurm scheduling.

Benchmarks
----------

Built‑in tasks are registered in ``core.evaluation.LIGHTEVAL_TASKS``:

- ``math_500``, ``aime24``, ``aime25``, ``gpqa:diamond``
- LCB code generation variants (extended suite)

Launching Jobs
--------------

The helper ``core.evaluation.run_benchmark_jobs(training_args, model_args)`` resolves the requested benchmark names (or ``all``) and submits jobs via Slurm using the vLLM OpenAI server.

Typical flow:

1) Train your model (or pick a Hub model id)
2) Ensure vLLM is available on your cluster image
3) From Python (or integrate in your pipeline), call:

.. code-block:: python

   from core.evaluation import run_benchmark_jobs

   # training_args.hub_model_id / hub_model_revision drive evaluation targets
   run_benchmark_jobs(training_args, model_args)

Notes
-----

- For large models (≥ 30B) or MATH‑heavy runs, the job script increases GPU count and uses tensor parallelism.
- Check the Slurm script under ``ops/slurm/evaluate.slurm`` if you want to customize cluster resources.
- To evaluate a single suite locally without Slurm, adapt ``run_lighteval_job`` to spawn ``vllm`` and ``lighteval`` processes on your workstation.
- For a quick parity check between GRPO and MaxEnt checkpoints on the shared ``math_500`` pipeline, use the offline stub runner:

  .. code-block:: bash

     make eval-math-delta  # tools/eval_math_delta.py (stub runner + fixed dataset)
