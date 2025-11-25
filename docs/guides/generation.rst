Generation (vLLM + Distilabel)
==============================

``src/generate.py`` provides a small CLI and helper to build a Distilabel pipeline that hits an OpenAI‑compatible endpoint (e.g., vLLM).

Start a vLLM Server
-------------------

Example (adjust model and ports as needed):

.. code-block:: bash

   python -m vllm.entrypoints.openai.api_server \
     --model "Qwen/Qwen2.5-1.5B-Instruct" \
     --port 8000

Run the Pipeline
----------------

.. code-block:: bash

   python -m src.generate \
     --hf-dataset "your/dataset" \
     --hf-dataset-config default \
     --hf-dataset-split train \
     --prompt-column instruction \
     --prompt-template "{{ instruction }}" \
     --model "Qwen/Qwen2.5-1.5B-Instruct" \
     --vllm-server-url http://localhost:8000/v1 \
     --temperature 0.8 \
     --top-p 0.9 \
     --max-new-tokens 8192 \
     --num-generations 1 \
     --input-batch-size 64 \
     --client-replicas 1

Optional: push results to the Hub:

.. code-block:: bash

   --hf-output-dataset your-org/generated-math \
   --private

Programmatic Use
----------------

.. code-block:: python

   from src.generate import DistilabelPipelineConfig, build_distilabel_pipeline

   cfg = DistilabelPipelineConfig(
       model="Qwen/Qwen2.5-1.5B-Instruct",
       base_url="http://localhost:8000/v1",
       prompt_template="{{ instruction }}",
       prompt_column="instruction",
   )
   pipe = build_distilabel_pipeline(cfg)
   # pipe.run(dataset, dataset_batch_size=..., use_cache=False)

Notes
-----

- The CLI prints the resolved configuration before running.
- Use ``--retries`` and ``--timeout`` for flaky or slow endpoints.
- To reduce server memory, prefer lower ``--max-new-tokens`` and fewer ``--num-generations``.
