# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build and run a distilabel text-generation pipeline.

This module provides a small CLI and a helper to configure and run a
distilabel Pipeline backed by an OpenAI-compatible endpoint (e.g., vLLM).
"""

from dataclasses import dataclass
from typing import Optional

from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration


@dataclass
class DistilabelPipelineConfig:
    """Configuration for building a distilabel generation pipeline.

    :ivar model: Model name to forward to the OpenAI-compatible endpoint.
    :vartype model: str
    :ivar base_url: Base URL of the server, defaults to ``http://localhost:8000/v1``.
    :vartype base_url: str
    :ivar prompt_column: Optional dataset column to map into the template.
    :vartype prompt_column: str | None
    :ivar prompt_template: Jinja template string used by the generation step.
    :vartype prompt_template: str
    :ivar temperature: Optional temperature override.
    :vartype temperature: float | None
    :ivar top_p: Optional nucleus sampling p.
    :vartype top_p: float | None
    :ivar max_new_tokens: Maximum tokens per generation.
    :vartype max_new_tokens: int
    :ivar num_generations: Number of completions per input example.
    :vartype num_generations: int
    :ivar input_batch_size: Distilabel input batch size.
    :vartype input_batch_size: int
    :ivar client_replicas: Number of parallel client replicas.
    :vartype client_replicas: int
    :ivar timeout: Request timeout in seconds.
    :vartype timeout: int
    :ivar retries: Number of HTTP retries per request.
    :vartype retries: int
    """

    model: str
    base_url: str = "http://localhost:8000/v1"
    prompt_column: Optional[str] = None
    prompt_template: str = "{{ instruction }}"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: int = 8192
    num_generations: int = 1
    input_batch_size: int = 64
    client_replicas: int = 1
    timeout: int = 900
    retries: int = 0


def build_distilabel_pipeline(cfg: DistilabelPipelineConfig | None = None, **kwargs) -> Pipeline:
    """Create and return a distilabel Pipeline based on ``cfg``.

    The returned pipeline performs text generation using an OpenAI-compatible
    endpoint. It groups generations and supports simple batching.

    :param cfg: Pipeline configuration parameters. If ``None``, any keyword
        arguments are used to construct :class:`DistilabelPipelineConfig`.
    :type cfg: DistilabelPipelineConfig | None
    :param kwargs: Optional fields used when ``cfg`` is ``None``.
    :type kwargs: Any
    :returns: A configured distilabel ``Pipeline``.
    :rtype: distilabel.pipeline.Pipeline
    """

    if cfg is None:
        cfg = DistilabelPipelineConfig(**kwargs)

    generation_kwargs = {"max_new_tokens": cfg.max_new_tokens}

    if cfg.temperature is not None:
        generation_kwargs["temperature"] = cfg.temperature

    if cfg.top_p is not None:
        generation_kwargs["top_p"] = cfg.top_p

    with Pipeline().ray() as pipe:  # avoid shadowing outer-scope name
        TextGeneration(
            llm=OpenAILLM(
                base_url=cfg.base_url,
                api_key="something",
                model=cfg.model,
                timeout=cfg.timeout,
                max_retries=cfg.retries,
                generation_kwargs=generation_kwargs,
            ),
            template=cfg.prompt_template,
            input_mappings=(
                {"instruction": cfg.prompt_column}
                if cfg.prompt_column is not None
                else {}
            ),
            input_batch_size=cfg.input_batch_size,
            num_generations=cfg.num_generations,
            group_generations=True,
            resources=StepResources(replicas=cfg.client_replicas),
        )

    return pipe


if __name__ == "__main__":
    import argparse

    from datasets import load_dataset

    parser = argparse.ArgumentParser(
        description=(
            "Run distilabel pipeline for generating responses with DeepSeek R1"
        )
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="HuggingFace dataset to load",
    )
    parser.add_argument(
        "--hf-dataset-config",
        type=str,
        required=False,
        help="Dataset config to use",
    )
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{{ instruction }}",
        help="Template string for formatting prompts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for generation",
    )
    parser.add_argument(
        "--vllm-server-url",
        type=str,
        default="http://localhost:8000/v1",
        help="URL of the vLLM server",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p value for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Number of generations per problem",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=64,
        help="Batch size for input processing",
    )
    parser.add_argument(
        "--client-replicas",
        type=int,
        default=1,
        help="Number of client replicas for parallel processing",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries for failed requests (default: 0)",
    )
    parser.add_argument(
        "--hf-output-dataset",
        type=str,
        required=False,
        help="HuggingFace repo to push results to",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to make the output dataset private when pushing to HF Hub",
    )

    args = parser.parse_args()

    print("\nRunning with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    print(
        (
            f"Loading '{args.hf_dataset}' (config: {args.hf_dataset_config}, "
            f"split: {args.hf_dataset_split}) dataset..."
        )
    )
    dataset = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
    print("Dataset loaded!")

    config = DistilabelPipelineConfig(
        model=args.model,
        base_url=args.vllm_server_url,
        prompt_template=args.prompt_template,
        prompt_column=args.prompt_column,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        input_batch_size=args.input_batch_size,
        client_replicas=args.client_replicas,
        timeout=args.timeout,
        retries=args.retries,
    )

    pipeline = build_distilabel_pipeline(config)

    print("Running generation pipeline...")
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,
        use_cache=False,
    )
    print("Generation pipeline finished!")

    if args.hf_output_dataset:
        print(f"Pushing resulting dataset to '{args.hf_output_dataset}'...")
        distiset.push_to_hub(args.hf_output_dataset, private=args.private)
        print("Dataset pushed!")
