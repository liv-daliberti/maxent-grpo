"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Application-layer helpers for running the distilabel generation pipeline.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Protocol, cast

from maxent_grpo.pipelines.base import PipelineResult, log_pipeline_banner
from maxent_grpo.generation.helpers import (
    DistilabelPipelineConfig,
    build_distilabel_pipeline,
)
from maxent_grpo.generation.errors import (
    GenerationServiceError,
    log_generation_service_error,
)

LOG = logging.getLogger(__name__)

__all__ = [
    "DistilabelGenerationConfig",
    "DistilabelPipelineConfig",
    "build_distilabel_pipeline",
    "run_generation_job",
]


class DistilabelPipeline(Protocol):
    """Minimal protocol for distilabel pipeline runners."""

    def run(
        self,
        *,
        dataset: Any,
        dataset_batch_size: int,
        use_cache: bool,
    ) -> Any:
        ...


@dataclass
class DistilabelGenerationConfig:
    """High-level configuration for the distilabel generation workflow.

    :param hf_dataset: Hugging Face dataset repo ID to read prompts from.
    :type hf_dataset: str
    :param model: Model name or endpoint identifier consumed by vLLM.
    :type model: str
    :param hf_dataset_config: Optional dataset config name.
    :type hf_dataset_config: str | None
    :param hf_dataset_split: Dataset split to stream for generation.
    :type hf_dataset_split: str
    :param prompt_column: Column containing the prompt text.
    :type prompt_column: str
    :param prompt_template: Jinja template applied to each prompt row.
    :type prompt_template: str
    :param vllm_server_url: Base URL of the vLLM server exposing OpenAI APIs.
    :type vllm_server_url: str
    :param temperature: Sampling temperature forwarded to vLLM.
    :type temperature: float | None
    :param top_p: Nucleus sampling parameter forwarded to vLLM.
    :type top_p: float | None
    :param max_new_tokens: Maximum tokens to generate per completion.
    :type max_new_tokens: int
    :param num_generations: Number of completions per prompt.
    :type num_generations: int
    :param input_batch_size: Batch size for pulling rows from the dataset.
    :type input_batch_size: int
    :param client_replicas: Number of HTTP clients to shard requests across.
    :type client_replicas: int
    :param timeout: Request timeout in seconds for vLLM calls.
    :type timeout: int
    :param retries: Retry attempts for failed requests.
    :type retries: int
    :param hf_output_dataset: Optional Hugging Face dataset repo to push outputs.
    :type hf_output_dataset: str | None
    :param private: Whether to push the output dataset as private.
    :type private: bool
    """

    hf_dataset: str
    model: str
    hf_dataset_config: Optional[str] = None
    hf_dataset_split: str = "train"
    prompt_column: str = "prompt"
    prompt_template: str = "{{ instruction }}"
    vllm_server_url: str = "http://localhost:8000/v1"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_new_tokens: int = 8192
    num_generations: int = 1
    input_batch_size: int = 64
    client_replicas: int = 1
    timeout: int = 600
    retries: int = 0
    hf_output_dataset: Optional[str] = None
    private: bool = False

    @classmethod
    def from_namespace(cls, args: Namespace) -> "DistilabelGenerationConfig":
        """Instantiate the config from an argparse namespace.

        :param args: Namespace containing fields matching the dataclass attributes.
        :type args: argparse.Namespace
        :returns: Materialized :class:`DistilabelGenerationConfig` instance.
        :rtype: DistilabelGenerationConfig
        """

        return cls(**vars(args))

    def to_namespace(self) -> Namespace:
        """Convert the config to an argparse namespace consumed by helpers.

        :returns: Namespace exposing the same attributes as the dataclass.
        :rtype: argparse.Namespace
        """

        return Namespace(**asdict(self))


def run_generation_job(
    cfg: DistilabelGenerationConfig | Namespace,
    builder: Optional[Callable[[DistilabelPipelineConfig], DistilabelPipeline]] = None,
) -> PipelineResult:
    """Execute the distilabel generation pipeline.

    :param cfg: Generation configuration or CLI namespace.
    :type cfg: DistilabelGenerationConfig | argparse.Namespace
    :param builder: Optional factory returning a configured distilabel pipeline.
        Defaults to :func:`maxent_grpo.generation.helpers.build_distilabel_pipeline`.
    :type builder: Callable[[DistilabelPipelineConfig], object] | None
    :returns: Summary of the pipeline execution; metrics/artifacts are currently
        unused but kept for parity with other pipelines.
    :rtype: maxent_grpo.pipelines.base.PipelineResult
    :raises RuntimeError: If the resulting distilabel dataset cannot be pushed.
    """
    from datasets import load_dataset  # type: ignore

    if isinstance(cfg, Namespace):
        args = cfg
    elif hasattr(cfg, "to_namespace"):
        args = cfg.to_namespace()
    else:
        args = Namespace(**vars(cfg))
    log_pipeline_banner("generation.distilabel", args)

    dataset = load_dataset(
        args.hf_dataset,
        args.hf_dataset_config,
        split=args.hf_dataset_split,
    )
    pipeline_cfg = DistilabelPipelineConfig(
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
    pipeline_builder = builder or build_distilabel_pipeline
    pipeline = pipeline_builder(pipeline_cfg)

    try:
        distiset = pipeline.run(
            dataset=cast(Any, dataset),
            dataset_batch_size=args.input_batch_size * 1000,
            use_cache=False,
        )
    except GenerationServiceError as exc:
        extra = {
            "dataset": args.hf_dataset,
            "dataset_config": args.hf_dataset_config,
            "dataset_split": args.hf_dataset_split,
            "model_id": args.model,
        }
        exc.payload = exc.payload.copy_with(extra=extra)
        log_generation_service_error(LOG, "distilabel", exc)
        raise RuntimeError(
            "Generation failed due to vLLM service error; see logs for payload."
        ) from exc
    if args.hf_output_dataset:
        push_fn = getattr(distiset, "push_to_hub", None)
        if not callable(push_fn):
            raise RuntimeError("distilabel dataset object does not expose push_to_hub")
        push_fn(args.hf_output_dataset, args.private)
    return PipelineResult(name="generation.distilabel")
