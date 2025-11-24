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

from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from maxent_grpo.pipelines.base import PipelineResult, log_pipeline_banner
from maxent_grpo.generation.helpers import (
    DistilabelPipelineConfig,
    build_distilabel_pipeline,
    run_distilabel_cli,
)

__all__ = [
    "DistilabelGenerationConfig",
    "DistilabelPipelineConfig",
    "build_distilabel_pipeline",
    "run_generation_job",
]


@dataclass
class DistilabelGenerationConfig:
    """High-level configuration for the distilabel generation workflow."""

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
        """Instantiate the config from an argparse namespace."""

        return cls(**vars(args))

    def to_namespace(self) -> Namespace:
        """Convert the config to an argparse namespace consumed by helpers."""

        return Namespace(**asdict(self))


def run_generation_job(
    cfg: DistilabelGenerationConfig | Namespace,
    builder: Optional[Callable[[DistilabelPipelineConfig], object]] = None,
) -> PipelineResult:
    """Execute the distilabel generation pipeline for ``cfg``."""

    args = cfg if isinstance(cfg, Namespace) else cfg.to_namespace()
    log_pipeline_banner("generation.distilabel", args)
    pipeline_builder = builder or build_distilabel_pipeline
    run_distilabel_cli(args, pipeline_builder)
    return PipelineResult(name="generation.distilabel")