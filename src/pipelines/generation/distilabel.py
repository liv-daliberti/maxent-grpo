"""Application-layer helpers for running the distilabel generation pipeline."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict, dataclass
from typing import Callable, Optional

from generation.helpers import (
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
) -> None:
    """Execute the distilabel generation pipeline for ``cfg``."""

    args = cfg if isinstance(cfg, Namespace) else cfg.to_namespace()
    pipeline_builder = builder or build_distilabel_pipeline
    run_distilabel_cli(args, pipeline_builder)
