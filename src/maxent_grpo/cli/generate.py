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

Typer-powered CLI + argparse helpers for the distilabel generation pipeline.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from typing import Any, Optional

try:  # Typer is optional for unit tests that only touch argparse helpers.
    import typer
except ImportError:  # pragma: no cover - Typer is optional for some workflows
    typer = None

from maxent_grpo.pipelines.generation.distilabel import (
    DistilabelGenerationConfig,
    run_generation_job,
)


def build_generate_parser() -> argparse.ArgumentParser:
    """Return the ArgumentParser used by ``src.generate``.

    The parser definition is centralized so tests (and future CLIs) can reuse
    the same options without duplicating help strings.

    :returns: Configured parser for the generation CLI.
    :rtype: argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(
        description="Run distilabel pipeline for generating responses with DeepSeek R1",
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
        help="Dataset column that contains the prompt/user instruction.",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{{ instruction }}",
        help="Jinja template string for formatting prompts.",
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
        default=None,
        help="Generation temperature override.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p / nucleus sampling parameter.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to generate per completion.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=1,
        help="Number of completions to generate per prompt.",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=64,
        help="Batch size for input processing.",
    )
    parser.add_argument(
        "--client-replicas",
        type=int,
        default=1,
        help="Number of distilabel client replicas to run in parallel.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Request timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries for failed requests (default: 0).",
    )
    parser.add_argument(
        "--hf-output-dataset",
        type=str,
        required=False,
        help="HuggingFace repo to push results to.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to make the output dataset private when pushing to the Hub.",
    )
    return parser


def run_cli(args: Namespace) -> None:
    """Execute the distilabel generation pipeline for parsed arguments.

    :param args: Parsed CLI namespace produced by :func:`build_generate_parser`
        or Typer, containing distilabel generation options.
    :returns: ``None``. The function is executed for side effects only.
    """

    run_generation_job(DistilabelGenerationConfig.from_namespace(args))


def _typer_entrypoint_fallback(*_args: Any, **_kwargs: Any) -> None:
    """Fallback entrypoint that surfaces missing Typer as a runtime error.

    :param _args: Unused positional arguments.
    :param _kwargs: Unused keyword arguments.
    :raises RuntimeError: Always raised to indicate Typer is required.
    """
    raise RuntimeError(
        "The Typer-based CLI requires the `typer` package. "
        "Install it via `pip install typer` to use the command-line interface."
    )


if typer is not None:

    def _typer_entrypoint_impl(
        hf_dataset: str = typer.Option(..., help="HuggingFace dataset to load."),
        hf_dataset_config: Optional[str] = typer.Option(
            None,
            help="Dataset config to use.",
        ),
        hf_dataset_split: str = typer.Option(
            "train",
            help="Dataset split to use.",
        ),
        prompt_column: str = typer.Option(
            "prompt",
            help="Dataset column that contains the prompt/user instruction.",
        ),
        prompt_template: str = typer.Option(
            "{{ instruction }}",
            help="Jinja template string for formatting prompts.",
        ),
        model: str = typer.Option(..., help="Model name to use for generation."),
        vllm_server_url: str = typer.Option(
            "http://localhost:8000/v1",
            help="URL of the vLLM server.",
        ),
        temperature: Optional[float] = typer.Option(
            None,
            help="Generation temperature override.",
        ),
        top_p: Optional[float] = typer.Option(
            None,
            help="Top-p / nucleus sampling parameter.",
        ),
        max_new_tokens: int = typer.Option(
            8192,
            help="Maximum number of tokens to generate per completion.",
        ),
        num_generations: int = typer.Option(
            1,
            help="Number of completions to generate per prompt.",
        ),
        input_batch_size: int = typer.Option(
            64,
            help="Batch size for input processing.",
        ),
        client_replicas: int = typer.Option(
            1,
            help="Number of distilabel client replicas to run in parallel.",
        ),
        timeout: int = typer.Option(
            600,
            help="Request timeout in seconds (default: 600).",
        ),
        retries: int = typer.Option(
            0,
            help="Number of retries for failed requests (default: 0).",
        ),
        hf_output_dataset: Optional[str] = typer.Option(
            None,
            help="HuggingFace repo to push results to.",
        ),
        private: bool = typer.Option(
            False,
            "--private",
            help="Whether to make the output dataset private when pushing to the Hub.",
            is_flag=True,
        ),
    ) -> None:
        """Typer command hooked to ``console_scripts`` for generation.

        :param hf_dataset: HuggingFace dataset to load.
        :param hf_dataset_config: Optional dataset config to use.
        :param hf_dataset_split: Dataset split to use (default: ``train``).
        :param prompt_column: Column containing the prompt/user instruction.
        :param prompt_template: Jinja template string used to render prompts.
        :param model: Model name to use for generation.
        :param vllm_server_url: URL of the vLLM server endpoint.
        :param temperature: Optional temperature override.
        :param top_p: Optional top-p / nucleus sampling parameter.
        :param max_new_tokens: Maximum number of tokens to generate.
        :param num_generations: Number of completions to generate per prompt.
        :param input_batch_size: Batch size for input processing.
        :param client_replicas: Number of distilabel client replicas.
        :param timeout: Request timeout in seconds.
        :param retries: Number of retries for failed requests.
        :param hf_output_dataset: Optional HuggingFace repo to push results to.
        :param private: Whether to create the Hub dataset as private.
        :returns: ``None``. Runs the generation job directly.
        """

        run_cli(
            Namespace(
                hf_dataset=hf_dataset,
                hf_dataset_config=hf_dataset_config,
                hf_dataset_split=hf_dataset_split,
                prompt_column=prompt_column,
                prompt_template=prompt_template,
                model=model,
                vllm_server_url=vllm_server_url,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_generations=num_generations,
                input_batch_size=input_batch_size,
                client_replicas=client_replicas,
                timeout=timeout,
                retries=retries,
                hf_output_dataset=hf_output_dataset,
                private=private,
            )
        )

else:  # pragma: no cover - Typer missing in minimal test envs
    _typer_entrypoint_impl = _typer_entrypoint_fallback


_typer_entrypoint = _typer_entrypoint_impl
try:  # Ensure the Typer entrypoint name is stable for tests/introspection.
    _typer_entrypoint.__name__ = "_typer_entrypoint"
except (AttributeError, TypeError):  # pragma: no cover - defensive
    pass


def app() -> None:
    """Invoke the Typer CLI (used by ``console_scripts`` + ``python -m``).

    :raises RuntimeError: If ``typer`` is not installed in the environment.
    :returns: ``None`` after dispatching to Typer.
    """

    if typer is None:  # pragma: no cover - Typer missing in minimal test envs
        raise RuntimeError(
            "The Typer-based CLI requires the `typer` package. "
            "Install it via `pip install typer` to use the command-line interface."
        )
    typer.run(_typer_entrypoint)


__all__ = ["app", "build_generate_parser", "run_cli"]
