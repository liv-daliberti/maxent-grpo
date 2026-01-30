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

CLI helper utilities shared across entrypoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple, cast
import sys

from .generate import build_generate_parser

if TYPE_CHECKING:
    from trl import ModelConfig  # type: ignore[reportMissingTypeStubs]

    from maxent_grpo.config import GRPOConfig, GRPOScriptArguments

__all__ = ["parse_grpo_args", "build_generate_parser"]


def parse_grpo_args() -> Tuple["GRPOScriptArguments", "GRPOConfig", "ModelConfig"]:
    """Parse GRPO CLI arguments.

    The import is deferred to keep optional training dependencies out of
    lightweight environments (tests, docs builds). If a stub CLI module is
    preloaded under ``maxent_grpo.training.cli`` it is used instead.

    :returns: Tuple of ``(script_args, training_args, model_args)`` parsed from
        the CLI.
    :rtype: tuple[GRPOScriptArguments, GRPOConfig, ModelConfig]
    :raises ImportError: If the training CLI parser cannot be imported.
    :raises RuntimeError: If the underlying parser fails to initialize.
    :raises SystemExit: When the argument parser aborts due to invalid input.
    """
    # Allow tests to monkeypatch the training CLI module and bypass the real parser.
    stub_cli = sys.modules.get("maxent_grpo.training.cli")
    if stub_cli is not None:
        delegate = getattr(stub_cli, "parse_grpo_args", None)
        if callable(delegate):
            parser = cast(
                Callable[[], Tuple["GRPOScriptArguments", "GRPOConfig", "ModelConfig"]],
                delegate,
            )
            return parser()
    from maxent_grpo.training.cli.trl import parse_grpo_args as _parse_grpo_args

    return _parse_grpo_args()
