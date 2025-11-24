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

Compatibility shim for the baseline GRPO CLI.

The actual console scripts now live under :mod:`maxent_grpo.cli`. Importing
this module or invoking its ``main``/``cli`` helpers delegates to the Hydra
entrypoint so legacy calls like ``python -m maxent_grpo.grpo`` keep working.
"""

from __future__ import annotations

from maxent_grpo.cli import hydra_cli, parse_grpo_args


def main(script_args=None, training_args=None, model_args=None) -> None:
    """Run the baseline trainer or delegate to Hydra when args are absent."""

    if script_args is None or training_args is None or model_args is None:
        try:
            script_args, training_args, model_args = parse_grpo_args()
        except Exception:
            return hydra_cli.baseline_entry()
    from maxent_grpo.pipelines.training.baseline import run_baseline_training

    return run_baseline_training(script_args, training_args, model_args)


def cli() -> None:
    """Invoke the baseline entrypoint (CLI style)."""

    main()

if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    cli()