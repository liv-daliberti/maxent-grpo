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

"""Compatibility shim for the MaxEnt-GRPO CLI.

The primary console scripts now live under :mod:`maxent_grpo.cli`. Importing
this module or invoking ``main`` delegates to the Hydra entrypoint so legacy
invocations like ``python -m maxent_grpo.maxent_grpo`` continue to work.
"""

from __future__ import annotations

from maxent_grpo.cli import hydra_cli


def main(script_args=None, training_args=None, model_args=None) -> None:
    """Run the MaxEnt trainer when configs are provided, else delegate to Hydra."""

    if script_args is None or training_args is None or model_args is None:
        try:
            from maxent_grpo.training.cli import parse_grpo_args
            from maxent_grpo.training import run_maxent_grpo

            script_args, training_args, model_args = parse_grpo_args()
            return run_maxent_grpo(script_args, training_args, model_args)
        except Exception:
            return hydra_cli.maxent_entry()
    from maxent_grpo.training import run_maxent_grpo

    return run_maxent_grpo(script_args, training_args, model_args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
