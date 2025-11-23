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

"""
Thin CLI wrapper for the MaxEnt-GRPO training loop.

This script simply parses the TRL configs (matching ``src/grpo.py``) and
hands control to ``training.run_maxent_grpo`` where the actual training
logic lives. Keeping the heavy implementation in ``training`` keeps this
entrypoint short and mirrors the layout of ``grpo.py`` for readability.
"""

from __future__ import annotations

import sys

from training.cli import parse_grpo_args
from training import run_maxent_grpo


def main() -> None:
    """Parse GRPO configs via TRL and hand control to ``run_maxent_grpo``.

    :returns: ``None``. The function hands execution to
        :func:`training.run_maxent_grpo`.
    :rtype: None
    """
    try:
        script_args, training_args, model_args = parse_grpo_args()
    except ImportError as exc:  # pragma: no cover - CLI guard
        print(str(exc), file=sys.stderr)
        raise
    try:
        run_maxent_grpo(script_args, training_args, model_args)
    except NotImplementedError as exc:  # pragma: no cover - forward guidance
        raise SystemExit(
            "MaxEnt training entrypoint has been removed. Use the Hydra "
            "pipelines or compose a runner with training.loop/training.pipeline."
        ) from exc


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
