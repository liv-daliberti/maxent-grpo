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
hands control to ``maxent_helpers.run_maxent_grpo`` where the actual training
logic lives. Keeping the heavy implementation in ``maxent_helpers`` keeps this
entrypoint short and mirrors the layout of ``grpo.py`` for readability.
"""

from __future__ import annotations

import sys

from configs import GRPOConfig, GRPOScriptArguments
from maxent_helpers import run_maxent_grpo


def main() -> None:
    """Parse GRPO configs via TRL and hand control to ``run_maxent_grpo``.

    The function mirrors :mod:`src.grpo` by delegating argument parsing to
    ``TrlParser`` and then invoking the shared MaxEnt-GRPO entrypoint.

    :raises ImportError: If TRL is not installed in the current environment.
    """
    try:
        from trl import ModelConfig, TrlParser
    except (ImportError, ModuleNotFoundError):  # pragma: no cover
        print(
            "This script requires TRL installed to parse configs (pip install trl).",
            file=sys.stderr,
        )
        raise

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    run_maxent_grpo(script_args, training_args, model_args)


if __name__ == "__main__":
    main()
