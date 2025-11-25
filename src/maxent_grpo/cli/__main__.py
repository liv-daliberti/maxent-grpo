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

Module entrypoint for ``python -m maxent_grpo.cli``.

Dispatches to the Hydra console scripts so users can run the CLI without
relying solely on console_scripts wiring. Equivalent to invoking
``maxent-grpo`` from the command line.
"""

from __future__ import annotations

from maxent_grpo.cli.hydra_cli import hydra_entry


def main() -> None:
    """Invoke the Hydra CLI entrypoint.

    :returns: ``None`` after running the console entrypoint.
    """

    hydra_entry()


if __name__ == "__main__":  # pragma: no cover - module execution
    main()
