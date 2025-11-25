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

Tests for the module entrypoint under maxent_grpo.cli.__main__.
"""

from __future__ import annotations


def test_cli_main_invokes_hydra_entry(monkeypatch):
    called = {}

    def _hydra_entry():
        called["ran"] = True

    import maxent_grpo.cli.__main__ as cli_main

    monkeypatch.setattr(cli_main, "hydra_entry", _hydra_entry)
    cli_main.main()
    assert called.get("ran") is True
