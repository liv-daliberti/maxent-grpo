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

Legacy stub installer for training module tests.
"""

from __future__ import annotations

from tests.helpers.run_setup_stubs import install_training_stubs


def _load_run_setup(monkeypatch):
    """Install lightweight training dependency stubs for downstream imports."""
    install_training_stubs(monkeypatch)
    return None


__all__ = ["_load_run_setup"]