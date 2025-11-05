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

"""Lightweight import-availability helpers.

These mirror transformers.utils.import_utils checks so callers can optionally
gate features based on optional dependencies being installed.
"""

from transformers.utils.import_utils import _is_package_available


# Use same as transformers.utils.import_utils
_e2b_available = _is_package_available("e2b")


def is_e2b_available() -> bool:
    """Return True if the optional 'e2b' package is available."""
    return _e2b_available


_morph_available = _is_package_available("morphcloud")


def is_morph_available() -> bool:
    """Return True if the optional 'morphcloud' package is available."""
    return _morph_available
