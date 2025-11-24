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
"""

# Dataset and script argument dataclasses used across training entrypoints.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from types import SimpleNamespace

try:  # Optional dependency in unit tests
    import trl
except ImportError:  # pragma: no cover - fallback for tests
    trl = SimpleNamespace()

if not hasattr(trl, "ScriptArguments"):  # pragma: no cover - test fallback

    class _BaseScriptArgs:
        """Minimal stand-in for ``trl.ScriptArguments`` when TRL is absent."""

        __slots__ = ()

    trl.ScriptArguments = _BaseScriptArgs
else:
    _BaseScriptArgs = trl.ScriptArguments

if not hasattr(trl, "GRPOConfig"):  # pragma: no cover - used by sibling module

    class _BaseGRPOConfig:
        """Minimal stand-in for ``trl.GRPOConfig`` used in test contexts."""

        __slots__ = ()

    trl.GRPOConfig = _BaseGRPOConfig


@dataclass
class DatasetConfig:
    """Configuration for a dataset inside a mixture.

    :ivar id: Dataset repository ID on the Hub (e.g., "org/name").
    :ivar config: Optional dataset configuration name.
    :ivar split: Split to load, defaults to "train".
    :ivar columns: Optional list of column names to keep; if provided all
        datasets in the mixture must share the same set of columns.
    :ivar weight: Optional sampling weight in (0, 1]; when specified, a
        subsample of that proportion is taken from the split.
    """

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: List[DatasetConfig]
    seed: int = field(default=0)
    test_split_size: Optional[float] = field(default=None)

    def __post_init__(self) -> None:
        if self.test_split_size is not None and not 0 < self.test_split_size < 1:
            raise ValueError("test_split_size must be between 0 and 1")


@dataclass
class ScriptArguments(_BaseScriptArgs):
    """Extended TRL ScriptArguments with dataset mixture support."""

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset name. Can be omitted if using dataset_mixture."},
    )
    dataset_mixture: Optional[DatasetMixtureConfig] = field(
        default=None,
        metadata={
            "help": (
                "Configuration for creating dataset mixtures with advanced options "
                "like shuffling."
            )
        },
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset config name when using dataset_name"},
    )

    def __post_init__(self) -> None:
        try:
            super().__post_init__()
        except AttributeError:
            pass
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError(
                "Either `dataset_name` or `dataset_mixture` must be provided"
            )

        if self.dataset_mixture is not None:
            if (
                not isinstance(self.dataset_mixture, dict)
                or "datasets" not in self.dataset_mixture
            ):
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            columns_sets = [
                set(dataset.columns)
                for dataset in datasets_list
                if dataset.columns is not None
            ]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


__all__ = ["DatasetConfig", "DatasetMixtureConfig", "ScriptArguments"]
