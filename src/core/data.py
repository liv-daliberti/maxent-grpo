"""
Dataset loading utilities with support for mixtures.

This module wraps Hugging Face ``datasets.load_dataset`` to handle either a
single dataset (``dataset_name``) or a declarative mixture with optional column
selection, subsampling via weights, shuffling, and an optional train/test split.
It returns a mapping compatible with downstream training/evaluation code (a
``datasets.DatasetDict`` when the library is installed, or a lightweight stub
during tests).

The import of ``datasets`` is guarded so this module can be imported in
environments where the library is unavailable; tests monkey‑patch the missing
symbols when needed.

License
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, TypeVar, cast

# Optional dependency: datasets. Keep imports lazy/guarded so that importing
# this module does not fail in environments without HF datasets. Tests monkey‑
# patch the module‑level symbols below.
try:  # pragma: no cover - exercised indirectly via monkeypatch in tests
    import datasets
    from datasets import DatasetDict, Dataset, concatenate_datasets
except (ImportError, ModuleNotFoundError):
    datasets = None

    T = TypeVar("T")

    class DatasetDict(Dict[str, T]):  # minimal placeholder for type hints
        """Simplified mapping used when ``datasets.DatasetDict`` is unavailable."""

    class Dataset:  # minimal placeholder for type hints
        """Minimal interface used when ``datasets.Dataset`` is unavailable."""

        def __len__(self) -> int:
            """Return the number of rows (placeholder implementation)."""
            raise NotImplementedError

        def shuffle(self, _seed: Optional[int] = None) -> "Dataset":
            """Return a shuffled view of the dataset."""
            raise NotImplementedError

        def select(self, _indices: Sequence[int]) -> "Dataset":
            """Return a subset of rows corresponding to ``_indices``."""
            raise NotImplementedError

        def select_columns(self, _column_names: Sequence[str]) -> "Dataset":
            """Return a dataset containing only ``_column_names``."""
            raise NotImplementedError

    def concatenate_datasets(
        datasets_list: Sequence[Dataset],
    ) -> Dataset:  # pragma: no cover
        """Fallback that raises when ``datasets`` is missing."""
        raise ImportError(
            "datasets library is not installed; provide a concatenate_datasets"
        )


from maxent_grpo.config import ScriptArguments


logger = logging.getLogger(__name__)


def get_dataset(args: ScriptArguments) -> DatasetDict[Dataset]:
    """Load a dataset or a mixture of datasets.

    :param args: Script arguments containing dataset configuration.
    :type args: maxent_grpo.config.ScriptArguments
    :returns: A dataset dictionary with ``train`` and optionally ``test``.
    :rtype: datasets.DatasetDict
    :raises ValueError: If no dataset information is provided or the mixture is empty.
    :raises ImportError: If datasets library is not installed.
    """
    if not datasets:
        raise ImportError("datasets library required but not installed")

    if args.dataset_name and not args.dataset_mixture:
        logger.info("Loading dataset: %s", args.dataset_name)
        return cast(
            DatasetDict[Dataset],
            datasets.load_dataset(args.dataset_name, args.dataset_config),
        )
    elif args.dataset_mixture:
        logger.info(
            "Creating dataset mixture with %d datasets",
            len(args.dataset_mixture.datasets),
        )
        seed: int = args.dataset_mixture.seed
        datasets_list: List[Dataset] = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(
                "Loading dataset for mixture: %s (config: %s)",
                dataset_config.id,
                dataset_config.config,
            )
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(
                    range(int(len(ds) * dataset_config.weight))
                )
                logger.info(
                    "Subsampled dataset '%s' (config: %s) with weight=%s to %d examples",
                    dataset_config.id,
                    dataset_config.config,
                    dataset_config.weight,
                    len(ds),
                )

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(
                "Created dataset mixture with %d examples", len(combined_dataset)
            )

            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    "Split dataset into train and test sets with test size: %s",
                    args.dataset_mixture.test_split_size,
                )
                return combined_dataset
            else:
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")


def load_dataset_split(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: str = "validation",
) -> Dataset:
    """Load a single split from a dataset independent of ScriptArguments.

    :param dataset_name: Dataset repository ID on the Hub.
    :type dataset_name: str
    :param dataset_config: Optional dataset config name.
    :type dataset_config: str | None
    :param split: Split to load (e.g., \"train\", \"validation\", \"test\").
    :type split: str
    :returns: The requested dataset split.
    :rtype: datasets.Dataset
    :raises ImportError: If the datasets library is unavailable.
    """
    if not datasets:
        raise ImportError("datasets library required but not installed")
    if not split:
        raise ValueError("split must be provided when loading an eval dataset")
    return cast(
        Dataset, datasets.load_dataset(dataset_name, dataset_config, split=split)
    )
