"""
Dataset loading utilities with support for mixtures.

This module wraps Hugging Face ``datasets.load_dataset`` to handle either a
single dataset (``dataset_name``) or a declarative mixture with optional column
selection, subsampling via weights, shuffling, and an optional train/test split.
It returns a mapping compatible with downstream training/evaluation code (a
``datasets.DatasetDict`` when the library is installed, or a lightweight stub
during tests).

The import of ``datasets`` is guarded so this module can be imported in
environments where the library is unavailable; tests monkey-patch the missing
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
import os
import random
import time
from typing import Any, List, Optional, cast

try:
    import datasets
    from datasets import Dataset, DatasetDict, concatenate_datasets
except ModuleNotFoundError:  # pragma: no cover - optional dependency

    class _DatasetsStub:
        """Lightweight stub so imports succeed when ``datasets`` is absent."""

        def load_dataset(self, *_args: Any, **_kwargs: Any):
            raise ModuleNotFoundError(
                "The 'datasets' package is required for dataset loading. "
                "Install with `pip install datasets`."
            )

        def __getattr__(self, _name: str) -> Any:
            raise ModuleNotFoundError(
                "The 'datasets' package is required for dataset loading. "
                "Install with `pip install datasets`."
            )

    datasets = _DatasetsStub()
    Dataset = Any  # type: ignore[assignment]
    DatasetDict = dict  # type: ignore[assignment]

    def concatenate_datasets(_datasets: List[Any]) -> Any:
        raise ModuleNotFoundError(
            "The 'datasets' package is required for dataset concatenation. "
            "Install with `pip install datasets`."
        )


from maxent_grpo.config import ScriptArguments


logger = logging.getLogger(__name__)

_DEFAULT_HF_RETRIES = 6
_DEFAULT_HF_RETRY_SLEEP = 2.0
_DEFAULT_HF_RETRY_MAX_SLEEP = 60.0


def _dataset_load_retry_settings() -> tuple[int, float, float]:
    def _read_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def _read_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    retries = max(0, _read_int("MAXENT_HF_DATASET_RETRIES", _DEFAULT_HF_RETRIES))
    sleep_s = max(0.0, _read_float("MAXENT_HF_DATASET_RETRY_SLEEP", _DEFAULT_HF_RETRY_SLEEP))
    max_sleep_s = max(
        sleep_s, _read_float("MAXENT_HF_DATASET_RETRY_MAX_SLEEP", _DEFAULT_HF_RETRY_MAX_SLEEP)
    )
    return retries, sleep_s, max_sleep_s


def _should_retry_dataset_load(exc: BaseException) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status <= 599):
        return True
    message = str(exc)
    for token in (" 502 ", " 503 ", " 504 ", " 500 ", " 429 "):
        if token in f" {message} ":
            return True
    return False


def _load_dataset_with_retries(*args: Any, **kwargs: Any) -> Any:
    retries, sleep_s, max_sleep_s = _dataset_load_retry_settings()
    last_exc: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            return datasets.load_dataset(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - network failures are environment dependent
            last_exc = exc
            if attempt >= retries or not _should_retry_dataset_load(exc):
                raise
            delay = min(max_sleep_s, sleep_s * (2**attempt))
            # Small jitter to avoid all ranks retrying in lockstep if this runs multi-process.
            delay = delay * (0.85 + 0.3 * random.random())
            logger.warning(
                "datasets.load_dataset failed (attempt %d/%d); retrying in %.1fs | error=%s",
                attempt + 1,
                retries + 1,
                delay,
                exc,
            )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("datasets.load_dataset failed unexpectedly without an exception")


def get_dataset(args: ScriptArguments) -> DatasetDict[Dataset]:
    """Load a dataset or a weighted mixture and return a dictionary.

    The function dispatches to ``datasets.load_dataset`` for simple cases or
    combines multiple datasets when ``args.dataset_mixture`` is provided. Each
    dataset in a mixture can specify a subset of columns, a fractional weight
    to subsample with deterministic shuffling, and an optional global test
    split on the concatenated result.

    :param args: Parsed script arguments that describe either a single dataset
        (``dataset_name`` / ``dataset_config``) or a declarative mixture
        (``dataset_mixture``).
    :type args: maxent_grpo.config.ScriptArguments
    :returns: Mapping with at least a ``train`` split, and possibly ``test`` if
        a split size was requested.
    :rtype: datasets.DatasetDict
    :raises ValueError: If neither a dataset name nor mixture is supplied, or
        when a mixture resolves to zero loaded datasets.
    """
    inline_ds = getattr(args, "dataset", None)
    if inline_ds is not None:
        if isinstance(inline_ds, dict):
            return inline_ds  # type: ignore[return-value]
        return {"train": inline_ds}  # type: ignore[return-value]
    if args.dataset_name and not args.dataset_mixture:
        logger.info("Loading dataset: %s", args.dataset_name)
        return cast(
            DatasetDict[Dataset],
            _load_dataset_with_retries(args.dataset_name, args.dataset_config),
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
            ds = _load_dataset_with_retries(
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
    """Load a single split from a dataset independent of ``ScriptArguments``.

    This helper is used by evaluation code that cannot rely on the full CLI
    argument object but still needs consistent column filtering and error
    handling.

    :param dataset_name: Dataset repository ID on the Hugging Face Hub.
    :type dataset_name: str
    :param dataset_config: Optional dataset configuration name to disambiguate
        multiple configurations.
    :type dataset_config: str | None
    :param split: Split to load (for example ``\"train\"``, ``\"validation\"``,
        or ``\"test\"``).
    :type split: str
    :returns: The requested dataset split as returned by ``datasets.load_dataset``.
    :rtype: datasets.Dataset
    :raises ValueError: If ``split`` is falsy, as evaluation requires an
        explicit split to avoid loading entire datasets inadvertently.
    """
    if not split:
        raise ValueError("split must be provided when loading an eval dataset")
    return cast(
        Dataset, datasets.load_dataset(dataset_name, dataset_config, split=split)
    )
