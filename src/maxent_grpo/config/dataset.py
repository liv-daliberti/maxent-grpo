"""
Dataset and script argument dataclasses shared across training entrypoints.

These helpers extend TRL's ``ScriptArguments`` to support dataset mixtures,
including per-dataset column selection, sampling weights, and optional
train/test splits. TRL is treated as an optional dependency so the config
objects remain importable in lightweight environments such as doc builds and
unit tests.

License
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

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from types import SimpleNamespace

try:  # Optional dependency in unit tests
    import trl
except ImportError:  # pragma: no cover - fallback for tests
    trl = SimpleNamespace()

LOG = logging.getLogger(__name__)

if not hasattr(trl, "ScriptArguments"):  # pragma: no cover - test fallback

    class _BaseScriptArgs:
        """Minimal stand-in for ``trl.ScriptArguments`` when TRL is absent."""

        __slots__ = ()

    trl.ScriptArguments = _BaseScriptArgs
else:
    _BaseScriptArgs = trl.ScriptArguments

if not hasattr(trl, "GRPOConfig"):  # pragma: no cover - used by sibling module

    @dataclass
    class _BaseGRPOConfig:
        """Minimal stand-in for ``trl.GRPOConfig`` used in test contexts."""

        per_device_train_batch_size: int = 1
        per_device_eval_batch_size: int = 1
        num_generations: int = 1
        num_train_epochs: int = 1
        max_steps: int = 0
        gradient_accumulation_steps: int = 1
        max_grad_norm: float = 0.0
        warmup_ratio: float = 0.0
        max_prompt_length: int = 1024
        max_completion_length: int = 256
        gen_temperature: float = 1.0
        gen_top_p: float = 1.0
        gen_best_of: Optional[int] = None
        gen_top_k: Optional[int] = None
        gen_frequency_penalty: float = 0.0
        gen_presence_penalty: float = 0.0
        use_vllm: bool = False
        vllm_url: str = "http://localhost:8000/generate"
        vllm_rounds_cfg: int = 0
        vllm_retry_sleep: float = 1.0
        vllm_backfill_local: bool = False
        vllm_request_logprobs: bool = True
        vllm_stop_sequences: Optional[list] = None
        vllm_request_timeout: float = 120.0
        vllm_max_retries: int = 3
        vllm_backoff: float = 1.0
        vllm_guided_json: Optional[str] = None
        vllm_guided_regex: Optional[str] = None
        vllm_logit_bias: Optional[dict] = None
        vllm_request_id_prefix: Optional[str] = None
        vllm_sync_weights: bool = False
        do_eval: bool = False
        eval_steps: Optional[int] = None
        clip_range: float = 0.2
        learning_rate: float = 1e-4
        train_grpo_objective: bool = True
        reward_weights: list = field(default_factory=list)
        reward_funcs: list = field(default_factory=list)
        output_dir: str = "."
        save_strategy: str = "no"
        save_steps: int = 0

        def get_process_log_level(self) -> int:
            return 20

    trl.GRPOConfig = _BaseGRPOConfig


@dataclass
class DatasetConfig:
    """Configuration for a dataset inside a mixture.

    This describes one dataset entry pulled from the Hub and optionally
    filtered/weighted before mixing.

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


class _DatasetMixtureMeta(type):
    """Metaclass that treats compatible mixtures as instances across reloads."""

    def __instancecheck__(cls, instance):
        base_check = type.__instancecheck__(cls, instance)
        duck_typed = (
            hasattr(instance, "datasets")
            and hasattr(instance, "seed")
            and hasattr(instance, "test_split_size")
        )
        return bool(base_check or duck_typed)


@dataclass
class DatasetMixtureConfig(metaclass=_DatasetMixtureMeta):
    """Configuration for a mixture of datasets.

    :ivar datasets: Ordered dataset entries combined into a single iterable.
    :ivar seed: Seed used for deterministic shuffling and sampling.
    :ivar test_split_size: Optional fraction moved to a held-out test split.
    :raises ValueError: If ``test_split_size`` is provided outside ``(0, 1)``.
    """

    datasets: List[DatasetConfig]
    seed: int = field(default=0)
    test_split_size: Optional[float] = field(default=None)

    def __post_init__(self) -> None:
        if self.test_split_size is not None and not 0 < self.test_split_size < 1:
            raise ValueError("test_split_size must be between 0 and 1")


@dataclass
class ScriptArguments(_BaseScriptArgs):
    """Extended TRL ScriptArguments with dataset mixture support.

    Accepts either a single dataset via ``dataset_name`` or a declarative
    mixture provided as a mapping. When a dictionary is supplied the payload is
    converted into :class:`DatasetMixtureConfig` entries and validated for
    consistent column naming across datasets.

    :ivar dataset_name: Dataset ID on the Hub when using a single source.
    :ivar dataset_mixture: Mixture configuration with constituent datasets.
    :ivar dataset_config: Optional config name associated with ``dataset_name``.
    :raises ValueError: If neither a dataset nor mixture is provided, or if the
        mixture payload is malformed.
    """

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
        except AttributeError as exc:
            LOG.debug("Skipping base ScriptArguments.__post_init__: %s", exc)
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError(
                "Either `dataset_name` or `dataset_mixture` must be provided"
            )

        if self.dataset_mixture is not None:
            self.dataset_mixture = self._coerce_dataset_mixture(self.dataset_mixture)

    def __setattr__(self, name, value):
        if name == "dataset_mixture" and value is not None:
            value = self._coerce_dataset_mixture(value)
        object.__setattr__(self, name, value)

    @staticmethod
    def _coerce_dataset_mixture(
        mixture: DatasetMixtureConfig | dict,
    ) -> DatasetMixtureConfig:
        if isinstance(mixture, DatasetMixtureConfig):
            return mixture
        if not isinstance(mixture, dict) or "datasets" not in mixture:
            raise ValueError(
                "dataset_mixture must be a dictionary with a 'datasets' key. "
                "Expected format: {'datasets': [...], 'seed': int}"
            )
        datasets_data = mixture.get("datasets", [])
        if not isinstance(datasets_data, list):
            raise ValueError("'datasets' must be a list of dataset configurations")
        datasets_list = [
            DatasetConfig(
                id=dataset_config.get("id"),
                config=dataset_config.get("config"),
                split=dataset_config.get("split", "train"),
                columns=dataset_config.get("columns"),
                weight=dataset_config.get("weight", 1.0),
            )
            for dataset_config in datasets_data
        ]
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
        return DatasetMixtureConfig(
            datasets=datasets_list,
            seed=mixture.get("seed", 0),
            test_split_size=mixture.get("test_split_size", None),
        )


__all__ = ["DatasetConfig", "DatasetMixtureConfig", "ScriptArguments"]
