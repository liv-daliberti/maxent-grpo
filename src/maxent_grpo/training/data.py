"""Dataset loading helpers for the InfoSeed runner."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import logging
from typing import Any, Callable, List, Mapping, Optional, Tuple, cast

try:  # pragma: no cover - optional dependency guard for stripped test envs
    from datasets import load_from_disk as _hf_load_from_disk
except (ImportError, ModuleNotFoundError):  # pragma: no cover - fallback when datasets is unavailable
    _hf_load_from_disk = None

from maxent_grpo.core.data import get_dataset, load_dataset_split
from maxent_grpo.training.runtime.prompts import (
    _prompt_char_limit_from_tokens,
    _to_prompt,
)

LOG = logging.getLogger(__name__)

def _stable_hash(value: Any) -> str:
    """Return a short, stable hash for arbitrarily typed values."""

    blob = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _dataset_cache_path(
    script_args: Any,
    training_args: Any,
    *,
    prompt_column: str,
    solution_column: str,
    train_split: str,
    char_limit: int,
) -> str:
    """Resolve an on-disk cache directory for the processed dataset."""

    base_dir = (
        getattr(training_args, "dataset_cache_dir", None)
        or getattr(script_args, "dataset_cache_dir", None)
        or os.environ.get("MAXENT_DATASET_CACHE_DIR")
    )
    if not base_dir:
        output_dir = getattr(training_args, "output_dir", None)
        if not output_dir:
            output_dir = os.getcwd()
        base_dir = os.path.join(output_dir, "dataset_cache")
    dataset_id = {
        "name": getattr(script_args, "dataset_name", None),
        "mixture": getattr(script_args, "dataset_mixture", None),
        "revision": getattr(script_args, "dataset_revision", None),
        "train_split": train_split,
        "prompt_column": prompt_column,
        "solution_column": solution_column,
        "char_limit": int(char_limit or 0),
        "system_prompt_hash": _stable_hash(
            getattr(training_args, "system_prompt", "") or ""
        ),
        "max_prompt_length": int(getattr(training_args, "max_prompt_length", 0) or 0),
    }
    cache_key = _stable_hash(dataset_id)
    return os.path.join(base_dir, cache_key)


def _format_eval_row(
    example: Mapping[str, Any],
    *,
    prompt_column: str,
    solution_column: str,
    tokenizer: Any,
    system_prompt: Optional[str],
    char_limit: int,
) -> dict:
    example_map = dict(example)
    prompt_col = prompt_column
    if prompt_col not in example_map and prompt_col == "problem" and "prompt" in example_map:
        prompt_col = "prompt"
    out = _to_prompt(
        example_map,
        tokenizer,
        prompt_col,
        system_prompt,
        char_limit=char_limit,
    )
    answer = example_map.get(solution_column, out.get("answer", ""))
    out["answer"] = "" if answer is None else str(answer)
    return out


def _normalize_eval_rows(rows: Any) -> Optional[List[dict]]:
    if rows is None:
        return None
    if isinstance(rows, list):
        return [dict(row) for row in rows]
    normalized: List[dict] = []
    try:
        iterator = iter(rows)
    except TypeError:
        return normalized
    for row in iterator:
        normalized.append(dict(row))
    return normalized


def _ensure_split_mapping(dataset: Any) -> Mapping[str, Any]:
    """Coerce a dataset-like object into a split->dataset mapping."""

    if isinstance(dataset, dict):
        return dataset
    if hasattr(dataset, "keys") and hasattr(dataset, "__getitem__"):
        return cast(Mapping[str, Any], dataset)
    return {"train": dataset}


def _sample_eval_rows(rows: List[dict], keep: int, seed: int) -> List[dict]:
    if keep <= 0 or keep >= len(rows):
        return rows
    indices = list(range(len(rows)))
    random.Random(int(seed or 0)).shuffle(indices)
    return [rows[idx] for idx in indices[:keep]]


def load_datasets(
    script_args: Any,
    training_args: Any,
    tokenizer: Any,
    *,
    accelerator: Any | None = None,
) -> Tuple[Any, list]:
    """Load train/eval datasets and return ``(train_dataset, eval_rows)``.

    The helper handles prompt/answer column normalization, optional dataset
    caching, and prompt truncation. Evaluation rows are normalized into a list
    of dictionaries with ``prompt``/``answer`` keys.

    :param script_args: Script arguments describing dataset identifiers and
        prompt/answer columns.
    :param training_args: Training configuration providing prompt limits and
        cache settings.
    :param tokenizer: Tokenizer used to format prompts.
    :param accelerator: Optional Accelerator used for process synchronization.
    :returns: Tuple of the processed training dataset and a list of evaluation
        rows (possibly empty when eval is disabled).
    :rtype: tuple[Any, list]
    :raises ValueError: If required dataset columns are missing.
    """

    pc = getattr(script_args, "dataset_prompt_column", "problem")
    sc = getattr(script_args, "dataset_solution_column", "answer")
    char_limit = _prompt_char_limit_from_tokens(
        getattr(training_args, "max_prompt_length", 0)
    )

    def _map_fn(ex: Mapping[str, Any]) -> dict:
        ex_map = dict(ex)
        prompt_col = pc
        if prompt_col not in ex_map and prompt_col == "problem" and "prompt" in ex_map:
            prompt_col = "prompt"
        out = _to_prompt(
            ex_map,
            tokenizer,
            prompt_col,
            getattr(training_args, "system_prompt", None),
            char_limit=char_limit,
        )
        answer = ex_map.get(sc, out.get("answer", ""))
        out["answer"] = "" if answer is None else str(answer)
        return out

    def _is_valid(row: dict) -> bool:
        """Drop rows missing a prompt or answer to keep collate happy."""

        prompt = row.get("prompt")
        answer = row.get("answer")
        return (
            isinstance(prompt, str)
            and isinstance(answer, str)
            and prompt.strip() != ""
            and answer.strip() != ""
        )

    train_split = getattr(script_args, "dataset_train_split", "train")
    test_split = getattr(script_args, "dataset_test_split", None)

    dataset = None
    cache_path = None
    is_main_process = bool(getattr(accelerator, "is_main_process", True))

    def _wait_for_everyone() -> None:
        if accelerator is None:
            return
        wait_for_all: Optional[Callable[[], None]] = getattr(
            accelerator, "wait_for_everyone", None
        )
        if wait_for_all is not None:
            wait_for_all()

    cache_path = _dataset_cache_path(
        script_args,
        training_args,
        prompt_column=pc,
        solution_column=sc,
        train_split=train_split,
        char_limit=char_limit,
    )
    cache_enabled = bool(cache_path and _hf_load_from_disk)
    if cache_enabled and os.path.isdir(cache_path) and _hf_load_from_disk:
        dataset = _hf_load_from_disk(cache_path)

    def _build_hf_dataset() -> Any:
        raw_ds = get_dataset(script_args)
        # Remove all original columns so the DataLoader only sees prompt/answer.
        remove_cols = getattr(raw_ds, "column_names", None)
        if isinstance(remove_cols, dict):
            # DatasetDict: merge column names across splits.
            merged = set()
            for cols in remove_cols.values():
                merged.update(cols)
            remove_cols = list(merged)
        if isinstance(remove_cols, list):
            remove_cols = [c for c in remove_cols if c not in {"prompt", "answer"}]
        else:
            remove_cols = None

        mapped = raw_ds.map(_map_fn, remove_columns=remove_cols, desc="Map")
        if hasattr(mapped, "filter"):
            mapped = mapped.filter(_is_valid, desc="Filter")
        return mapped

    if dataset is None and accelerator is not None and not is_main_process and cache_enabled:
        _wait_for_everyone()
        if os.path.isdir(cache_path) and _hf_load_from_disk:
            dataset = _hf_load_from_disk(cache_path)
        else:  # pragma: no cover - indicates main process failed before caching
            raise RuntimeError(
                f"Expected dataset cache at {cache_path} but it was not created."
            )

    if dataset is None:
        raw_ds = cast(Any, get_dataset(script_args))
        if hasattr(raw_ds, "map"):
            if accelerator is None or is_main_process:
                dataset = _build_hf_dataset()
                if cache_enabled and hasattr(dataset, "save_to_disk"):
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    tmp_dir = f"{cache_path}.tmp"
                    if os.path.isdir(tmp_dir):
                        shutil.rmtree(tmp_dir)
                    dataset.save_to_disk(tmp_dir)
                    os.replace(tmp_dir, cache_path)
            _wait_for_everyone()
            if dataset is None:
                if cache_enabled and os.path.isdir(cache_path) and _hf_load_from_disk:
                    dataset = _hf_load_from_disk(cache_path)
                else:
                    dataset = _build_hf_dataset()
        elif isinstance(raw_ds, dict):
            dataset = {}
            for split, rows in raw_ds.items():
                mapped = [_map_fn(cast(Mapping[str, Any], row)) for row in rows]
                dataset[split] = [row for row in mapped if _is_valid(row)]
        else:
            mapped_rows = [_map_fn(cast(Mapping[str, Any], row)) for row in raw_ds]
            dataset = {"train": [row for row in mapped_rows if _is_valid(row)]}

    dataset_map = _ensure_split_mapping(dataset)
    if test_split is None:
        test_split = "validation" if "validation" in dataset_map else None
        if test_split is None and "test" in dataset_map:
            test_split = "test"

    if train_split not in dataset_map:
        train_split = "train" if "train" in dataset_map else list(dataset_map.keys())[0]
    train_ds = dataset_map[train_split]

    eval_rows = _normalize_eval_rows(getattr(script_args, "eval_rows", None))
    if eval_rows is None and getattr(training_args, "do_eval", False):
        eval_dataset_name = getattr(script_args, "eval_dataset_name", None)
        eval_prompt_col = (
            getattr(script_args, "eval_dataset_prompt_column", None) or pc
        )
        eval_solution_col = (
            getattr(script_args, "eval_dataset_solution_column", None) or sc
        )
        if eval_dataset_name:
            eval_split = getattr(script_args, "eval_dataset_split", "validation")
            eval_ds_raw = load_dataset_split(
                eval_dataset_name,
                getattr(script_args, "eval_dataset_config", None),
                eval_split,
            )
            eval_rows = [
                _format_eval_row(
                    cast(Mapping[str, Any], row),
                    prompt_column=eval_prompt_col,
                    solution_column=eval_solution_col,
                    tokenizer=tokenizer,
                    system_prompt=getattr(training_args, "system_prompt", None),
                    char_limit=char_limit,
                )
                for row in eval_ds_raw
            ]
        elif test_split is not None and test_split in dataset_map:
            full_eval = dataset_map[test_split]
            try:
                n_total = len(full_eval)
            except (TypeError, AttributeError):
                n_total = 0
            n_keep = min(1000, max(1, int(0.1 * n_total))) if n_total > 0 else 0
            shuffle_fn = getattr(full_eval, "shuffle", None)
            if callable(shuffle_fn) and n_keep > 0:
                try:
                    shuffled = shuffle_fn(seed=training_args.seed)
                    select_fn = getattr(shuffled, "select", None)
                    if callable(select_fn):
                        subset = select_fn(range(n_keep))
                        eval_rows = _normalize_eval_rows(subset)
                    else:
                        eval_rows = _normalize_eval_rows(shuffled)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    eval_rows = None
            if eval_rows is None:
                eval_rows = _normalize_eval_rows(full_eval)
                if eval_rows and n_keep > 0:
                    eval_rows = _sample_eval_rows(
                        eval_rows, n_keep, getattr(training_args, "seed", 0)
                    )
    if eval_rows is None:
        eval_rows = []
    return train_ds, eval_rows


def resolve_dataloader_kwargs(training_args: Any) -> dict:
    """Return ``torch.utils.data.DataLoader`` kwargs derived from training_args.

    :param training_args: Training config or namespace containing DataLoader
        knobs such as ``dataloader_num_workers`` and ``dataloader_pin_memory``.
    :returns: Dictionary of keyword arguments suitable for ``DataLoader``.
    :rtype: dict
    """

    kwargs: dict[str, Any] = {}
    num_workers = int(getattr(training_args, "dataloader_num_workers", 0) or 0)
    if num_workers < 0:
        num_workers = 0
    kwargs["num_workers"] = num_workers
    pin_memory = getattr(training_args, "dataloader_pin_memory", None)
    if pin_memory is not None:
        kwargs["pin_memory"] = bool(pin_memory)
    prefetch = getattr(training_args, "dataloader_prefetch_factor", None)
    if prefetch is not None:
        if num_workers > 0:
            try:
                kwargs["prefetch_factor"] = int(prefetch)
            except (TypeError, ValueError):
                LOG.warning("Invalid dataloader_prefetch_factor=%s; ignoring.", prefetch)
        else:
            LOG.debug("Ignoring dataloader_prefetch_factor because num_workers=0.")
    persistent = getattr(training_args, "dataloader_persistent_workers", None)
    if persistent is not None:
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(persistent)
        else:
            LOG.debug("Ignoring dataloader_persistent_workers because num_workers=0.")
    return kwargs


__all__ = ["load_datasets", "resolve_dataloader_kwargs"]
