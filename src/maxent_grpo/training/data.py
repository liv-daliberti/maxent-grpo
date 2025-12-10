"""Dataset loading helpers for the InfoSeed runner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any, Tuple

try:  # pragma: no cover - optional dependency guard for stripped test envs
    from datasets import load_from_disk as _hf_load_from_disk
except Exception:  # pragma: no cover - fallback when datasets is unavailable
    _hf_load_from_disk = None

from maxent_grpo.core.data import get_dataset
from maxent_grpo.training.runtime.prompts import (
    _prompt_char_limit_from_tokens,
    _to_prompt,
)


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


def load_datasets(
    script_args: Any,
    training_args: Any,
    tokenizer: Any,
    *,
    accelerator: Any | None = None,
) -> Tuple[Any, list]:
    """Load train/eval datasets and return (train_dataset, eval_rows)."""

    raw_ds = get_dataset(script_args)
    pc = getattr(script_args, "dataset_prompt_column", "problem")
    sc = getattr(script_args, "dataset_solution_column", "answer")
    char_limit = _prompt_char_limit_from_tokens(
        getattr(training_args, "max_prompt_length", 0)
    )

    def _map_fn(ex: dict) -> dict:
        out = _to_prompt(
            ex,
            tokenizer,
            pc,
            getattr(training_args, "system_prompt", None),
            char_limit=char_limit,
        )
        answer = ex.get(sc, out.get("answer", ""))
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
    if test_split is None:
        test_split = "validation" if hasattr(raw_ds, "keys") and "validation" in raw_ds else None
        if test_split is None and hasattr(raw_ds, "keys") and "test" in raw_ds:
            test_split = "test"

    dataset = None
    cache_path = None
    wait_fn = getattr(accelerator, "wait_for_everyone", None)
    is_main_process = bool(getattr(accelerator, "is_main_process", True))

    def _build_hf_dataset():
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

    if hasattr(raw_ds, "map"):
        cache_path = _dataset_cache_path(
            script_args,
            training_args,
            prompt_column=pc,
            solution_column=sc,
            train_split=train_split,
            char_limit=char_limit,
        )
        cache_enabled = bool(cache_path)
        if cache_enabled and os.path.isdir(cache_path) and _hf_load_from_disk:
            dataset = _hf_load_from_disk(cache_path)
        else:
            if accelerator is None or is_main_process:
                dataset = _build_hf_dataset()
                if cache_enabled and hasattr(dataset, "save_to_disk"):
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    tmp_dir = f"{cache_path}.tmp"
                    if os.path.isdir(tmp_dir):
                        shutil.rmtree(tmp_dir)
                    dataset.save_to_disk(tmp_dir)
                    os.replace(tmp_dir, cache_path)
            if callable(wait_fn):
                wait_fn()
            if dataset is None:
                if cache_enabled and os.path.isdir(cache_path) and _hf_load_from_disk:
                    dataset = _hf_load_from_disk(cache_path)
                else:
                    dataset = _build_hf_dataset()
    elif isinstance(raw_ds, dict):
        dataset = {}
        for split, rows in raw_ds.items():
            mapped = [_map_fn(row) for row in rows]
            dataset[split] = [row for row in mapped if _is_valid(row)]
    else:
        mapped_rows = [_map_fn(row) for row in raw_ds]
        dataset = {"train": [row for row in mapped_rows if _is_valid(row)]}

    train_ds = dataset[train_split]

    eval_rows = getattr(script_args, "eval_rows", None)
    if eval_rows is None and test_split is not None and test_split in dataset:
        eval_rows = dataset[test_split]
    if eval_rows is None:
        eval_rows = []
    return train_ds, eval_rows


__all__ = ["load_datasets"]
