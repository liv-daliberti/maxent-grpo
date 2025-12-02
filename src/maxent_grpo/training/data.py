"""Dataset loading helpers for the InfoSeed runner."""

from __future__ import annotations

from typing import Any, Tuple

from maxent_grpo.core.data import get_dataset
from maxent_grpo.training.runtime.prompts import (
    _prompt_char_limit_from_tokens,
    _to_prompt,
)


def load_datasets(
    script_args: Any, training_args: Any, tokenizer: Any
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
        out["answer"] = str(ex.get(sc, out.get("answer", "")))
        return out

    if hasattr(raw_ds, "map"):
        dataset = raw_ds.map(_map_fn)
    elif isinstance(raw_ds, dict):
        dataset = {
            split: [_map_fn(row) for row in rows] for split, rows in raw_ds.items()
        }
    else:
        dataset = {"train": [_map_fn(row) for row in raw_ds]}

    train_split = getattr(script_args, "dataset_train_split", "train")
    test_split = getattr(script_args, "dataset_test_split", None)
    if test_split is None:
        test_split = (
            "validation"
            if "validation" in dataset.keys()
            else ("test" if "test" in dataset.keys() else None)
        )
    train_ds = dataset[train_split]

    eval_rows = getattr(script_args, "eval_rows", None)
    if eval_rows is None and test_split is not None and test_split in dataset:
        eval_rows = dataset[test_split]
    if eval_rows is None:
        eval_rows = []
    return train_ds, eval_rows


__all__ = ["load_datasets"]
