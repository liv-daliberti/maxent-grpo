"""Unit tests for maxent_grpo.training.data.load_datasets."""

from __future__ import annotations

import random
from types import SimpleNamespace

from maxent_grpo.training import data as training_data


class _ChatTokenizerStub:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [msg["content"] for msg in messages]
        prompt = " || ".join(parts)
        if add_generation_prompt:
            prompt += " <assistant>"
        return prompt


def _base_script_args() -> SimpleNamespace:
    return SimpleNamespace(
        dataset=None,
        dataset_prompt_column="problem",
        dataset_solution_column="answer",
        dataset_train_split="train",
        dataset_test_split="test",
        eval_dataset_name=None,
        eval_dataset_split="validation",
        eval_dataset_prompt_column=None,
        eval_dataset_solution_column=None,
        eval_dataset_config=None,
        eval_rows=None,
    )


def _base_training_args() -> SimpleNamespace:
    return SimpleNamespace(
        max_prompt_length=0,
        system_prompt="[SYS]",
        seed=42,
        do_eval=True,
    )


def test_load_datasets_prefers_eval_dataset(monkeypatch):
    script_args = _base_script_args()
    script_args.dataset = {
        "train": [{"problem": "train prompt", "answer": "train_answer"}],
        "test": [{"problem": "held-out prompt", "answer": "held_answer"}],
    }
    script_args.eval_dataset_name = "hf/math"
    script_args.eval_dataset_prompt_column = "question"
    script_args.eval_dataset_solution_column = "solution"
    eval_rows_raw = [
        {"question": "Eval Q1", "solution": "Ans1"},
        {"question": "Eval Q2", "solution": "Ans2"},
    ]
    training_args = _base_training_args()
    tokenizer = _ChatTokenizerStub()

    monkeypatch.setattr(training_data, "get_dataset", lambda args: args.dataset)
    monkeypatch.setattr(
        training_data,
        "load_dataset_split",
        lambda name, config, split: eval_rows_raw,
    )

    train_ds, eval_rows = training_data.load_datasets(
        script_args,
        training_args,
        tokenizer,
    )
    assert len(train_ds) == 1
    assert len(eval_rows) == len(eval_rows_raw)
    assert [row["answer"] for row in eval_rows] == ["Ans1", "Ans2"]
    assert all("Eval Q" in row["prompt"] for row in eval_rows)


def test_load_datasets_samples_test_split_when_no_eval(monkeypatch):
    script_args = _base_script_args()
    train_rows = [{"problem": "train prompt", "answer": "train_answer"}]
    test_rows = [
        {"problem": f"eval prompt {idx}", "answer": f"ans{idx}"} for idx in range(50)
    ]
    script_args.dataset = {"train": train_rows, "test": test_rows}
    script_args.eval_dataset_name = None
    training_args = _base_training_args()
    training_args.seed = 7
    tokenizer = _ChatTokenizerStub()

    monkeypatch.setattr(training_data, "get_dataset", lambda args: args.dataset)

    def _fail(*_args, **_kwargs):
        raise AssertionError("eval dataset loader should not be called")

    monkeypatch.setattr(training_data, "load_dataset_split", _fail)

    train_ds, eval_rows = training_data.load_datasets(
        script_args,
        training_args,
        tokenizer,
    )
    assert len(train_ds) == len(train_rows)
    expected_keep = min(1000, max(1, int(0.1 * len(test_rows))))
    assert len(eval_rows) == expected_keep
    assert all(row["answer"].startswith("ans") for row in eval_rows)
    rng = random.Random(training_args.seed)
    indices = list(range(len(test_rows)))
    rng.shuffle(indices)
    expected_answers = {test_rows[idx]["answer"] for idx in indices[:expected_keep]}
    assert {row["answer"] for row in eval_rows} == expected_answers
