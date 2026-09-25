"""Frozen MultiRC one-valid-option tasks with source-gold semantic mode maps.

This is an explicit adaptation, not the official MultiRC select-all metric.
Only a single uppercase option letter is accepted. Paraphrases share a mode.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


SCHEMA_VERSION = "multirc_one_valid_option_v1"
SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's answer format exactly."


def render_user_prompt(record: dict[str, Any]) -> str:
    choices = "\n".join(f"{o['display']}. {o['text']}" for o in record["options"])
    return (
        "Read the passage and question. Several answer options may be correct. "
        "Choose any ONE correct answer option supported by the passage. "
        "An option may give one of the requested examples or facts; you do not need to "
        "list every correct option. Return only its uppercase letter, with no explanation.\n\n"
        f"Passage:\n{record['passage']}\n\nQuestion:\n{record['question']}\n\n"
        f"Answer options:\n{choices}\n\nAnswer with one uppercase letter only."
    )


def render_prompt(record: dict[str, Any]) -> str:
    """Exact Qwen2.5 ChatML bytes; no tokenizer or model needed for CPU verification."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{render_user_prompt(record)}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@dataclass
class Task:
    record: dict[str, Any]

    @property
    def task_id(self) -> str:
        return self.record["id"]

    @property
    def prompt(self) -> str:
        return render_prompt(self.record)

    @property
    def family(self) -> str:
        return "multirc_science_one_valid_option"

    @property
    def split(self) -> str:
        return self.record["split"]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source_passage_id": self.record["source_passage_id"],
            "source_question_id": self.record["source_question_id"],
            "num_options": len(self.record["options"]),
            "gold_mode_count": len(set(self.record["gold_mode_map"].values())),
            "known_mode_count": len(set(self.record["gold_mode_map"].values())),
            "gold_mode_keys": sorted(set(self.record["gold_mode_map"].values())),
            "review_status": self.record["review_status"],
            "schema_version": SCHEMA_VERSION,
        }

    def verify(self, text: str) -> dict[str, Any]:
        answer = text.strip() if isinstance(text, str) else ""
        options = {o["display"]: o for o in self.record["options"]}
        violations = []
        chosen = None
        if not re.fullmatch(r"[A-Z]", answer):
            violations.append("expected_one_uppercase_letter")
        elif answer not in options:
            violations.append("letter_not_in_options")
        else:
            chosen = options[answer]
            if not chosen["label"]:
                violations.append("source_label_incorrect")
        accepted = not violations
        key = self.record["gold_mode_map"][chosen["id"]] if accepted else None
        return {
            "accepted": accepted,
            "canonical_key": key,
            "hard_violations": [],
            "receipt": {
                "rejection_reasons": violations,
                "verifier": SCHEMA_VERSION,
                "task_id": self.task_id,
                "display_letter": answer,
                "source_option_id": chosen["id"] if chosen else None,
                "source_label": int(chosen["label"]) if chosen else None,
                "semantic_mode": key,
                "response_sha256": hashlib.sha256(str(text).encode()).hexdigest(),
            },
        }


def _path(config: dict[str, Any]) -> Path:
    return Path(config["records_path"])


def validate_record(record: dict[str, Any]) -> None:
    options = record["options"]
    assert 3 <= len(options) <= 26, "invalid number of options"
    assert [o["display"] for o in options] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)])
    assert len({o["id"] for o in options}) == len(options), "duplicate source option"
    gold = {o["id"] for o in options if o["label"] == 1}
    assert all(type(o["label"]) is int and o["label"] in (0, 1) for o in options)
    assert len(gold) >= 2 and len(gold) < len(options)
    assert set(record["gold_mode_map"]) == gold, "gold mode map must cover exactly source-positive options"
    assert len(set(record["gold_mode_map"].values())) >= 2, "not semantically multimodal"
    assert record["review_status"] == "assistant_audited_source_labels_and_semantic_modes"
    assert record["split"] in {"train", "dev", "test"}
    for value in (record["passage"], record["question"], *(o["text"] for o in options)):
        assert "<|im_start|>" not in value and "<|im_end|>" not in value


def load_tasks(config: dict[str, Any]) -> list[Task]:
    records = [json.loads(line) for line in _path(config).read_text().splitlines() if line.strip()]
    tasks = []
    splits = config.get("splits")
    if isinstance(splits, str):
        splits = [splits]
    ids = config.get("task_ids")
    for record in records:
        validate_record(record)
        if splits is not None and record["split"] not in splits:
            continue
        if ids is not None and record["id"] not in ids:
            continue
        tasks.append(Task(record))
    assert len({t.task_id for t in tasks}) == len(tasks), "duplicate task id"
    return tasks


def dataset_identity(config: dict[str, Any]) -> dict[str, Any]:
    path = _path(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "records_path": str(path.resolve()),
        "records_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompt_format": "qwen2.5_chatml",
        "system_prompt": SYSTEM_PROMPT,
        "task": "explicit_one_valid_option_adaptation_of_MultiRC",
        "semantic_modes": "reviewed_gold_statement_classes_not_display_letters",
    }
