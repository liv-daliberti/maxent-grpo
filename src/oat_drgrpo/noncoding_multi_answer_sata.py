"""SATA-Bench Reuters topic tagging adapted to one valid answer per sample."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

VERSION = "sata_reuters_one_valid_topic_v1"
SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's answer format exactly."


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def user_prompt(record: dict[str, Any]) -> str:
    choices = "\n".join(f"{o['display']}. {o['text']}" for o in record["options"])
    return (
        "Read the document and choose ONE correct topic from the listed options. "
        "Other listed topics may also be correct. Use the topic labels as provided. "
        "Return only the uppercase letter of your chosen topic, with no explanation.\n\n"
        f"Document:\n{record['paragraph']}\n\n"
        f"Question:\n{record['source_question']}\n\n"
        f"Options:\n{choices}\n\nChoose one correct topic. Answer with one uppercase letter only."
    )


def render_prompt(record: dict[str, Any]) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt(record)}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def validate_record(record: dict[str, Any]) -> None:
    options = record["options"]
    if record["schema_version"] != VERSION or record["domain"] != "news":
        raise ValueError("Unrecognized task schema/domain")
    if record["split"] not in {"train", "dev", "test"}:
        raise ValueError("Invalid split")
    if not 3 <= len(options) <= 26:
        raise ValueError("Invalid option count")
    if [o["display"] for o in options] != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]):
        raise ValueError("Invalid display-letter permutation")
    if len({o["id"] for o in options}) != len(options):
        raise ValueError("Duplicate semantic option IDs")
    if any(type(o["label"]) is not int or o["label"] not in (0, 1) for o in options):
        raise ValueError("Labels must be native binary targets")
    gold = {o["id"] for o in options if o["label"]}
    if not 2 <= len(gold) < len(options) or gold != set(record["gold_topic_ids"]):
        raise ValueError("Gold support must contain multiple distinct labels and a distractor")
    for o in options:
        if o["id"] != "reuters_topic:" + o["text"]:
            raise ValueError("Mode identity must be the unchanged native topic label")
    if set(record["source_answer_groups"]) != {o["text"] for o in options if o["label"]}:
        raise ValueError("Source positive annotations were changed")
    if set(record["source_distractor_groups"]) != {o["text"] for o in options if not o["label"]}:
        raise ValueError("Source negative annotations were changed")
    for s in (record["paragraph"], record["source_question"], *(o["text"] for o in options)):
        if "<|im_start|>" in s or "<|im_end|>" in s:
            raise ValueError("Embedded chat delimiter")


@dataclass
class Task:
    record: dict[str, Any]

    @property
    def task_id(self):
        return self.record["id"]

    @property
    def prompt(self):
        return render_prompt(self.record)

    @property
    def family(self):
        return "sata_reuters_news_topic"

    @property
    def split(self):
        return self.record["split"]

    @property
    def metadata(self):
        return {
            "known_mode_count": len(self.record["gold_topic_ids"]),
            "gold_mode_count": len(self.record["gold_topic_ids"]),
            "gold_mode_keys": self.record["gold_topic_ids"],
            "num_options": len(self.record["options"]),
            "domain": self.record["domain"],
            "source_row_index": self.record["source_row_index"],
            "document_group": self.record["document_group"],
        }

    def verify(self, text):
        answer = text.strip() if isinstance(text, str) else ""
        by_letter = {o["display"]: o for o in self.record["options"]}
        option, reasons = None, []
        if not re.fullmatch(r"[A-Z]", answer):
            reasons.append("expected_one_uppercase_letter")
        elif answer not in by_letter:
            reasons.append("letter_not_in_options")
        else:
            option = by_letter[answer]
            if not option["label"]:
                reasons.append("topic_not_in_source_gold")
        accepted = not reasons
        return {
            "accepted": accepted,
            "canonical_key": option["id"] if accepted else None,
            "hard_violations": [],
            "receipt": {
                "verifier": VERSION,
                "source": "SATA-Bench human-validated target annotations",
                "task_id": self.task_id,
                "response_sha256": hashlib.sha256(str(text).encode()).hexdigest(),
                "display_letter": answer,
                "native_topic": option["text"] if option else None,
                "source_label": option["label"] if option else None,
                "rejection_reasons": reasons,
            },
        }


def load_tasks(config):
    path = Path(config["records_path"])
    if config.get("records_sha256") and digest(path) != config["records_sha256"]:
        raise ValueError("Frozen records SHA256 mismatch")
    splits = config.get("splits", ["train", "dev"])
    if isinstance(splits, str):
        splits = [splits]
    if "test" in splits and not config.get("allow_test", False):
        raise ValueError("Reserved test requires explicit allow_test; use dev during pilot")
    ids = config.get("task_ids")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tasks = []
    for record in records:
        validate_record(record)
        if record["split"] in splits and (ids is None or record["id"] in ids):
            tasks.append(Task(record))
    if len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    return tasks


def dataset_identity(config):
    path = Path(config["records_path"])
    manifest_path = Path(config.get("manifest_path", path.parent / "noncoding_multi_answer_sata_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    if manifest["records_sha256"] != digest(path):
        raise ValueError("Manifest/records integrity mismatch")
    return {
        "schema_version": VERSION,
        "records_path": str(path.resolve()),
        "records_sha256": digest(path),
        "manifest_sha256": digest(manifest_path),
        "adapter_sha256": digest(Path(__file__)),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "config": config,
        "source": manifest["source"],
        "split_counts": manifest["split_counts"],
        "prompt_format": "qwen2.5_chatml",
        "system_prompt": SYSTEM_PROMPT,
        "adaptation": "one_valid_listed_topic_per_sample; not official select-all evaluation",
        "mode_identity": "unchanged Reuters topic ID, invariant to display-letter permutation",
    }
