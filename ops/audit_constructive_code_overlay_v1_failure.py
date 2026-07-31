#!/usr/bin/env python3
"""Classify the frozen ConstructiveCode v1 overlay admission failure."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SLATE = ROOT / "var/data/constructive_code_review_slate_v1"
REPLAY_ROOT = ROOT / "var/artifacts/constructive_code_overlay_replay_v1"
OUTPUT = ROOT / "var/artifacts/constructive_code_overlay_v1_failure_classification.json"
TASKS = ("482_a", "988_a", "1153_b", "149_c")
PYTHON3_LABELS = frozenset({"py3", "python3", "pypy3"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    receipt_path = REPLAY_ROOT / "receipt.json"
    equivalence_path = REPLAY_ROOT / "checker_equivalence_audit.json"
    receipt = _json(receipt_path)
    equivalence = _json(equivalence_path)
    task_counts: list[dict[str, Any]] = []
    incompatible_total = 0
    for task in TASKS:
        replay_path = SLATE / task / "python_replays.jsonl"
        counts: Counter[tuple[str, str]] = Counter()
        for line in replay_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            language = str(row.get("language", "")).strip().lower()
            label = str(row.get("known_label", ""))
            counts[(label, language)] += 1
            if language not in PYTHON3_LABELS:
                incompatible_total += 1
        task_counts.append(
            {
                "task": task.upper(),
                "counts_by_label_and_language": {
                    f"{label}:{language}": count
                    for (label, language), count in sorted(counts.items())
                },
                "replay_file_sha256": _sha256(replay_path),
            }
        )

    violations = list(equivalence.get("violations", []))
    payload = {
        "schema_version": "constructive-code-overlay-v1-failure-classification-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "fail",
        "decision": "constructive_code_v1_ineligible",
        "policy_sampling_authorized": False,
        "main_cohort_launch_authorized": False,
        "classification": {
            "runtime_language_contract": "fail",
            "reason": (
                "The frozen replay slate contains submissions not explicitly "
                "labeled Python 3, but the admission runner executed every "
                "submission in the pinned Python 3.10 runtime."
            ),
            "incompatible_replay_count": incompatible_total,
            "released_checker_equivalence": equivalence.get("status"),
            "released_checker_violation_count": len(violations),
            "throughput": receipt.get("throughput", {}).get("status"),
        },
        "tasks": task_counts,
        "next_allowed_design": (
            "A separately preregistered v2 may rematerialize an explicitly "
            "Python-3-only slate and rerun all admission gates before any "
            "model sample. v1 may not be repaired or substituted post hoc."
        ),
        "hashes": {
            "receipt_sha256": _sha256(receipt_path),
            "equivalence_audit_sha256": _sha256(equivalence_path),
            "replay_ledger_sha256": _sha256(REPLAY_ROOT / "replays.jsonl"),
            "audit_source_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUTPUT)
    print(
        "[constructive-overlay-v1-failure] "
        f"status=fail incompatible_replays={incompatible_total}"
    )


if __name__ == "__main__":
    main()
