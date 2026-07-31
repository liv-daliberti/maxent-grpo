from __future__ import annotations

import hashlib
import json

import pytest

from replay_constructive_code_v2 import (
    _canonical_sha256,
    _v1_hashes,
    _validate_submission_records,
    _validate_v1_logical_root,
)


import pytest as _pytest

from conftest import reload_constructive_code_v2_bases as _reload_v2_bases


@_pytest.fixture(autouse=True)
def _pristine_constructive_code_v2_bases():
    """Keep this module's v2 assertions independent of test execution order."""

    _reload_v2_bases()
    yield


def _record(label: str, index: int, language: str = "py3"):
    code = f"print({index!r}) # {label}"
    return {
        "known_label": label,
        "language": language,
        "code": code,
        "submission_sha256": hashlib.sha256(code.encode()).hexdigest(),
    }


def _records():
    records = [
        _record("correct", index, ("py3", "python3", "pypy3")[index % 3])
        for index in range(100)
    ] + [
        _record("incorrect", index, ("py3", "python3", "pypy3")[index % 3])
        for index in range(100)
    ]
    return sorted(
        records,
        key=lambda row: (row["known_label"], row["submission_sha256"]),
    )


def test_v2_replay_selection_accepts_exact_frozen_contract():
    selected = _validate_submission_records(_records(), set())
    assert len(selected) == 200
    assert sum(row.known_label == "correct" for row in selected) == 100


def test_v2_replay_selection_rejects_v1_hash_overlap():
    records = _records()
    with pytest.raises(ValueError, match="non-held-out"):
        _validate_submission_records(records, {records[0]["submission_sha256"]})


def test_v2_replay_selection_rejects_ambiguous_python_label():
    records = _records()
    records[0]["language"] = "python"
    with pytest.raises(ValueError, match="malformed"):
        _validate_submission_records(records, set())


def test_v2_replay_verifies_sealed_v1_exclusion_ledger(tmp_path):
    task_dir = tmp_path / "327_b"
    task_dir.mkdir()
    code = "print(1)"
    digest = hashlib.sha256(code.encode()).hexdigest()
    replay_path = task_dir / "python_replays.jsonl"
    replay_path.write_text(
        json.dumps({"code": code, "submission_sha256": digest}) + "\n"
    )
    replay_sha = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    task = {
        "schema_version": "constructive-code-review-task-v1",
        "source_problem_id": "327_B",
        "replays": {"python_replays_jsonl_sha256": replay_sha},
    }
    task["task_record_sha256"] = _canonical_sha256(task)
    (task_dir / "task.json").write_text(json.dumps(task))

    assert _v1_hashes(task_dir, "327_B") == {digest}
    replay_path.write_text(replay_path.read_text() + "\n")
    with pytest.raises(ValueError, match="sealed v1 exclusion ledger drift"):
        _v1_hashes(task_dir, "327_B")


def test_v2_replay_v1_root_contract_is_relocation_safe_but_logically_fixed():
    manifest = {
        "sources": {"v1_root": "var/data/constructive_code_review_slate_v1"}
    }
    _validate_v1_logical_root(manifest)

    for drifted in (
        "/tmp/copied-v1",
        "var/data/another-ledger",
        "../constructive_code_review_slate_v1",
    ):
        manifest["sources"]["v1_root"] = drifted
        with pytest.raises(ValueError, match="v1 exclusion root drift"):
            _validate_v1_logical_root(manifest)
