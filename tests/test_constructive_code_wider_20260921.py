from __future__ import annotations

import pytest

import build_constructive_code_wider_20260921 as builder
from oat_drgrpo.constructive_code import ConstructiveCodeError, ReleasedCheckerDecision, sha256_bytes
from oat_drgrpo import constructive_code_wider_adapters_20260921 as adapters


def witness(problem, stdin, output):
    decision = ReleasedCheckerDecision("a" * 64, sha256_bytes(stdin.encode()), sha256_bytes(output.encode()), True, 0)
    return adapters.canonicalize_task_witness(problem_id=problem, adapter_id=builder.ALL_TASKS[problem][1], input_data=stdin, output=output, decision=decision)


@pytest.mark.parametrize("problem", list(builder.ALL_TASKS))
def test_semantic_alternatives_and_declared_equivalences(problem):
    stdin, first, different, equivalent = builder.CANONICAL_PROBES[problem]
    first_key = witness(problem, stdin, first).canonical_key
    assert first_key != witness(problem, stdin, different).canonical_key
    assert first_key == witness(problem, stdin, equivalent).canonical_key


def test_named_children_are_not_anonymous_partition_labels():
    stdin = "2 2\n1 2\n"
    a = adapters.canonical_value("244_A", stdin, "1 3 2 4")
    b = adapters.canonical_value("244_A", stdin, "2 4 1 3")
    assert a != b


def test_multisets_retain_multiplicity():
    assert adapters.canonical_value("1352_B", "1\n12 3\n", "YES 2 2 8") != adapters.canonical_value("1352_B", "1\n12 3\n", "YES 2 4 6")


def test_input_checker_rejects_out_of_statement_matrix_dimensions():
    with pytest.raises(ConstructiveCodeError, match="outside task constraints"):
        adapters.input_cases("1016_D", "1 2\n0\n0 0\n")


def test_accepted_checker_binding_cannot_be_transferred_to_another_output():
    stdin = "1\n3\n"
    decision = ReleasedCheckerDecision("a" * 64, sha256_bytes(stdin.encode()), sha256_bytes(b"2 3 1"), True, 0)
    with pytest.raises(ConstructiveCodeError, match="exact accepted checker binding"):
        adapters.canonicalize_task_witness(problem_id="1454_A", adapter_id="wider_1454_a_v1", input_data=stdin, output="3 1 2", decision=decision)


def test_original_heldout_ids_are_never_broader_development_tasks():
    assert not set(builder.TASKS) & set(builder.ORIGINAL_HELDOUT)
    assert "483_C" not in builder.TASKS
    assert len(builder.TASKS) == 24


def test_reserved_tasks_are_prospectively_separated():
    assert len(builder.NEW_HELDOUT) == 16
    assert len(builder.TRAIN_RESERVE) == 12
    assert not set(builder.NEW_HELDOUT) & (set(builder.TASKS) | set(builder.TRAIN_RESERVE))


def test_statement_transcription_is_bound_to_exact_source_tokens():
    repaired, provenance = builder.repair_statement("1016_D", "109 109 109")
    assert repaired == "10^9 10^9 10^9"
    assert provenance["source_statement_sha256"] != provenance["effective_statement_sha256"]
    with pytest.raises(ValueError, match="precondition drift"):
        builder.repair_statement("1016_D", "10^9")


def test_reserved_input_prime_promise_is_checked():
    with pytest.raises(ConstructiveCodeError, match="prime is composite"):
        adapters.input_cases("1549_A", "1\n9\n")


def test_statement_finalization_rebinds_manifest_without_losing_source(tmp_path, monkeypatch):
    import json
    reservation = tmp_path / "reservation.json"
    builder.write_json(reservation, {"test": ["1023_C"]})
    monkeypatch.setattr(builder, "RESERVATION", reservation)
    monkeypatch.setattr(builder, "RESERVATION_SHA256", builder.canonical_hash({"test": ["1023_C"]}))
    builder.write_json(tmp_path / "545_b/task.json", {"source_problem_id": "545_B", "statement": "n <= 105"})
    builder.write_json(tmp_path / "manifest.json", {"status": "audited", "tasks": [{"relative_path": "545_b"}]})
    builder.freeze_statement_repairs(tmp_path)
    row = json.loads((tmp_path / "545_b/task.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert row["source_statement"] == "n <= 105"
    assert row["statement"] == "n <= 10^5"
    assert row["task_record_sha256"] == manifest["tasks"][0]["task_record_sha256"]
    builder.freeze_statement_repairs(tmp_path)
    assert json.loads((tmp_path / "545_b/task.json").read_text()) == row


def test_reviewed_binary_input_contract_rejects_literal_regex_anchors():
    assert adapters.input_cases("545_B", "00\n11\n") == [{"s": "00", "t": "11"}]
    for text in ("^00$\n^11$\n", "00 11\n", "00\n111\n", "00\n11\n\n"):
        with pytest.raises(ConstructiveCodeError):
            adapters.input_cases("545_B", text)
