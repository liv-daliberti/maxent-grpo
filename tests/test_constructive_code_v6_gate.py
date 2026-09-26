from pathlib import Path

import audit_constructive_code_v6_gate as gate


ROOT = Path(__file__).resolve().parents[1]


def test_v6_split_is_ten_problem_disjoint_tasks_and_train_covers_four_families():
    assert len(gate.SPLIT_ASSIGNMENT) == 10
    assert set(gate.SPLIT_ASSIGNMENT) == set(gate.SELECTED_SUITES)
    assert list(gate.SPLIT_ASSIGNMENT.values()).count("train") == 4
    assert list(gate.SPLIT_ASSIGNMENT.values()).count("development") == 3
    assert list(gate.SPLIT_ASSIGNMENT.values()).count("evaluation") == 3
    assert set(gate.EXCLUDED_TASKS) == {"1208_C", "1408_A"}


def test_v6_protocol_discloses_failed_v5_and_exact_replay_boundary():
    text = (ROOT / "paper/preregistration/constructive_code_executable_slate_v6_20260730.md").read_text()
    for phrase in (
        "ConstructiveCode v5 remains a failed experiment",
        "V6 is a new benchmark version",
        "exact gate contains 960 records",
        "no language-model sampling occurs in this gate",
    ):
        assert phrase in text


def test_shared_auditor_honors_the_explicit_required_suite_set():
    text = (ROOT / "ops/audit_constructive_code_v2.py").read_text()
    assert "len(suites) != len(required_suite_ids)" in text
