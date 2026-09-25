from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate_constructive_code_pilot_20260921 as pilot


ROOT = Path(__file__).resolve().parents[1]


def test_pass_at_k_matches_exact_draw_probability():
    assert pilot._pass_at_k(64, 0, 32) == 0
    assert pilot._pass_at_k(64, 64, 8) == 1
    assert pilot._pass_at_k(64, 1, 8) == pytest.approx(8 / 64)
    assert pilot._pass_at_k(64, 1, 32) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        pilot._pass_at_k(64, 65, 8)


def test_pcmd_conditions_on_success_and_enforces_30_draw_threshold():
    assert pilot._mode_metrics(["a"] * 29, 64)["pcmd"] is None
    one = pilot._mode_metrics(["a"] * 30, 64)
    assert one["pcmd"] == 0 and one["pcmd_eligible"]
    two = pilot._mode_metrics(["a"] * 15 + ["b"] * 15, 64)
    assert two["pcmd"] == pytest.approx(1 - 2 * 15 * 14 / (30 * 29))
    assert two["pass_at_1"] == pytest.approx(30 / 64)
    assert pilot._mode_metrics([str(i) for i in range(30)], 64)["pcmd"] == 1


def _summary_fixture():
    public = [{"source_problem_id": name, "problem_key": name, "statement_sha256": "s", "suite_id": "suite", "suite_sha256": "h"} for name in pilot.DEVELOPMENT_PROBLEMS]
    attempts = []
    for row in range(3):
        for sample in range(64):
            accepted = row < 2 and sample < 10
            attempts.append({"row_index": row, "sample_index": sample, "accepted": accepted, "canonical_key": str(sample % 2) if accepted else None, "hard_violations": [], "terminal_worker_record": True})
    return attempts, public


def test_capability_boundary_has_exact_denominator_and_missing_diversity():
    attempts, public = _summary_fixture()
    result = pilot._summarize(attempts, public)
    assert result["status"] == "pass"
    assert result["summary"]["aggregate_accuracy"] == pytest.approx(20 / 192)
    assert result["summary"]["multimode_tasks"] == 2
    assert result["summary"]["pcmd_eligible_tasks"] == 0
    assert result["summary"]["macro_pcmd_over_eligible_tasks"] is None
    attempts[0]["accepted"] = False
    assert pilot._summarize(attempts, public)["status"] == "capability_fail"


def test_audit_failure_is_distinct_and_duplicate_requests_fail_closed():
    attempts, public = _summary_fixture()
    attempts[-1]["hard_violations"] = ["sandbox failure"]
    assert pilot._summarize(attempts, public)["status"] == "audit_fail"
    with pytest.raises(ValueError, match="request identities"):
        pilot._summarize(attempts[:-1] + [attempts[0]], public)


def _replay_fixture():
    task = SimpleNamespace(problem_id="359_B", problem_key="key", suite_id="suite", suite_sha256="suitehash", checker_sha256="checkerhash", tests=[1, 2])
    candidate = {"row_index": 0, "sample_index": 0, "request_seed": 77101, "emitted_text_sha256": "sourcehash", "executed_source_sha256": "sourcehash", "fence_stripped": False, "token_count": 8, "finish_reason": "stop"}
    replay = {"source_problem_id": task.problem_id, "problem_key": task.problem_key, "suite_id": task.suite_id, "suite_sha256": task.suite_sha256, "checker_sha256": task.checker_sha256, "submission_sha256": "sourcehash", "released_checker_accepted": True, "wrapper_accepted": True, "behavior_key": "canonical", "execution": {"candidate_invocation_wall_seconds": [0.1, 0.1], "checker_wall_seconds": 0.1, "executed_tests": 2, "suite_tests": 2, "first_failure": None}}
    return candidate, task, replay


@pytest.mark.parametrize("field", ["checker_sha256", "suite_sha256", "submission_sha256", "source_problem_id"])
def test_replay_identity_mismatch_cannot_be_counted_as_accepted(field):
    candidate, task, replay = _replay_fixture()
    assert pilot._attempt(candidate, task, replay)["accepted"]
    replay[field] = "wrong"
    attempt = pilot._attempt(candidate, task, replay)
    assert not attempt["accepted"] and attempt["hard_violations"]


def test_partial_suite_and_worker_exception_fail_closed():
    candidate, task, replay = _replay_fixture()
    replay["execution"]["executed_tests"] = 1
    assert not pilot._attempt(candidate, task, replay)["accepted"]
    failed = pilot._attempt(candidate, task, None, "sandbox unavailable")
    assert not failed["terminal_worker_record"] and not failed["accepted"]


def test_raw_receipts_round_trip_source_without_loss(tmp_path):
    path = tmp_path / "raw.jsonl"
    row = {"emitted_text": "```python\nprint('é')\n```", "code": "print('é')"}
    with path.open("x", encoding="utf-8") as handle:
        pilot._append_jsonl(handle, [row])
        assert json.loads(path.read_text()) == row


def test_actual_overlay_changes_only_missing_public_equation(tmp_path):
    source = json.loads((ROOT / "var/data/constructive_code_v5/359_b/task.json").read_text())["statement"]
    row = {"source_problem_id": "359_B", "statement": source, "statement_sha256": hashlib.sha256(source.encode()).hexdigest()}
    overlay = ROOT / "docs/codecontests_pilot_prompt_overlay_20260921.json"
    revised, receipt = pilot._apply_prompt_overlay([row], overlay)
    assert "<image>" not in revised[0]["statement"]
    assert revised[0]["source_statement_sha256"] == row["statement_sha256"]
    assert revised[0]["statement_sha256"] != row["statement_sha256"]
    assert receipt["historical_359_B_prompt_matches"] is False
    tampered = json.loads(overlay.read_text())
    tampered["overlays"]["359_B"]["replacement_statement"] += " hint"
    path = tmp_path / "bad_overlay.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="identity drift"):
        pilot._apply_prompt_overlay([row], path)


def test_retained_replay_ledger_matches_gate_selection():
    gate = json.loads((ROOT / "var/artifacts/constructive_code_v6_gate_audit.json").read_text())
    rows = pilot._selected_audit_replays(ROOT / "var/artifacts/constructive_code_v5_replays.jsonl", gate)
    assert len(rows) == 960
    gate["selected_replays_sha256"] = "bad"
    with pytest.raises(ValueError, match="ledger identity drift"):
        pilot._selected_audit_replays(ROOT / "var/artifacts/constructive_code_v5_replays.jsonl", gate)


@pytest.mark.parametrize("changed_mode", [False, True])
def test_accepted_candidate_requires_a_stable_independent_replay(changed_mode):
    candidate, task, first = _replay_fixture()
    candidate["code"] = "print(1)"
    second = copy.deepcopy(first)
    if changed_mode:
        second["behavior_key"] = "different canonical mode"
    replies = iter([first, second])
    base = SimpleNamespace(Submission=lambda *a: a, _replay_submission=lambda **kw: next(replies))
    args = SimpleNamespace(launcher=None, runtime_root=None, scratch_root=None)
    attempt = pilot._verify_candidate(base, args, candidate, task)
    assert attempt["stability_recheck_required"]
    assert attempt["stability_recheck"]["replay"] == second
    assert attempt["accepted"] is (not changed_mode)
    assert bool(attempt["hard_violations"]) is changed_mode


def test_independent_replay_failure_is_retained_and_fails_audit():
    candidate, task, first = _replay_fixture()
    candidate["code"] = "print(1)"
    replies = iter([first])
    base = SimpleNamespace(Submission=lambda *a: a, _replay_submission=lambda **kw: next(replies))
    args = SimpleNamespace(launcher=None, runtime_root=None, scratch_root=None)
    attempt = pilot._verify_candidate(base, args, candidate, task)
    assert not attempt["accepted"]
    assert not attempt["stability_recheck"]["terminal_worker_record"]
    assert any("independent full-suite" in message for message in attempt["hard_violations"])
