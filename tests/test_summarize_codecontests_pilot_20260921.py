from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import summarize_codecontests_pilot_20260921 as report


def _write_fixture(root: Path):
    responses, attempts = [], []
    for r, problem in enumerate(report.PROBLEMS):
        for sample in range(64):
            code = "print(1)"
            digest = hashlib.sha256(code.encode()).hexdigest()
            shared = {"row_index": r, "sample_index": sample, "source_problem_id": problem, "request_seed": 77101 + 10000 * r + sample, "emitted_text_sha256": digest, "executed_source_sha256": digest, "token_count": 1}
            responses.append({**shared, "emitted_text": code, "code": code, "token_ids": [1], "finish_reason": "stop"})
            attempts.append({**shared, "accepted": False, "canonical_key": None, "hard_violations": [], "terminal_worker_record": True, "execution": {"first_failure": {"stage": "released_checker"}}, "stability_recheck_required": False, "stability_recheck": None})
    artifacts = {}
    for name, rows in (("responses", responses), ("attempts", attempts)):
        path = root / (name + ".jsonl")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        artifacts[name] = {"path": str(path), "sha256": report.sha256(path)}
    receipt = {
        "schema_version": "constructive-code-coder-7b-capability-pilot-20260921-v1",
        "status": "capability_fail", "artifacts": artifacts, "attempts": attempts,
        "information_boundary": {"loaded_problem_ids": list(report.PROBLEMS), "evaluation_rows_loaded": False},
        "prompt_results": [{"accepted": 0, "distinct_valid_modes": 0, "pcmd": None, "pass_at_1": 0, "pass_at_8": 0, "pass_at_32": 0} for _ in range(3)],
        "model": "Coder-7B", "model_tree_sha256": "modelhash", "source_hash": "sourcehash", "execution_hash": "exechash", "prompt_overlay": {},
        "gpu": {"names": ["A6000"], "visible_count": 1},
        "timing": {"generation_wall_seconds": 10, "generation_samples_per_second": 19.2, "generation_tokens_per_second": 19.2, "execution_wall_seconds": 1, "model_load_seconds": 1, "estimated_allocated_gpu_hours_during_evaluator": 0.01},
    }
    path = root / "capability.json"
    path.write_text(json.dumps(receipt))
    return path


def test_no_success_reports_missing_diversity_and_zero_curves(tmp_path):
    path = _write_fixture(tmp_path)
    receipt, responses, attempts = report.load_verified(path)
    summary = report.summarize(receipt, responses, attempts, {"allocated_gpu_count": 1, "elapsed_seconds": 1800})
    assert summary["status"] == "capability_fail"
    assert summary["macro_pcmd_over_eligible_tasks"] is None
    assert summary["scheduler_allocated_gpu_hours"] == 0.5
    assert all(point["pass_at_k"] == 0 and point["expected_distinct_correct_modes_at_k"] == 0 for curve in summary["sampling_curves"] for point in curve["points"])
    text = report.report(summary)
    assert "capability gate did not pass" in text
    assert "not a pure model-size comparison" in text
    assert "No model training" in text


def test_raw_sidecar_modification_is_rejected(tmp_path):
    path = _write_fixture(tmp_path)
    raw = tmp_path / "responses.jsonl"
    raw.write_text(raw.read_text().replace("print(1)", "print(2)", 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        report.load_verified(path)


def test_mutated_final_acceptance_is_rejected(tmp_path):
    path = _write_fixture(tmp_path)
    receipt = json.loads(path.read_text())
    receipt["attempts"][0]["accepted"] = True
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="differs from final"):
        report.load_verified(path)


def test_without_replacement_curves_do_not_assume_independence():
    assert report.draw_probability(64, 1, 32) == 0.5
    assert report.draw_probability(64, 1, 64) == 1
    assert report.draw_probability(64, 0, 64) == 0
