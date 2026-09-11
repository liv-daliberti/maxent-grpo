from pathlib import Path

import evaluate_constructive_code_v6_coder_viability as v6


ROOT = Path(__file__).resolve().parents[1]


def _gate():
    tasks = []
    suites = []
    for problem_id, suite_id in v6.v6_gate.SELECTED_SUITES.items():
        problem_key = f"key-{problem_id}"
        tasks.append(
            {
                "source_problem_id": problem_id,
                "problem_key": problem_key,
                "selected_suite_id": suite_id,
                "status": "pass",
            }
        )
        suites.append(
            {"problem_key": problem_key, "suite_id": suite_id, "status": "pass"}
        )
    return {
        "status": "pass",
        "expected_replay_count": 960,
        "observed_replay_count": 960,
        "violations": [],
        "checker_equivalence_violations": [],
        "evaluation_rows_loaded": False,
        "language_model_sampling": False,
        "task_results": tasks,
        "suite_results": suites,
    }


def test_v6_viability_uses_three_development_tasks_and_no_evaluation_rows():
    assert v6.DEVELOPMENT_PROBLEMS == ("359_B", "988_A", "1399_D")
    assert v6.EVALUATION_PROBLEMS == ("361_B", "1294_C", "149_C")
    keys = {problem: f"key-{problem}" for problem in v6.DEVELOPMENT_PROBLEMS}
    assert v6._validate_v6_gate_and_choose_suites(_gate(), keys) == {
        problem: v6.v6_gate.SELECTED_SUITES[problem]
        for problem in v6.DEVELOPMENT_PROBLEMS
    }


def test_v6_viability_protocol_freezes_192_requests():
    text = (ROOT / "paper/preregistration/constructive_code_v6_coder_05b_viability_20260730.md").read_text()
    assert "192 requests total" in text
    assert "evaluation tasks 361B, 1294C, and 149C are never loaded" in text


def test_v6_slurm_leaves_runtime_extraction_destination_absent():
    text = (
        ROOT / "ops/slurm/evaluate_constructive_code_v6_coder_viability.slurm"
    ).read_text()
    mkdir_line = next(line for line in text.splitlines() if line.startswith("mkdir -p "))
    assert '"$RUN_ROOT/build"' in mkdir_line
    assert '"$RUN_ROOT/scratch"' in mkdir_line
    assert '"$RUN_ROOT/runtime"' not in mkdir_line
