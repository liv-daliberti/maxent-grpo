from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e75r3_point_maze_waypoint_final_eval.py"
ANALYZER = ROOT / "ops/exp_scaling/analyze_e75r3_point_maze_waypoint_final_eval.py"
PROTOCOL = ROOT / "paper/preregistration/e75r3_point_maze_waypoint_final_eval_20260804.md"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("e75r3_final_analyzer", ANALYZER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_plan_is_frozen_before_any_submission_and_has_no_exclusions():
    source = LAUNCHER.read_text()
    freeze = source.index("atomic_json(PLAN, plan)")
    submit = source.index("jobs[arm] = submit_held")
    release = source.index('run(["scontrol", "release"')
    assert freeze < submit < release
    assert '"exclusions": []' in source
    assert '"one_evaluation_per_arm": True' in source
    assert '"common_random_numbers_across_arms": True' in source
    assert "operational_resume_after_pre_release_cancellation" in source


def test_final_jobs_use_all_eval_maps_once_and_analysis_waits_for_all_arms():
    evaluation = (ROOT / "ops/slurm/e75r3_point_maze_waypoint_final_eval.slurm").read_text()
    analysis = (ROOT / "ops/slurm/e75r3_point_maze_waypoint_final_analysis.slurm").read_text()
    assert "--evaluation-only" in evaluation
    assert "--evaluation-prompts 64 --evaluation-split eval" in evaluation
    assert "verify-arm" in evaluation
    assert "--output-model" not in evaluation
    assert "analyze" in analysis
    launcher = LAUNCHER.read_text()
    assert '"--dependency=afterok:"' in launcher
    assert 'dependencies = [jobs[arm] for arm in ARMS]' in launcher


def test_protocol_freezes_terminal_checkpoints_and_paired_analysis():
    text = " ".join(PROTOCOL.read_text().split())
    assert "three terminal update-64 checkpoints" in text
    assert "There are no run, map, family, trajectory, or outcome exclusions" in text
    assert "same per-map, per-trajectory, per-decision sampling seeds" in text
    assert "10,000 resamples" in text
    assert "current minus GRPO and delayed minus GRPO" in text


def test_paired_bootstrap_is_deterministic_and_preserves_constant_delta():
    analyzer = load_analyzer()
    first = analyzer.paired_bootstrap([0.25] * 64, seed=756402)
    second = analyzer.paired_bootstrap([0.25] * 64, seed=756402)
    assert first == second == (0.25, 0.25)
