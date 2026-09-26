"""Contract tests for the staged, validation-selected E76 campaign."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launch = _load("e76_launch", "ops/exp_scaling/launch_e76_tuned_scale.py")
select = _load("e76_select", "ops/exp_scaling/select_e76_tuned_scale.py")


def _stage_a_selection(tmp_path: Path) -> Path:
    payload = {
        "complete": True,
        "models": {
            model: {
                "learning_rate": 1e-7,
                "beta": 0.01,
                "domains": {
                    "graph_coloring": {"stopping_pass": 2},
                    "pantry_plan": {"stopping_pass": 1},
                },
            }
            for model in launch.MODELS
        },
    }
    path = tmp_path / "stage_a.json"
    path.write_text(json.dumps(payload))
    return path


def _stage_b_selection(tmp_path: Path) -> Path:
    payload = {
        "complete": True,
        "models": {
            model: {
                "domains": {
                    domain: {
                        "xmode": {"dose": 0.10},
                        "rehearsal": {"dose": 0.05},
                    }
                    for domain in launch.DOMAINS
                }
            }
            for model in launch.MODELS
        },
    }
    path = tmp_path / "stage_b.json"
    path.write_text(json.dumps(payload))
    return path


def test_registered_stage_sizes_and_unique_cells(tmp_path, monkeypatch):
    stage_a = launch.stage_cells(ROOT, "a")
    assert len(stage_a) == 48
    assert {
        (cell["model"], cell["domain"], cell["arm"], cell["lr"], cell["beta"])
        for cell in stage_a
    } == {
        (model, domain, arm, lr, beta)
        for model in launch.MODELS
        for domain in launch.DOMAINS
        for arm in ("grpo", "xmode")
        for lr in launch.LEARNING_RATES
        for beta in launch.KL_BETAS
    }

    monkeypatch.setitem(launch.SELECTIONS, "a", str(_stage_a_selection(tmp_path)))
    stage_b = launch.stage_cells(ROOT, "b")
    assert len(stage_b) == 28
    assert sum(cell["arm"] == "grpo" for cell in stage_b) == 4
    assert sum(cell["arm"] == "xmode" for cell in stage_b) == 12
    assert sum(cell["arm"] == "rehearsal" for cell in stage_b) == 12

    monkeypatch.setitem(launch.SELECTIONS, "b", str(_stage_b_selection(tmp_path)))
    stage_c = launch.stage_cells(ROOT, "c")
    assert len(stage_c) == 36
    assert {cell["seed"] for cell in stage_c} == {55, 56, 57}


def test_tuning_environment_never_receives_reported_test_path(tmp_path):
    sources = launch.references(ROOT)
    cell = {
        "model": "falcon1b", "domain": "graph_coloring", "arm": "xmode",
        "lr": 5e-8, "beta": 0.01, "dose": 0.10, "seed": 53,
        "target_passes": 6,
    }
    env, _, _ = launch.build_env(ROOT, sources["graph_coloring"], cell, "a", tmp_path)
    assert env["OAT_ZERO_PROMPT_DATA"].endswith("e76_tuned_scale/graph_coloring/train")
    assert env["OAT_ZERO_EVAL_DATA"].endswith("e76_tuned_scale/graph_coloring/validation")
    assert env["OAT_ZERO_EVAL_DATA"] != sources["graph_coloring"]["inherited_eval_config"]["eval_data"]
    assert env["OAT_ZERO_MAX_TRAIN"] == "320"
    assert env["OAT_ZERO_EVAL_PROMPT_INTERVAL"] == "80"
    assert env["OAT_ZERO_NUM_PROMPT_EPOCH"] == "6"
    assert float(env["OAT_ZERO_LEARNING_RATE"]) == 5e-8
    assert float(env["OAT_ZERO_BETA"]) == 0.01
    assert env["OAT_ZERO_PROMPT_TEMPLATE"] == "falcon_boxed"


def test_falcon_pantry_uses_registered_prompt_template_and_action_task(tmp_path):
    source = launch.references(ROOT)["pantry_plan"]
    cell = {
        "model": "falcon1b", "domain": "pantry_plan", "arm": "grpo",
        "lr": 5e-8, "beta": 0.0, "dose": None, "seed": 53,
        "target_passes": 6,
    }
    env, _, _ = launch.build_env(ROOT, source, cell, "a", tmp_path)
    assert env["OAT_ZERO_PROMPT_TEMPLATE"] == "falcon_pantry_support_mask"
    assert env["OAT_ZERO_CANONICAL_ACTION_TASK"] == "pantry_support_mask"


def test_falcon_placements_use_the_available_cross_partition_gpu_pools():
    graph = launch.placement({"model": "falcon1b", "domain": "graph_coloring"})
    pantry = launch.placement({"model": "falcon1b", "domain": "pantry_plan"})
    assert graph["partition"] == "all"
    assert graph["nodelist"] == "node105,node202,node203,node204"
    assert pantry["partition"] == "all"
    assert pantry["nodelist"] == "node205,node206,node208"


def test_cross_partition_submission_is_held_for_site_normalization(capsys):
    cell = {
        "model": "falcon1b", "domain": "pantry_plan",
        "arm": "grpo", "seed": 53,
    }
    assert launch.submit_cell(ROOT, "a", {}, cell, [], True) is None
    command = capsys.readouterr().out
    assert "--partition=all" in command
    assert "--nodelist=node205,node206,node208" in command
    assert "--nice=0" in command
    assert "--hold" in command


def test_control_and_rehearsal_are_objectively_distinct(tmp_path):
    source = launch.references(ROOT)["pantry_plan"]
    common = {
        "model": "qwen3b", "domain": "pantry_plan", "lr": 1e-7,
        "beta": 0.01, "seed": 54, "target_passes": 2,
    }
    grpo, _, _ = launch.build_env(
        ROOT, source, {**common, "arm": "grpo", "dose": None}, "b", tmp_path
    )
    rehearsal, _, _ = launch.build_env(
        ROOT, source, {**common, "arm": "rehearsal", "dose": 0.05}, "b", tmp_path
    )
    assert grpo["OAT_ZERO_VARIANT"] == "grpo_compute_matched"
    assert grpo["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "1"
    assert float(grpo["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) == 0.0
    assert rehearsal["OAT_ZERO_VARIANT"] == "verified_first_replay_rehearsal_only"
    assert float(rehearsal["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) == 0.05
    assert float(rehearsal["OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA"]) == 0.05


def test_stage_a_checkpoint_rule_enforces_pass_floor():
    def run(arm: str, metrics):
        return {
            "arm": arm,
            "run_stamp": arm,
            "target_steps": 640,
            "validation_metrics": metrics,
        }

    grpo = run(
        "grpo",
        {
            0: {"pass_at_8": 0.60, "distinct_at_8": 1.0},
            320: {"pass_at_8": 0.59, "distinct_at_8": 1.4},
            640: {"pass_at_8": 0.40, "distinct_at_8": 2.0},
        },
    )
    xmode = run(
        "xmode",
        {
            0: {"pass_at_8": 0.60, "distinct_at_8": 1.0},
            320: {"pass_at_8": 0.60, "distinct_at_8": 1.5},
            640: {"pass_at_8": 0.60, "distinct_at_8": 2.1},
        },
    )
    chosen = select.choose_domain_checkpoint([grpo, xmode])
    assert chosen["step"] == 320
    assert chosen["fallback_no_feasible_checkpoint"] is False


def test_final_stage_restores_full_training_and_reported_test(tmp_path):
    source = launch.references(ROOT)["graph_coloring"]
    cell = {
        "model": "qwen3b", "domain": "graph_coloring", "arm": "xmode",
        "lr": 1e-7, "beta": 0.01, "dose": 0.10, "seed": 55,
        "target_passes": 2,
    }
    env, _, _ = launch.build_env(ROOT, source, cell, "c", tmp_path)
    assert env["OAT_ZERO_PROMPT_DATA"] == source["inherited_eval_config"]["prompt_data"]
    assert env["OAT_ZERO_EVAL_DATA"] == source["inherited_eval_config"]["eval_data"]
    assert env["OAT_ZERO_MAX_TRAIN"] == "384"
    assert env["OAT_ZERO_EVAL_PROMPT_INTERVAL"] == "96"
