from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str, name: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage = load("ops/exp_scaling/launch_e117_stage1_development.py", "e117s1_launch_test")
builder = load("ops/exp_scaling/build_e117_stage1_development_results.py", "e117s1_builder_test")


def test_registered_grid_and_balanced_node_blocks():
    assert len(stage.SENTINELS) * len(stage.SEEDS) * len(stage.ARMS) == 36
    assert stage.CHECKPOINTS == tuple(range(0, 3073, 192))
    assert stage.START_ORDERS == {
        201: ("c", "p", "f"),
        202: ("p", "f", "c"),
        203: ("f", "c", "p"),
    }
    assert list(stage.NODES.values()).count("node202") == 6
    assert list(stage.NODES.values()).count("node203") == 6
    assert stage.MAX_RESUME_CHECKPOINTS == 1
    ledger = json.loads((ROOT / stage.R2_LEDGER).read_text(encoding="utf-8"))
    cells = stage.planned_cells(ROOT, ROOT / stage.BASE_SNAPSHOT, ledger)
    for node in ("node202", "node203"):
        lane = [cell for cell in cells if cell["node"] == node]
        assert len(lane) == 18
        assert [cell["node_lane_position"] for cell in lane] == list(range(18))
        for scale, domain, seed in stage.NODES:
            block = [
                cell["arm"] for cell in lane
                if (cell["scale"], cell["domain"], cell["seed"]) == (scale, domain, seed)
            ]
            assert not block or tuple(block) == stage.START_ORDERS[seed]


def test_stage1_environment_is_a_narrow_r2_projection():
    ledger = json.loads(
        (ROOT / stage.R2_LEDGER).read_text(encoding="utf-8")
    )
    templates = stage.r2_templates(ledger)
    snapshot = ROOT / stage.BASE_SNAPSHOT
    environments = {}
    for arm in stage.ARMS:
        env, target = stage.build_environment(
            ROOT,
            snapshot,
            templates[("qwen05b", "countdown", arm)],
            scale="qwen05b",
            domain="countdown",
            seed=201,
            arm=arm,
        )
        assert "confirmation" not in env["OAT_ZERO_EVAL_DATA"]
        assert env["OAT_ZERO_EXPORT_STEPS"] == "-1"
        assert env["OAT_ZERO_RESUME_STEPS"] == "64"
        assert env["OAT_ZERO_MAX_RESUME_NUM"] == "1"
        assert env["OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS"] == "16"
        assert env["OAT_ZERO_EVAL_MODE_COVERAGE_SEED"] == "117900"
        assert str(target).endswith(f"_{stage.run_stamp('qwen05b', 'countdown', 201, arm)}")
        environments[arm] = env
    ignored = {
        "SAVE_PATH",
        "RUN_STAMP",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY",
    }
    common = {
        arm: {key: value for key, value in env.items() if key not in ignored}
        for arm, env in environments.items()
    }
    assert common["c"] == common["p"] == common["f"]


def prompt_surface(seed: int = 117900):
    return {
        "seed": seed,
        "prompts": [
            {
                "answer_mode_count": 3,
                "option_ids": [None] * 8,
                "prompt": f"prompt {index}",
                "prompt_index": index,
                "reference": f"reference {index}",
                "request_seeds_by_option": [seed],
                "answer_keys": ["response-derived"],
                "metrics": {"any_correct_at_k": 1.0},
                "responses": ["secret response"],
                "rewards": [1.0],
            }
            for index in range(128)
        ],
    }


def test_response_free_projection_excludes_outcomes_and_binds_request_seed(tmp_path):
    path = tmp_path / "draw.jsonl"
    original = prompt_surface()
    prompt_hash, request_hash = builder._projection(original, path=path, line_number=1)
    mutated = copy.deepcopy(original)
    for prompt in mutated["prompts"]:
        prompt["answer_keys"] = ["different"]
        prompt["metrics"] = {"any_correct_at_k": 0.0}
        prompt["responses"] = ["different"]
        prompt["rewards"] = [0.0]
    assert builder._projection(mutated, path=path, line_number=2) == (
        prompt_hash,
        request_hash,
    )
    different_draw = prompt_surface(117901)
    assert builder._projection(different_draw, path=path, line_number=3)[0] == prompt_hash
    assert builder._projection(different_draw, path=path, line_number=3)[1] != request_hash


def test_actor_retains_already_used_fixed_draw_seed():
    text = (ROOT / "src/oat_drgrpo/actor.py").read_text(encoding="utf-8")
    assert "request_seeds_by_prompt = [" in text
    assert '"request_seeds_by_prompt": request_seeds_by_prompt' in text


def test_execution_freeze_binds_storage_and_confirmation_boundary():
    text = (ROOT / stage.PROTOCOL).read_text(encoding="utf-8")
    assert "OAT_ZERO_EXPORT_STEPS=-1" in text
    assert "confirmation reserve must not be read" in text
    assert "after` dependencies" in text
    storage = (ROOT / stage.STORAGE_AMENDMENT).read_text(encoding="utf-8")
    assert "exactly one rolling resume checkpoint" in storage
    assert "at most two run campaign-wide" in storage
    assert "afterok" in storage
    assert "104 GiB" in storage
