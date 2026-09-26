from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_pantry_stage_b_05b_12pass.py"
AUDITOR = ROOT / "ops/audit_pantry_stage_b_05b_12pass.py"
PROTOCOL = ROOT / "paper/preregistration/pantry_stage_b_05b_12pass_20260730.md"


def load(name: str, path: Path):
    script_directory = str(path.parent)
    if script_directory not in sys.path:
        sys.path.insert(0, script_directory)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_cartesian_product_and_budget() -> None:
    launcher = load("pantry_stage_b_launcher_contract", LAUNCHER)
    auditor = load("pantry_stage_b_auditor_contract", AUDITOR)
    assert launcher.ARMS == auditor.ARMS
    assert launcher.SEEDS == auditor.SEEDS == (43, 44, 45, 46, 47)
    assert launcher.UPDATES == auditor.UPDATES == 384
    assert launcher.MAX_QUERIES == (launcher.UPDATES - 1) * launcher.ROLLOUTS
    assert launcher.EVAL_INTERVAL == auditor.EVAL_INTERVAL == 8
    assert launcher.EVAL_DRAWS == auditor.EVAL_DRAWS == 4
    assert len(launcher.expected_cells()) == 10


def test_environment_is_final_fresh_compute_matched_recipe() -> None:
    launcher = load("pantry_stage_b_launcher_environment", LAUNCHER)
    _, source_root, ops_root = launcher.prerequisites()
    values = launcher.environment(source_root, ops_root)
    assert values["OAT_ZERO_TRAIN_SEEDS"] == "43,44,45,46,47"
    assert values["OAT_ZERO_DRGRPO_VARIANT"] == "grpo_compute_matched"
    assert values["OAT_ZERO_ONLY_ARMS"] == "grpo,verified_first_global_replay_canonical"
    assert values["OAT_ZERO_MAX_PROMPT_EPOCHS"] == "12"
    assert values["OAT_ZERO_NUM_PROMPT_EPOCH"] == "12"
    assert values["OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES"] == "384"
    assert values["OAT_ZERO_MAX_QUERIES"] == "6128"
    assert values["OAT_ZERO_EVAL_PROMPT_INTERVAL"] == "8"
    assert values["OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS"] == "4"
    assert values["OAT_ZERO_AUTO_RESUME"] == "0"
    assert values["OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP"] == "1"


def test_protocol_names_gate_anchors_and_no_substitution() -> None:
    text = PROTOCOL.read_text()
    for required in (
        "eligible_for_ten_stage_b_jobs",
        "Seeds: 43, 44, 45, 46, and 47",
        "exactly 384 optimizer updates",
        "max_queries=(384-1)*16=6128",
        "four\ndeterministic temperature-one draws at K=8",
        "passes 0, 1, 2, 3, 4, 5, 6, 8, 10, and 12",
        "No resume, seed replacement",
    ):
        assert required in text


def test_terminal_audit_is_fail_closed_over_all_cells_and_cadence() -> None:
    text = AUDITOR.read_text()
    for required in (
        '"manifest_exact_ten_cells"',
        '"compute_traversal_match_by_seed"',
        "exact quarter-pass evaluation cadence absent",
        "control replay derivative not zero",
        "eligible replay never applied",
        '"pantry_stage_b_stopped"',
    ):
        assert required in text
