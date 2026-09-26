from __future__ import annotations

import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e83_semantic_maxent_without_replay_05b.py"
E81_LAUNCHER = ROOT / "ops/exp_scaling/launch_e81_semantic_maxent_verified_replay_05b.py"
E78_LAUNCHER = ROOT / "ops/exp_scaling/launch_e78_verified_replay_only_05b.py"
PROTOCOL = ROOT / (
    "paper/preregistration/e83_semantic_maxent_without_replay_05b_20260807.md"
)
EXPERIMENT_SH = ROOT / "ops/run_experiment.sh"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shell_case_body(name: str) -> list[str]:
    text = EXPERIMENT_SH.read_text(encoding="utf-8")
    match = re.search(rf"^  {re.escape(name)}\)\n(.*?)^    ;;$", text, re.M | re.S)
    assert match is not None, f"variant {name} is absent from run_experiment.sh"
    return [
        line.strip()
        for line in match.group(1).splitlines()
        if line.strip().startswith("export ") or line.strip().startswith("VARIANT_TAG=")
    ]


def test_e83_is_the_fourth_cell_of_the_factorial():
    e83 = load(LAUNCHER, "e83_launcher")
    e81 = load(E81_LAUNCHER, "e83_e81")
    assert e83.ARM == "semantic_only"
    assert tuple(e83.DOMAINS) == tuple(e81.DOMAINS)
    assert e83.SEEDS == e81.SEEDS == (43, 44, 45, 46, 47)
    assert e83.PASSES == 8
    assert e83.TARGET_STEPS == 3072
    assert e83.CHECKPOINT_INTERVAL == 192
    assert e83.SEMANTIC_COEF == e81.SEMANTIC_COEF == 0.10


def test_e83_differs_from_e81_only_in_the_applied_replay_derivative():
    e81 = load(E81_LAUNCHER, "e83_e81_objective")
    e83 = load(LAUNCHER, "e83_launcher_objective")
    with_replay = e81.fixed_objective()
    without = e83.fixed_objective()

    moved = {
        key
        for key in set(with_replay) | set(without)
        if with_replay.get(key) != without.get(key)
    }
    # exactly two keys move: the variant label and the derivative switch
    assert moved == {
        "OAT_ZERO_VARIANT",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY",
    }
    assert with_replay["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "0"
    assert without["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "1"
    # the semantic term itself is untouched, so the factorial varies one thing
    for key in (
        "OAT_ZERO_SEMANTIC_SHANNON_COEF",
        "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP",
        "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE",
    ):
        assert with_replay[key] == without[key]


def test_e83_differs_from_e78_control_only_in_the_semantic_term():
    e78 = load(E78_LAUNCHER, "e83_e78")
    e83 = load(LAUNCHER, "e83_launcher_control")
    control = e78.fixed_objective("control")
    semantic_only = e83.fixed_objective()

    semantic_keys = {
        "OAT_ZERO_VARIANT",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF",
        "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP",
        "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE",
        "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE",
    }
    assert {k: v for k, v in control.items() if k not in semantic_keys} == {
        k: v for k, v in semantic_only.items() if k not in semantic_keys
    }
    # both sides keep the replay derivative off; only the semantic term appears
    assert control["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "1"
    assert semantic_only["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "1"
    assert control["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"
    assert semantic_only["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.1"


def test_e83_shell_variant_is_the_control_branch_plus_semantic_exports_only():
    control = shell_case_body("grpo_compute_matched")
    semantic = shell_case_body("compute_matched_semantic_maxent")

    def strip(lines: list[str]) -> list[str]:
        return [
            line
            for line in lines
            if "SEMANTIC_SHANNON" not in line and not line.startswith("VARIANT_TAG=")
        ]

    assert strip(control) == strip(semantic)
    assert 'VARIANT_TAG="compute_matched_semantic_maxent"' in semantic
    assert "export OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1" in semantic
    assert "export OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1" in semantic
    # the published control branch is untouched
    assert not any("SEMANTIC_SHANNON" in line for line in control)


def test_e83_run_stamps_do_not_collide_with_the_other_arms():
    e78 = load(E78_LAUNCHER, "e83_e78_stamp")
    e81 = load(E81_LAUNCHER, "e83_e81_stamp")
    e83 = load(LAUNCHER, "e83_launcher_stamp")
    stamps = {
        e83.run_stamp(domain, seed) for domain in e83.DOMAINS for seed in e83.SEEDS
    }
    assert len(stamps) == 25
    others = {
        e78.run_stamp(domain, arm, seed)
        for domain in e78.DOMAINS
        for arm in e78.ARMS
        for seed in e78.SEEDS
    } | {
        e81.run_stamp(domain, seed) for domain in e81.DOMAINS for seed in e81.SEEDS
    }
    assert not (stamps & others)


def test_e83_protocol_registers_the_interaction_reading_in_advance():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "Frozen before submission",
        "does not re-run E78 or E81",
        "exactly eight passes",
        "3,072 optimizer updates",
        "every 192 updates",
        "passes 0, 0.5, 1.0, ..., 8.0",
        "5 domains x 1 arm x 5 seeds = 25 runs",
        "eta = 0.10",
        "does not identify an optimal value",
        "There is no reward for discovering an outcome for the first",
        "`semantic_only - control`",
        "(E81 - replay) - (E83 - control)",
        "exactly zero",
        "Do not pool domains",
        "do not select a best checkpoint",
        "PointMaze is excluded",
    ):
        assert literal in text, literal
