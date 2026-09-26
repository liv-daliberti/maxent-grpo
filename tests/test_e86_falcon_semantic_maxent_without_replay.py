from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "ops/exp_scaling"
LAUNCHER = EXP / "launch_e86_falcon_semantic_maxent_without_replay.py"
E79_LAUNCHER = EXP / "launch_e79_falcon1b_aligned_verified_replay.py"
E82_LAUNCHER = EXP / "launch_e82_falcon_semantic_maxent_verified_replay.py"
E83_LAUNCHER = EXP / "launch_e83_semantic_maxent_without_replay_05b.py"
E85_LAUNCHER = EXP / "launch_e85_pantry_semantic_repair.py"
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e86_semantic_maxent_without_replay_falcon1b_20260809.md"
)


def load(path: Path, name: str):
    if str(EXP) not in sys.path:
        sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e86_is_the_fourth_cell_of_the_falcon_factorial():
    e86 = load(LAUNCHER, "e86_launcher")
    e79 = load(E79_LAUNCHER, "e86_e79")
    assert e86.ARM == "semantic_only"
    assert tuple(e86.DOMAINS) == tuple(e79.DOMAINS)
    assert e86.SEEDS == e79.SEEDS == (55, 56, 57, 58, 59)
    assert e86.PASSES == 8
    assert e86.TARGET_STEPS == 3072
    assert e86.CHECKPOINT_INTERVAL == 192
    assert e86.MODEL_REVISION == e79.MODEL_REVISION
    assert e86.SEMANTIC_COEF == 0.10


def test_e86_takes_e83s_objective_verbatim():
    """The Falcon arm is a replication, not a second specification."""

    e83 = load(E83_LAUNCHER, "e86_e83_objective")
    e86 = load(LAUNCHER, "e86_launcher_objective")
    assert e86.fixed_objective() == e83.fixed_objective()
    assert e86.VARIANT == e83.VARIANT == "compute_matched_semantic_maxent"


def test_e86_differs_from_e82_only_in_the_applied_replay_derivative():
    e82 = load(E82_LAUNCHER, "e86_e82_objective")
    e86 = load(LAUNCHER, "e86_launcher_replay")
    with_replay = e82.fixed_objective()
    without = e86.fixed_objective()

    moved = {
        key
        for key in set(with_replay) | set(without)
        if with_replay.get(key) != without.get(key)
    }
    assert moved == {
        "OAT_ZERO_VARIANT",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY",
    }
    assert with_replay["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "0"
    assert without["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "1"
    # the declared dose is still 0.1; compute-only is what makes it inert
    assert without["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"
    for key in (
        "OAT_ZERO_SEMANTIC_SHANNON_COEF",
        "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP",
        "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE",
    ):
        assert with_replay[key] == without[key]


def test_e86_is_pinned_to_the_same_placement_as_its_e79_and_e82_cells():
    e86 = load(LAUNCHER, "e86_launcher_place")
    e79 = load(E79_LAUNCHER, "e86_e79_place")
    for domain in e86.DOMAINS:
        for seed in e86.SEEDS:
            assert e79.placement(domain, seed) == e79.placement(domain, seed)

    ledger = ROOT / e86.E82_LEDGER
    if not ledger.is_file():
        return
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    for run in payload["runs"]:
        node, gpu = e79.placement(str(run["domain"]), int(run["seed"]))
        assert (node, gpu) == (str(run["node"]), str(run["gpu"]))


def test_e86_carries_the_e85_repair_so_pantry_is_live_from_the_first_update():
    e85 = load(E85_LAUNCHER, "e86_e85_patch")
    e86 = load(LAUNCHER, "e86_launcher_patch")
    assert e86.PATCHED_FILES == e85.PATCHED_FILES
    assert "src/oat_drgrpo/learner/grpo.py" in e86.PATCHED_FILES
    assert e86.REPAIRED_DOMAIN == "pantry_plan"


def test_e86_run_stamps_do_not_collide_with_the_other_falcon_arms():
    e79 = load(E79_LAUNCHER, "e86_e79_stamp")
    e82 = load(E82_LAUNCHER, "e86_e82_stamp")
    e86 = load(LAUNCHER, "e86_launcher_stamp")
    stamps = {
        e86.run_stamp(domain, seed) for domain in e86.DOMAINS for seed in e86.SEEDS
    }
    assert len(stamps) == 25
    others = {
        e79.run_stamp(domain, arm, seed)
        for domain in e79.DOMAINS
        for arm in e79.ARMS
        for seed in e79.SEEDS
    } | {
        e82.run_stamp(domain, seed) for domain in e82.DOMAINS for seed in e82.SEEDS
    }
    assert not (stamps & others)


def test_e86_protocol_registers_the_cross_family_reading_in_advance():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "Frozen before submission",
        "does not re-run E79 or E82",
        "exactly eight passes",
        "3,072 optimizer updates",
        "every 192 updates",
        "passes 0, 0.5, 1.0, ..., 8.0",
        "5 domains x 1 arm x 5 seeds = 25 runs",
        "eta = 0.10",
        "does not identify an optimal value",
        "There is no reward for discovering an outcome for the first time",
        "`semantic_only - control`",
        "(E82 - replay) - (E86 - control)",
        "exactly zero",
        "Do not pool domains",
        "do not pool model families",
        "do not select a best checkpoint",
        "PointMaze is excluded",
        # the repair is registered, not discovered later
        "PantryPlan is born repaired",
        "need no repair cohort",
        "`canonical_action_task=none`",
    ):
        assert literal in text, literal
