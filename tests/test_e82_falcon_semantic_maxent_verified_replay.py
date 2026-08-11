from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_e82_falcon_semantic_maxent_verified_replay.py"
E81_LAUNCHER = ROOT / "ops/exp_scaling/launch_e81_semantic_maxent_verified_replay_05b.py"
E79_LAUNCHER = ROOT / "ops/exp_scaling/launch_e79_falcon1b_aligned_verified_replay.py"
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e82_semantic_maxent_on_verified_replay_falcon1b_20260807.md"
)

SEMANTIC_KEYS = {
    "OAT_ZERO_VARIANT",
    "OAT_ZERO_SEMANTIC_SHANNON_COEF",
    "OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP",
    "OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT",
    "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE",
    "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE",
    "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE",
}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e82_is_the_falcon_cohort_with_one_new_arm():
    e79 = load(E79_LAUNCHER, "e82_e79")
    e82 = load(LAUNCHER, "e82_launcher")
    assert e82.ARM == "semantic"
    assert tuple(e82.DOMAINS) == tuple(e79.DOMAINS)
    assert e82.SEEDS == e79.SEEDS == (55, 56, 57, 58, 59)
    assert e82.PASSES == 8
    assert e82.TRAIN_ROWS == 384
    assert e82.TARGET_STEPS == 3072
    assert e82.CHECKPOINT_INTERVAL == 192
    assert e82.MODEL_REVISION == e79.MODEL_REVISION


def test_e82_objective_is_e81s_objective_verbatim():
    e81 = load(E81_LAUNCHER, "e82_e81")
    e82 = load(LAUNCHER, "e82_launcher_objective")
    assert e82.fixed_objective() == e81.fixed_objective()
    assert e82.SEMANTIC_COEF == e81.SEMANTIC_COEF == 0.10
    assert e82.VARIANT == e81.VARIANT == "verified_replay_semantic_maxent"


def test_e82_differs_from_e79_replay_only_in_the_semantic_term():
    e79 = load(E79_LAUNCHER, "e82_e79_objective")
    e82 = load(LAUNCHER, "e82_launcher_diff")
    replay = e79.fixed_objective("replay")
    semantic = e82.fixed_objective()

    assert {k: v for k, v in replay.items() if k not in SEMANTIC_KEYS} == {
        k: v for k, v in semantic.items() if k not in SEMANTIC_KEYS
    }
    assert semantic["OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY"] == "0"
    assert semantic["OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"] == (
        "verified_likelihood_per_rollout"
    )
    assert semantic["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"] == "0.1"
    assert semantic["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.1"
    assert replay["OAT_ZERO_SEMANTIC_SHANNON_COEF"] == "0.0"


def test_e82_pins_every_cell_to_its_e79_pairs_placement():
    e79 = load(E79_LAUNCHER, "e82_e79_place")
    e82 = load(LAUNCHER, "e82_launcher_place")
    # Placement is a deterministic function of domain and seed, which is what
    # makes each paired difference a within-node, within-GPU-model comparison.
    for domain in e82.DOMAINS:
        for seed in e82.SEEDS:
            assert e79.placement(domain, seed) == e79.placement(domain, seed)
            node, gpu = e79.placement(domain, seed)
            assert node.startswith("node")
            assert gpu in {"a5000", "a6000"}


def test_e82_run_stamps_do_not_collide_with_e79_or_e81():
    e79 = load(E79_LAUNCHER, "e82_e79_stamp")
    e81 = load(E81_LAUNCHER, "e82_e81_stamp")
    e82 = load(LAUNCHER, "e82_launcher_stamp")
    stamps = {
        e82.run_stamp(domain, seed)
        for domain in e82.DOMAINS
        for seed in e82.SEEDS
    }
    assert len(stamps) == 25
    others = {
        e79.run_stamp(domain, arm, seed)
        for domain in e79.DOMAINS
        for arm in e79.ARMS
        for seed in e79.SEEDS
    } | {
        e81.run_stamp(domain, seed)
        for domain in e81.DOMAINS
        for seed in e81.SEEDS
    }
    assert not (stamps & others)


def test_e82_snapshot_patch_set_matches_e81s():
    e81 = load(E81_LAUNCHER, "e82_e81_patch")
    e82 = load(LAUNCHER, "e82_launcher_patch")
    assert e82.SNAPSHOT_PREFIX == "e82_falcon_semantic_maxent"
    assert sorted(e81.PATCHED_FILES) == [
        "ops/run_experiment.sh",
        "src/oat_drgrpo/args.py",
    ]


def test_e82_protocol_registers_the_cross_family_reading_in_advance():
    text = PROTOCOL.read_text(encoding="utf-8")
    for literal in (
        "Frozen before submission",
        "does not re-run either E79 arm",
        "exactly eight passes",
        "3,072 optimizer updates",
        "every 192 updates",
        "passes 0, 0.5, 1.0, ..., 8.0",
        "5 domains x 1 arm x 5 seeds = 25 runs",
        "eta = 0.10",
        "does not identify an optimal",
        "There is no reward for discovering an outcome for the first",
        "`semantic - replay`",
        "Do not pool domains",
        "do not pool the two model families",
        "do not select a best checkpoint",
        "A Qwen-only",
        "PointMaze is excluded",
        "nothing in this",
    ):
        assert literal in text, literal
