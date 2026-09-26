from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "ops/exp_scaling/launch_pantry_support_mask_drgrpo_smoke_v1_r1.py"


def _launcher_module():
    """Load the shared launcher pinned to the r1 repair suffix.

    The r2 and paired launchers set ``PANTRY_REPAIR_SUFFIX`` at import time, so
    whichever test loads first would otherwise decide which repair this module
    configures. Pinning the variable here keeps the r1 contract independent of
    test execution order.
    """

    spec = importlib.util.spec_from_file_location("pantry_r1_launcher", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(os.environ, {"PANTRY_REPAIR_SUFFIX": "r1"}):
        spec.loader.exec_module(module)
    return module


def test_pantry_r1_is_bound_to_failed_v1_and_fresh_artifacts():
    module = _launcher_module()
    source = LAUNCHER.read_text()
    protocol = (
        ROOT
        / "paper/preregistration/pantry_support_mask_drgrpo_smoke_v1_r1_horizon_repair_20260730.md"
    ).read_text()

    assert module.PREFIX == "ppsmoke_support_mask_drgrpo_v1_r1"
    assert module.FAILED_AUDIT.name == "pantry_support_mask_drgrpo_smoke_v1_audit.json"
    assert module.FAILED_STDOUT.name == "xdr_train-30187473.out"
    # The launcher now serves both repair suffixes, so the bound job id is a
    # constant rather than a literal in the emitted record.
    assert module.FAILED_JOB_ID == 30187473
    assert "FAILED_JOB_ID = 30199460 if IS_R2 else 30187473" in source
    assert '"failed_job_id": FAILED_JOB_ID' in source
    assert 'f"fresh {REPAIR_SUFFIX} artifact required: {path}"' in source
    assert "No prompt, data row, split, seed" in protocol
    assert "no fixed node" in protocol


def test_pantry_r1_scientific_cell_matches_frozen_v1_values():
    values = _launcher_module().scientific_environment()

    assert values["OAT_ZERO_ONLY_ARMS"] == "grpo"
    assert values["OAT_ZERO_TRAIN_SEEDS"] == "76201"
    assert values["OAT_ZERO_NUM_SAMPLES"] == "16"
    assert values["OAT_ZERO_LEARNING_RATE"] == "0.0000002"
    assert values["OAT_ZERO_MAX_TRAIN"] == "32"
    assert values["OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES"] == "32"
    assert values["OAT_ZERO_CANONICAL_ACTION_TASK"] == "pantry_support_mask"
    assert values["OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT"] == "6"
    assert values["OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING"] == "1"
    assert values["OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING"] == "1"
    assert values["OAT_ZERO_AUTO_RESUME"] == "0"
    assert values["OAT_ZERO_WATCHDOG_REQUEUE"] == "0"
    for name in (
        "MAXENT",
        "SEMANTIC_SHANNON",
        "ONLINE_CANONICAL_MAXENT",
        "VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL",
    ):
        assert values[f"OAT_ZERO_INCLUDE_{name}_ARM"] == "0"


def test_pantry_r1_placement_changes_only_before_release():
    source = LAUNCHER.read_text()

    amend = source.index('"Partition=all", "Gres=gpu:a5000:1", "NodeList="')
    identity = source.index("atomic(SUBMISSION")
    release = source.index('run_command(["scontrol", "release"')
    assert amend < identity < release
    assert '"placement_only": True' in source
    assert '"scientific_change": False' in source
    assert '"RunTime=00:00:00"' in source


def test_pantry_r1_audit_is_dynamic_afterany_and_extends_v1_checks():
    launcher = LAUNCHER.read_text()
    wrapper = (
        ROOT / "ops/audit_pantry_support_mask_drgrpo_smoke_v1_r1.py"
    ).read_text()
    slurm = (
        ROOT / "ops/slurm/audit_pantry_support_mask_drgrpo_smoke_v1_r1.slurm"
    ).read_text()

    assert "--dependency=afterany:{job_id}" in launcher
    assert "PANTRY_R1_TRAIN_JOB_ID={job_id}" in launcher
    assert "base.PREFIX = PREFIX" in wrapper
    assert "repair_protocol_matches" in wrapper
    assert "failed_stdout_matches" in wrapper
    assert 'TRAIN_JOB_ID="${PANTRY_R1_TRAIN_JOB_ID:' in slurm
    assert '--job-id "$TRAIN_JOB_ID"' in slurm
