#!/usr/bin/env python3
"""Launch PantryPlan's fresh-split support-retention calibration pair."""

from __future__ import annotations

import json
import os
from pathlib import Path

import launch_pantry_support_mask_paired_integration_v3 as v3


ROOT = Path(__file__).resolve().parents[2]
paired = v3.base
foundation = paired.base
PROTOCOL = ROOT / "paper/preregistration/pantry_support_retention_repair_v1_20260730.md"
DATA = ROOT / "var/data/pantry_plan_modebench_v3_repair"
DATA_AUDIT = ROOT / "var/artifacts/pantry_plan_modebench_v3_repair_audit.json"
DISJOINT_AUDIT = (
    ROOT / "var/artifacts/pantry_plan_modebench_v3_repair_disjointness_audit.json"
)
PREFIX = "pprepair_support_retention_v1"

foundation.DATA = DATA
paired.PROTOCOL = PROTOCOL
paired.PREFIX = PREFIX
paired.IDENTITY = ROOT / "var/artifacts/pantry_support_retention_repair_v1_identity.json"
paired.SUBMISSION = (
    ROOT / "var/artifacts/pantry_support_retention_repair_v1_submission.json"
)
paired.MANIFEST = ROOT / f"var/artifacts/{PREFIX}_comparative_jobs.tsv"
paired.AUDIT = ROOT / "var/artifacts/pantry_support_retention_repair_v1_audit.json"
paired.AUDIT_RUNNER = (
    ROOT / "var/artifacts/pantry_support_retention_repair_v1_audit_runner_identity.json"
)
paired.AUDIT_SUBMISSION = (
    ROOT / "var/artifacts/pantry_support_retention_repair_v1_audit_submission.json"
)
paired.LAUNCHER_ENTRYPOINT = Path(__file__).resolve()
paired.AUDIT_PROGRAM = ROOT / "ops/audit_pantry_support_retention_gate_v1.py"
paired.AUDIT_BATCH = (
    ROOT / "ops/slurm/audit_pantry_support_retention_repair_v1.slurm"
)
paired.AUDIT_SUPPORT_FILES = [
    ROOT / "ops/audit_pantry_support_mask_paired_smoke_v1.py"
]
paired.SEED = 76401
paired.OPTIMIZER_UPDATES = 96
paired.ROLLOUTS_PER_PROMPT = 16
paired.EXPECTED_MAX_QUERIES = 1520


def prerequisites():
    for path in (PROTOCOL, DATA / "identity.json", DATA_AUDIT, DISJOINT_AUDIT):
        if not path.is_file():
            raise FileNotFoundError(path)
    data_audit = json.loads(DATA_AUDIT.read_text())
    disjoint = json.loads(DISJOINT_AUDIT.read_text())
    if (
        data_audit.get("status") != "pass"
        or disjoint.get("status") != "pass"
        or disjoint.get("decision") != "fresh_repair_split_admitted"
    ):
        raise RuntimeError("fresh PantryPlan repair data did not pass admission")
    return v3.prerequisites()


def environment(source_root: Path, ops_root: Path) -> dict[str, str]:
    values = foundation.scientific_environment()
    values.update(
        RUN_STAMP_PREFIX=PREFIX,
        OAT_ZERO_COMPARATIVE_DATA_ROOT=str(DATA),
        OAT_ZERO_TRAIN_SEEDS="76401",
        OAT_ZERO_ONLY_ARMS="grpo,verified_first_global_replay_canonical",
        OAT_ZERO_DRGRPO_VARIANT="grpo_compute_matched",
        OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM="1",
        OAT_ZERO_MAXENT_ALPHA="0",
        OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA="0",
        OAT_ZERO_SEMANTIC_SHANNON_COEF="0.10",
        OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP="5.0",
        OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT="1.0",
        OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP="0.05",
        OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS="64",
        OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY="0.90",
        OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA="0",
        OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA="0.50",
        OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT="1.0",
        OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP="5.0",
        OAT_ZERO_ONLINE_CANONICAL_KEY_MODE="modebench_outcome",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA="0.20",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY="16",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP="1",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS="64",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY="0.90",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA="0.20",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS="64",
        OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY="0.90",
        OAT_ZERO_MAX_QUERIES="1520",
        OAT_ZERO_MAX_PROMPT_EPOCHS="3",
        OAT_ZERO_NUM_PROMPT_EPOCH="3",
        OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES="96",
        OAT_ZERO_EVAL_PROMPT_INTERVAL="8",
        OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS="4",
        OAT_ZERO_EVAL_MODE_COVERAGE_SEED="76499",
        OAT_ZERO_SAVE_STEPS="96",
        OAT_ZERO_SAVE_FROM="96",
        OAT_ZERO_CAMPAIGN_SOURCE_ROOT=str(source_root),
        OAT_ZERO_OPS_SNAPSHOT_ROOT=str(ops_root),
        OAT_ZERO_PROTOCOL_IDENTITY=str(paired.IDENTITY),
    )
    return values


def validate() -> None:
    foundation.run_command(
        [
            str(foundation.PYTHON),
            "-m",
            "py_compile",
            str(Path(__file__).resolve()),
            str(paired.AUDIT_PROGRAM),
        ]
    )
    foundation.run_command(["bash", "-n", str(paired.AUDIT_BATCH)])
    test_env = dict(os.environ)
    test_env["PYTHONPATH"] = str(ROOT / "src")
    library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    test_env["LD_LIBRARY_PATH"] = library + (
        ":" + test_env["LD_LIBRARY_PATH"] if test_env.get("LD_LIBRARY_PATH") else ""
    )
    foundation.run_command(
        [
            str(foundation.PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / "tests/test_pantry_support_mask_paired_integration_v3.py"),
            str(ROOT / "tests/test_pantry_support_mask.py"),
            str(ROOT / "tests/test_e58_global_verified_replay_contract.py"),
        ],
        env=test_env,
    )


paired.IDENTITY_EXTRA = {
    "qualification_version": "pantry_support_retention_repair_v1",
    "prompt_passes": 3,
    "development_only": True,
    "final_seed": False,
    "fresh_data_seed": 74103,
    "fresh_data_audit_sha256": foundation.sha(DATA_AUDIT),
    "cross_version_disjointness_audit_sha256": foundation.sha(DISJOINT_AUDIT),
    "replay_alpha": 0.20,
    "replay_mass_alpha": 0.20,
    "novelty_beta": 0.50,
    "learning_rate": 2e-7,
    "evaluation_split": "fresh_development_only",
}
paired.prerequisites = prerequisites
paired.environment = environment
paired.validate = validate


if __name__ == "__main__":
    paired.main()

