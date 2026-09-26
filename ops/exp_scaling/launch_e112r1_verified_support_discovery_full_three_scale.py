#!/usr/bin/env python3
"""Submit E112-R1: sampler-contract-corrected MaxEnt at all three scales."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e111_verified_support_discovery_mechanism_gate as e111_audit  # noqa: E402
import launch_e105_group_centered_semantic_repair_full_three_scale as e105  # noqa: E402
import launch_e111_verified_support_discovery_mechanism_gate_three_scale as e111  # noqa: E402


DOMAINS = e111.DOMAINS
SCALE_SEEDS = e105.SCALE_SEEDS
MODEL_TAGS = e111.MODEL_TAGS
VARIANT = e111.VARIANT
ARM = e111.ARM
TRAIN_ROWS = 384
PASSES = 8
TARGET_STEPS = TRAIN_ROWS * PASSES
CHECKPOINT_INTERVAL = 192
QWEN3_A6000_CHECKPOINT_INTERVAL = 64
LEDGER = "var/artifacts/e112r1_verified_support_discovery_full_three_scale_jobs.json"
PROTOCOL = (
    "paper/preregistration/"
    "e112r1_sampler_contract_repair_and_e112_retirement_20260819.md"
)
UNIT_EVIDENCE = "var/artifacts/e112r1_verified_support_discovery_unit_tests.json"
E112_RETIREMENT = "var/artifacts/e112_sampler_contract_failure_retirement.json"
E112_ORIGINAL_LEDGER = (
    "var/artifacts/e112_verified_support_discovery_full_three_scale_jobs.json"
)
E111_LEDGER = e111.LEDGER
E111_AUDIT = e111.AUDIT
E105_LEDGER = e105.LEDGER
E105_SUPERSESSION = "paper/preregistration/e105_v6_superseded_by_e111_20260818.md"
QWEN3_A6000_NODE_LIST = e105.QWEN3_A6000_NODE_LIST


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return e111.digest(path)


def validate_unit_evidence(root: Path, snapshot: Path) -> dict[str, Any]:
    path = root / UNIT_EVIDENCE
    if not path.is_file():
        raise SystemExit(f"E112-R1 unit evidence is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "e112r1_verified_support_discovery_unit_tests_v1",
        "passed": True,
        "returncode": 0,
        "snapshot_root": str(snapshot),
        "outcomes_used": False,
        "pointmaze": "excluded",
    }
    failed = [key for key, value in expected.items() if payload.get(key) != value]
    if failed:
        raise SystemExit(f"E112-R1 unit evidence failed: {failed}")
    if "303 passed" not in str(payload.get("stdout", "")):
        raise SystemExit("E112-R1 unit evidence lacks the exact production summary")
    for field, base in (
        ("test_sha256", root),
        ("snapshot_source_sha256", snapshot),
    ):
        hashes = payload.get(field)
        if not isinstance(hashes, dict) or not hashes:
            raise SystemExit(f"E112-R1 unit evidence lacks {field}")
        for relative, expected_sha256 in hashes.items():
            path = base / str(relative)
            if not path.is_file() or digest(path) != expected_sha256:
                raise SystemExit(f"E112-R1 unit evidence digest drifted: {relative}")
    relative_launcher = (
        "ops/exp_scaling/"
        "launch_e112r1_verified_support_discovery_full_three_scale.py"
    )
    if digest(root / relative_launcher) != digest(snapshot / relative_launcher):
        raise SystemExit("E112-R1 root and snapshot launchers differ")
    return payload


def require_terminal_e111_gate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = subprocess.run(
        [sys.executable, str(root / "ops/exp_scaling/audit_e111_verified_support_discovery_mechanism_gate.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("E111 terminal mechanism audit did not pass")
    ledger_path = root / E111_LEDGER
    audit_path = root / E111_AUDIT
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "e111_verified_support_discovery_mechanism_gate_audit_v1",
        "terminal": True,
        "outcomes_used_for_gate": False,
        "post_freeze_training_reward_exposure": True,
        "pointmaze": "excluded",
        "passed": True,
    }
    failed = [key for key, value in expected.items() if audit.get(key) != value]
    if failed:
        raise SystemExit(f"E111 terminal audit failed: {failed}")
    scale_cells = audit.get("scale_chain_cells", {})
    if any(not scale_cells.get(scale) for scale in SCALE_SEEDS):
        raise SystemExit("E111 lacks a complete causal-chain cell at every scale")
    amendments = audit.get("scheduler_amendments", {})
    if amendments.get("scheduler_only") is not True or amendments.get("environment_changed") is not False:
        raise SystemExit("E111 scheduler amendment validation failed")
    recovery = audit.get("runtime_recovery_patch", {})
    if (
        recovery.get("passed") is not True
        or recovery.get("checkpoint_deserialization_only") is not True
        or recovery.get("optimizer_update_changed") is not False
        or recovery.get("treatment_changed") is not False
        or recovery.get("outcomes_inspected") is not False
    ):
        raise SystemExit("E111 runtime recovery validation failed")
    pantry_recovery = audit.get("pantry_partial_checkpoint_recovery", {})
    if (
        pantry_recovery.get("passed") is not True
        or pantry_recovery.get("checkpoint_storage_only") is not True
        or pantry_recovery.get("verified_after_requeue") is not True
        or pantry_recovery.get("optimizer_update_changed") is not False
        or pantry_recovery.get("treatment_changed") is not False
        or pantry_recovery.get("outcomes_inspected") is not False
    ):
        raise SystemExit("E111 Pantry checkpoint recovery validation failed")
    pantry_timeout = audit.get("qwen3_pantry_timeout_continuation", {})
    pantry_chain = pantry_timeout.get("continuation_chains", {}).get(
        "30674762", []
    )
    pantry_effective = pantry_timeout.get(
        "continuation_by_original_job_id", {}
    ).get("30674762", {})
    if (
        pantry_timeout.get("passed") is not True
        or pantry_timeout.get("installed") is not True
        or pantry_timeout.get("released") is not True
        or pantry_timeout.get("same_scientific_cell") is not True
        or pantry_timeout.get("same_run_directory") is not True
        or pantry_timeout.get("checkpoint_interval") != 2
        or pantry_timeout.get("outcomes_inspected") is not False
        or pantry_timeout.get("pointmaze") != "excluded"
        or not isinstance(pantry_chain, list)
        or len(pantry_chain) != 2
        or pantry_chain[0] != 30674762
        or pantry_effective.get("continuation_job_id") != pantry_chain[-1]
    ):
        raise SystemExit("E111 Pantry timeout continuation validation failed")
    timeout_replacement = audit.get(
        "qwen3_python_mathir_timeout_replacement", {}
    )
    if (
        timeout_replacement.get("passed") is not True
        or timeout_replacement.get("installed") is not True
        or timeout_replacement.get("released") is not True
        or timeout_replacement.get("same_scientific_cells") is not True
        or timeout_replacement.get("same_run_directories") is not True
        or timeout_replacement.get("checkpoint_interval") != 2
        or timeout_replacement.get("outcomes_inspected") is not False
        or timeout_replacement.get("pointmaze") != "excluded"
    ):
        raise SystemExit("E111 purged-timeout replacement validation failed")
    python_chain = timeout_replacement.get("continuation_chains", {}).get(
        "30674760", []
    )
    python_effective = timeout_replacement.get(
        "continuation_by_original_job_id", {}
    ).get("30674760", {})
    if (
        not isinstance(python_chain, list)
        or len(python_chain) != 3
        or python_chain[:2] != [30674760, 30739797]
        or timeout_replacement.get("second_continuation_job_id")
        != python_chain[-1]
        or python_effective.get("continuation_job_id") != python_chain[-1]
    ):
        raise SystemExit(
            "E111 Python second-continuation chain validation failed"
        )
    if ledger.get("released") is not True or ledger.get("pointmaze") != "excluded":
        raise SystemExit("E111 release ledger drifted")
    return ledger, audit


def require_e105_retired(root: Path) -> dict[str, Any]:
    ledger_path = root / E105_LEDGER
    note = root / E105_SUPERSESSION
    if not ledger_path.is_file() or not note.is_file():
        raise SystemExit("E105 retirement evidence is absent")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    runs = list(ledger.get("runs", []))
    ids = [int(run.get("job_id")) for run in runs]
    if len(ids) != 75 or len(set(ids)) != 75:
        raise SystemExit("E105 retirement job set drifted")
    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(str(value) for value in ids), "-o", "%A"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("could not verify E105 retirement")
    active = sorted(
        {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}
    )
    if active:
        raise SystemExit(
            f"E105 remains active ({len(active)} jobs); E112 release is blocked"
        )
    return {
        "ledger": str(ledger_path),
        "ledger_sha256": digest(ledger_path),
        "supersession": str(note),
        "supersession_sha256": digest(note),
        "job_ids": ids,
        "active_jobs": [],
        "outcomes_inspected": False,
    }


def require_original_e112_retired(root: Path) -> dict[str, Any]:
    record_path = root / E112_RETIREMENT
    ledger_path = root / E112_ORIGINAL_LEDGER
    if not record_path.is_file() or not ledger_path.is_file():
        raise SystemExit("original E112 retirement evidence is absent")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    job_ids = [int(run.get("job_id")) for run in ledger.get("runs", [])]
    expected = {
        "schema": "e112_sampler_contract_failure_retirement_v1",
        "installed": True,
        "original_ledger": str(ledger_path),
        "original_ledger_sha256": digest(ledger_path),
        "exact_job_ids": job_ids,
        "original_e112_retired": True,
        "original_artifacts_preserved": True,
        "pool_with_replacement": False,
        "active_after_job_ids": [],
        "active_after_count": 0,
        "efficacy_outcomes_inspected": False,
        "replacement_ledger": str(root / LEDGER),
        "replacement_released": False,
    }
    failed = [key for key, value in expected.items() if record.get(key) != value]
    if len(job_ids) != 75 or len(set(job_ids)) != 75 or failed:
        raise SystemExit(f"original E112 retirement validation failed: {failed}")
    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(str(value) for value in job_ids), "-o", "%A"],
        capture_output=True,
        text=True,
        check=False,
    )
    active = sorted(
        {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}
    )
    if result.returncode != 0 or active:
        raise SystemExit(f"original E112 still has active jobs: {active}")
    return record


def references(root: Path, scale: str) -> list[dict[str, Any]]:
    return e105.references(root, scale)


def run_stamp(scale: str, domain: str, seed: int) -> str:
    return f"e112r1_{scale}_{e111.e81.DOMAIN_TAGS[domain]}_{ARM}_s{seed}"


def save_path(root: Path, scale: str, domain: str, seed: int) -> Path:
    return root / "var/data" / (
        f"xdr_{MODEL_TAGS[scale]}_{VARIANT}_{run_stamp(scale, domain, seed)}"
    )


def recovery_checkpoint_interval(
    scale: str,
    domain: str,
    seed: int,
    qwen3_a6000_cells: set[tuple[str, int]],
) -> int:
    """Return the storage-only recovery cadence for one frozen placement."""

    if scale == "qwen3b" and (domain, seed) in qwen3_a6000_cells:
        return QWEN3_A6000_CHECKPOINT_INTERVAL
    return CHECKPOINT_INTERVAL


def build_env(
    root: Path,
    scale: str,
    run: dict[str, Any],
    snapshot: Path,
    qwen3_a6000_cells: set[tuple[str, int]],
) -> tuple[dict[str, str], Path]:
    env, _ = e105.build_env(root, scale, run, snapshot)
    domain = str(run.get("domain"))
    seed = int(run.get("seed"))
    recovery_interval = recovery_checkpoint_interval(
        scale,
        domain,
        seed,
        qwen3_a6000_cells,
    )
    target = save_path(root, scale, domain, seed)
    canonical_learner = (
        env.get("OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING") == "1"
    )
    if canonical_learner != (domain == "pantry_plan"):
        raise SystemExit(f"E112-R1 sampler surface drifted for {domain}")
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(scale, domain, seed),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_STEPS": str(recovery_interval),
            "OAT_ZERO_SAVE_FROM": str(recovery_interval),
            "OAT_ZERO_RESUME_STEPS": str(recovery_interval),
            "OAT_ZERO_MAX_SAVE_NUM": "1",
            "OAT_ZERO_MAX_RESUME_NUM": "1",
            "OAT_ZERO_AUTO_RESUME": "1",
            "OAT_ZERO_WATCHDOG_REQUEUE": "1",
            "OAT_ZERO_SAVE_CKPT": "1",
            "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING": (
                "0" if canonical_learner else "1"
            ),
            "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC": (
                "0" if canonical_learner else "1"
            ),
            "OAT_ZERO_USE_WB": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(e111.fixed_objective())
    return env, target


def job_name(scale: str, domain: str, seed: int) -> str:
    scale_tag = {"qwen05b": "q05", "falcon1b": "f1", "qwen3b": "q3"}[scale]
    return f"e112r1-{scale_tag}-{e111.e81.DOMAIN_TAGS[domain][:5]}-s{seed}"


def sbatch_command(
    root: Path,
    scale: str,
    run: dict[str, Any],
    env: dict[str, str],
    qwen3_a6000_cells: set[tuple[str, int]],
) -> list[str]:
    command = e105.sbatch_command(root, scale, run, env, qwen3_a6000_cells)
    domain = str(run.get("domain"))
    seed = int(run.get("seed"))
    return [
        f"--job-name={job_name(scale, domain, seed)}"
        if token.startswith("--job-name=")
        else token
        for token in command
    ]


def held_job_audit(
    job_id: str,
    scale: str,
    run: dict[str, Any],
    snapshot: Path,
    qwen3_a6000_cells: set[tuple[str, int]],
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E112-R1 job {job_id}")
    record = result.stdout
    seed = int(run.get("seed"))
    domain = str(run.get("domain"))
    recovery_interval = recovery_checkpoint_interval(
        scale,
        domain,
        seed,
        qwen3_a6000_cells,
    )
    canonical_learner = domain == "pantry_plan"
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={seed}",
        f"OAT_ZERO_MAX_TRAIN={TRAIN_ROWS}",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={PASSES}",
        f"OAT_ZERO_EVAL_PROMPT_INTERVAL={CHECKPOINT_INTERVAL}",
        f"OAT_ZERO_SAVE_STEPS={recovery_interval}",
        f"OAT_ZERO_SAVE_FROM={recovery_interval}",
        f"OAT_ZERO_RESUME_STEPS={recovery_interval}",
        "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING="
        + ("0" if canonical_learner else "1"),
        "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC="
        + ("0" if canonical_learner else "1"),
        "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING="
        + ("1" if canonical_learner else "0"),
        "OAT_ZERO_SOURCE_ROOT=" + str(snapshot / "src"),
        "OAT_ZERO_OPS_SNAPSHOT_ROOT=" + str(snapshot / "ops"),
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK=1",
        "OAT_ZERO_SEMANTIC_RMS_CONTROL=0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS=0",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS=0",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER=1.0",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING=1",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY=0",
        "OAT_ZERO_MAXENT_ALPHA=0.0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
    )
    missing = [needle for needle in required if needle not in record]
    if scale == "qwen3b":
        cell = (str(run.get("domain")), seed)
        if cell in qwen3_a6000_cells:
            placement = (
                "Partition=lowprio",
                f"ReqNodeList={QWEN3_A6000_NODE_LIST}",
                "TresPerNode=gres/gpu:a6000:1",
            )
        else:
            placement = (
                "Partition=mltheory",
                "ReqNodeList=node302",
                "TresPerNode=gres/gpu:a100:1",
            )
        missing.extend(needle for needle in placement if needle not in record)
    if missing:
        raise RuntimeError(f"held E112-R1 job {job_id} lacks {missing}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path, required=True)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    root = repo_root()
    protocol = root / PROTOCOL
    if not protocol.is_file():
        raise SystemExit(f"E112-R1 protocol is absent: {protocol}")
    snapshot = args.snapshot_root.resolve()
    e111.verify_snapshot(snapshot)
    unit = validate_unit_evidence(root, snapshot)
    gate_ledger: dict[str, Any] = {}
    gate_audit: dict[str, Any] = {}
    e105_retirement: dict[str, Any] = {}
    e112_retirement: dict[str, Any] = {}
    if args.submit:
        gate_ledger, gate_audit = require_terminal_e111_gate(root)
        e105_retirement = require_e105_retired(root)
        e112_retirement = require_original_e112_retired(root)
    ledger_path = root / LEDGER
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate E112-R1 submission: {ledger_path}")

    placement, qwen3_a6000_cells = e105.require_qwen3_paired_placement(root)
    comparators = {
        scale: e105.comparator_index(
            root,
            scale,
            allow_legacy_python=not args.submit,
        )
        for scale in SCALE_SEEDS
    }
    planned: list[dict[str, Any]] = []
    for scale in SCALE_SEEDS:
        for run in references(root, scale):
            domain = str(run.get("domain"))
            seed = int(run.get("seed"))
            env, target = build_env(
                root,
                scale,
                run,
                snapshot,
                qwen3_a6000_cells,
            )
            if args.submit and target.exists():
                raise SystemExit(f"refusing to overwrite E112-R1 run: {target}")
            comparator = comparators[scale][(domain, seed)]
            planned.append(
                {
                    "scale": scale,
                    "model_tag": MODEL_TAGS[scale],
                    "arm": ARM,
                    "domain": domain,
                    "seed": seed,
                    "run_stamp": run_stamp(scale, domain, seed),
                    "run_dir": str(target),
                    "checkpoint_interval_steps": recovery_checkpoint_interval(
                        scale,
                        domain,
                        seed,
                        qwen3_a6000_cells,
                    ),
                    "paired_replay": {
                        "run_stamp": str(comparator.get("run_stamp")),
                        "run_dir": str(comparator.get("run_dir")),
                        "job_id": int(comparator.get("job_id")),
                    },
                    "command": sbatch_command(
                        root, scale, run, env, qwen3_a6000_cells
                    ),
                    "template": run,
                }
            )
    if len(planned) != 75 or Counter(cell["scale"] for cell in planned) != {
        "qwen05b": 25,
        "falcon1b": 25,
        "qwen3b": 25,
    }:
        raise SystemExit("E112-R1 did not materialize exactly 25 cells per scale")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(token) for token in cell["command"]))
        print(f"[e112r1] dry_run=True cells=75 snapshot={snapshot}")
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in planned:
            result = subprocess.run(
                cell["command"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "submission failed for {}: {}".format(
                        cell.get("run_stamp"), result.stderr.strip()
                    )
                )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"invalid job id: {result.stdout!r}")
            submitted.append(job_id)
            held = held_job_audit(
                job_id,
                str(cell["scale"]),
                cell["template"],
                snapshot,
                qwen3_a6000_cells,
            )
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "scale", "model_tag", "arm", "domain", "seed",
                        "run_stamp", "run_dir", "checkpoint_interval_steps",
                        "paired_replay",
                    )
                }
                | {"job_id": int(job_id), "held_scheduler_record": held}
            )
        e111_ledger_path = root / E111_LEDGER
        e111_audit_path = root / E111_AUDIT
        payload = {
            "schema": "e112r1_verified_support_discovery_full_three_scale_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": digest(protocol),
            "launcher_sha256": digest(Path(__file__)),
            "unit_evidence": str(root / UNIT_EVIDENCE),
            "unit_evidence_sha256": digest(root / UNIT_EVIDENCE),
            "snapshot_root": str(snapshot),
            "snapshot_identity_sha256": digest(snapshot / "SNAPSHOT_IDENTITY.json"),
            "e111_ledger": str(e111_ledger_path),
            "e111_ledger_sha256": digest(e111_ledger_path),
            "e111_audit": str(e111_audit_path),
            "e111_audit_sha256": digest(e111_audit_path),
            "e111_scale_chain_cells": gate_audit.get("scale_chain_cells"),
            "e111_runtime_recovery_patch": gate_audit.get(
                "runtime_recovery_patch"
            ),
            "e111_pantry_partial_checkpoint_recovery": gate_audit.get(
                "pantry_partial_checkpoint_recovery"
            ),
            "e111_qwen3_pantry_timeout_continuation": gate_audit.get(
                "qwen3_pantry_timeout_continuation"
            ),
            "e111_qwen3_python_mathir_timeout_replacement": gate_audit.get(
                "qwen3_python_mathir_timeout_replacement"
            ),
            "e105_retirement": e105_retirement,
            "original_e112_retirement": e112_retirement,
            "original_e112_retirement_sha256": digest(root / E112_RETIREMENT),
            "qwen3_paired_placement_artifact": str(root / e105.QWEN3_PLACEMENT_ARTIFACT),
            "qwen3_paired_placement_artifact_sha256": digest(root / e105.QWEN3_PLACEMENT_ARTIFACT),
            "qwen3_a6000_cells": [
                {"domain": domain, "seed": seed}
                for domain, seed in sorted(qwen3_a6000_cells)
            ],
            "comparator_ledgers": {
                scale: {
                    "path": str(root / path),
                    "sha256": digest(root / path),
                }
                for scale, path in e105.COMPARATOR_LEDGERS.items()
            },
            "repaired_python_comparator_ledger": str(root / e105.REPAIRED_PYTHON_COMPARATOR_LEDGER),
            "repaired_python_comparator_ledger_sha256": digest(root / e105.REPAIRED_PYTHON_COMPARATOR_LEDGER),
            "models": list(SCALE_SEEDS),
            "domains": list(DOMAINS),
            "seeds": {key: list(value) for key, value in SCALE_SEEDS.items()},
            "arms": [ARM],
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "qwen3_a6000_checkpoint_interval_steps": (
                QWEN3_A6000_CHECKPOINT_INTERVAL
            ),
            "semantic_coefficient": e111.SEMANTIC_COEFFICIENT,
            "replay_weight": e111.REPLAY_WEIGHT,
            "proposal_temperature": e111.PROPOSAL_TEMPERATURE,
            "proposal_max_attempts": e111.PROPOSAL_MAX_ATTEMPTS,
            "objective": "verified_support_v7_plus_uniform_replaydr_plus_support_discovery",
            "pointmaze": "excluded",
            "outcomes_inspected_for_release": False,
            "runs": records,
            "released": False,
        }
        e111.e81.atomic_json(ledger_path, payload)
        for job_id in submitted:
            released = subprocess.run(
                ["scontrol", "release", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if released.returncode != 0:
                raise RuntimeError(f"release failed for {job_id}")
        payload["released"] = True
        e111.e81.atomic_json(ledger_path, payload)
    except Exception:
        e111.e81.cancel(submitted)
        if ledger_path.exists():
            ledger_path.unlink()
        raise
    print(f"[e112r1] cells=75 released=75 snapshot={snapshot} ledger={ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
