#!/usr/bin/env python3
"""Validate E111 scheduler-only amendments without reading outcome endpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e111_verified_support_discovery_mechanism_gate_jobs.json"
PARTITION_RECORD = ROOT / "var/artifacts/e111_lowprio_partition_amendment.json"
TIME_RECORD = ROOT / "var/artifacts/e111_qwen3_two_hour_backfill_amendment.json"
DURABILITY_RECORD = ROOT / "var/artifacts/e111_qwen3_restart_durability_amendment.json"
WRAPPER = ROOT / "ops/slurm/train_node302.slurm"
DURABILITY_BEGIN = "# BEGIN E111_QWEN3_RESTART_DURABILITY_AMENDMENT"
DURABILITY_END = "# END E111_QWEN3_RESTART_DURABILITY_AMENDMENT"
RUNTIME_OPS_RECORD = ROOT / "var/artifacts/e111_qwen3_runtime_ops_durability_amendment.json"
RUNTIME_TRAIN = ROOT / (
    "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/ops/train.sh"
)
RUNTIME_OPS_BEGIN = "# BEGIN E111_QWEN3_RUNTIME_OPS_DURABILITY_AMENDMENT"
RUNTIME_OPS_END = "# END E111_QWEN3_RUNTIME_OPS_DURABILITY_AMENDMENT"
L40_RECORD = ROOT / "var/artifacts/e111_qwen3_l40_placement_amendment.json"
L40_BACKFILL_RECORD = ROOT / "var/artifacts/e111_qwen3_l40_45m_backfill_amendment.json"
RETURN_A6000_RECORD = ROOT / "var/artifacts/e111_qwen3_return_a6000_amendment.json"
SMALL_BACKFILL_RECORD = ROOT / "var/artifacts/e111_small_scale_node026_backfill_amendment.json"
SMALL_HEALTH_RECORD = ROOT / "var/artifacts/e111_small_scale_node026_health_widening.json"
TWO_STEP_RECORD = ROOT / "var/artifacts/e111_qwen3_two_step_durability_amendment.json"
TWO_STEP_START_RECORD = ROOT / "var/artifacts/e111_qwen3_two_step_start_amendment.json"
TIMEOUT_REQUEUE_RECORD = ROOT / "var/artifacts/e111_exact_timeout_requeue_after_recovery.json"
CHECKPOINT_VALIDATOR_RECORD = ROOT / "var/artifacts/e111_checkpoint_zip_validation_runtime_amendment.json"
VARIANT = "verified_replay_semantic_maxent_verified_support_discovery"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, label: str, violations: list[str]) -> dict[str, Any]:
    if not path.is_file():
        violations.append(f"{label} is absent")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        violations.append(f"{label} is invalid JSON: {exc}")
        return {}
    if not isinstance(payload, dict):
        violations.append(f"{label} is not an object")
        return {}
    return payload


def _submit_line(record: str) -> str:
    marker = "SubmitLine="
    if marker not in record:
        return ""
    return record.split(marker, 1)[1].split(" WorkDir=", 1)[0]


def _durability_block(wrapper: str) -> str:
    if wrapper.count(DURABILITY_BEGIN) != 1 or wrapper.count(DURABILITY_END) != 1:
        raise ValueError("live wrapper lacks one exact E111 durability block")
    start = wrapper.index(DURABILITY_BEGIN)
    finish = wrapper.index(DURABILITY_END, start) + len(DURABILITY_END)
    return wrapper[start:finish]


def _runtime_ops_block(value: str) -> str:
    if value.count(RUNTIME_OPS_BEGIN) != 1 or value.count(RUNTIME_OPS_END) != 1:
        raise ValueError("runtime train script lacks one exact durability block")
    start = value.index(RUNTIME_OPS_BEGIN)
    finish = value.index(RUNTIME_OPS_END, start) + len(RUNTIME_OPS_END)
    return value[start:finish]


def _check_metadata(
    payload: dict[str, Any],
    *,
    label: str,
    schema: str,
    job_ids: list[int],
    changed_fields: list[str],
    ledger_sha256: str,
    violations: list[str],
) -> None:
    expected = {
        "schema": schema,
        "scheduler_only": True,
        "environment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "job_ids": job_ids,
        "changed_fields": changed_fields,
        "ledger_sha256": ledger_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            violations.append(f"{label} has invalid {key}")
    ledger = Path(str(payload.get("ledger", ""))).resolve()
    if ledger != LEDGER.resolve():
        violations.append(f"{label} names a different E111 ledger")
    amendment = Path(str(payload.get("amendment", ""))).resolve()
    if not amendment.is_file() or payload.get("amendment_sha256") != digest(amendment):
        violations.append(f"{label} markdown digest mismatch")


def _common_record_needles(run: dict[str, Any]) -> tuple[str, ...]:
    return (
        "JobId=" + str(int(run["job_id"])),
        "Account=mltheory",
        f"OAT_ZERO_VARIANT={VARIANT}",
        "OAT_ZERO_SEED=" + str(int(run["seed"])),
        "OAT_ZERO_MAX_TRAIN=8",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS=0",
    )


def _check_records(
    payload: dict[str, Any],
    *,
    label: str,
    runs: list[dict[str, Any]],
    before_needles: tuple[str, ...],
    after_needles: tuple[str, ...],
    violations: list[str],
) -> None:
    before = payload.get("before_scheduler_records")
    after = payload.get("after_scheduler_records")
    expected_keys = {str(int(run["job_id"])) for run in runs}
    if not isinstance(before, dict) or set(before) != expected_keys:
        violations.append(f"{label} before-record job set mismatch")
        return
    if not isinstance(after, dict) or set(after) != expected_keys:
        violations.append(f"{label} after-record job set mismatch")
        return
    for run in runs:
        key = str(int(run["job_id"]))
        old = str(before[key])
        new = str(after[key])
        for needle in _common_record_needles(run) + before_needles:
            if needle not in old:
                violations.append(f"{label} {key} before record lacks {needle}")
        for needle in _common_record_needles(run) + after_needles:
            if needle not in new:
                violations.append(f"{label} {key} after record lacks {needle}")
        if not _submit_line(old) or _submit_line(old) != _submit_line(new):
            violations.append(f"{label} {key} submit line changed")


def validate(
    ledger: dict[str, Any],
    runs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    ledger_sha256 = digest(LEDGER) if LEDGER.is_file() else ""
    all_ids = [int(run["job_id"]) for run in runs]
    qwen3_runs = [run for run in runs if str(run.get("scale")) == "qwen3b"]
    qwen3_ids = [int(run["job_id"]) for run in qwen3_runs]
    small_ids = [30674729, 30674733, 30674754]
    small_runs = [run for run in runs if int(run["job_id"]) in set(small_ids)]
    if [int(run["job_id"]) for run in small_runs] != small_ids:
        violations.append("E111 small-scale amendment job order/set mismatch")

    partition = _load(PARTITION_RECORD, "E111 partition amendment", violations)
    _check_metadata(
        partition,
        label="E111 partition amendment",
        schema="e111_lowprio_partition_amendment_v1",
        job_ids=all_ids,
        changed_fields=["Partition"],
        ledger_sha256=ledger_sha256,
        violations=violations,
    )
    for key, expected in (
        ("old_partition", "mltheory"),
        ("new_partition", "lowprio"),
        ("account", "mltheory"),
    ):
        if partition.get(key) != expected:
            violations.append(f"E111 partition amendment has invalid {key}")
    _check_records(
        partition,
        label="E111 partition amendment",
        runs=runs,
        before_needles=(
            "JobState=PENDING", "Reason=BadConstraints", "RunTime=00:00:00",
            "Partition=mltheory",
        ),
        after_needles=("Partition=lowprio",),
        violations=violations,
    )

    timing = _load(TIME_RECORD, "E111 3B time amendment", violations)
    _check_metadata(
        timing,
        label="E111 3B time amendment",
        schema="e111_qwen3_two_hour_backfill_amendment_v1",
        job_ids=qwen3_ids,
        changed_fields=["TimeLimit"],
        ledger_sha256=ledger_sha256,
        violations=violations,
    )
    for key, expected in (
        ("old_time_limit", "12:00:00"),
        ("new_time_limit", "02:00:00"),
    ):
        if timing.get(key) != expected:
            violations.append(f"E111 3B time amendment has invalid {key}")
    if timing.get("prior_partition_amendment_sha256") != (
        digest(PARTITION_RECORD) if PARTITION_RECORD.is_file() else ""
    ):
        violations.append("E111 3B time amendment prior-record digest mismatch")
    prior = Path(str(timing.get("prior_partition_amendment", ""))).resolve()
    if prior != PARTITION_RECORD.resolve():
        violations.append("E111 3B time amendment names a different prior record")
    _check_records(
        timing,
        label="E111 3B time amendment",
        runs=qwen3_runs,
        before_needles=(
            "JobState=PENDING", "RunTime=00:00:00", "Partition=lowprio",
            "ReqNodeList=node[103-104,205-208]",
            "TresPerNode=gres/gpu:a6000:1", "TimeLimit=12:00:00",
        ),
        after_needles=(
            "Partition=lowprio", "ReqNodeList=node[103-104,205-208]",
            "TresPerNode=gres/gpu:a6000:1", "TimeLimit=02:00:00",
        ),
        violations=violations,
    )

    durability = _load(DURABILITY_RECORD, "E111 3B durability amendment", violations)
    expected_durability = {
        "schema": "e111_qwen3_restart_durability_amendment_v1",
        "job_ids": qwen3_ids,
        "changed_fields": [
            "OAT_ZERO_SAVE_STEPS",
            "OAT_ZERO_SAVE_FROM",
            "OAT_ZERO_RESUME_STEPS",
        ],
        "checkpoint_before": {
            "OAT_ZERO_SAVE_STEPS": "32",
            "OAT_ZERO_SAVE_FROM": "32",
            "OAT_ZERO_RESUME_STEPS": "32",
        },
        "checkpoint_after": {
            "OAT_ZERO_SAVE_STEPS": "8",
            "OAT_ZERO_SAVE_FROM": "8",
            "OAT_ZERO_RESUME_STEPS": "8",
        },
        "source_root": str(
            ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/src"
        ),
        "ops_root": str(
            ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/ops"
        ),
        "variant": VARIANT,
        "ledger_sha256": ledger_sha256,
        "storage_only": True,
        "treatment_environment_changed": False,
        "recovery_environment_changed": True,
        "endpoint_outcomes_inspected": False,
        "pointmaze": "excluded",
        "installed": True,
    }
    for key, value in expected_durability.items():
        if durability.get(key) != value:
            violations.append(f"E111 3B durability amendment has invalid {key}")
    if Path(str(durability.get("ledger", ""))).resolve() != LEDGER.resolve():
        violations.append("E111 3B durability amendment names a different ledger")
    amendment = Path(str(durability.get("amendment", ""))).resolve()
    if (
        not amendment.is_file()
        or durability.get("amendment_sha256") != digest(amendment)
    ):
        violations.append("E111 3B durability amendment markdown digest mismatch")
    if Path(str(durability.get("wrapper", ""))).resolve() != WRAPPER.resolve():
        violations.append("E111 3B durability amendment names a different wrapper")

    try:
        block = _durability_block(WRAPPER.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        violations.append(f"E111 3B durability amendment wrapper error: {exc}")
        block = ""
    if block:
        block_digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
        if durability.get("amendment_block_sha256") != block_digest:
            violations.append("E111 3B durability amendment block digest mismatch")
        required_block_fragments = (
            "30674758|30674759|30674760|30674761|30674762)",
            "e76_tuned_scale_2febfdc12e36650d/src",
            "e76_tuned_scale_2febfdc12e36650d/ops",
            f'"{VARIANT}"',
            '[[ "${OAT_ZERO_SAVE_STEPS:-}" != "32" ]]',
            '[[ "${OAT_ZERO_SAVE_FROM:-}" != "32" ]]',
            '[[ "${OAT_ZERO_RESUME_STEPS:-}" != "32" ]]',
            "export OAT_ZERO_SAVE_STEPS=8",
            "export OAT_ZERO_SAVE_FROM=8",
            "export OAT_ZERO_RESUME_STEPS=8",
        )
        for fragment in required_block_fragments:
            if fragment not in block:
                violations.append(
                    f"E111 3B durability amendment block lacks {fragment}"
                )

    _check_records(
        durability,
        label="E111 3B durability amendment",
        runs=qwen3_runs,
        before_needles=(
            "OAT_ZERO_SAVE_STEPS=32",
            "OAT_ZERO_SAVE_FROM=32",
            "OAT_ZERO_RESUME_STEPS=32",
        ),
        after_needles=(
            "OAT_ZERO_SAVE_STEPS=32",
            "OAT_ZERO_SAVE_FROM=32",
            "OAT_ZERO_RESUME_STEPS=32",
        ),
        violations=violations,
    )
    before_restarts = durability.get("restart_counts_before")
    after_restarts = durability.get("restart_counts_after")
    expected_restart_keys = {str(job_id) for job_id in qwen3_ids}
    if not isinstance(before_restarts, dict) or set(before_restarts) != expected_restart_keys:
        violations.append("E111 3B durability before-restart set mismatch")
    if not isinstance(after_restarts, dict) or set(after_restarts) != expected_restart_keys:
        violations.append("E111 3B durability after-restart set mismatch")
    if isinstance(before_restarts, dict) and isinstance(after_restarts, dict):
        for key in expected_restart_keys:
            before_value = before_restarts.get(key)
            after_value = after_restarts.get(key)
            if (
                not isinstance(before_value, int)
                or not isinstance(after_value, int)
                or before_value < 0
                or after_value < before_value
            ):
                violations.append(
                    f"E111 3B durability has invalid restart counts for {key}"
                )

    runtime = _load(
        RUNTIME_OPS_RECORD, "E111 3B runtime-ops durability amendment", violations
    )
    expected_runtime = {
        "schema": "e111_qwen3_runtime_ops_durability_amendment_v1",
        "job_ids": qwen3_ids,
        "source_root": str(
            ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/src"
        ),
        "ops_root": str(
            ROOT / "var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/ops"
        ),
        "variant": VARIANT,
        "ledger_sha256": ledger_sha256,
        "checkpoint_before": {
            "OAT_ZERO_SAVE_STEPS": "32",
            "OAT_ZERO_SAVE_FROM": "32",
            "OAT_ZERO_RESUME_STEPS": "32",
        },
        "checkpoint_after": {
            "OAT_ZERO_SAVE_STEPS": "8",
            "OAT_ZERO_SAVE_FROM": "8",
            "OAT_ZERO_RESUME_STEPS": "8",
        },
        "stored_batch_script_job_id": 30674760,
        "stored_batch_script_rereads_runtime_train": True,
        "storage_only": True,
        "python_source_changed": False,
        "treatment_changed": False,
        "endpoint_outcomes_inspected": False,
        "pointmaze": "excluded",
        "installed": True,
    }
    for key, value in expected_runtime.items():
        if runtime.get(key) != value:
            violations.append(
                f"E111 3B runtime-ops durability amendment has invalid {key}"
            )
    if Path(str(runtime.get("ledger", ""))).resolve() != LEDGER.resolve():
        violations.append(
            "E111 3B runtime-ops durability amendment names a different ledger"
        )
    runtime_amendment = Path(str(runtime.get("amendment", ""))).resolve()
    if (
        not runtime_amendment.is_file()
        or runtime.get("amendment_sha256") != digest(runtime_amendment)
    ):
        violations.append(
            "E111 3B runtime-ops durability amendment markdown digest mismatch"
        )
    if Path(str(runtime.get("runtime_train", ""))).resolve() != RUNTIME_TRAIN.resolve():
        violations.append(
            "E111 3B runtime-ops durability amendment names a different train script"
        )
    for key in ("runtime_train_after_sha256", "amendment_block_sha256"):
        value = runtime.get(key)
        if not isinstance(value, str) or len(value) != 64:
            violations.append(
                f"E111 3B runtime-ops durability amendment has invalid {key}"
            )

    stored_batch = str(runtime.get("stored_batch_script", ""))
    if runtime.get("stored_batch_script_sha256") != hashlib.sha256(
        stored_batch.encode("utf-8")
    ).hexdigest():
        violations.append(
            "E111 3B runtime-ops durability stored-batch digest mismatch"
        )
    for fragment in (
        "RUNTIME_OPS_ROOT=",
        'cp "${RUNTIME_OPS_ROOT}/train.sh" "$RUNTIME_SCRIPT_DIR/train.sh"',
        'exec "$RUNTIME_SCRIPT_DIR/run_experiment.sh"',
    ):
        if fragment not in stored_batch:
            violations.append(
                f"E111 3B runtime-ops durability stored batch lacks {fragment}"
            )
    if DURABILITY_BEGIN in stored_batch:
        violations.append(
            "checkout-wrapper durability block unexpectedly entered stored batch"
        )

    _check_records(
        runtime,
        label="E111 3B runtime-ops durability amendment",
        runs=qwen3_runs,
        before_needles=(
            "OAT_ZERO_SAVE_STEPS=32",
            "OAT_ZERO_SAVE_FROM=32",
            "OAT_ZERO_RESUME_STEPS=32",
        ),
        after_needles=(
            "OAT_ZERO_SAVE_STEPS=32",
            "OAT_ZERO_SAVE_FROM=32",
            "OAT_ZERO_RESUME_STEPS=32",
        ),
        violations=violations,
    )
    runtime_before_restarts = runtime.get("restart_counts_before")
    runtime_after_restarts = runtime.get("restart_counts_after")
    if (
        not isinstance(runtime_before_restarts, dict)
        or set(runtime_before_restarts) != expected_restart_keys
    ):
        violations.append("E111 3B runtime-ops before-restart set mismatch")
    if (
        not isinstance(runtime_after_restarts, dict)
        or set(runtime_after_restarts) != expected_restart_keys
    ):
        violations.append("E111 3B runtime-ops after-restart set mismatch")
    if isinstance(runtime_before_restarts, dict) and isinstance(
        runtime_after_restarts, dict
    ):
        for key in expected_restart_keys:
            before_value = runtime_before_restarts.get(key)
            after_value = runtime_after_restarts.get(key)
            if (
                not isinstance(before_value, int)
                or not isinstance(after_value, int)
                or before_value < 0
                or after_value < before_value
            ):
                violations.append(
                    f"E111 3B runtime-ops has invalid restart counts for {key}"
                )

    two_step = _load(
        TWO_STEP_RECORD, "E111 3B two-step durability amendment", violations
    )
    expected_two_step = {
        "schema": "e111_qwen3_two_step_durability_amendment_v1",
        "job_ids": qwen3_ids,
        "old_interval": 8,
        "new_interval": 2,
        "storage_only": True,
        "treatment_changed": False,
        "jobs_signaled": False,
        "jobs_reset": False,
        "endpoint_outcomes_inspected": False,
        "pointmaze": "excluded",
        "ledger_sha256": ledger_sha256,
        "prior_record_sha256": (
            digest(RUNTIME_OPS_RECORD) if RUNTIME_OPS_RECORD.is_file() else ""
        ),
        "runtime_train_before_sha256": runtime.get("runtime_train_after_sha256"),
        "prior_block_sha256": runtime.get("amendment_block_sha256"),
        "installed": True,
    }
    for key, value in expected_two_step.items():
        if two_step.get(key) != value:
            violations.append(f"E111 3B two-step amendment has invalid {key}")
    if Path(str(two_step.get("ledger", ""))).resolve() != LEDGER.resolve():
        violations.append("E111 3B two-step amendment names a different ledger")
    if Path(str(two_step.get("prior_record", ""))).resolve() != RUNTIME_OPS_RECORD.resolve():
        violations.append("E111 3B two-step amendment names a different prior record")
    if Path(str(two_step.get("runtime_train", ""))).resolve() != RUNTIME_TRAIN.resolve():
        violations.append("E111 3B two-step amendment names a different train script")
    if not isinstance(two_step.get("runtime_train_after_sha256"), str) or len(
        two_step.get("runtime_train_after_sha256", "")
    ) != 64:
        violations.append("E111 3B two-step amended train digest is invalid")
    two_step_amendment = Path(str(two_step.get("amendment", ""))).resolve()
    if (
        not two_step_amendment.is_file()
        or two_step.get("amendment_sha256") != digest(two_step_amendment)
    ):
        violations.append("E111 3B two-step amendment markdown digest mismatch")

    two_step_start = _load(
        TWO_STEP_START_RECORD, "E111 3B two-step start amendment", violations
    )
    expected_two_step_start = {
        "schema": "e111_qwen3_two_step_start_amendment_v1",
        "job_ids": qwen3_ids,
        "old_resume_from": 32,
        "new_resume_from": 2,
        "checkpoint_interval": 2,
        "storage_only": True,
        "treatment_changed": False,
        "jobs_signaled": False,
        "jobs_reset": False,
        "endpoint_outcomes_inspected": False,
        "pointmaze": "excluded",
        "ledger_sha256": ledger_sha256,
        "prior_record_sha256": (
            digest(TWO_STEP_RECORD) if TWO_STEP_RECORD.is_file() else ""
        ),
        "runtime_train_before_sha256": two_step.get("runtime_train_after_sha256"),
        "prior_block_sha256": two_step.get("amendment_block_sha256"),
        "installed": True,
    }
    for key, value in expected_two_step_start.items():
        if two_step_start.get(key) != value:
            violations.append(f"E111 3B two-step start amendment has invalid {key}")
    if Path(str(two_step_start.get("ledger", ""))).resolve() != LEDGER.resolve():
        violations.append("E111 3B two-step start amendment names a different ledger")
    if Path(str(two_step_start.get("prior_record", ""))).resolve() != TWO_STEP_RECORD.resolve():
        violations.append("E111 3B two-step start amendment names a different prior record")
    if Path(str(two_step_start.get("runtime_train", ""))).resolve() != RUNTIME_TRAIN.resolve():
        violations.append("E111 3B two-step start amendment names a different train script")
    elif two_step_start.get("runtime_train_after_sha256") != digest(RUNTIME_TRAIN):
        violations.append("E111 3B two-step start amended train digest mismatch")
    start_amendment = Path(str(two_step_start.get("amendment", ""))).resolve()
    if (
        not start_amendment.is_file()
        or two_step_start.get("amendment_sha256") != digest(start_amendment)
    ):
        violations.append("E111 3B two-step start amendment markdown digest mismatch")

    try:
        current_runtime_block = _runtime_ops_block(
            RUNTIME_TRAIN.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        violations.append(f"E111 3B two-step runtime block error: {exc}")
        current_runtime_block = ""
    if current_runtime_block:
        current_block_digest = hashlib.sha256(
            current_runtime_block.encode("utf-8")
        ).hexdigest()
        if two_step_start.get("amendment_block_sha256") != current_block_digest:
            violations.append("E111 3B two-step start runtime block digest mismatch")
        for fragment in (
            "30674758|30674759|30674760|30674761|30674762)",
            "e76_tuned_scale_2febfdc12e36650d/src",
            "e76_tuned_scale_2febfdc12e36650d/ops",
            f'"{VARIANT}"',
            "OAT_ZERO_SAVE_STEPS:-",
            "OAT_ZERO_SAVE_FROM:-",
            "OAT_ZERO_RESUME_STEPS:-",
            "OAT_ZERO_RESUME_FROM:-",
            "export OAT_ZERO_SAVE_STEPS=2",
            "export OAT_ZERO_SAVE_FROM=2",
            "export OAT_ZERO_RESUME_STEPS=2",
            "export OAT_ZERO_RESUME_FROM=2",
            "checkpoint_interval=2 checkpoint_start=2",
        ):
            if fragment not in current_runtime_block:
                violations.append(
                    f"E111 3B two-step start runtime block lacks {fragment}"
                )

    _check_records(
        two_step,
        label="E111 3B two-step durability amendment",
        runs=qwen3_runs,
        before_needles=(
            "OAT_ZERO_SAVE_STEPS=32",
            "OAT_ZERO_SAVE_FROM=32",
            "OAT_ZERO_RESUME_STEPS=32",
        ),
        after_needles=(
            "OAT_ZERO_SAVE_STEPS=32",
            "OAT_ZERO_SAVE_FROM=32",
            "OAT_ZERO_RESUME_STEPS=32",
        ),
        violations=violations,
    )
    two_before = two_step.get("restart_counts_before")
    two_after = two_step.get("restart_counts_after")
    if not isinstance(two_before, dict) or set(two_before) != expected_restart_keys:
        violations.append("E111 3B two-step before-restart set mismatch")
    if not isinstance(two_after, dict) or set(two_after) != expected_restart_keys:
        violations.append("E111 3B two-step after-restart set mismatch")
    if isinstance(two_before, dict) and isinstance(two_after, dict):
        for key in expected_restart_keys:
            before_value = two_before.get(key)
            after_value = two_after.get(key)
            if (
                not isinstance(before_value, int)
                or not isinstance(after_value, int)
                or before_value < 0
                or after_value < before_value
            ):
                violations.append(
                    f"E111 3B two-step has invalid restart counts for {key}"
                )

    _check_records(
        two_step_start,
        label="E111 3B two-step start amendment",
        runs=qwen3_runs,
        before_needles=(
            "OAT_ZERO_SAVE_STEPS=32",
            "OAT_ZERO_SAVE_FROM=32",
            "OAT_ZERO_RESUME_STEPS=32",
        ),
        after_needles=(
            "OAT_ZERO_SAVE_STEPS=32",
            "OAT_ZERO_SAVE_FROM=32",
            "OAT_ZERO_RESUME_STEPS=32",
        ),
        violations=violations,
    )
    start_before = two_step_start.get("restart_counts_before")
    start_after = two_step_start.get("restart_counts_after")
    if not isinstance(start_before, dict) or set(start_before) != expected_restart_keys:
        violations.append("E111 3B two-step start before-restart set mismatch")
    if not isinstance(start_after, dict) or set(start_after) != expected_restart_keys:
        violations.append("E111 3B two-step start after-restart set mismatch")
    if isinstance(start_before, dict) and isinstance(start_after, dict):
        for key in expected_restart_keys:
            before_value = start_before.get(key)
            after_value = start_after.get(key)
            if (
                not isinstance(before_value, int)
                or not isinstance(after_value, int)
                or before_value < 0
                or after_value < before_value
            ):
                violations.append(
                    f"E111 3B two-step start has invalid restart counts for {key}"
                )

    l40 = _load(L40_RECORD, "E111 3B L40 placement amendment", violations)
    expected_l40 = {
        "schema": "e111_qwen3_l40_placement_amendment_v1",
        "job_ids": qwen3_ids,
        "changed_fields": ["ReqNodeList", "Gres"],
        "old_required_nodes": "node[103-104,205-208]",
        "new_required_nodes": "node403",
        "old_gres": "gpu:a6000:1",
        "new_gres": "gpu:l40:1",
        "partition": "lowprio",
        "account": "mltheory",
        "cpus_per_task": 16,
        "memory": "128G",
        "time_limit": "02:00:00",
        "scheduler_only": True,
        "environment_changed": False,
        "mechanism_gate_only": True,
        "e112_paired_hardware_changed": False,
        "endpoint_outcomes_inspected": False,
        "pointmaze": "excluded",
        "ledger_sha256": ledger_sha256,
        "applied": True,
    }
    for key, value in expected_l40.items():
        if l40.get(key) != value:
            violations.append(f"E111 3B L40 placement amendment has invalid {key}")
    if Path(str(l40.get("ledger", ""))).resolve() != LEDGER.resolve():
        violations.append("E111 3B L40 placement amendment names a different ledger")
    l40_amendment = Path(str(l40.get("amendment", ""))).resolve()
    if (
        not l40_amendment.is_file()
        or l40.get("amendment_sha256") != digest(l40_amendment)
    ):
        violations.append("E111 3B L40 placement amendment markdown digest mismatch")
    _check_records(
        l40,
        label="E111 3B L40 placement amendment",
        runs=qwen3_runs,
        before_needles=(
            "JobState=PENDING",
            "Partition=lowprio",
            "ReqNodeList=node[103-104,205-208]",
            "TresPerNode=gres/gpu:a6000:1",
            "MinMemoryNode=128G",
            "TimeLimit=02:00:00",
        ),
        after_needles=(
            "Partition=lowprio",
            "ReqNodeList=node403",
            "TresPerNode=gres/gpu:l40:1",
            "MinMemoryNode=128G",
            "TimeLimit=02:00:00",
        ),
        violations=violations,
    )
    for label, node_record in (
        ("before", str(l40.get("node403_before", ""))),
        ("after", str(l40.get("node403_after", ""))),
    ):
        for needle in ("NodeName=node403", "Gres=gpu:l40:8", "lowprio"):
            if needle not in node_record:
                violations.append(
                    f"E111 3B L40 placement {label} node record lacks {needle}"
                )

    backfill = _load(
        L40_BACKFILL_RECORD, "E111 3B L40 backfill amendment", violations
    )
    expected_backfill = {
        "schema": "e111_qwen3_l40_45m_backfill_amendment_v1",
        "job_ids": qwen3_ids,
        "changed_fields": ["TimeLimit"],
        "old_time_limit": "02:00:00",
        "new_time_limit": "00:45:00",
        "required_nodes": "node403",
        "gres": "gpu:l40:1",
        "scheduler_only": True,
        "environment_changed": False,
        "mechanism_gate_only": True,
        "e112_changed": False,
        "endpoint_outcomes_inspected": False,
        "pointmaze": "excluded",
        "ledger_sha256": ledger_sha256,
        "applied": True,
    }
    for key, value in expected_backfill.items():
        if backfill.get(key) != value:
            violations.append(f"E111 3B L40 backfill amendment has invalid {key}")
    if Path(str(backfill.get("ledger", ""))).resolve() != LEDGER.resolve():
        violations.append("E111 3B L40 backfill amendment names a different ledger")
    backfill_amendment = Path(str(backfill.get("amendment", ""))).resolve()
    if (
        not backfill_amendment.is_file()
        or backfill.get("amendment_sha256") != digest(backfill_amendment)
    ):
        violations.append("E111 3B L40 backfill amendment markdown digest mismatch")
    _check_records(
        backfill,
        label="E111 3B L40 backfill amendment",
        runs=qwen3_runs,
        before_needles=(
            "JobState=PENDING",
            "Partition=lowprio",
            "ReqNodeList=node403",
            "TresPerNode=gres/gpu:l40:1",
            "TimeLimit=02:00:00",
        ),
        after_needles=(
            "Partition=lowprio",
            "ReqNodeList=node403",
            "TresPerNode=gres/gpu:l40:1",
            "TimeLimit=00:45:00",
        ),
        violations=violations,
    )

    returned = _load(
        RETURN_A6000_RECORD, "E111 3B A6000 return amendment", violations
    )
    expected_return = {
        "schema": "e111_qwen3_return_a6000_amendment_v1",
        "job_ids": qwen3_ids,
        "changed_fields": ["ReqNodeList", "Gres"],
        "old_required_nodes": "node403",
        "new_required_nodes": "node[103-104,205-208]",
        "old_gres": "gpu:l40:1",
        "new_gres": "gpu:a6000:1",
        "time_limit": "00:45:00",
        "l40_training_steps": 0,
        "scheduler_only": True,
        "environment_changed": False,
        "e112_changed": False,
        "endpoint_outcomes_inspected": False,
        "pointmaze": "excluded",
        "ledger_sha256": ledger_sha256,
        "applied": True,
    }
    for key, value in expected_return.items():
        if returned.get(key) != value:
            violations.append(f"E111 3B A6000 return amendment has invalid {key}")
    if Path(str(returned.get("ledger", ""))).resolve() != LEDGER.resolve():
        violations.append("E111 3B A6000 return amendment names a different ledger")
    return_amendment = Path(str(returned.get("amendment", ""))).resolve()
    if (
        not return_amendment.is_file()
        or returned.get("amendment_sha256") != digest(return_amendment)
    ):
        violations.append("E111 3B A6000 return amendment markdown digest mismatch")
    _check_records(
        returned,
        label="E111 3B A6000 return amendment",
        runs=qwen3_runs,
        before_needles=(
            "JobState=PENDING",
            "Partition=lowprio",
            "ReqNodeList=node403",
            "TresPerNode=gres/gpu:l40:1",
            "TimeLimit=00:45:00",
        ),
        after_needles=(
            "Partition=lowprio",
            "ReqNodeList=node[103-104,205-208]",
            "TresPerNode=gres/gpu:a6000:1",
            "TimeLimit=00:45:00",
        ),
        violations=violations,
    )

    small = _load(
        SMALL_BACKFILL_RECORD, "E111 small-scale node026 amendment", violations
    )
    expected_small = {
        "schema": "e111_small_scale_node026_backfill_amendment_v1",
        "job_ids": small_ids,
        "changed_fields": ["ReqNodeList", "TimeLimit"],
        "old_required_nodes": "node[020-022,025,103-104,202-208,403]",
        "new_required_nodes": "node026",
        "old_time_limit": "08:00:00",
        "new_time_limit": "00:45:00",
        "gres": "gpu:1",
        "scheduler_only": True,
        "environment_changed": False,
        "endpoint_outcomes_inspected": False,
        "pointmaze": "excluded",
        "ledger_sha256": ledger_sha256,
        "applied": True,
    }
    for key, value in expected_small.items():
        if small.get(key) != value:
            violations.append(f"E111 small-scale node026 amendment has invalid {key}")
    if Path(str(small.get("ledger", ""))).resolve() != LEDGER.resolve():
        violations.append("E111 small-scale node026 amendment names a different ledger")
    small_amendment = Path(str(small.get("amendment", ""))).resolve()
    if (
        not small_amendment.is_file()
        or small.get("amendment_sha256") != digest(small_amendment)
    ):
        violations.append("E111 small-scale node026 amendment markdown digest mismatch")
    _check_records(
        small,
        label="E111 small-scale node026 amendment",
        runs=small_runs,
        before_needles=(
            "JobState=PENDING",
            "Partition=lowprio",
            "ReqNodeList=node[020-022,025,103-104,202-208,403]",
            "TresPerNode=gres/gpu:1",
            "MinMemoryNode=64G",
            "TimeLimit=08:00:00",
        ),
        after_needles=(
            "Partition=lowprio",
            "ReqNodeList=node026",
            "TresPerNode=gres/gpu:1",
            "MinMemoryNode=64G",
            "TimeLimit=00:45:00",
        ),
        violations=violations,
    )
    for label, node_record in (
        ("before", str(small.get("node026_before", ""))),
        ("after", str(small.get("node026_after", ""))),
    ):
        for needle in ("NodeName=node026", "Gres=gpu:rtx_3090:10", "lowprio"):
            if needle not in node_record:
                violations.append(
                    f"E111 small-scale node026 {label} record lacks {needle}"
                )

    small_health = _load(
        SMALL_HEALTH_RECORD,
        "E111 small-scale node026 health widening",
        violations,
    )
    expected_small_health = {
        "schema": "e111_small_scale_node026_health_widening_v1",
        "job_ids": small_ids,
        "old_required_nodes": "node026",
        "new_required_nodes": "node[202-204,403]",
        "time_limit": "00:45:00",
        "gres": "gpu:1",
        "changed_fields": ["ReqNodeList"],
        "same_job_ids": True,
        "jobs_requeued": False,
        "jobs_reset": False,
        "replacement_jobs_submitted": False,
        "environment_changed": False,
        "treatment_changed": False,
        "e112_paired_hardware_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "ledger_sha256": ledger_sha256,
        "installed": True,
    }
    for key, value in expected_small_health.items():
        if small_health.get(key) != value:
            violations.append(
                f"E111 small-scale node026 health widening has invalid {key}"
            )
    if Path(str(small_health.get("ledger", ""))).resolve() != LEDGER.resolve():
        violations.append(
            "E111 small-scale node026 health widening names a different ledger"
        )
    health_protocol = Path(str(small_health.get("protocol", ""))).resolve()
    if (
        not health_protocol.is_file()
        or small_health.get("protocol_sha256") != digest(health_protocol)
    ):
        violations.append(
            "E111 small-scale node026 health-widening protocol digest mismatch"
        )
    _check_records(
        small_health,
        label="E111 small-scale node026 health widening",
        runs=small_runs,
        before_needles=(
            "JobState=PENDING",
            "Partition=lowprio",
            "ReqNodeList=node026",
            "TresPerNode=gres/gpu:1",
            "MinMemoryNode=64G",
            "TimeLimit=00:45:00",
        ),
        after_needles=(
            "JobState=PENDING",
            "Partition=lowprio",
            "ReqNodeList=node[202-204,403]",
            "TresPerNode=gres/gpu:1",
            "MinMemoryNode=64G",
            "TimeLimit=00:45:00",
        ),
        violations=violations,
    )
    node026_trigger = str(small_health.get("node026_before", ""))
    for needle in ("NodeName=node026", "DRAIN", "overheated"):
        if needle not in node026_trigger:
            violations.append(
                f"E111 small-scale node026 health trigger lacks {needle}"
            )
    replacement_nodes = small_health.get("replacement_nodes_before")
    expected_replacement_nodes = {"node202", "node203", "node204", "node403"}
    if not isinstance(replacement_nodes, dict) or set(replacement_nodes) != (
        expected_replacement_nodes
    ):
        violations.append(
            "E111 small-scale health-widening replacement node set mismatch"
        )
    elif any(
        marker in str(record)
        for record in replacement_nodes.values()
        for marker in ("DRAIN", "DOWN", "FAIL")
    ):
        violations.append(
            "E111 small-scale health-widening replacement node was unhealthy"
        )
    update_results = small_health.get("update_results")
    expected_update_keys = {str(value) for value in small_ids}
    if not isinstance(update_results, dict) or set(update_results) != (
        expected_update_keys
    ):
        violations.append(
            "E111 small-scale health-widening update-result set mismatch"
        )
    elif any(
        not isinstance(value, dict) or value.get("returncode") != 0
        for value in update_results.values()
    ):
        violations.append("E111 small-scale health-widening update failed")

    checkpoint_validator = _load(
        CHECKPOINT_VALIDATOR_RECORD,
        "E111 checkpoint ZIP-validation amendment",
        violations,
    )
    expected_checkpoint_validator = {
        "schema": "e111_checkpoint_zip_validation_runtime_amendment_v1",
        "root_live_helper_identical": True,
        "selection_rule": (
            "highest-step/newest-tie among complete model+optimizer ZIP checkpoints"
        ),
        "checkpoint_storage_only": True,
        "python_training_source_changed": False,
        "optimizer_update_changed": False,
        "treatment_changed": False,
        "running_jobs_signaled_or_reset": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "installed": True,
    }
    for key, value in expected_checkpoint_validator.items():
        if checkpoint_validator.get(key) != value:
            violations.append(
                f"E111 checkpoint ZIP-validation amendment has invalid {key}"
            )
    validator_protocol = Path(
        str(checkpoint_validator.get("protocol", ""))
    ).resolve()
    if (
        not validator_protocol.is_file()
        or checkpoint_validator.get("protocol_sha256") != digest(validator_protocol)
    ):
        violations.append("E111 checkpoint ZIP-validation protocol digest mismatch")
    for field in ("root_helper", "root_runner", "live_helper", "live_runner"):
        path = Path(str(checkpoint_validator.get(field, ""))).resolve()
        if (
            not path.is_file()
            or checkpoint_validator.get(field + "_sha256") != digest(path)
        ):
            violations.append(
                f"E111 checkpoint ZIP-validation {field} digest mismatch"
            )
    root_helper = Path(str(checkpoint_validator.get("root_helper", "")))
    live_helper = Path(str(checkpoint_validator.get("live_helper", "")))
    if (
        not root_helper.is_file()
        or not live_helper.is_file()
        or root_helper.read_bytes() != live_helper.read_bytes()
    ):
        violations.append("E111 checkpoint ZIP-validation helpers differ")

    timeout_requeue = _load(
        TIMEOUT_REQUEUE_RECORD,
        "E111 exact timeout requeue recovery",
        violations,
    )
    timeout_ids = [30674729, 30674733, 30674754, 30674762]
    expected_timeout = {
        "schema": "e111_exact_timeout_requeue_after_recovery_v1",
        "job_ids": timeout_ids,
        "same_job_ids": True,
        "replacement_jobs_submitted": False,
        "state_reset": False,
        "environment_changed": False,
        "treatment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "ledger_sha256": ledger_sha256,
        "installed": True,
    }
    for key, value in expected_timeout.items():
        if timeout_requeue.get(key) != value:
            violations.append(f"E111 timeout requeue recovery has invalid {key}")
    if Path(str(timeout_requeue.get("ledger", ""))).resolve() != LEDGER.resolve():
        violations.append("E111 timeout requeue recovery names a different ledger")
    timeout_protocol = Path(str(timeout_requeue.get("protocol", ""))).resolve()
    if (
        not timeout_protocol.is_file()
        or timeout_requeue.get("protocol_sha256") != digest(timeout_protocol)
    ):
        violations.append("E111 timeout requeue protocol digest mismatch")
    expected_keys = {str(job_id) for job_id in timeout_ids}
    pre = timeout_requeue.get("pre_action_accounting")
    post = timeout_requeue.get("post_action_scheduler_records")
    commands = timeout_requeue.get("commands")
    results = timeout_requeue.get("requeue_results")
    if not isinstance(pre, dict) or set(pre) != expected_keys:
        violations.append("E111 timeout requeue pre-action job set mismatch")
    if not isinstance(post, dict) or set(post) != expected_keys:
        violations.append("E111 timeout requeue post-action job set mismatch")
    if commands != [["scontrol", "requeue", str(job_id)] for job_id in timeout_ids]:
        violations.append("E111 timeout requeue command set/order mismatch")
    if not isinstance(results, dict) or set(results) != expected_keys:
        violations.append("E111 timeout requeue result set mismatch")
    for job_id in timeout_ids:
        key = str(job_id)
        if isinstance(pre, dict) and (
            key not in pre or f"{job_id}|TIMEOUT|" not in str(pre[key]).splitlines()[0]
        ):
            violations.append(f"E111 timeout requeue {key} lacks TIMEOUT evidence")
        if isinstance(post, dict):
            record = str(post.get(key, ""))
            run = next((row for row in runs if int(row["job_id"]) == job_id), None)
            if run is None:
                violations.append(f"E111 timeout requeue {key} is absent from ledger")
            else:
                for needle in _common_record_needles(run):
                    if needle not in record:
                        violations.append(
                            f"E111 timeout requeue {key} post record lacks {needle}"
                        )
                if not any(
                    f"JobState={state}" in record for state in ("PENDING", "RUNNING")
                ):
                    violations.append(
                        f"E111 timeout requeue {key} post record lacks queued state"
                    )
        if isinstance(results, dict) and (
            not isinstance(results.get(key), dict)
            or results[key].get("returncode") != 0
        ):
            violations.append(f"E111 timeout requeue {key} command failed")

    return {
        "partition_record": str(PARTITION_RECORD),
        "partition_record_sha256": digest(PARTITION_RECORD) if PARTITION_RECORD.is_file() else None,
        "qwen3_time_record": str(TIME_RECORD),
        "qwen3_time_record_sha256": digest(TIME_RECORD) if TIME_RECORD.is_file() else None,
        "qwen3_durability_record": str(DURABILITY_RECORD),
        "qwen3_durability_record_sha256": (
            digest(DURABILITY_RECORD) if DURABILITY_RECORD.is_file() else None
        ),
        "qwen3_runtime_ops_durability_record": str(RUNTIME_OPS_RECORD),
        "qwen3_runtime_ops_durability_record_sha256": (
            digest(RUNTIME_OPS_RECORD) if RUNTIME_OPS_RECORD.is_file() else None
        ),
        "qwen3_two_step_durability_record": str(TWO_STEP_RECORD),
        "qwen3_two_step_durability_record_sha256": (
            digest(TWO_STEP_RECORD) if TWO_STEP_RECORD.is_file() else None
        ),
        "qwen3_two_step_start_record": str(TWO_STEP_START_RECORD),
        "qwen3_two_step_start_record_sha256": (
            digest(TWO_STEP_START_RECORD) if TWO_STEP_START_RECORD.is_file() else None
        ),
        "qwen3_l40_placement_record": str(L40_RECORD),
        "qwen3_l40_placement_record_sha256": (
            digest(L40_RECORD) if L40_RECORD.is_file() else None
        ),
        "qwen3_l40_backfill_record": str(L40_BACKFILL_RECORD),
        "qwen3_l40_backfill_record_sha256": (
            digest(L40_BACKFILL_RECORD) if L40_BACKFILL_RECORD.is_file() else None
        ),
        "qwen3_return_a6000_record": str(RETURN_A6000_RECORD),
        "qwen3_return_a6000_record_sha256": (
            digest(RETURN_A6000_RECORD) if RETURN_A6000_RECORD.is_file() else None
        ),
        "small_scale_node026_record": str(SMALL_BACKFILL_RECORD),
        "small_scale_node026_record_sha256": (
            digest(SMALL_BACKFILL_RECORD) if SMALL_BACKFILL_RECORD.is_file() else None
        ),
        "small_scale_node026_health_record": str(SMALL_HEALTH_RECORD),
        "small_scale_node026_health_record_sha256": (
            digest(SMALL_HEALTH_RECORD) if SMALL_HEALTH_RECORD.is_file() else None
        ),
        "small_scale_node026_health_widening_applied": (
            small_health.get("installed") is True
        ),
        "checkpoint_validator_record": str(CHECKPOINT_VALIDATOR_RECORD),
        "checkpoint_validator_record_sha256": (
            digest(CHECKPOINT_VALIDATOR_RECORD)
            if CHECKPOINT_VALIDATOR_RECORD.is_file()
            else None
        ),
        "checkpoint_validator_installed": checkpoint_validator.get("installed") is True,
        "exact_timeout_requeue_record": str(TIMEOUT_REQUEUE_RECORD),
        "exact_timeout_requeue_record_sha256": (
            digest(TIMEOUT_REQUEUE_RECORD) if TIMEOUT_REQUEUE_RECORD.is_file() else None
        ),
        "exact_timeout_requeue_applied": timeout_requeue.get("installed") is True,
        "exact_timeout_requeue_job_ids": timeout_requeue.get("job_ids"),
        "exact_timeout_requeue_recorded_after_at": timeout_requeue.get(
            "recorded_after_at"
        ),
        "checkout_wrapper_amendment_effective_for_submitted_jobs": False,
        "runtime_ops_amendment_effective_on_restart": True,
        "mechanism_gate_qwen3_checkpoint_interval": 2,
        "mechanism_gate_qwen3_checkpoint_start": 2,
        "mechanism_gate_qwen3_hardware": "a6000",
        "mechanism_gate_qwen3_time_limit": "00:45:00",
        "mechanism_gate_small_scale_node": "node[202-204,403]",
        "mechanism_gate_small_scale_time_limit": "00:45:00",
        "e112_paired_hardware_changed": False,
        "scheduler_only": not violations,
        "storage_only_recovery_changed": True,
        "treatment_environment_changed": False,
        "environment_changed": False,
        "outcomes_inspected": False,
    }, violations
