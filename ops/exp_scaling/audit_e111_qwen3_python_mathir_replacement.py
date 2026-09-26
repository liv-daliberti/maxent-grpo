#!/usr/bin/env python3
"""Validate the exact E111 purged-timeout continuations without outcomes."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e111_qwen3_python_mathir_replacement_after_purged_timeout as launch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RECORD = launch.RECORD
SECOND_RECORD = ROOT / "var/artifacts/e111_qwen3_python_second_continuation_jobs.json"
SECOND_PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e111_qwen3_python_second_continuation_after_purged_timeout_20260818.md"
)
SECOND_LAUNCHER = ROOT / (
    "ops/exp_scaling/"
    "launch_e111_qwen3_python_second_continuation_after_purged_timeout.py"
)
SECOND_ORIGINAL_JOB_ID = 30674760
SECOND_PRIOR_JOB_ID = 30739797


def _load(path: Path, label: str, violations: list[str]) -> dict[str, Any]:
    if not path.is_file():
        violations.append(f"{label} is absent")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        violations.append(f"{label} is invalid JSON: {exc}")
        return {}
    if not isinstance(value, dict):
        violations.append(f"{label} is not an object")
        return {}
    return value


def _record_needles(
    continuation: dict[str, Any],
    original: dict[str, Any],
    snapshot: Path,
) -> list[str]:
    job_id = int(continuation["continuation_job_id"])
    return [
        f"JobId={job_id}",
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Account=mltheory",
        f"Partition={launch.PARTITION}",
        f"ReqNodeList={launch.NODELIST}",
        "TresPerNode=gres/gpu:a6000:1",
        "TimeLimit=00:45:00",
        f"SAVE_PATH={original['run_dir']}",
        f"RUN_STAMP={original['run_stamp']}",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
        "OAT_ZERO_SEED=70",
        "OAT_ZERO_MAX_TRAIN=8",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=64",
        "OAT_ZERO_SAVE_STEPS=2",
        "OAT_ZERO_SAVE_FROM=2",
        "OAT_ZERO_RESUME_STEPS=2",
        "OAT_ZERO_RESUME_FROM=2",
        *(
            f"{key}={value}"
            for key, value in launch.e111.fixed_objective().items()
        ),
    ]


def validate(
    ledger: dict[str, Any] | None = None,
    runs: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    violations: list[str] = []
    if ledger is None:
        ledger = _load(launch.LEDGER, "E111 ledger", violations)
    if runs is None:
        raw_runs = ledger.get("runs", [])
        runs = raw_runs if isinstance(raw_runs, list) else []
    record = _load(RECORD, "E111 Qwen-3B replacement record", violations)
    failed = _load(
        launch.FAILED_REQUEUE,
        "E111 failed same-ID requeue record",
        violations,
    )
    try:
        if ledger and failed:
            launch.validate_failed_requeue(failed, ledger)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        violations.append(f"failed same-ID requeue evidence is invalid: {exc}")

    ledger_sha256 = launch.e111.digest(launch.LEDGER) if launch.LEDGER.is_file() else None
    failed_sha256 = (
        launch.e111.digest(launch.FAILED_REQUEUE)
        if launch.FAILED_REQUEUE.is_file()
        else None
    )
    expected = {
        "schema": "e111_qwen3_python_mathir_replacement_jobs_v1",
        "ledger_sha256": ledger_sha256,
        "failed_same_id_requeue_record_sha256": failed_sha256,
        "snapshot_root": str(ledger.get("snapshot_root", "")),
        "original_job_ids": list(launch.ORIGINAL_BY_DOMAIN.values()),
        "domains": list(launch.DOMAINS),
        "checkpoint_interval": 2,
        "partition": "lowprio",
        "nodelist": launch.e111.QWEN3_A6000_NODES,
        "gres": "gpu:a6000:1",
        "time_limit": "00:45:00",
        "same_scientific_cells": True,
        "same_run_directories": True,
        "state_reset": False,
        "optimizer_update_changed": False,
        "environment_changed_except_storage_and_placement": False,
        "treatment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "released": True,
        "installed": True,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            violations.append(f"E111 Qwen-3B replacement has invalid {key}")
    path_fields = {
        "ledger": launch.LEDGER,
        "protocol": launch.PROTOCOL,
        "launcher": Path(launch.__file__).resolve(),
        "failed_same_id_requeue_record": launch.FAILED_REQUEUE,
    }
    for field, expected_path in path_fields.items():
        if Path(str(record.get(field, ""))).resolve() != expected_path.resolve():
            violations.append(f"E111 Qwen-3B replacement has invalid {field}")
    digest_fields = {
        "protocol_sha256": launch.PROTOCOL,
        "launcher_sha256": Path(launch.__file__).resolve(),
    }
    for field, path in digest_fields.items():
        if not path.is_file() or record.get(field) != launch.e111.digest(path):
            violations.append(f"E111 Qwen-3B replacement {field} mismatch")

    snapshot = Path(str(ledger.get("snapshot_root", ""))).resolve()
    originals = {int(run.get("job_id", -1)): run for run in runs}
    continuation_rows = record.get("continuations")
    if not isinstance(continuation_rows, list) or len(continuation_rows) != 2:
        violations.append("E111 Qwen-3B replacement is not exactly two continuations")
        continuation_rows = []
    expected_originals = set(launch.ORIGINAL_BY_DOMAIN.values())
    observed_originals: set[int] = set()
    continuation_ids: set[int] = set()
    effective: dict[str, dict[str, Any]] = {}
    for continuation in continuation_rows:
        if not isinstance(continuation, dict):
            violations.append("E111 Qwen-3B continuation row is not an object")
            continue
        try:
            original_id = int(continuation["original_job_id"])
            continuation_id = int(continuation["continuation_job_id"])
        except (KeyError, TypeError, ValueError):
            violations.append("E111 Qwen-3B continuation has invalid job identity")
            continue
        if original_id in observed_originals:
            violations.append(f"duplicate original continuation identity {original_id}")
        if continuation_id in continuation_ids:
            violations.append(f"duplicate continuation identity {continuation_id}")
        observed_originals.add(original_id)
        continuation_ids.add(continuation_id)
        original = originals.get(original_id)
        if original is None:
            violations.append(f"continuation maps unknown original job {original_id}")
            continue
        domain = str(original.get("domain"))
        if launch.ORIGINAL_BY_DOMAIN.get(domain) != original_id:
            violations.append(f"continuation {original_id} has invalid domain mapping")
        expected_row = {
            "scale": "qwen3b",
            "domain": domain,
            "seed": 70,
            "run_stamp": str(original.get("run_stamp")),
            "run_dir": str(original.get("run_dir")),
            "stdout": str(
                ROOT
                / "var/artifacts/logs"
                / f"{launch.e111.job_name('qwen3b', domain)}-{continuation_id}.out"
            ),
            "stderr": str(
                ROOT
                / "var/artifacts/logs"
                / f"{launch.e111.job_name('qwen3b', domain)}-{continuation_id}.err"
            ),
        }
        for key, value in expected_row.items():
            if continuation.get(key) != value:
                violations.append(
                    f"continuation {original_id}->{continuation_id} has invalid {key}"
                )
        selected = Path(str(continuation.get("selected_checkpoint_before_submission", "")))
        run_dir = Path(str(original.get("run_dir", "")))
        if run_dir not in selected.parents or not selected.name.startswith("step_"):
            violations.append(
                f"continuation {original_id}->{continuation_id} has invalid checkpoint"
            )
        command = continuation.get("command")
        command_needles = {
            "--hold",
            "--partition=lowprio",
            f"--nodelist={launch.NODELIST}",
            "--gres=gpu:a6000:1",
            "--time=00:45:00",
        }
        if not isinstance(command, list) or not command_needles.issubset(set(command)):
            violations.append(
                f"continuation {original_id}->{continuation_id} command drifted"
            )
        held = str(continuation.get("held_scheduler_record", ""))
        for needle in _record_needles(continuation, original, snapshot):
            if needle not in held:
                violations.append(
                    f"continuation {original_id}->{continuation_id} held record lacks {needle}"
                )
        effective[str(original_id)] = {
            "original_job_id": original_id,
            "continuation_job_id": continuation_id,
            "domain": domain,
            "run_dir": str(original.get("run_dir")),
            "stdout": str(continuation.get("stdout", "")),
            "stderr": str(continuation.get("stderr", "")),
        }
    if observed_originals != expected_originals:
        violations.append("E111 Qwen-3B replacement original-job set mismatch")
    if continuation_ids & expected_originals or len(continuation_ids) != 2:
        violations.append("E111 Qwen-3B replacement continuation IDs are not fresh")

    releases = record.get("release_results")
    expected_release_keys = {str(value) for value in continuation_ids}
    if not isinstance(releases, dict) or set(releases) != expected_release_keys:
        violations.append("E111 Qwen-3B replacement release-result set mismatch")
    elif any(
        not isinstance(value, dict) or value.get("returncode") != 0
        for value in releases.values()
    ):
        violations.append("E111 Qwen-3B replacement release failed")

    # The first Python continuation itself timed out after a complete step-54
    # checkpoint and was purged from Slurm. Validate its exact continuation as
    # a second immutable link while retaining the original science-cell ID.
    second_violations: list[str] = []
    second: dict[str, Any] = {}
    second_id: int | None = None
    python_original = originals.get(SECOND_ORIGINAL_JOB_ID)
    first_python = effective.get(str(SECOND_ORIGINAL_JOB_ID), {})
    if SECOND_RECORD.is_file():
        second = _load(
            SECOND_RECORD,
            "E111 Qwen-3B Python second-continuation record",
            second_violations,
        )
        try:
            first_python_id = int(first_python.get("continuation_job_id", -1))
        except (TypeError, ValueError):
            first_python_id = -1
        if first_python_id != SECOND_PRIOR_JOB_ID:
            second_violations.append("first Python continuation identity drifted")
        if python_original is None:
            second_violations.append("original Python cell is absent")
        first_record_sha256 = (
            launch.e111.digest(RECORD) if RECORD.is_file() else None
        )
        expected_second = {
            "schema": "e111_qwen3_python_second_continuation_jobs_v1",
            "ledger_sha256": ledger_sha256,
            "first_continuation_record_sha256": first_record_sha256,
            "original_job_id": SECOND_ORIGINAL_JOB_ID,
            "prior_continuation_job_id": SECOND_PRIOR_JOB_ID,
            "domain": "python_factors",
            "scale": "qwen3b",
            "seed": 70,
            "run_dir": (
                str(python_original.get("run_dir")) if python_original else None
            ),
            "run_stamp": (
                str(python_original.get("run_stamp")) if python_original else None
            ),
            "checkpoint_interval": 2,
            "partition": launch.PARTITION,
            "nodelist": launch.NODELIST,
            "gres": launch.GRES,
            "time_limit": launch.TIME_LIMIT,
            "same_scientific_cell": True,
            "same_run_directory": True,
            "state_reset": False,
            "optimizer_update_changed": False,
            "environment_changed_except_storage_and_placement": False,
            "treatment_changed": False,
            "outcomes_inspected": False,
            "pointmaze": "excluded",
            "released": True,
            "installed": True,
        }
        for key, value in expected_second.items():
            if second.get(key) != value:
                second_violations.append(
                    f"E111 Qwen-3B Python second continuation has invalid {key}"
                )
        second_path_fields = {
            "ledger": launch.LEDGER,
            "first_continuation_record": RECORD,
            "protocol": SECOND_PROTOCOL,
            "launcher": SECOND_LAUNCHER,
        }
        for field, expected_path in second_path_fields.items():
            if (
                Path(str(second.get(field, ""))).resolve()
                != expected_path.resolve()
            ):
                second_violations.append(
                    "E111 Qwen-3B Python second continuation has invalid "
                    f"{field}"
                )
        for field, path in {
            "protocol_sha256": SECOND_PROTOCOL,
            "launcher_sha256": SECOND_LAUNCHER,
        }.items():
            if (
                not path.is_file()
                or second.get(field) != launch.e111.digest(path)
            ):
                second_violations.append(
                    "E111 Qwen-3B Python second continuation "
                    f"{field} mismatch"
                )
        try:
            second_id = int(second["continuation_job_id"])
        except (KeyError, TypeError, ValueError):
            second_violations.append(
                "E111 Qwen-3B Python second continuation has invalid job identity"
            )
        if second_id is not None:
            if second_id in expected_originals or second_id in continuation_ids:
                second_violations.append(
                    "E111 Qwen-3B Python second continuation job ID is not fresh"
                )
            if python_original is not None:
                expected_checkpoint = (
                    Path(str(python_original["run_dir"]))
                    / f"debug_job{SECOND_PRIOR_JOB_ID}"
                    / "checkpoints/step_00054"
                )
                if (
                    Path(
                        str(
                            second.get(
                                "selected_checkpoint_before_submission", ""
                            )
                        )
                    ).resolve()
                    != expected_checkpoint.resolve()
                ):
                    second_violations.append(
                        "E111 Qwen-3B Python second continuation checkpoint drifted"
                    )
                expected_stdout = str(
                    ROOT
                    / "var/artifacts/logs"
                    / (
                        f"{launch.e111.job_name('qwen3b', 'python_factors')}"
                        f"-{second_id}.out"
                    )
                )
                expected_stderr = str(
                    ROOT
                    / "var/artifacts/logs"
                    / (
                        f"{launch.e111.job_name('qwen3b', 'python_factors')}"
                        f"-{second_id}.err"
                    )
                )
                if second.get("stdout") != expected_stdout:
                    second_violations.append(
                        "E111 Qwen-3B Python second continuation stdout drifted"
                    )
                if second.get("stderr") != expected_stderr:
                    second_violations.append(
                        "E111 Qwen-3B Python second continuation stderr drifted"
                    )
                expected_command, _env, _template = launch.continuation_command(
                    python_original, snapshot
                )
                if second.get("command") != expected_command:
                    second_violations.append(
                        "E111 Qwen-3B Python second continuation command drifted"
                    )
                held = str(second.get("held_scheduler_record", ""))
                for needle in _record_needles(
                    second, python_original, snapshot
                ):
                    if needle not in held:
                        second_violations.append(
                            "E111 Qwen-3B Python second continuation held "
                            f"record lacks {needle}"
                        )
        accounting = str(second.get("prior_accounting", "")).splitlines()
        accounting_fields = accounting[0].split("|") if accounting else []
        if (
            len(accounting_fields) < 2
            or accounting_fields[0] != str(SECOND_PRIOR_JOB_ID)
            or accounting_fields[1].split("+", 1)[0] != "TIMEOUT"
        ):
            second_violations.append(
                "E111 Qwen-3B Python second continuation prior accounting drifted"
            )
        release = second.get("release_result")
        if not isinstance(release, dict) or release.get("returncode") != 0:
            second_violations.append(
                "E111 Qwen-3B Python second continuation release failed"
            )
        if (
            not second_violations
            and second_id is not None
            and python_original is not None
        ):
            effective[str(SECOND_ORIGINAL_JOB_ID)] = {
                "original_job_id": SECOND_ORIGINAL_JOB_ID,
                "prior_continuation_job_id": SECOND_PRIOR_JOB_ID,
                "continuation_job_id": second_id,
                "domain": "python_factors",
                "run_dir": str(python_original["run_dir"]),
                "stdout": str(second["stdout"]),
                "stderr": str(second["stderr"]),
            }
    violations.extend(second_violations)

    all_continuation_ids = set(continuation_ids)
    if second_id is not None:
        all_continuation_ids.add(second_id)
    continuation_chains = {
        str(SECOND_ORIGINAL_JOB_ID): [
            SECOND_ORIGINAL_JOB_ID,
            SECOND_PRIOR_JOB_ID,
            *([second_id] if second_id is not None else []),
        ],
        "30674761": [
            30674761,
            int(
                effective.get("30674761", {}).get(
                    "continuation_job_id", -1
                )
            ),
        ],
    }

    report = {
        "record": str(RECORD),
        "record_sha256": launch.e111.digest(RECORD) if RECORD.is_file() else None,
        "second_continuation_record": str(SECOND_RECORD),
        "second_continuation_record_sha256": (
            launch.e111.digest(SECOND_RECORD)
            if SECOND_RECORD.is_file()
            else None
        ),
        "failed_same_id_requeue_valid": not any(
            value.startswith("failed same-ID") for value in violations
        ),
        "original_job_ids": sorted(expected_originals),
        "continuation_job_ids": sorted(all_continuation_ids),
        "continuation_by_original_job_id": effective,
        "continuation_chains": continuation_chains,
        "second_continuation_job_id": second_id,
        "same_scientific_cells": (
            record.get("same_scientific_cells") is True
            and second.get("same_scientific_cell") is True
        ),
        "same_run_directories": (
            record.get("same_run_directories") is True
            and second.get("same_run_directory") is True
        ),
        "checkpoint_interval": record.get("checkpoint_interval"),
        "outcomes_inspected": (
            record.get("outcomes_inspected") is True
            or second.get("outcomes_inspected") is True
        ),
        "pointmaze": (
            "excluded"
            if record.get("pointmaze") == second.get("pointmaze") == "excluded"
            else "invalid"
        ),
        "released": (
            record.get("released") is True
            and second.get("released") is True
        ),
        "installed": (
            record.get("installed") is True
            and second.get("installed") is True
        ),
        "passed": not violations,
    }
    return report, violations


def main() -> int:
    report, violations = validate()
    payload = report | {"violations": violations}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
