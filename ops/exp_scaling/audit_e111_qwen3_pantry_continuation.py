#!/usr/bin/env python3
"""Validate the exact E111 Qwen-3B Pantry timeout continuation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e111_qwen3_python_mathir_replacement as shared_audit  # noqa: E402
import launch_e111_qwen3_pantry_continuation_after_purged_timeout as launch  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RECORD = launch.RECORD


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
    record = _load(RECORD, "E111 Qwen-3B Pantry continuation record", violations)

    originals = {
        int(run.get("job_id", -1)): run
        for run in runs
        if isinstance(run, dict)
    }
    original = originals.get(launch.ORIGINAL_JOB_ID)
    if original is None:
        violations.append("original E111 Qwen-3B Pantry cell is absent")
    else:
        for key, value in {
            "scale": "qwen3b",
            "domain": launch.DOMAIN,
            "seed": 70,
        }.items():
            if original.get(key) != value:
                violations.append(f"original Pantry cell has invalid {key}")

    ledger_sha256 = (
        launch.shared.e111.digest(launch.LEDGER)
        if launch.LEDGER.is_file()
        else None
    )
    expected = {
        "schema": "e111_qwen3_pantry_continuation_jobs_v1",
        "ledger_sha256": ledger_sha256,
        "original_job_id": launch.ORIGINAL_JOB_ID,
        "prior_job_id": launch.ORIGINAL_JOB_ID,
        "domain": launch.DOMAIN,
        "scale": "qwen3b",
        "seed": 70,
        "run_dir": str(original.get("run_dir")) if original else None,
        "run_stamp": str(original.get("run_stamp")) if original else None,
        "checkpoint_interval": 2,
        "partition": launch.shared.PARTITION,
        "nodelist": launch.shared.NODELIST,
        "gres": launch.shared.GRES,
        "time_limit": launch.shared.TIME_LIMIT,
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
    for key, value in expected.items():
        if record.get(key) != value:
            violations.append(
                f"E111 Qwen-3B Pantry continuation has invalid {key}"
            )

    for field, path in {
        "ledger": launch.LEDGER,
        "protocol": launch.PROTOCOL,
        "launcher": Path(launch.__file__).resolve(),
    }.items():
        if Path(str(record.get(field, ""))).resolve() != path.resolve():
            violations.append(
                f"E111 Qwen-3B Pantry continuation has invalid {field}"
            )
    for field, path in {
        "protocol_sha256": launch.PROTOCOL,
        "launcher_sha256": Path(launch.__file__).resolve(),
    }.items():
        if (
            not path.is_file()
            or record.get(field) != launch.shared.e111.digest(path)
        ):
            violations.append(
                f"E111 Qwen-3B Pantry continuation {field} mismatch"
            )

    continuation_id: int | None = None
    try:
        continuation_id = int(record["continuation_job_id"])
    except (KeyError, TypeError, ValueError):
        violations.append("Pantry continuation has invalid job identity")
    if continuation_id is not None and continuation_id in originals:
        violations.append("Pantry continuation job ID is not fresh")

    snapshot = Path(str(ledger.get("snapshot_root", ""))).resolve()
    if original is not None and continuation_id is not None:
        expected_checkpoint = (
            Path(str(original["run_dir"]))
            / f"debug_job{launch.ORIGINAL_JOB_ID}"
            / "checkpoints/step_00054"
        )
        if (
            Path(
                str(record.get("selected_checkpoint_before_submission", ""))
            ).resolve()
            != expected_checkpoint.resolve()
        ):
            violations.append("Pantry continuation checkpoint drifted")
        expected_stdout = str(
            ROOT
            / "var/artifacts/logs"
            / (
                f"{launch.shared.e111.job_name('qwen3b', launch.DOMAIN)}"
                f"-{continuation_id}.out"
            )
        )
        expected_stderr = str(
            ROOT
            / "var/artifacts/logs"
            / (
                f"{launch.shared.e111.job_name('qwen3b', launch.DOMAIN)}"
                f"-{continuation_id}.err"
            )
        )
        if record.get("stdout") != expected_stdout:
            violations.append("Pantry continuation stdout drifted")
        if record.get("stderr") != expected_stderr:
            violations.append("Pantry continuation stderr drifted")
        expected_command, _env, _template = launch.shared.continuation_command(
            original, snapshot
        )
        if record.get("command") != expected_command:
            violations.append("Pantry continuation command drifted")
        held = str(record.get("held_scheduler_record", ""))
        for needle in shared_audit._record_needles(
            record, original, snapshot
        ):
            if needle not in held:
                violations.append(
                    f"Pantry continuation held record lacks {needle}"
                )

    accounting = str(record.get("prior_accounting", "")).splitlines()
    fields = accounting[0].split("|") if accounting else []
    if (
        len(fields) < 2
        or fields[0] != str(launch.ORIGINAL_JOB_ID)
        or fields[1].split("+", 1)[0] != "TIMEOUT"
    ):
        violations.append("Pantry continuation prior accounting drifted")
    release = record.get("release_result")
    if not isinstance(release, dict) or release.get("returncode") != 0:
        violations.append("Pantry continuation release failed")

    effective: dict[str, dict[str, Any]] = {}
    if original is not None and continuation_id is not None:
        effective[str(launch.ORIGINAL_JOB_ID)] = {
            "original_job_id": launch.ORIGINAL_JOB_ID,
            "continuation_job_id": continuation_id,
            "domain": launch.DOMAIN,
            "run_dir": str(original["run_dir"]),
            "stdout": str(record.get("stdout", "")),
            "stderr": str(record.get("stderr", "")),
        }
    report = {
        "record": str(RECORD),
        "record_sha256": (
            launch.shared.e111.digest(RECORD) if RECORD.is_file() else None
        ),
        "original_job_ids": [launch.ORIGINAL_JOB_ID],
        "continuation_job_ids": (
            [continuation_id] if continuation_id is not None else []
        ),
        "continuation_by_original_job_id": effective,
        "continuation_chains": {
            str(launch.ORIGINAL_JOB_ID): [
                launch.ORIGINAL_JOB_ID,
                *([continuation_id] if continuation_id is not None else []),
            ]
        },
        "same_scientific_cell": record.get("same_scientific_cell") is True,
        "same_run_directory": record.get("same_run_directory") is True,
        "checkpoint_interval": record.get("checkpoint_interval"),
        "outcomes_inspected": record.get("outcomes_inspected") is True,
        "pointmaze": record.get("pointmaze"),
        "released": record.get("released") is True,
        "installed": record.get("installed") is True,
        "passed": not violations,
    }
    return report, violations


def main() -> int:
    report, violations = validate()
    print(json.dumps(report | {"violations": violations}, indent=2, sort_keys=True))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
