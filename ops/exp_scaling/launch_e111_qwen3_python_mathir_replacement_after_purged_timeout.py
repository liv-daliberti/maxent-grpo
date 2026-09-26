#!/usr/bin/env python3
"""Continue the two purged E111 Qwen-3B timeout cells, fail closed."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e111_verified_support_discovery_mechanism_gate_three_scale as e111  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / e111.LEDGER
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e111_qwen3_python_mathir_replacement_after_purged_timeout_20260818.md"
)
FAILED_REQUEUE = ROOT / "var/artifacts/e111_qwen3_python_mathir_timeout_requeue.json"
RECORD = ROOT / "var/artifacts/e111_qwen3_python_mathir_replacement_jobs.json"
CHECKPOINT_VALIDATOR = ROOT / "ops/validate_deepspeed_checkpoint.py"
ORIGINAL_BY_DOMAIN = {"python_factors": 30674760, "mathir": 30674761}
DOMAINS = tuple(ORIGINAL_BY_DOMAIN)
CHECKPOINT_INTERVAL = 2
PARTITION = "lowprio"
NODELIST = e111.QWEN3_A6000_NODES
GRES = "gpu:a6000:1"
TIME_LIMIT = "00:45:00"


def now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_failed_requeue(payload: dict[str, Any], ledger: dict[str, Any]) -> None:
    expected = {
        "schema": "e111_qwen3_python_mathir_timeout_requeue_v1",
        "job_ids": list(ORIGINAL_BY_DOMAIN.values()),
        "same_job_ids": True,
        "replacement_jobs_submitted": False,
        "state_reset": False,
        "environment_changed": False,
        "treatment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "installed": False,
        "ledger_sha256": e111.digest(LEDGER),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"failed same-ID record has invalid {key}")
    if Path(str(payload.get("ledger", ""))).resolve() != LEDGER.resolve():
        raise RuntimeError("failed same-ID record names another ledger")
    old_protocol = Path(str(payload.get("protocol", ""))).resolve()
    if (
        not old_protocol.is_file()
        or payload.get("protocol_sha256") != e111.digest(old_protocol)
    ):
        raise RuntimeError("failed same-ID record protocol digest mismatch")
    pre = payload.get("pre_action_accounting")
    if not isinstance(pre, dict) or set(pre) != {
        str(value) for value in ORIGINAL_BY_DOMAIN.values()
    }:
        raise RuntimeError("failed same-ID record accounting set mismatch")
    for job_id in ORIGINAL_BY_DOMAIN.values():
        if f"{job_id}|TIMEOUT|" not in str(pre[str(job_id)]):
            raise RuntimeError(f"original job {job_id} lacks TIMEOUT evidence")
    results = payload.get("requeue_results")
    first = str(next(iter(ORIGINAL_BY_DOMAIN.values())))
    if not isinstance(results, dict) or set(results) != {first}:
        raise RuntimeError("failed same-ID attempt result set mismatch")
    result = results[first]
    if (
        not isinstance(result, dict)
        or int(result.get("returncode", 0)) == 0
        or result.get("stdout") != ""
        or result.get("stderr") != "Invalid job id specified for job 30674760\n"
    ):
        raise RuntimeError("failed same-ID attempt is not the frozen no-op failure")
    runs = {int(run["job_id"]): run for run in ledger.get("runs", [])}
    if set(ORIGINAL_BY_DOMAIN.values()) - set(runs):
        raise RuntimeError("original continuation cells are absent from E111 ledger")


def original_runs(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    runs = {int(run["job_id"]): run for run in ledger.get("runs", [])}
    selected: list[dict[str, Any]] = []
    for domain, job_id in ORIGINAL_BY_DOMAIN.items():
        run = runs.get(job_id)
        if run is None:
            raise RuntimeError(f"missing original E111 job {job_id}")
        expected = {
            "scale": "qwen3b",
            "domain": domain,
            "seed": e111.SCALE_SEEDS["qwen3b"],
        }
        for key, value in expected.items():
            if run.get(key) != value:
                raise RuntimeError(f"original job {job_id} has invalid {key}")
        selected.append(run)
    return selected


def template_for(domain: str) -> dict[str, Any]:
    matches = [
        run
        for run in e111.references(ROOT, "qwen3b")
        if str(run["domain"]) == domain
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one frozen Qwen-3B template for {domain}")
    return matches[0]


def continuation_environment(
    run: dict[str, Any], snapshot: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    domain = str(run["domain"])
    template = template_for(domain)
    env, generated_target = e111.build_env(ROOT, "qwen3b", template, snapshot)
    run_dir = Path(str(run["run_dir"]))
    if generated_target.resolve() != run_dir.resolve():
        raise RuntimeError(f"{domain}: generated target differs from original run")
    if not run_dir.is_dir():
        raise RuntimeError(f"{domain}: original run directory is absent")
    env.update(
        {
            "SAVE_PATH": str(run_dir),
            "RUN_STAMP": str(run["run_stamp"]),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_FROM": str(CHECKPOINT_INTERVAL),
        }
    )
    return env, template


def continuation_command(
    run: dict[str, Any], snapshot: Path
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    env, template = continuation_environment(run, snapshot)
    command = e111.sbatch_command(ROOT, "qwen3b", template, env)
    replacements = {
        "--partition=": f"--partition={PARTITION}",
        "--nodelist=": f"--nodelist={NODELIST}",
        "--gres=": f"--gres={GRES}",
        "--time=": f"--time={TIME_LIMIT}",
    }
    output: list[str] = []
    seen: set[str] = set()
    for token in command:
        replacement = None
        for prefix, value in replacements.items():
            if token.startswith(prefix):
                replacement = value
                seen.add(prefix)
                break
        output.append(replacement if replacement is not None else token)
    if set(replacements) != seen:
        raise RuntimeError(f"cannot apply exact continuation placement: {seen}")
    if "--hold" not in output:
        raise RuntimeError("continuation submission is not held")
    return output, env, template


def checkpoint_selection(run: dict[str, Any]) -> str:
    result = subprocess.run(
        [sys.executable, str(CHECKPOINT_VALIDATOR), "--select-under", str(run["run_dir"])],
        capture_output=True,
        text=True,
        check=False,
    )
    selected = result.stdout.strip()
    if result.returncode != 0 or not selected:
        raise RuntimeError(
            f"no valid checkpoint for original job {run['job_id']}: "
            f"{result.stderr.strip()}"
        )
    path = Path(selected)
    if not path.is_dir() or Path(str(run["run_dir"])) not in path.parents:
        raise RuntimeError(f"checkpoint selection escaped original run: {selected}")
    return selected


def held_job_audit(
    job_id: int,
    *,
    run: dict[str, Any],
    env: dict[str, str],
    snapshot: Path,
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or f"JobId={job_id}" not in result.stdout:
        raise RuntimeError(f"cannot inspect held continuation job {job_id}")
    record = result.stdout.strip()
    required = [
        f"JobId={job_id}",
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Account=mltheory",
        f"Partition={PARTITION}",
        f"ReqNodeList={NODELIST}",
        "TresPerNode=gres/gpu:a6000:1",
        "TimeLimit=00:45:00",
        f"SAVE_PATH={run['run_dir']}",
        f"RUN_STAMP={run['run_stamp']}",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
        "OAT_ZERO_MAX_TRAIN=8",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=64",
        "OAT_ZERO_SAVE_STEPS=2",
        "OAT_ZERO_SAVE_FROM=2",
        "OAT_ZERO_RESUME_STEPS=2",
        "OAT_ZERO_RESUME_FROM=2",
    ]
    required.extend(f"{key}={value}" for key, value in e111.fixed_objective().items())
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"held continuation job {job_id} lacks {missing}")
    if env["SAVE_PATH"] != str(run["run_dir"]):
        raise RuntimeError("continuation environment changed the run directory")
    return record


def cancel(job_ids: list[int]) -> None:
    if job_ids:
        subprocess.run(
            ["scancel", *(str(value) for value in job_ids)],
            capture_output=True,
            text=True,
            check=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    if args.submit and RECORD.exists():
        raise SystemExit(f"refusing duplicate continuation: {RECORD}")
    for path in (LEDGER, PROTOCOL, FAILED_REQUEUE, CHECKPOINT_VALIDATOR):
        if not path.is_file():
            raise SystemExit(f"required frozen input is absent: {path}")

    ledger = load_json(LEDGER)
    failed_requeue = load_json(FAILED_REQUEUE)
    validate_failed_requeue(failed_requeue, ledger)
    snapshot = Path(str(ledger["snapshot_root"])).resolve()
    e111.verify_snapshot(snapshot)
    runs = original_runs(ledger)
    planned: list[dict[str, Any]] = []
    for run in runs:
        command, env, template = continuation_command(run, snapshot)
        selected = checkpoint_selection(run)
        planned.append(
            {
                "run": run,
                "command": command,
                "env": env,
                "template": template,
                "selected_checkpoint_before_submission": selected,
            }
        )
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(token) for token in cell["command"]))
            print(
                "# checkpoint="
                + str(cell["selected_checkpoint_before_submission"])
            )
        print(f"[e111-continuation] dry_run=True cells={len(planned)}")
        return 0

    submitted: list[int] = []
    continuations: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema": "e111_qwen3_python_mathir_replacement_jobs_v1",
        "recorded_before_at": now(),
        "protocol": str(PROTOCOL),
        "protocol_sha256": e111.digest(PROTOCOL),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": e111.digest(Path(__file__).resolve()),
        "ledger": str(LEDGER),
        "ledger_sha256": e111.digest(LEDGER),
        "failed_same_id_requeue_record": str(FAILED_REQUEUE),
        "failed_same_id_requeue_record_sha256": e111.digest(FAILED_REQUEUE),
        "snapshot_root": str(snapshot),
        "original_job_ids": list(ORIGINAL_BY_DOMAIN.values()),
        "domains": list(DOMAINS),
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "partition": PARTITION,
        "nodelist": NODELIST,
        "gres": GRES,
        "time_limit": TIME_LIMIT,
        "same_scientific_cells": True,
        "same_run_directories": True,
        "state_reset": False,
        "optimizer_update_changed": False,
        "environment_changed_except_storage_and_placement": False,
        "treatment_changed": False,
        "outcomes_inspected": False,
        "pointmaze": "excluded",
        "continuations": continuations,
        "released": False,
        "installed": False,
    }
    try:
        for cell in planned:
            result = subprocess.run(
                cell["command"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "sbatch failed")
            job_text = result.stdout.strip().split(";", 1)[0]
            if not job_text.isdigit():
                raise RuntimeError(f"invalid continuation job id: {result.stdout!r}")
            job_id = int(job_text)
            if job_id in ORIGINAL_BY_DOMAIN.values() or job_id in submitted:
                raise RuntimeError(f"invalid replacement identity: {job_id}")
            submitted.append(job_id)
            run = cell["run"]
            held = held_job_audit(
                job_id,
                run=run,
                env=cell["env"],
                snapshot=snapshot,
            )
            domain = str(run["domain"])
            continuations.append(
                {
                    "scale": "qwen3b",
                    "domain": domain,
                    "seed": e111.SCALE_SEEDS["qwen3b"],
                    "original_job_id": int(run["job_id"]),
                    "continuation_job_id": job_id,
                    "run_stamp": str(run["run_stamp"]),
                    "run_dir": str(run["run_dir"]),
                    "stdout": str(
                        ROOT
                        / "var/artifacts/logs"
                        / f"{e111.job_name('qwen3b', domain)}-{job_id}.out"
                    ),
                    "stderr": str(
                        ROOT
                        / "var/artifacts/logs"
                        / f"{e111.job_name('qwen3b', domain)}-{job_id}.err"
                    ),
                    "selected_checkpoint_before_submission": cell[
                        "selected_checkpoint_before_submission"
                    ],
                    "command": cell["command"],
                    "held_scheduler_record": held,
                }
            )
        if len(continuations) != 2:
            raise RuntimeError("continuation set is not exactly two cells")
        e111.e81.atomic_json(RECORD, payload)
        release_results: dict[str, dict[str, Any]] = {}
        for job_id in submitted:
            result = subprocess.run(
                ["scontrol", "release", str(job_id)],
                capture_output=True,
                text=True,
                check=False,
            )
            release_results[str(job_id)] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            if result.returncode != 0:
                raise RuntimeError(f"release failed for continuation {job_id}")
        payload.update(
            {
                "recorded_after_at": now(),
                "release_results": release_results,
                "released": True,
                "installed": True,
            }
        )
        e111.e81.atomic_json(RECORD, payload)
    except Exception as exc:
        cancel(submitted)
        payload.update(
            {
                "failed_at": now(),
                "submitted_job_ids": submitted,
                "error": str(exc),
                "released": False,
                "installed": False,
            }
        )
        e111.e81.atomic_json(RECORD, payload)
        raise
    print(
        f"[e111-continuation] installed=True released=2 jobs={submitted} "
        f"record={RECORD}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
