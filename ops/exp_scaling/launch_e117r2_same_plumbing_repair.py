#!/usr/bin/env python3
"""Freeze, smoke, and transactionally submit the repaired E117-R2 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_e117s1_lowprio_partition_repair as scheduler  # noqa: E402
import launch_e117_same_plumbing_component_preflight as e117  # noqa: E402


PROTOCOL = "paper/preregistration/e117r2_same_plumbing_repair_20260829.md"
LEDGER = "var/artifacts/e117r2_same_plumbing_component_preflight_jobs.json"
AUDIT = "var/artifacts/e117r2_same_plumbing_component_preflight_audit.json"
AUDIT_JOB = "var/artifacts/e117r2_same_plumbing_component_preflight_audit_job.json"
SMOKE = "var/artifacts/e117r2_same_plumbing_runtime_smoke.json"
PARENT_LEDGER = "var/artifacts/e117r1_same_plumbing_component_preflight_jobs.json"
PARENT_AUDIT = "var/artifacts/e117r1_same_plumbing_component_preflight_audit.json"
NODES = {
    ("qwen05b", "countdown"): "node202",
    ("qwen05b", "graph_coloring"): "node203",
    ("qwen05b", "python_factors"): "node203",
    ("falcon1b", "mathir"): "node203",
}


def configure_campaign() -> None:
    e117.CAMPAIGN_TAG = "e117r2"
    e117.JOB_NAME_PREFIX = "e117r2"
    e117.LEDGER = LEDGER
    e117.PROTOCOL = PROTOCOL
    e117.SENTINEL_NODES = dict(NODES)


def run(
    command: list[str], *, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    return result


def environment_text(record: str) -> str:
    marker = "--export="
    start = record.find(marker)
    end = record.find(" --partition=", start)
    if start < 0 or end < 0:
        raise RuntimeError("held scheduler record lacks a bounded export surface")
    return record[start + len(marker) : end]


def schedule_audit(
    *, root: Path, snapshot: Path, ledger: Path, job_ids: list[int]
) -> tuple[int, str]:
    dependency = "afterany:" + ":".join(str(job_id) for job_id in job_ids)
    audit_script = (
        snapshot / "ops/exp_scaling/audit_e117_same_plumbing_component_preflight.py"
    )
    output = root / AUDIT
    command = [
        "sbatch",
        "--parsable",
        f"--dependency={dependency}",
        "--job-name=e117r2-audit",
        "--partition=all",
        "--account=allcs",
        "--cpus-per-task=2",
        "--mem=4G",
        "--time=01:00:00",
        "--nice=0",
        "--no-requeue",
        f"--chdir={root}",
        "--output=" + str(root / "var/artifacts/logs/e117r2-audit-%j.out"),
        "--error=" + str(root / "var/artifacts/logs/e117r2-audit-%j.err"),
        "--wrap=" + f"python {audit_script} --ledger {ledger} --output {output}",
    ]
    result = run(command)
    raw = result.stdout.strip().split(";", 1)[0]
    if not raw.isdigit():
        raise RuntimeError(f"invalid E117-R2 audit job id: {result.stdout!r}")
    job_id = int(raw)
    record = scheduler.show(job_id)
    if (
        scheduler.field(record, "JobState") != "PENDING"
        or scheduler.field(record, "Reason") != "Dependency"
        or scheduler.field(record, "Requeue") != "0"
        or scheduler.field(record, "Account") != "allcs"
        or scheduler.field(record, "Partition") != "all"
        or str(audit_script) not in record
        or any(
            str(value) not in scheduler.field(record, "Dependency") for value in job_ids
        )
    ):
        raise RuntimeError("E117-R2 audit dependency failed held inspection")
    return job_id, record


def run_snapshot_smoke(root: Path, snapshot: Path, output: Path) -> dict[str, Any]:
    smoke_script = snapshot / "ops/exp_scaling/smoke_e117_same_plumbing_runtime.py"
    if not smoke_script.is_file():
        raise RuntimeError(f"snapshot smoke is absent: {smoke_script}")
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
    else:
        training_python = root / "var/seed_paper_eval/paper310/bin/python"
        training_library = root / "var/seed_paper_eval/paper310/lib"
        if not training_python.is_file() or not training_library.is_dir():
            raise RuntimeError("frozen E117 training Python is unavailable")
        environment = os.environ.copy()
        existing_library_path = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = str(training_library) + (
            f":{existing_library_path}" if existing_library_path else ""
        )
        run(
            [
                str(training_python),
                str(smoke_script),
                "--snapshot-root",
                str(snapshot),
                "--output",
                str(output),
            ],
            environment=environment,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
    if (
        payload.get("passed") is not True
        or payload.get("efficacy_outcomes_used") is not False
        or payload.get("endpoint_contrasts_computed") is not False
        or payload.get("snapshot_root") != str(snapshot)
        or payload.get("snapshot_identity_sha256")
        != e117.digest(snapshot / "SNAPSHOT_IDENTITY.json")
    ):
        raise RuntimeError("E117-R2 snapshot smoke receipt is invalid")
    return payload


def base_payload(
    *,
    root: Path,
    snapshot: Path,
    protocol: Path,
    smoke: Path,
    parent_ledger: Path,
    parent_audit: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "e117_same_plumbing_component_preflight_jobs_v1",
        "protocol": str(protocol),
        "protocol_sha256": e117.digest(protocol),
        "launcher": str(Path(__file__).resolve()),
        "launcher_sha256": e117.digest(Path(__file__).resolve()),
        "snapshot_root": str(snapshot),
        "snapshot_identity_sha256": e117.digest(snapshot / "SNAPSHOT_IDENTITY.json"),
        "runtime_smoke": str(smoke),
        "runtime_smoke_sha256": e117.digest(smoke),
        "original_ledger": str(parent_ledger),
        "original_ledger_sha256": e117.digest(parent_ledger),
        "repair_parent_audit": str(parent_audit),
        "repair_parent_audit_sha256": e117.digest(parent_audit),
        "variant": e117.VARIANT,
        "seed": e117.SEED,
        "sentinels": [
            {"scale": scale, "domain": domain, "node": NODES[(scale, domain)]}
            for scale, domain in e117.SENTINELS
        ],
        "arms": list(e117.ARMS),
        "arm_labels": e117.ARM_LABELS,
        "train_rows": e117.TRAIN_ROWS,
        "passes": e117.PASSES,
        "target_steps": e117.TARGET_STEPS,
        "checkpoint_interval_steps": e117.CHECKPOINT_INTERVAL,
        "evaluation_draws": e117.EVAL_DRAWS,
        "proposal_fixed_control_groups": e117.PROPOSAL_GROUPS,
        "proposal_max_attempts": e117.e111.PROPOSAL_MAX_ATTEMPTS,
        "proposal_temperature": e117.e111.PROPOSAL_TEMPERATURE,
        "replay_weight": e117.e111.REPLAY_WEIGHT,
        "semantic_coefficient_f": e117.SEMANTIC_COEFFICIENT,
        "runtime_repairs": [
            "explicit_zero_coefficient_tracker_initialization",
            "complete_shared_semantic_telemetry_namespace",
            "positive_bitwise_zero_semantic_policy_advantage",
            "validated_terminal_bookkeeping_row_classification",
        ],
        "efficacy_gate": False,
        "outcomes_inspected_for_release": False,
        "pointmaze": "excluded",
        "runs": records,
        "released": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit == args.dry_run:
        raise SystemExit("choose exactly one of --submit or --dry-run")

    configure_campaign()
    root = e117.repo_root()
    protocol = root / PROTOCOL
    ledger = root / LEDGER
    audit = root / AUDIT
    audit_job = root / AUDIT_JOB
    smoke = root / SMOKE
    parent_ledger = root / PARENT_LEDGER
    parent_audit = root / PARENT_AUDIT
    for required in (protocol, parent_ledger, parent_audit):
        if not required.is_file():
            raise SystemExit(f"required E117-R2 provenance is absent: {required}")
    parent_audit_payload = json.loads(parent_audit.read_text(encoding="utf-8"))
    if (
        parent_audit_payload.get("efficacy_outcomes_used") is not False
        or parent_audit_payload.get("endpoint_contrasts_computed") is not False
    ):
        raise SystemExit("E117-R1 efficacy boundary is not intact")
    if args.submit:
        existing = [path for path in (ledger, audit, audit_job) if path.exists()]
        if existing:
            raise SystemExit(f"refusing duplicate E117-R2 installation: {existing}")

    e117.assert_factorization()
    snapshot = e117.snapshot_util.ensure_snapshot(root, args.snapshot_root)
    e117.verify_snapshot(snapshot)
    smoke_output = smoke if args.submit else Path("/tmp/e117r2_dry_run_smoke.json")
    run_snapshot_smoke(root, snapshot, smoke_output)
    cells = e117.planned_cells(root, snapshot)
    existing_runs = [
        cell["run_dir"] for cell in cells if Path(cell["run_dir"]).exists()
    ]
    if existing_runs:
        raise SystemExit(f"refusing to overwrite E117-R2 runs: {existing_runs}")
    if args.dry_run:
        for cell in cells:
            print(shlex.join(cell["command"]))
        print(f"[e117r2] dry_run=True smoke=pass snapshot={snapshot} cells=12")
        return 0

    submitted: list[int] = []
    records: list[dict[str, Any]] = []
    audit_job_id = 0
    released = False
    try:
        for cell in cells:
            result = run(list(cell["command"]))
            raw = result.stdout.strip().split(";", 1)[0]
            if not raw.isdigit():
                raise RuntimeError(f"invalid E117-R2 job id: {result.stdout!r}")
            job_id = int(raw)
            submitted.append(job_id)
            held = e117.held_job_audit(
                str(job_id),
                scale=str(cell["scale"]),
                domain=str(cell["domain"]),
                arm=str(cell["arm"]),
                snapshot=snapshot,
            )
            export = environment_text(held)
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "scale",
                        "model_tag",
                        "domain",
                        "seed",
                        "arm",
                        "arm_label",
                        "node",
                        "run_stamp",
                        "run_dir",
                    )
                }
                | {
                    "job_id": job_id,
                    "held_scheduler_record": held,
                    "scientific_environment_sha256": hashlib.sha256(
                        export.encode()
                    ).hexdigest(),
                }
            )
        if len(records) != 12:
            raise RuntimeError("E117-R2 did not hold exactly 12 cells")
        payload = base_payload(
            root=root,
            snapshot=snapshot,
            protocol=protocol,
            smoke=smoke,
            parent_ledger=parent_ledger,
            parent_audit=parent_audit,
            records=records,
        )
        e117.e111.e81.atomic_json(ledger, payload)
        audit_job_id, audit_record = schedule_audit(
            root=root,
            snapshot=snapshot,
            ledger=ledger,
            job_ids=submitted,
        )
        payload["audit_job_id"] = audit_job_id
        payload["audit_script"] = str(
            snapshot / "ops/exp_scaling/audit_e117_same_plumbing_component_preflight.py"
        )
        payload["audit_script_sha256"] = e117.digest(Path(payload["audit_script"]))
        audit_payload = {
            "schema": "e117r2_same_plumbing_component_preflight_audit_job_v1",
            "audit_job_id": audit_job_id,
            "dependency_job_ids": submitted,
            "ledger": str(ledger),
            "audit_output": str(audit),
            "snapshot_root": str(snapshot),
            "audit_script_sha256": payload["audit_script_sha256"],
            "scheduler_record": audit_record,
            "efficacy_outcomes_used": False,
            "released": False,
        }
        e117.e111.e81.atomic_json(ledger, payload)
        e117.e111.e81.atomic_json(audit_job, audit_payload)
        for job_id in submitted:
            run(["scontrol", "release", str(job_id)])
        released = True
        payload["released"] = True
        audit_payload["released"] = True
        e117.e111.e81.atomic_json(ledger, payload)
        e117.e111.e81.atomic_json(audit_job, audit_payload)
    except Exception:
        if not released:
            e117.e111.e81.cancel(
                [
                    str(value)
                    for value in submitted + ([audit_job_id] if audit_job_id else [])
                ]
            )
            for path in (ledger, audit_job):
                if path.exists():
                    path.unlink()
        raise

    jobs_csv = ",".join(str(value) for value in submitted)
    print(
        f"[e117r2] released=12 audit={audit_job_id} "
        f"jobs={jobs_csv} snapshot={snapshot}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
