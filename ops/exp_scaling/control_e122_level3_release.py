#!/usr/bin/env python3
"""Read-only E122 status, or explicitly advance its audited held cohort once.

Only this new campaign's exact ledger IDs can receive ``scontrol release``.
Each ID has a durable, exclusive intent before that command; an uncertain or
failed command consumes the intent and stops automatic advancement. There is
no submission, cancellation, requeue, deletion, or retry operation here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
JOURNAL_ROOT = ROOT / "var/artifacts/e122_level3_factorial/release_controller"
DATASET_IDENTITY_SHA256 = "890d7697af7789e0ae53c803586ec7f722685b7eb2239643175a1933fa45650d"
GIB = 1024**3
SHARED_HEADROOM_BYTES = 64 * GIB
MAX_ACTIVE = 4
PROFILE_BYTES = {"3b": (84 * GIB, 25 * GIB // 4), "05b": (16 * GIB, 5 * GIB // 4)}
CELL_FIELDS = ("domain", "dataset_domain", "arm", "seed", "run_stamp", "run_dir")
TERMINAL_STATES = {
    "COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
}
KNOWN_STATES = TERMINAL_STATES | {
    "PENDING", "RUNNING", "SUSPENDED", "COMPLETING", "CONFIGURING",
    "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "RESV_DEL_HOLD",
    "SIGNALING", "SPECIAL_EXIT", "STAGE_OUT", "STOPPED",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_json(path: Path, expected: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected or ""):
        raise ValueError("an explicit lowercase SHA256 pin is required")
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    payload = json.loads(contents)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def cell_key(cell: dict[str, Any]) -> tuple[str, str, int]:
    return str(cell["domain"]), str(cell["arm"]), int(cell["seed"])


def field(record: str, name: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]*)", record)
    if match is None:
        raise ValueError(f"held record lacks {name}")
    return match.group(1)


def load_campaign(args: argparse.Namespace, launcher: Any) -> dict[str, Any]:
    """Reauthenticate source, model, data, admission and both reviewed inputs."""
    plan_path, ledger_path = Path(args.plan).resolve(), Path(args.held_ledger).resolve()
    plan = pinned_json(plan_path, args.plan_sha256)
    ledger = pinned_json(ledger_path, args.held_ledger_sha256)
    verified = launcher.verify_plan(
        path=plan_path, expected_sha256=args.plan_sha256,
        model_choice=args.model_choice, require_admission=True,
    )
    if verified != plan or digest(plan_path) != args.plan_sha256:
        raise ValueError("plan changed during admission verification")
    if plan.get("schema") != "e122_level3_factorial_plan_v1":
        raise ValueError("unexpected E122 plan schema")
    if ledger.get("schema") != "e122_level3_factorial_jobs_v1":
        raise ValueError("unexpected E122 ledger schema")
    if plan.get("model_choice") != args.model_choice or ledger.get("model_choice") != args.model_choice:
        raise ValueError("explicit model choice does not match both frozen inputs")
    if ledger.get("model_choice_pending") is not False:
        raise ValueError("model choice has not been frozen")
    if ledger.get("status") != "held_audited" or ledger.get("released") is not False:
        raise ValueError("the complete audited held ledger is required")
    if ledger.get("plan_sha256") != args.plan_sha256 or Path(ledger["plan_path"]).resolve() != plan_path:
        raise ValueError("held ledger does not bind the pinned plan")
    if ledger.get("admission_proof") != plan.get("admission_proof"):
        raise ValueError("held ledger admission proof differs from the plan")
    if plan.get("admission_proof", {}).get("status") != "matched_fixed_reference":
        raise ValueError("canonical confirmation admission has not passed")
    if plan.get("dataset_identity_sha256") != DATASET_IDENTITY_SHA256:
        raise ValueError("E122 dataset identity mismatch")
    profile = plan.get("storage_profile", {})
    peak, terminal = PROFILE_BYTES[args.model_choice]
    if (profile.get("model_choice"), profile.get("peak_bytes"), profile.get("terminal_bytes")) != (args.model_choice, peak, terminal):
        raise ValueError("frozen storage profile differs from the conservative model policy")
    if profile.get("measurement_verified") is not True or not profile.get("evidence"):
        raise ValueError("storage profile requires verified model-specific measurement evidence")
    cells = plan.get("cells", [])
    rows = ledger.get("runs", [])
    if len(cells) != 100 or len(rows) != 100:
        raise ValueError("all 100 planned and audited held jobs are required")
    indexed = {cell_key(cell): cell for cell in cells}
    if len(indexed) != 100 or len({cell_key(row) for row in rows}) != 100:
        raise ValueError("duplicate factorial cell identity")
    ids: set[str] = set()
    ordered = []
    for row in rows:
        job_id = str(row["job_id"])
        if not re.fullmatch(r"[1-9][0-9]*", job_id) or job_id in ids:
            raise ValueError("job IDs must be distinct exact numeric allocation IDs")
        ids.add(job_id)
        cell = indexed.get(cell_key(row))
        if cell is None or any(row.get(key) != cell[key] for key in CELL_FIELDS):
            raise ValueError(f"held ledger cell differs from plan: {job_id}")
        record = str(row.get("held_scheduler_record", ""))
        if (field(record, "JobId"), field(record, "JobState"), field(record, "Reason")) != (job_id, "PENDING", "JobHeldUser"):
            raise ValueError(f"ledger lacks original user-held identity: {job_id}")
        launcher.audit_held_record(record, int(job_id), cell)
        ordered.append({"job_id": job_id, "cell": cell, "row": row})
    projected = {cell_key(cell): {key: cell[key] for key in CELL_FIELDS} for cell in cells}
    planned_rows = ledger.get("planned_runs", [])
    recorded = {cell_key(row): row for row in planned_rows}
    if len(planned_rows) != 100 or recorded != projected:
        raise ValueError("ledger planned rows differ from frozen cells")
    if digest(ledger_path) != args.held_ledger_sha256:
        raise ValueError("held ledger changed during verification")
    return {
        "plan": plan, "jobs": ordered, "profile": profile,
        "binding": {
            "plan_path": str(plan_path), "plan_sha256": args.plan_sha256,
            "held_ledger_path": str(ledger_path), "held_ledger_sha256": args.held_ledger_sha256,
            "model_choice": args.model_choice, "controller_sha256": digest(Path(__file__)),
        },
    }


def command(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=45)


def scheduler_snapshot(job_ids: list[str]) -> dict[str, Any]:
    joined = ",".join(job_ids)
    queue = command(["squeue", "--noheader", "--jobs", joined, "--format=%i|%T|%r"])
    account = command(["sacct", "--noheader", "--parsable2", "--jobs", joined, "--format=JobIDRaw,State,ExitCode"])
    for label, result in (("squeue", queue), ("sacct", account)):
        if result.returncode:
            raise RuntimeError(f"{label} failed ({result.returncode}): {result.stderr.strip()}")
    return parse_scheduler(job_ids, queue.stdout, account.stdout)


def normalize_state(value: str) -> str:
    return value.strip().split(" ", 1)[0].rstrip("+")


def parse_scheduler(job_ids: list[str], queue_text: str, account_text: str) -> dict[str, Any]:
    """Require exact allocation accounting; absence never proves completion."""
    wanted = set(job_ids)
    queue: dict[str, tuple[str, str]] = {}
    account: dict[str, tuple[str, str]] = {}
    for output, records in ((queue_text, queue), (account_text, account)):
        for line in output.splitlines():
            parts = [part.strip() for part in line.split("|")]
            if not parts or parts[0] not in wanted:
                continue  # Ignore .batch/.extern/step rows, never normalize them to a parent.
            if len(parts) < 3 or parts[0] in records:
                raise ValueError("duplicate or malformed scheduler allocation record")
            records[parts[0]] = (normalize_state(parts[1]), parts[2])
    observations = {}
    for job_id in job_ids:
        q, a = queue.get(job_id), account.get(job_id)
        unknown = a is None or a[0] not in KNOWN_STATES
        if q is not None:
            unknown |= q[0] not in KNOWN_STATES or q[0] in TERMINAL_STATES
            unknown |= a is not None and a[0] in TERMINAL_STATES
            state = q[0]
            terminal = False
        else:
            terminal = a is not None and a[0] in TERMINAL_STATES
            unknown |= not terminal
            state = a[0] if a else "UNKNOWN"
        observations[job_id] = {
            "state": state, "reason": q[1] if q else None,
            "accounting_state": a[0] if a else None,
            "exit_code": a[1] if a else None,
            "terminal": terminal and not unknown,
            "held": q == ("PENDING", "JobHeldUser") and not unknown,
            "unknown": bool(unknown),
        }
    return {"observed_at": now(), "jobs": observations, "squeue": queue_text, "sacct": account_text}


def successful_endpoint(row: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any] | None:
    if not observed["terminal"] or observed["state"] != "COMPLETED" or observed["exit_code"] != "0:0":
        return None
    try:
        run_dir = Path(row["run_dir"]).resolve()
        marker_path = run_dir / "TRAINING_COMPLETE.json"
        marker = json.loads(marker_path.read_text())
        export = Path(marker["terminal_export"]).resolve()
        step = marker.get("terminal_step")
        if (marker.get("schema") != "oat_zero_training_complete_v1"
                or not isinstance(step, int) or step < 3072
                or not export.is_relative_to(run_dir)
                or export.name != f"step_{step:05d}" or export.parent.name != "saved_models"):
            return None
        if not any(path.is_file() and path.stat().st_size > 0 for path in export.glob("*.safetensors")):
            return None
        return {"terminal_step": step, "terminal_export": str(export),
                "completion_marker": str(marker_path), "completion_marker_sha256": digest(marker_path)}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def storage_decision(*, free_bytes: int, unfinished: int, active: int, peak_bytes: int, terminal_bytes: int) -> dict[str, Any]:
    terminal_reserve = unfinished * terminal_bytes
    peak_reserve = active * peak_bytes
    remaining = free_bytes - SHARED_HEADROOM_BYTES - terminal_reserve - peak_reserve
    return {
        "free_bytes": free_bytes, "shared_headroom_bytes": SHARED_HEADROOM_BYTES,
        "unfinished_terminal_reserve_bytes": terminal_reserve,
        "active_peak_reserve_bytes": peak_reserve,
        "remaining_headroom_bytes": remaining, "additional_peak_bytes": peak_bytes,
        "can_release": active < MAX_ACTIVE and remaining >= peak_bytes,
    }


def disk_space(path: Path) -> dict[str, Any]:
    stat = os.statvfs(path)
    return {"path": str(path), "device": path.stat().st_dev, "observed_at": now(),
            "free_bytes": stat.f_bavail * stat.f_frsize, "fragment_bytes": stat.f_frsize}


def read_journals(root: Path, binding: dict[str, Any], job_ids: set[str]) -> dict[str, Any]:
    if not root.exists():
        return {}
    context_path = root / "context.json"
    if not context_path.is_file():
        raise ValueError("journal directory exists without a complete immutable context")
    if json.loads(context_path.read_text()).get("binding") != binding:
        raise ValueError("controller journal context differs from the reviewed campaign")
    entries = {}
    for path in sorted((root / "jobs").glob("*.intent.json")):
        job_id = path.name.removesuffix(".intent.json")
        intent = json.loads(path.read_text())
        if job_id not in job_ids or intent.get("job_id") != job_id or intent.get("binding") != binding:
            raise ValueError("release intent identity or campaign binding differs")
        result_path = path.with_name(f"{job_id}.result.json")
        result = json.loads(result_path.read_text()) if result_path.is_file() else None
        if result is not None and (result.get("job_id") != job_id or result.get("intent_sha256") != digest(path)):
            raise ValueError("release result does not match its immutable intent")
        entries[job_id] = {"intent": intent, "result": result}
    for path in (root / "jobs").glob("*.result.json"):
        if path.name.removesuffix(".result.json") not in entries:
            raise ValueError("orphan release result")
    return entries


def evaluate(campaign: dict[str, Any], snapshot: dict[str, Any], entries: dict[str, Any], disk: dict[str, Any]) -> dict[str, Any]:
    reserved, nonterminal, running, held, terminal, unknown, candidates, issues, review = ([] for _ in range(9))
    endpoints = {}
    for job in campaign["jobs"]:
        job_id = job["job_id"]
        observed = snapshot["jobs"][job_id]
        entry = entries.get(job_id)
        endpoint = successful_endpoint(job["row"], observed)
        if endpoint is not None:
            endpoints[job_id] = endpoint
        if observed["unknown"]:
            unknown.append(job_id)
        if observed["terminal"]:
            terminal.append(job_id)
        if observed["state"] == "RUNNING":
            running.append(job_id)
        released = entry is not None or not observed["held"]
        if released and not observed["terminal"]:
            nonterminal.append(job_id)
        # Requeue=1 is preserved. Even a failed terminal allocation may restart;
        # only authenticated successful final export retires its peak and slot.
        if released and endpoint is None:
            reserved.append(job_id)
            if observed["terminal"]:
                review.append(job_id)
        if observed["held"] and entry is None:
            held.append(job_id)
            candidates.append(job_id)
        if entry is not None:
            result = entry["result"]
            if result is None or result.get("returncode") != 0 or result.get("error"):
                issues.append(f"ambiguous_release:{job_id}")
            elif observed["held"]:
                issues.append(f"released_job_is_held:{job_id}")
        elif not observed["held"] and not observed["unknown"]:
            issues.append(f"unjournaled_job_activity:{job_id}")
    profile = campaign["profile"]
    storage = storage_decision(free_bytes=disk["free_bytes"], unfinished=len(campaign["jobs"]) - len(endpoints),
                               active=len(reserved), peak_bytes=profile["peak_bytes"], terminal_bytes=profile["terminal_bytes"])
    reason = ("unknown_scheduler_state" if unknown else "ambiguous_or_external_activity" if issues
              else "needs_operator_review" if review
              else "complete" if len(endpoints) == len(campaign["jobs"])
              else "concurrency_cap" if len(reserved) >= MAX_ACTIVE
              else "waiting_disk" if not storage["can_release"]
              else "no_staged_held_jobs" if not candidates else None)
    return {
        "schema": "e122_level3_release_status_v1", "observed_at": snapshot["observed_at"],
        "binding": campaign["binding"], "model_choice": campaign["binding"]["model_choice"],
        "staged_held": len(held), "released_nonterminal": len(nonterminal), "terminal": len(terminal),
        "reserved_unfinished_slots": len(reserved), "running": len(running),
        "successful_endpoints": len(endpoints), "endpoint_evidence": endpoints, "unknown": len(unknown),
        "unfinished_terminal_exports": len(campaign["jobs"]) - len(endpoints),
        "max_released_nonterminal": MAX_ACTIVE, "blocked_reason": reason,
        "issues": issues, "needs_operator_review_job_ids": review, "storage": storage, "disk": disk,
        "staged_held_job_ids": held, "released_nonterminal_job_ids": nonterminal,
        "reserved_unfinished_job_ids": reserved, "running_job_ids": running,
        "terminal_job_ids": terminal, "unknown_job_ids": unknown,
        "next_job_id": candidates[0] if candidates and reason is None else None,
        "scheduler": snapshot,
    }


def immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


@contextmanager
def locked(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def status(campaign: dict[str, Any], root: Path) -> dict[str, Any]:
    ids = [job["job_id"] for job in campaign["jobs"]]
    entries = read_journals(root, campaign["binding"], set(ids))
    snapshot = scheduler_snapshot(ids)
    # All run directories are on the repository's shared filesystem. Refuse a
    # mounted child run path, which would invalidate this free-space reading.
    disk = disk_space(ROOT)
    for job in campaign["jobs"]:
        path = Path(job["row"]["run_dir"]).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("E122 run directory lies outside the measured filesystem")
        while not path.exists():
            path = path.parent
        if path.stat().st_dev != disk["device"]:
            raise ValueError("E122 run directory is on a different filesystem")
    return evaluate(campaign, snapshot, entries, disk)


def publish_status(root: Path, payload: dict[str, Any]) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    immutable_json(root / "status" / f"{stamp}_{uuid4().hex}.json", payload)


def advance_once(args: argparse.Namespace, launcher: Any, root: Path = JOURNAL_ROOT) -> dict[str, Any]:
    # Validate before creating the lock or any persistent controller state.
    campaign = load_campaign(args, launcher)
    with locked(root):
        context = root / "context.json"
        if not context.exists():
            if any((root / "jobs").glob("*.json")):
                raise ValueError("existing release journals have no immutable context")
            immutable_json(context, {"schema": "e122_level3_release_context_v1", "created_at": now(), "binding": campaign["binding"]})
        before = status(campaign, root)
        job_id = before["next_job_id"]
        if job_id is None:
            publish_status(root, before)
            return before
        # Refresh admission/source pins and exact held identity before each
        # single-job release, then take a new scheduler and statvfs decision.
        campaign = load_campaign(args, launcher)
        job = next(job for job in campaign["jobs"] if job["job_id"] == job_id)
        held_record = launcher.audit_held(int(job_id), job["cell"])
        before = status(campaign, root)
        if before["next_job_id"] != job_id:
            publish_status(root, before)
            return before
        intent_path = root / "jobs" / f"{job_id}.intent.json"
        argv = ["scontrol", "release", job_id]
        immutable_json(intent_path, {
            "schema": "e122_level3_release_intent_v1", "created_at": now(),
            "binding": campaign["binding"], "job_id": job_id, "cell": job["cell"],
            "held_scheduler_record": held_record, "command": argv, "decision": before,
        })
        result: dict[str, Any] = {
            "schema": "e122_level3_release_result_v1", "job_id": job_id,
            "intent_sha256": digest(intent_path), "command": argv,
        }
        try:
            completed = command(argv)
            result.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr, error=None)
        except Exception as exc:
            # Even a timeout or exec failure may have reached Slurm. Never retry.
            result.update(returncode=None, error=f"{type(exc).__name__}: {exc}")
        result["recorded_at"] = now()
        immutable_json(root / "jobs" / f"{job_id}.result.json", result)
        after = status(campaign, root)
        after["last_release_job_id"] = job_id
        publish_status(root, after)
        return after


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--held-ledger", type=Path, required=True)
    parser.add_argument("--held-ledger-sha256", required=True)
    parser.add_argument("--model-choice", choices=sorted(PROFILE_BYTES), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="read only (the default)")
    mode.add_argument("--advance", action="store_true", help="release at most one eligible held job")
    mode.add_argument("--watch", action="store_true", help="explicitly advance periodically within the storage policy")
    parser.add_argument("--interval-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    if not 1 <= args.interval_seconds <= 60:
        parser.error("interval must be between 1 and 60 seconds")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    launcher = importlib.import_module("launch_e122_level3_factorial")
    try:
        while True:
            payload = (advance_once(args, launcher) if args.advance or args.watch
                       else status(load_campaign(args, launcher), JOURNAL_ROOT))
            print(json.dumps(payload, sort_keys=True), flush=True)
            if not args.watch or payload["blocked_reason"] == "complete":
                return 0
            if payload["blocked_reason"] in {"unknown_scheduler_state", "ambiguous_or_external_activity", "needs_operator_review"}:
                return 2
            time.sleep(args.interval_seconds)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"schema": "e122_level3_release_error_v1", "observed_at": now(),
                          "error": f"{type(exc).__name__}: {exc}", "releases_stopped": True}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
