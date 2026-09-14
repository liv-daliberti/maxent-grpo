#!/usr/bin/env python3
"""Prioritize existing E118 cells on usable capacity; default is read-only.

--apply starts one audited transaction. --resume continues that exact transaction
and refuses uncertain submissions, changed identities, or unexpected writers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e118q3_maxrl_verified_replay_extension_jobs.json"
AUDIT = ROOT / "var/artifacts/e118_capacity_priority_20260905.json"
PROTOCOL = ROOT / "paper/preregistration/e118_capacity_priority_20260905.md"
PVL = "node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]"
CS_NODES = "node202,node205,node206,node207"
NODE302_IDS = (31048128, 31048130, 31048132, 31048113,
               31048141, 31048142, 31048111, 31048112)
ROOT_EXPORTS = {
    "ROOT_DIR": str(ROOT), "OAT_ZERO_REPO_ROOT": str(ROOT),
    "MAXENT_GRPO_ROOT": str(ROOT), "MAXENT_GRPO_VAR_ROOT": str(ROOT / "var"),
}
IDENTITY = ("domain", "arm", "seed", "run_dir", "run_stamp")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encoded(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def command(parts: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(parts, capture_output=True, text=True, check=check)


def field(record: str, name: str) -> str:
    match = re.search(r"(?:^|\s)" + re.escape(name) + r"=([^\s]*)", record)
    if match is None:
        raise RuntimeError(f"scheduler record lacks {name}")
    return match.group(1)


def show(job_id: int) -> str:
    return command(["scontrol", "show", "job", "-dd", "-o", str(job_id)]).stdout


def queue() -> dict[int, str]:
    text = command(["squeue", "-h", "-u", str(command(["id", "-un"]).stdout.strip()),
                    "-o", "%i|%T"]).stdout
    return {int(line.split("|")[0]): line.split("|")[1]
            for line in text.splitlines() if line.split("|")[0].isdigit()}


def submit_tokens(record: str) -> list[str]:
    match = re.search(r"(?:^|\s)SubmitLine=(.*?)\s+WorkDir=", record)
    if match is None:
        raise RuntimeError("cannot extract the full original SubmitLine")
    tokens = shlex.split(match.group(1))
    if not tokens or tokens[0] != "sbatch":
        raise RuntimeError("original SubmitLine is not an sbatch invocation")
    return tokens


def exports(tokens: list[str]) -> dict[str, str]:
    matches = [item.split("=", 1)[1] for item in tokens if item.startswith("--export=")]
    if len(matches) != 1 or not matches[0].startswith("ALL,"):
        raise RuntimeError("expected one --export=ALL,... argument")
    result: dict[str, str] = {}
    # A comma inside a value is retained unless followed by another KEY=.
    for item in re.split(r",(?=[A-Za-z_][A-Za-z0-9_]*=)", matches[0][4:]):
        key, value = item.split("=", 1)
        if key in result:
            raise RuntimeError(f"duplicate exported variable: {key}")
        result[key] = value
    return result


def nonroot_exports(tokens: list[str]) -> dict[str, str]:
    return {key: value for key, value in exports(tokens).items()
            if key not in ROOT_EXPORTS}


def placed(original: list[str], lane: str, old_id: int) -> list[str]:
    account, partition, nodes = (
        ("mltheory", "mltheory", "node302") if lane == "node302"
        else ("allcs", "lowprio", "node208")
    )
    updates = {"--account": account, "--partition": partition,
               "--nodelist": nodes, "--exclude": PVL,
               "--comment": f"e118-capacity-20260905-old{old_id}"}
    env = exports(original)
    env.update(ROOT_EXPORTS)
    updates["--export"] = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    result = []
    for token in original[:-1]:
        key = token.split("=", 1)[0]
        if key in updates or token == "--hold":
            if key in updates and "=" not in token:
                raise RuntimeError(f"unsupported split sbatch option: {key}")
            continue
        result.append(token)
    result.extend(f"{key}={value}" for key, value in updates.items())
    result.extend(["--hold", original[-1]])
    if nonroot_exports(result) != nonroot_exports(original):
        raise RuntimeError("replacement changes scientific or runtime exports")
    return result


def audit_record(record: str, item: dict[str, Any], *, held: bool) -> None:
    lane = item["lane"]
    expected = {
        "Account": "mltheory" if lane == "node302" else "allcs",
        "Partition": "mltheory" if lane == "node302" else "lowprio",
        "ReqNodeList": "node302" if lane == "node302" else "node208",
        "ExcNodeList": PVL, "MinMemoryNode": "128G",
        "NumCPUs": "16", "TimeLimit": "12:00:00",
    }
    if held:
        expected.update(JobState="PENDING", Reason="JobHeldUser")
    for key, value in expected.items():
        if field(record, key) != value:
            raise RuntimeError(f"job placement drift: {key}: {field(record, key)} != {value}")
    if "gres/gpu=1" not in field(record, "ReqTRES").split(","):
        raise RuntimeError("replacement must request exactly one GPU")
    observed = submit_tokens(record)
    if nonroot_exports(observed) != nonroot_exports(item["original_command"]):
        raise RuntimeError("replacement changed original scientific/runtime exports")
    env = exports(observed)
    if any(env.get(key) != value for key, value in ROOT_EXPORTS.items()):
        raise RuntimeError("replacement lost an explicit repository-root export")
    if env.get("SAVE_PATH") != item["run_dir"] or env.get("RUN_STAMP") != item["run_stamp"]:
        raise RuntimeError("replacement changed the run directory or stamp")
    if env.get("OAT_ZERO_AUTO_RESUME") != "1":
        raise RuntimeError("replacement disabled automatic checkpoint resume")


def plan() -> dict[str, Any]:
    source_bytes = LEDGER.read_bytes()
    source = json.loads(source_bytes)
    if not source.get("released") or source.get("target_steps") != 3072:
        raise RuntimeError("source ledger contract drift")
    states = queue()
    pending = [row for row in source["runs"] if states.get(int(row["job_id"])) == "PENDING"]
    if len(pending) != 42:
        raise RuntimeError(f"expected 42 pending E118 Qwen-3B cells, found {len(pending)}")
    by_id = {int(row["job_id"]): row for row in pending}
    if not set(NODE302_IDS).issubset(by_id):
        raise RuntimeError("one of the eight priority continuations is no longer pending")
    pantry = {int(row["job_id"]) for row in pending
              if row["domain"] == "pantry_plan" and int(row["seed"]) in (72, 73, 74)}
    if len(pantry) != 6 or not {31047580, 31047581, 31047582}.issubset(pantry):
        raise RuntimeError("Pantry s72-s74 identity drift")
    replacements, updates = [], []
    priority_order = {job: index for index, job in enumerate(NODE302_IDS)}
    ordered = sorted(pending, key=lambda row: (
        0 if int(row["job_id"]) in priority_order else 1,
        priority_order.get(int(row["job_id"]), int(row["job_id"]))))
    for row in ordered:
        job_id = int(row["job_id"])
        record = show(job_id)
        for key, value in {"JobState": "PENDING", "Account": "allcs",
                           "Partition": "cs", "ExcNodeList": PVL,
                           "MinMemoryNode": "128G", "NumCPUs": "16",
                           "TimeLimit": "12:00:00"}.items():
            if field(record, key) != value:
                raise RuntimeError(f"old job {job_id} drift in {key}")
        item = {key: row[key] for key in IDENTITY}
        item.update(old_job_id=job_id, before_record=record)
        if job_id in NODE302_IDS or job_id in pantry:
            item["lane"] = "node302" if job_id in NODE302_IDS else "node208"
            item["original_command"] = submit_tokens(record)
            item["command"] = placed(item["original_command"], item["lane"], job_id)
            item["new_job_id"] = None
            if job_id in NODE302_IDS:
                validator = (Path(source["snapshot_root"]) / "ops/validate_deepspeed_checkpoint.py")
                selected = command(["python3", str(validator), "--select-under", row["run_dir"]])
                if not selected.stdout.strip():
                    raise RuntimeError(f"priority continuation {job_id} lacks a valid checkpoint")
                item["resume_checkpoint"] = selected.stdout.strip()
            replacements.append(item)
        else:
            updates.append(item)
    return {
        "schema": "e118-capacity-priority-20260905-v1", "created_at": now(),
        "protocol": str(PROTOCOL), "source_ledger": str(LEDGER),
        "original_ledger_sha256": sha(source_bytes), "status": "planned",
        "scheduler_only": True, "same_scientific_cells": True,
        "same_run_directories": True, "treatment_changed": False,
        "pvl_exclusion": PVL, "pvl_compute_allowed": False,
        "outcomes_inspected": False, "replacements": replacements,
        "cs_placement_updates": updates, "events": [],
    }


def save(audit: dict[str, Any], event: str) -> None:
    audit["updated_at"] = now()
    audit["events"].append({"at": audit["updated_at"], "event": event})
    atomic(AUDIT, audit)


def safely_held_old(record: str, item: dict[str, Any]) -> bool:
    """Accept only our user holds or the three originally invalid pinned jobs."""
    if field(record, "JobState") != "PENDING" or field(record, "Priority") != "0":
        return False
    reason = field(record, "Reason")
    if reason == "JobHeldUser":
        return True
    return (field(item["before_record"], "Reason") == "BadConstraints"
            and reason in {"JobHeldAdmin", "BadConstraints"})


def apply(audit: dict[str, Any], *, prepare_source=None) -> None:
    rows = audit["replacements"]
    # Once the authoritative ledger is committed, never re-enable old jobs.
    if not audit.get("ledger_committed"):
        for item in rows:
            old_id = item["old_job_id"]
            current = show(old_id)
            if field(current, "JobState") != "PENDING":
                raise RuntimeError(f"old E118 job {old_id} is no longer pending; stop before replacement")
            if not safely_held_old(current, item):
                command(["scontrol", "hold", str(old_id)])
            held_record = show(old_id)
            if not safely_held_old(held_record, item):
                raise RuntimeError(f"old E118 job {old_id} did not become safely held")
            item["original_pending_reason"] = field(item["before_record"], "Reason")
            item["old_held_record"] = held_record
            item["old_held"] = True
            save(audit, f"held superseded pending job {old_id}")

        for item in rows:
            if item.get("new_job_id") is None:
                if item.get("submission_uncertain"):
                    marker = f"e118-capacity-20260905-old{item['old_job_id']}"
                    listing = command(["squeue", "-h", "-u", command(["id", "-un"]).stdout.strip(),
                                       "-o", "%i|%k"]).stdout
                    found = [int(line.split("|", 1)[0]) for line in listing.splitlines()
                             if line.split("|", 1)[-1] == marker]
                    if len(found) != 1:
                        raise RuntimeError(f"uncertain prior submission for {marker}; inspect scheduler before resuming")
                    item["new_job_id"] = found[0]
                    save(audit, f"reconciled replacement {found[0]}")
                else:
                    item["submission_uncertain"] = True
                    save(audit, f"submitting held replacement for {item['old_job_id']}")
                    result = command(item["command"], check=False)
                    item["submission_returncode"] = result.returncode
                    item["submission_stdout"] = result.stdout
                    item["submission_stderr"] = result.stderr
                    if result.returncode:
                        # Keep uncertain set: a remote failure is not proof no job exists.
                        save(audit, f"replacement submission failed for {item['old_job_id']}")
                        raise RuntimeError(result.stderr.strip() or "sbatch failed")
                    item["new_job_id"] = int(result.stdout.strip().split(";", 1)[0])
                    item["submission_uncertain"] = False
                    save(audit, f"submitted held replacement {item['new_job_id']}")
            record = show(int(item["new_job_id"]))
            audit_record(record, item, held=True)
            item["held_scheduler_record"] = record
            save(audit, f"validated held replacement {item['new_job_id']}")

        current_bytes = LEDGER.read_bytes()
        source = json.loads(current_bytes)
        current_hash = sha(current_bytes)
        if current_hash == audit.get("intended_ledger_sha256"):
            audit["ledger_committed"] = True
            save(audit, "reconciled committed source ledger")
        else:
            if current_hash != audit["original_ledger_sha256"]:
                raise RuntimeError("source ledger changed during transaction; refusing to overwrite")
            by_id = {int(row["job_id"]): row for row in source["runs"]}
            for item in rows:
                row = by_id[item["old_job_id"]]
                if any(row[key] != item[key] for key in IDENTITY):
                    raise RuntimeError("source ledger scientific identity changed")
                row["previous_job_ids"] = [*row.get("previous_job_ids", []), item["old_job_id"]]
                row["job_id"] = item["new_job_id"]
                row["held_scheduler_record"] = item["held_scheduler_record"]
            source.setdefault("repair_history", []).append({
                "at": now(), "audit": str(AUDIT), "protocol": str(PROTOCOL),
                "scheduler_only": True,
                "replacements": [{"old_job_id": item["old_job_id"], "new_job_id": item["new_job_id"]}
                                 for item in rows],
            })
            if prepare_source is not None:
                prepare_source(source)
            audit["intended_ledger_sha256"] = sha(encoded(source))
            save(audit, "committing source ledger before cancel/release")
            atomic(LEDGER, source)
            audit["ledger_committed"] = True
            save(audit, "committed source ledger")
    elif sha(LEDGER.read_bytes()) != audit["intended_ledger_sha256"]:
        raise RuntimeError("committed source ledger changed; review before resuming")

    command(["python3", str(ROOT / "ops/exp_scaling/build_e118_aggregate_ledger.py")])
    audit["aggregate_rebuilt"] = True
    save(audit, "rebuilt aggregate from authoritative source ledgers")

    for item in rows:
        old_id = item["old_job_id"]
        if item.get("old_cancelled"):
            continue
        states = queue()
        if old_id in states:
            current = show(old_id)
            if not safely_held_old(current, item):
                raise RuntimeError(f"superseded job {old_id} unexpectedly active/unheld; do not release replacement")
            item["old_before_cancel_record"] = current
            item["cancel_requested"] = True
            save(audit, f"cancelling superseded held job {old_id}")
            command(["scancel", str(old_id)])
            if old_id in queue():
                raise RuntimeError(f"superseded job {old_id} still visible; resume after cancellation completes")
        elif not item.get("cancel_requested"):
            raise RuntimeError(f"superseded job {old_id} disappeared without this transaction cancelling it")
        item["old_cancelled"] = True
        save(audit, f"cancelled superseded job {old_id}")

    for item in audit["cs_placement_updates"]:
        job_id = item["old_job_id"]
        if item.get("updated"):
            continue
        current = show(job_id)
        if field(current, "JobState") != "PENDING":
            item["skipped_state"] = field(current, "JobState")
            save(audit, f"preserved nonpending cs job {job_id}")
            continue
        if field(current, "Account") != "allcs" or field(current, "Partition") != "cs" or field(current, "ExcNodeList") != PVL:
            raise RuntimeError(f"cs job {job_id} scheduler contract drift")
        command(["scontrol", "update", f"JobId={job_id}", f"NodeList={CS_NODES}"])
        after = show(job_id)
        admitted = set(command(["scontrol", "show", "hostnames", field(after, "ReqNodeList")]).stdout.split())
        if admitted != set(CS_NODES.split(",")) or field(after, "ExcNodeList") != PVL:
            raise RuntimeError(f"cs job {job_id} placement validation failed")
        item["after_record"] = after
        item["updated"] = True
        save(audit, f"updated cs placement {job_id}")

    for item in rows:
        if item.get("released"):
            continue
        # All old targets must be absent before any replacement is released.
        live_states = queue()
        if any(row["old_job_id"] in live_states for row in rows):
            raise RuntimeError("a superseded job remains queued; refusing replacement release")
        current = show(int(item["new_job_id"]))
        if item.get("release_requested") and field(current, "Reason") != "JobHeldUser":
            audit_record(current, item, held=False)
            if field(current, "JobState") not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "COMPLETED"}:
                raise RuntimeError(f"replacement {item['new_job_id']} is unexpectedly terminal")
            item["released"] = True
            item["after_record"] = current
            save(audit, f"reconciled replacement release {item['new_job_id']}")
            continue
        audit_record(current, item, held=True)
        item["release_requested"] = True
        save(audit, f"releasing replacement {item['new_job_id']}")
        command(["scontrol", "release", str(item["new_job_id"])])
        after = show(int(item["new_job_id"]))
        if field(after, "Reason") == "JobHeldUser":
            raise RuntimeError(f"replacement {item['new_job_id']} remains held")
        item["after_record"] = after
        item["released"] = True
        save(audit, f"released replacement {item['new_job_id']}")
    audit["status"] = "complete"
    save(audit, "transaction complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume:
        audit = json.loads(AUDIT.read_text())
        if audit.get("status") == "complete":
            raise SystemExit("transaction already complete; inspect the recorded audit")
    else:
        if args.apply and AUDIT.exists():
            raise SystemExit(f"audit already exists; review it and use --resume: {AUDIT}")
        audit = plan()
    print(json.dumps({
        "replacements": [{key: item[key] for key in ("old_job_id", "lane", *IDENTITY)}
                         for item in audit["replacements"]],
        "cs_placement_updates": [item["old_job_id"] for item in audit["cs_placement_updates"]],
        "apply": args.apply, "resume": args.resume,
    }, indent=2))
    if not (args.apply or args.resume):
        return
    if not PROTOCOL.is_file():
        raise SystemExit(f"required scheduling amendment absent: {PROTOCOL}")
    if not args.resume:
        save(audit, "started transaction")
    try:
        apply(audit)
    except BaseException as exc:
        audit["status"] = "interrupted"
        audit["error"] = repr(exc)
        save(audit, "transaction stopped; existing holds and ledger state preserved")
        raise


if __name__ == "__main__":
    main()
