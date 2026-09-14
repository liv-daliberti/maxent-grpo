#!/usr/bin/env python3
"""Constrain only pending E119 Pantry cells to their existing A6000 routes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
import campaign_stats as campaign

ART = ROOT / "var/artifacts/e119_health_recovery_20260905"
AUDIT = ART / "pending_pantry_a6000.json"
SAFE = ("node205", "node207")
LOW_MEMORY = {"node203", "node204"}
PRESERVE = ("Account", "Partition", "NumNodes", "NumCPUs", "NumTasks", "CPUs/Task",
            "MinMemoryNode", "ReqTRES", "TimeLimit", "ExcNodeList", "Dependency",
            "Command", "WorkDir", "StdOut", "StdErr", "Requeue")


def call(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def field(record: str, name: str) -> str:
    found = re.search(r"(?:^| )" + re.escape(name) + r"=([^ ]*)", record)
    if not found:
        raise RuntimeError(f"missing scheduler field {name}")
    return found.group(1)


def submitline(record: str) -> str:
    return record.split(" SubmitLine=", 1)[1].split(" WorkDir=", 1)[0]


def show(job_id: int) -> str:
    return call("scontrol", "show", "job", "-dd", "-o", str(job_id))


def hosts(record: str) -> set[str]:
    return set(call("scontrol", "show", "hostnames", field(record, "ReqNodeList")).splitlines())


def target_nodes(record: str) -> str | None:
    if field(record, "JobState") != "PENDING":
        return None
    pool = hosts(record)
    if not pool.intersection(LOW_MEMORY):
        return None
    target = [node for node in SAFE if node in pool]
    if not target:
        raise RuntimeError("pending Pantry cell has no previously permitted A6000 route")
    if field(record, "Reason") in {"JobHeldUser", "JobHeldAdmin"}:
        raise RuntimeError("refusing to change an existing held job")
    assert field(record, "Account") == "allcs"
    assert field(record, "Partition") == "cs"
    assert field(record, "NumNodes") == "1-1"
    return ",".join(target)


def persist(record: dict, path: Path = AUDIT) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def prepare() -> dict:
    ledger = json.loads(campaign.E119_LEDGER.read_text())
    mapping = campaign.e119_continuation_jobs(campaign.E119_LEDGER)
    baseline = []
    for run in ledger["runs"]:
        if run["domain"] != "pantry_plan":
            continue
        original = int(run["job_id"])
        job_id = mapping.get(original, original)
        record = show(job_id)
        assert f"RUN_STAMP={run['run_stamp']}" in record
        baseline.append({"original_job_id": original, "job_id": job_id,
                         "arm": run["arm"], "seed": run["seed"],
                         "run_dir": run["run_dir"], "before": record,
                         "before_state": field(record, "JobState"),
                         "target_nodes": target_nodes(record)})
    assert len(baseline) == 20
    assert len({row["job_id"] for row in baseline}) == 20
    for node in SAFE:
        record = call("scontrol", "show", "node", "-o", node)
        assert "gpu:a6000:" in field(record, "Gres")
    return {"schema": "e119-pending-pantry-a6000-placement-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_configuration_changed": False,
            "run_directories_changed": False, "runtime_changed": False,
            "other_cohorts_changed": False,
            "selection_basis": "pending E119 Pantry cells eligible for the 24GB GPU class implicated in five learner CUDA OOMs",
            "baseline_pending_count": sum(r["before_state"] == "PENDING" for r in baseline),
            "baseline_running_count": sum(r["before_state"] == "RUNNING" for r in baseline),
            "target_job_ids": [r["job_id"] for r in baseline if r["target_nodes"]],
            "baseline": baseline, "changes": [], "applied": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if AUDIT.exists():
        raise RuntimeError(f"prior transaction requires explicit inspection: {AUDIT}")
    audit = prepare()
    persist(audit, ART / "pending_pantry_a6000_plan.json")
    if not args.apply:
        print(json.dumps({"dry_run": True, "baseline_pending": audit["baseline_pending_count"],
                          "targets": audit["target_job_ids"]}))
        return
    persist(audit)
    for baseline in audit["baseline"]:
        if not baseline["target_nodes"]:
            continue
        jid = baseline["job_id"]
        latest = show(jid)
        target = target_nodes(latest)
        row = {"job_id": jid, "pre_mutation": latest}
        audit["changes"].append(row)
        if target is None:
            row["skipped"] = "job started or placement already safe"
            persist(audit)
            continue
        # Holding a pending job prevents an allocation racing the node change.
        call("scontrol", "hold", str(jid))
        held = show(jid)
        row["held"] = held
        if field(held, "JobState") != "PENDING":
            call("scontrol", "release", str(jid))
            row["skipped"] = "allocation raced the hold; node constraints untouched"
            persist(audit)
            continue
        assert field(held, "Reason") == "JobHeldUser"
        assert submitline(held) == submitline(latest)
        persist(audit)
        call("scontrol", "update", f"JobId={jid}", f"ReqNodeList={target}")
        after = show(jid)
        assert field(after, "JobState") == "PENDING"
        assert field(after, "Reason") == "JobHeldUser"
        assert hosts(after) == set(target.split(","))
        for key in PRESERVE:
            assert field(after, key) == field(latest, key), key
        assert submitline(after) == submitline(latest)
        row["audited_held_after"] = after
        row["target_nodes"] = target
        persist(audit)
        call("scontrol", "release", str(jid))
        row["released"] = True
        row["after"] = show(jid)
        persist(audit)
    audit["applied"] = True
    audit["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist(audit)
    print(json.dumps({"audit": str(AUDIT), "changed": [r["job_id"] for r in audit["changes"] if r.get("released")],
                      "skipped": [r["job_id"] for r in audit["changes"] if r.get("skipped")]}))


if __name__ == "__main__":
    main()
