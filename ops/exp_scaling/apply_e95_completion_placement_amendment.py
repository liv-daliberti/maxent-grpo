#!/usr/bin/env python3
"""Release the thirteen incomplete E95 cells onto same-GPU-family pools."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/e95_completion_placement_amendment_20260818.md"
)
OUT = ROOT / "var/artifacts/e95_completion_placement_amendment.json"
QWEN_LEDGER = ROOT / "var/artifacts/e95_plain_grpo_Qwen25-05B_jobs.json"
FALCON_LEDGER = ROOT / "var/artifacts/e95_plain_grpo_Falcon3-1B_jobs.json"

QWEN_TARGETS = {
    30516445: ("graph_coloring", 43),
    30516453: ("countdown", 46),
    30516454: ("countdown", 47),
    30516455: ("python_factors", 43),
    30516456: ("python_factors", 44),
    30516463: ("mathir", 46),
    30516464: ("mathir", 47),
    30516465: ("pantry_plan", 43),
    30516466: ("pantry_plan", 44),
}
FALCON_TARGETS = {
    30516440: ("pantry_plan", 55),
    30516441: ("pantry_plan", 56),
    30516443: ("pantry_plan", 58),
    30516444: ("pantry_plan", 59),
}
QWEN_NODES = "node[105,202-204]"
FALCON_NODES = "node[103-104,205-208,805]"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def command(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def scheduler_record(job_id: int) -> str:
    return command(["scontrol", "show", "job", "-dd", "-o", str(job_id)])


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing E95 ledger: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def target_runs() -> dict[int, dict[str, Any]]:
    payloads = {"qwen05b": load(QWEN_LEDGER), "falcon1b": load(FALCON_LEDGER)}
    if any(payload.get("released") is not True for payload in payloads.values()):
        raise RuntimeError("E95 source ledgers were not released")
    selected: dict[int, dict[str, Any]] = {}
    for family, targets in (("qwen05b", QWEN_TARGETS), ("falcon1b", FALCON_TARGETS)):
        ledger = payloads[family]
        for job_id, (domain, seed) in targets.items():
            matches = [
                row
                for row in ledger["runs"]
                if int(row["job_id"]) == job_id
                and str(row["domain"]) == domain
                and int(row["seed"]) == seed
            ]
            if len(matches) != 1:
                raise RuntimeError(f"E95 target identity drifted: {job_id}")
            selected[job_id] = {
                **matches[0],
                "family": family,
                "ledger_path": str(QWEN_LEDGER if family == "qwen05b" else FALCON_LEDGER),
            }
    return selected


def inventory() -> str:
    return command(
        [
            "sinfo",
            "-h",
            "-N",
            "-n",
            "node103,node104,node105,node202,node203,node204,node205,node206,node207,node208,node805",
            "-o",
            "%N|%P|%G|%m",
        ]
    )


def validate_inventory(text: str) -> None:
    expected = {
        "node105": "a5000",
        "node202": "a5000",
        "node203": "a5000",
        "node204": "a5000",
        "node103": "a6000",
        "node104": "a6000",
        "node205": "a6000",
        "node206": "a6000",
        "node207": "a6000",
        "node208": "a6000",
        "node805": "a6000",
    }
    lines = text.splitlines()
    for node, gpu in expected.items():
        matches = [line for line in lines if line.startswith(f"{node}|all|")]
        if len(matches) != 1 or f"gpu:{gpu}:" not in matches[0]:
            raise RuntimeError(f"E95 same-family inventory drifted: {node}")
        if int(matches[0].rsplit("|", 1)[1]) < 64 * 1024:
            raise RuntimeError(f"E95 node lacks 64 GiB host memory: {node}")


def require_science(run: dict[str, Any], record: str, *, initial: bool) -> None:
    family = str(run["family"])
    job_id = int(run["job_id"])
    required = [
        "JobState=PENDING",
        "NumCPUs=8",
        "MinMemoryNode=64G",
        f"OAT_ZERO_SEED={run['seed']}",
        "OAT_ZERO_VARIANT=grpo_plain_control",
        "OAT_ZERO_MAX_TRAIN=384",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
        "OAT_ZERO_LEARNING_RATE=2e-07",
        f"SAVE_PATH={run['run_dir']}",
    ]
    if family == "qwen05b":
        required.extend(("TresPerNode=gres/gpu:a5000:1", "Nice=4000" if initial else "Nice=0"))
        if initial:
            required.extend(("Partition=cs", "ReqNodeList=node105", "RunTime=00:00:00"))
        else:
            required.extend(("Partition=all", f"ReqNodeList={QWEN_NODES}"))
    else:
        required.extend(("TresPerNode=gres/gpu:a6000:1", "Nice=4000" if initial else "Nice=0"))
        if initial:
            required.extend(("Partition=cs", "Reason=JobHeldUser"))
        else:
            required.extend(("Partition=all", f"ReqNodeList={FALCON_NODES}"))
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"E95 job {job_id} lacks {missing}")


def update_ledger(path: Path, amendment: dict[str, Any]) -> None:
    payload = load(path)
    entries = payload.setdefault("scheduler_amendments", [])
    if any(str(item.get("artifact")) == str(OUT) for item in entries):
        raise RuntimeError(f"E95 ledger already records {OUT}")
    entries.append(amendment)
    atomic_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"missing E95 amendment protocol: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate E95 amendment: {OUT}")

    runs = target_runs()
    inventory_text = inventory()
    validate_inventory(inventory_text)
    before = {job_id: scheduler_record(job_id) for job_id in runs}
    for job_id, run in runs.items():
        require_science(run, before[job_id], initial=True)

    if not args.apply:
        print(f"hold {len(runs)} E95 jobs")
        print(f"Qwen -> Partition=all NodeList={QWEN_NODES} Nice=0")
        print(f"Falcon -> Partition=all NodeList={FALCON_NODES} Nice=0")
        print("audit held records, record amendment, release all jobs")
        return 0

    held: list[int] = []
    changed: list[int] = []
    originally_held = {
        job_id: "Reason=JobHeldUser" in record for job_id, record in before.items()
    }
    try:
        for job_id in runs:
            command(["scontrol", "hold", str(job_id)])
            held.append(job_id)
        for job_id, run in runs.items():
            nodes = QWEN_NODES if run["family"] == "qwen05b" else FALCON_NODES
            command(
                [
                    "scontrol",
                    "update",
                    f"JobId={job_id}",
                    "Partition=all",
                    "Account=allcs",
                    f"NodeList={nodes}",
                    "Nice=0",
                ]
            )
            changed.append(job_id)
        held_after = {job_id: scheduler_record(job_id) for job_id in runs}
        for job_id, run in runs.items():
            require_science(run, held_after[job_id], initial=False)
            if "Reason=JobHeldUser" not in held_after[job_id]:
                raise RuntimeError(f"E95 job {job_id} escaped held audit")

        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "schema": "e95_completion_placement_amendment_v1",
            "applied_at": timestamp,
            "released": False,
            "protocol": str(PROTOCOL),
            "protocol_sha256": digest(PROTOCOL),
            "inventory": inventory_text,
            "qwen_nodes": QWEN_NODES,
            "falcon_nodes": FALCON_NODES,
            "jobs": [
                {
                    "job_id": job_id,
                    "family": run["family"],
                    "domain": run["domain"],
                    "seed": run["seed"],
                    "run_dir": run["run_dir"],
                    "before": before[job_id],
                    "held_after": held_after[job_id],
                }
                for job_id, run in runs.items()
            ],
        }
        atomic_json(OUT, payload)
        amendment = {
            "artifact": str(OUT),
            "protocol": str(PROTOCOL),
            "applied_at": timestamp,
            "scientific_change": False,
            "change": "same-GPU-family pool widening and Nice 4000 to 0",
        }
        update_ledger(QWEN_LEDGER, amendment)
        update_ledger(FALCON_LEDGER, amendment)
        for job_id in runs:
            command(["scontrol", "release", str(job_id)])
        released_after = {job_id: scheduler_record(job_id) for job_id in runs}
        for job_id, record in released_after.items():
            if "Reason=JobHeldUser" in record:
                raise RuntimeError(f"E95 job {job_id} remained user-held")
        payload["released"] = True
        payload["released_after"] = released_after
        atomic_json(OUT, payload)
    except Exception:
        if not OUT.exists():
            for job_id in changed:
                run = runs[job_id]
                old_nodes = "node105" if run["family"] == "qwen05b" else (
                    "node206" if job_id in (30516440, 30516443) else "node207"
                )
                subprocess.run(
                    [
                        "scontrol",
                        "update",
                        f"JobId={job_id}",
                        "Partition=cs",
                        "Account=allcs",
                        f"NodeList={old_nodes}",
                        "Nice=4000",
                    ],
                    check=False,
                )
            for job_id in held:
                if not originally_held[job_id]:
                    subprocess.run(["scontrol", "release", str(job_id)], check=False)
        raise

    print(
        f"released {len(QWEN_TARGETS)} Qwen and {len(FALCON_TARGETS)} Falcon "
        f"E95 cells; artifact={OUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
