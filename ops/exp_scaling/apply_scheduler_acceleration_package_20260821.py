#!/usr/bin/env python3
"""Apply the approved E113-R4/E116/E112-R1 scheduler-only package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e113r4_e116_e112r1_scheduler_acceleration_package_20260821.md"
)
DAPO_LEDGER = ROOT / "var/artifacts/e113r4_official_verl_dapo_jobs.json"
E116_LEDGER = ROOT / (
    "var/artifacts/e116_sparse_rlep_qwen05b_domain_extension_jobs.json"
)
E112_LEDGER = ROOT / (
    "var/artifacts/e112r1_verified_support_discovery_full_three_scale_jobs.json"
)
OUT = ROOT / "var/artifacts/scheduler_acceleration_package_20260821.json"

A6000_POOL = "node[103-104,205-208]"
A6000_NODES = (
    "node103", "node104", "node205", "node206", "node207", "node208"
)

DAPO = {
    30800804: {"family": "qwen05b", "seed": 43},
    30800805: {"family": "falcon1b", "seed": 55},
}
E116 = {
    30790388: {"domain": "countdown", "seed": 46},
    30790390: {"domain": "countdown", "seed": 47},
    30790399: {"domain": "mathir", "seed": 46},
    30790401: {"domain": "mathir", "seed": 47},
}
E112 = {
    30791519: {"domain": "python_factors", "seed": 55},
    30791520: {"domain": "python_factors", "seed": 56},
    30791522: {"domain": "python_factors", "seed": 58},
    30791523: {"domain": "python_factors", "seed": 59},
    30791529: {"domain": "pantry_plan", "seed": 55},
    30791530: {"domain": "pantry_plan", "seed": 56},
    30791531: {"domain": "pantry_plan", "seed": 57},
    30791532: {"domain": "pantry_plan", "seed": 58},
    30791533: {"domain": "pantry_plan", "seed": 59},
}


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(command)}: {detail}")
    return result.stdout.strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scheduler_record(job_id: int) -> str:
    return run(["scontrol", "show", "job", "-dd", "-o", str(job_id)])


def field(record: str, key: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S*)", record)
    return match.group(1) if match else None


def export_map(record: str) -> dict[str, str]:
    match = re.search(r"--export=ALL,(\S+)", record)
    if not match:
        raise RuntimeError("scheduler record carries no frozen --export block")
    pairs: dict[str, str] = {}
    for item in match.group(1).split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            pairs[key] = value
    if not pairs:
        raise RuntimeError("frozen --export block is empty")
    return pairs


def require_equal_environment(job_id: int, live: str, frozen: str) -> None:
    live_env = export_map(live)
    frozen_env = export_map(frozen)
    if live_env != frozen_env:
        drift = {
            key: (frozen_env.get(key), live_env.get(key))
            for key in sorted(set(frozen_env) | set(live_env))
            if frozen_env.get(key) != live_env.get(key)
        }
        raise RuntimeError(f"job {job_id} environment drifted: {drift}")


def inventory() -> dict[str, str]:
    node_inventory = run(
        [
            "sinfo", "-h", "-N", "-n",
            ",".join((*A6000_NODES, "node105")),
            "-o", "%N|%P|%G|%m|%T",
        ]
    )
    lines = node_inventory.splitlines()
    for node in A6000_NODES:
        matches = [
            line for line in lines
            if line.startswith(f"{node}|all|") and "gpu:a6000:" in line
        ]
        if len(matches) != 1:
            raise RuntimeError(f"A6000 pool inventory drifted for {node}")
        if int(matches[0].split("|")[3].rstrip("+")) < 128 * 1024:
            raise RuntimeError(f"A6000 pool node lacks 128 GiB: {node}")
    a5000 = [
        line for line in lines
        if line.startswith("node105|mltheory|") and "gpu:a5000:" in line
    ]
    if len(a5000) != 1 or int(a5000[0].split("|")[3].rstrip("+")) < 64 * 1024:
        raise RuntimeError("node105 A5000/mltheory inventory drifted")

    all_partition = run(["scontrol", "show", "partition", "all", "-o"])
    ml_partition = run(
        ["scontrol", "show", "partition", "mltheory", "-o"]
    )
    if "AllowAccounts=" not in all_partition or "allcs" not in field(
        all_partition, "AllowAccounts"
    ).split(","):
        raise RuntimeError("account allcs is not allowed in partition all")
    if field(all_partition, "PreemptMode") != "OFF":
        raise RuntimeError("partition all is no longer non-preempting")
    if field(ml_partition, "AllowAccounts") != "mltheory":
        raise RuntimeError("mltheory partition account contract drifted")
    return {
        "nodes": node_inventory,
        "all_partition": all_partition,
        "mltheory_partition": ml_partition,
    }


def frozen_records() -> dict[int, dict[str, Any]]:
    dapo = json.loads(DAPO_LEDGER.read_text(encoding="utf-8"))
    e116 = json.loads(E116_LEDGER.read_text(encoding="utf-8"))
    e112 = json.loads(E112_LEDGER.read_text(encoding="utf-8"))
    if not all(payload.get("released") is True for payload in (dapo, e116, e112)):
        raise RuntimeError("one of the source ledgers is not durably released")

    found: dict[int, dict[str, Any]] = {}
    dapo_smokes = dapo.get("smokes", {})
    smoke_records = (
        dapo_smokes.values() if isinstance(dapo_smokes, dict) else dapo_smokes
    )
    for record in smoke_records:
        job_id = int(record.get("job_id", -1))
        if job_id not in DAPO:
            continue
        target = DAPO[job_id]
        if str(record.get("family")) != target["family"] or int(
            record.get("seed", -1)
        ) != target["seed"]:
            raise RuntimeError(f"DAPO ledger target drifted: {job_id}")
        found[job_id] = {
            "component": "e113r4_smoke",
            "frozen_record": str(record["held_scheduler_record"]),
            "path": str(record["run_dir"]),
            **target,
        }

    for record in e116.get("collection", {}).get("records", []):
        job_id = int(record.get("collection_job_id", -1))
        if job_id not in E116:
            continue
        target = E116[job_id]
        if str(record.get("domain")) != target["domain"] or int(
            record.get("seed", -1)
        ) != target["seed"]:
            raise RuntimeError(f"E116 ledger target drifted: {job_id}")
        found[job_id] = {
            "component": "e116_pool",
            "frozen_record": str(record["held_collection_scheduler_record"]),
            "path": str(record["pool_root"]),
            "audit_job_id": int(record["audit_job_id"]),
            **target,
        }

    for record in e112.get("runs", []):
        job_id = int(record.get("job_id", -1))
        if job_id not in E112:
            continue
        target = E112[job_id]
        if str(record.get("domain")) != target["domain"] or int(
            record.get("seed", -1)
        ) != target["seed"]:
            raise RuntimeError(f"E112-R1 ledger target drifted: {job_id}")
        found[job_id] = {
            "component": "e112r1_science",
            "frozen_record": str(record["held_scheduler_record"]),
            "path": str(record["run_dir"]),
            **target,
        }

    expected = set(DAPO) | set(E116) | set(E112)
    if set(found) != expected:
        raise RuntimeError(f"source ledgers lack targets: {sorted(expected - set(found))}")
    return found


def validate_before(job_id: int, record: str, frozen: dict[str, Any]) -> None:
    component = str(frozen["component"])
    common = {
        "JobState": "PENDING",
        "RunTime": "00:00:00",
        "Restarts": "0",
        "Dependency": "(null)",
    }
    if component == "e113r4_smoke":
        expected = {
            **common, "Partition": "cs", "Account": "allcs",
            "ReqNodeList": "(null)", "NumCPUs": "16",
            "MinMemoryNode": "128G", "TimeLimit": "1-00:00:00",
            "TresPerNode": "gres/gpu:a6000:1", "Reason": "Priority",
            "Nice": "0",
        }
    elif component == "e116_pool":
        expected = {
            **common, "Partition": "cs", "Account": "allcs",
            "ReqNodeList": "node105", "NumCPUs": "8",
            "MinMemoryNode": "64G", "TimeLimit": "1-12:00:00",
            "TresPerNode": "gres/gpu:a5000:1", "Reason": "BadConstraints",
            "Nice": "100",
        }
    else:
        expected = {
            **common, "Partition": "cs", "Account": "allcs",
            "ReqNodeList": field(str(frozen["frozen_record"]), "ReqNodeList"),
            "NumCPUs": "8", "MinMemoryNode": "64G",
            "TimeLimit": "3-00:00:00",
            "TresPerNode": "gres/gpu:a6000:1", "Reason": "JobHeldUser",
            "Priority": "0", "Nice": "100",
        }
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    if wrong:
        raise RuntimeError(f"job {job_id} preflight drifted: {wrong}")
    require_equal_environment(job_id, record, str(frozen["frozen_record"]))
    if component in {"e113r4_smoke", "e116_pool"} and Path(
        str(frozen["path"])
    ).exists():
        raise RuntimeError(f"zero-runtime target already has output path: {job_id}")


def validate_after(job_id: int, record: str, frozen: dict[str, Any]) -> None:
    component = str(frozen["component"])
    state = field(record, "JobState")
    if state not in {"PENDING", "RUNNING"}:
        raise RuntimeError(f"job {job_id} has unexpected post-state: {state}")
    expected = {
        "Account": "mltheory" if component == "e116_pool" else "allcs",
        "Partition": "mltheory" if component == "e116_pool" else "all",
        "ReqNodeList": "node105" if component == "e116_pool" else A6000_POOL,
        "TresPerNode": (
            "gres/gpu:a5000:1" if component == "e116_pool"
            else "gres/gpu:a6000:1"
        ),
    }
    wrong = {
        key: (value, field(record, key))
        for key, value in expected.items()
        if field(record, key) != value
    }
    if wrong:
        raise RuntimeError(f"job {job_id} postflight drifted: {wrong}")
    require_equal_environment(job_id, record, str(frozen["frozen_record"]))
    if component == "e112r1_science" and state == "PENDING" and field(
        record, "Reason"
    ) == "JobHeldUser":
        raise RuntimeError(f"E112-R1 job stayed held after release: {job_id}")
    if component == "e116_pool" and state == "PENDING" and field(
        record, "Reason"
    ) == "BadConstraints":
        raise RuntimeError(f"E116 pool job remains unsatisfiable: {job_id}")


def update(job_id: int, **changes: str) -> None:
    run(
        ["scontrol", "update", f"JobId={job_id}"]
        + [f"{key}={value}" for key, value in changes.items()]
    )


def release(job_id: int) -> None:
    run(["scontrol", "release", str(job_id)])


def hold(job_id: int) -> None:
    run(["scontrol", "hold", str(job_id)])


def dependency_snapshot() -> dict[str, Any]:
    dapo_science = {
        str(job_id): scheduler_record(job_id) for job_id in (30800707, 30800760)
    }
    e116_audits = {
        str(job_id): scheduler_record(job_id)
        for job_id in (30790389, 30790391, 30790400, 30790402)
    }
    return {"dapo_science_boundary": dapo_science, "e116_audits": e116_audits}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"protocol is absent: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate application: {OUT}")

    node_inventory = inventory()
    frozen = frozen_records()
    before = {job_id: scheduler_record(job_id) for job_id in frozen}
    for job_id in sorted(frozen):
        validate_before(job_id, before[job_id], frozen[job_id])
    dependencies_before = dependency_snapshot()

    if not args.apply:
        for job_id in DAPO:
            print(
                f"scontrol update JobId={job_id} Partition=all "
                f"NodeList={A6000_POOL}"
            )
        for job_id in E116:
            print(
                f"scontrol update JobId={job_id} Partition=mltheory "
                "Account=mltheory"
            )
        for job_id in E112:
            print(
                f"scontrol update JobId={job_id} Partition=all "
                f"NodeList={A6000_POOL}; scontrol release {job_id}"
            )
        print(f"[scheduler-acceleration] dry_run=True jobs={len(frozen)}")
        return 0

    changed: list[int] = []
    released: list[int] = []
    try:
        for job_id in DAPO:
            update(job_id, Partition="all", NodeList=A6000_POOL)
            changed.append(job_id)
        for job_id in E116:
            update(job_id, Partition="mltheory", Account="mltheory")
            changed.append(job_id)
        for job_id in E112:
            update(job_id, Partition="all", NodeList=A6000_POOL)
            changed.append(job_id)
        for job_id in E112:
            release(job_id)
            released.append(job_id)

        time.sleep(2)
        after = {job_id: scheduler_record(job_id) for job_id in frozen}
        for job_id in sorted(frozen):
            validate_after(job_id, after[job_id], frozen[job_id])
    except Exception:
        for job_id in reversed(released):
            try:
                current = scheduler_record(job_id)
                if field(current, "JobState") == "PENDING":
                    hold(job_id)
            except Exception:  # noqa: BLE001 - restore as much as possible
                pass
        for job_id in reversed(changed):
            try:
                current = scheduler_record(job_id)
                if field(current, "JobState") != "PENDING":
                    continue
                if job_id in E116:
                    update(job_id, Partition="cs", Account="allcs")
                elif job_id in E112:
                    update(
                        job_id,
                        Partition="cs",
                        NodeList=str(
                            field(str(frozen[job_id]["frozen_record"]), "ReqNodeList")
                        ),
                    )
                else:
                    update(job_id, Partition="cs", NodeList="")
            except Exception:  # noqa: BLE001 - restore as much as possible
                pass
        raise

    dependencies_after = dependency_snapshot()
    for job_id, record in dependencies_after["dapo_science_boundary"].items():
        dependency = field(record, "Dependency") or ""
        if "30800804" not in dependency or "30800805" not in dependency:
            raise RuntimeError(f"DAPO science dependency drifted: {job_id}")
    for job_id, record in dependencies_after["e116_audits"].items():
        pool_id = {
            "30790389": "30790388", "30790391": "30790390",
            "30790400": "30790399", "30790402": "30790401",
        }[job_id]
        if pool_id not in (field(record, "Dependency") or ""):
            raise RuntimeError(f"E116 audit dependency drifted: {job_id}")

    filesystem = {
        str(job_id): {
            "path": str(record["path"]),
            "exists": Path(str(record["path"])).exists(),
        }
        for job_id, record in frozen.items()
    }
    payload = {
        "schema": "scheduler_acceleration_package_20260821_v1",
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "author_approved": True,
        "scheduler_only": True,
        "scientific_environment_changed": False,
        "gpu_type_changed": False,
        "stopping_rule_changed": False,
        "dependencies_changed": False,
        "outcomes_inspected": False,
        "run_directories_touched": False,
        "pointmaze": "excluded",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": digest(PROTOCOL),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": digest(Path(__file__)),
        "ledgers": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (DAPO_LEDGER, E116_LEDGER, E112_LEDGER)
        },
        "inventory": node_inventory,
        "filesystem": filesystem,
        "dependencies_before": dependencies_before,
        "dependencies_after": dependencies_after,
        "jobs": [
            {
                "job_id": job_id,
                "component": frozen[job_id]["component"],
                "domain": frozen[job_id].get("domain"),
                "family": frozen[job_id].get("family"),
                "seed": frozen[job_id]["seed"],
                "before": before[job_id],
                "after": after[job_id],
            }
            for job_id in sorted(frozen)
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[scheduler-acceleration] applied=True jobs={len(frozen)} "
        f"artifact={OUT.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
