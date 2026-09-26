#!/usr/bin/env python3
"""Clean-restart preempted E106 Qwen-3B and widen only its partition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e104_group_centered_semantic_repair_three_scale as e104  # noqa: E402
import launch_e106_python_lambda_normalization_three_scale as e106  # noqa: E402
import status_e78 as status  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = (ROOT / e106.SNAPSHOT).resolve()
PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e106_qwen3_preemption_clean_restart_all_partition_amendment_20260818.md"
)
LEDGER = ROOT / e106.LEDGER
PRIOR_PLACEMENT = ROOT / "var/artifacts/e106_qwen3_a6000_placement.json"
TIME_LIMIT_AMENDMENT = ROOT / (
    "var/artifacts/e106_python_smoke_time_limit_amendment.json"
)
RESUME_EVIDENCE = ROOT / "var/artifacts/e105_semantic_replay_identity_contract_tests.json"
OUT = ROOT / (
    "var/artifacts/"
    "e106_qwen3_preemption_clean_restart_all_partition_amendment.json"
)
JOB_ID = 30640331
OLD_PARTITION = "lowprio"
NEW_PARTITION = "all"
NODE_LIST = "node[103-104,205-208,805]"
AUTHORIZED_NODES = (
    "node103",
    "node104",
    "node205",
    "node206",
    "node207",
    "node208",
    "node805",
)
INTERRUPTED_STEP = 22
ARCHIVE_SUFFIX = "_preempted_no_checkpoint_step22_restart1"
ALLOWED_PREFIX_FILES = {
    "debug_job30640331/eval_results/0_multi_answer.json",
    "debug_job30640331/eval_results/16_multi_answer.json",
    "debug_job30640331/eval_mode_coverage_draws.jsonl",
    "debug_job30640331/train_metrics.jsonl",
}


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required artifact is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed: {detail}")
    return result.stdout.strip()


def scheduler_record() -> str:
    return command(["scontrol", "show", "job", "-dd", "-o", str(JOB_ID)])


def require_record(record: str, *, partition: str, held: bool) -> None:
    required = [
        "JobState=PENDING",
        f"Partition={partition}",
        "Account=mltheory",
        "RunTime=00:00:00",
        "TimeLimit=02:00:00",
        "Requeue=1",
        "Restarts=1",
        "ExitCode=0:0",
        "NumCPUs=16",
        "MinMemoryNode=128G",
        f"ReqNodeList={NODE_LIST}",
        "TresPerNode=gres/gpu:a6000:1",
        f"OAT_ZERO_SOURCE_ROOT={SNAPSHOT_ROOT}/src",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={SNAPSHOT_ROOT}/ops",
        "OAT_ZERO_VARIANT=verified_replay_semantic_maxent_group_centered",
        "OAT_ZERO_SEED=70",
        "OAT_ZERO_MAX_TRAIN=64",
        "OAT_ZERO_SAVE_FROM=32",
        "OAT_ZERO_SAVE_STEPS=32",
        "OAT_ZERO_AUTO_RESUME=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
    ]
    if held:
        required.append("Reason=JobHeldUser")
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"Qwen-3B scheduler record lacks {missing}")


def target_run() -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = load(LEDGER)
    if ledger.get("released") is not True or ledger.get("pointmaze") != "excluded":
        raise RuntimeError("E106 ledger release/exclusion drifted")
    if ledger.get("snapshot_root") != str(SNAPSHOT_ROOT):
        raise RuntimeError("E106 snapshot root drifted")
    matches = [
        item
        for item in ledger.get("runs", [])
        if int(item.get("job_id", -1)) == JOB_ID
    ]
    if len(matches) != 1:
        raise RuntimeError("E106 Qwen-3B target set drifted")
    run = matches[0]
    if (
        run.get("scale"),
        run.get("domain"),
        int(run.get("seed", -1)),
    ) != ("qwen3b", "python_factors", 70):
        raise RuntimeError("E106 Qwen-3B scientific cell drifted")
    return ledger, run


def inventory_record() -> str:
    return command(
        [
            "sinfo",
            "-h",
            "-N",
            "-n",
            ",".join(AUTHORIZED_NODES),
            "-o",
            "%N|%P|%G|%m",
        ]
    )


def require_inventory(record: str) -> None:
    for node in AUTHORIZED_NODES:
        matches = [
            line
            for line in record.splitlines()
            if line.startswith(f"{node}|all|") and "gpu:a6000:" in line
        ]
        if len(matches) != 1:
            raise RuntimeError(f"authorized all/A6000 inventory drifted: {node}")
        if int(matches[0].rsplit("|", 1)[1]) < 128 * 1024:
            raise RuntimeError(f"authorized node lacks 128 GiB: {node}")


def prefix_manifest(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.is_dir():
        raise RuntimeError(f"interrupted run directory is absent: {run_dir}")
    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    relative = {str(path.relative_to(run_dir)) for path in files}
    if relative != ALLOWED_PREFIX_FILES:
        raise RuntimeError(
            "interrupted prefix file set drifted; refusing an ambiguous restart: "
            f"{sorted(relative)}"
        )
    if status.run_step(run_dir) != INTERRUPTED_STEP:
        raise RuntimeError("interrupted prefix is not exactly step 22")
    forbidden = (
        "checkpoint",
        "global_step",
        "optimizer",
        "resume",
        "replay_bank",
        "semantic_history",
    )
    if any(any(token in str(path.relative_to(run_dir)).lower() for token in forbidden) for path in files):
        raise RuntimeError("interrupted prefix unexpectedly contains resumable state")
    return [
        {
            "path": str(path.relative_to(run_dir)),
            "size": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in files
    ]


def validate_frozen_evidence() -> None:
    placement = load(PRIOR_PLACEMENT)
    if (
        placement.get("schema") != "e106_qwen3_a6000_placement_v1"
        or placement.get("job_id") != JOB_ID
        or placement.get("environment_changed") is not False
        or placement.get("post_update_outcomes_inspected") is not False
    ):
        raise RuntimeError("prior Qwen-3B placement evidence drifted")
    time_limit = load(TIME_LIMIT_AMENDMENT)
    if time_limit.get("schema") != "e106_python_smoke_time_limit_amendment_v1":
        raise RuntimeError("E106 time-limit evidence drifted")
    resume = load(RESUME_EVIDENCE)
    assertions = resume.get("assertions", {})
    if (
        resume.get("passed") is not True
        or resume.get("snapshot_sha256") != e106.SNAPSHOT_SHA256
        or assertions.get("joint_auto_resume_is_exact") is not True
    ):
        raise RuntimeError("joint-resume identity evidence drifted")


def update_partition(partition: str) -> None:
    command(
        [
            "scontrol",
            "update",
            f"JobId={JOB_ID}",
            f"Partition={partition}",
            "Account=mltheory",
            f"NodeList={NODE_LIST}",
            "Gres=gpu:a6000:1",
            "TimeLimit=02:00:00",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not PROTOCOL.is_file():
        raise SystemExit(f"protocol is absent: {PROTOCOL}")
    if args.apply and OUT.exists():
        raise SystemExit(f"refusing duplicate amendment: {OUT}")
    ledger, run = target_run()
    validate_frozen_evidence()
    inventory = inventory_record()
    require_inventory(inventory)
    run_dir = Path(str(run["run_dir"]))
    archive = Path(f"{run_dir}{ARCHIVE_SUFFIX}")
    if archive.exists():
        raise RuntimeError(f"interrupted-prefix archive already exists: {archive}")
    manifest = prefix_manifest(run_dir)
    before = scheduler_record()
    require_record(before, partition=OLD_PARTITION, held=False)
    if not args.apply:
        print(f"scontrol hold {JOB_ID}")
        print(f"mv {shlex.quote(str(run_dir))} {shlex.quote(str(archive))}")
        print(
            f"scontrol update JobId={JOB_ID} Partition={NEW_PARTITION} "
            f"Account=mltheory NodeList={NODE_LIST} Gres=gpu:a6000:1 "
            "TimeLimit=02:00:00"
        )
        print(f"scontrol release {JOB_ID}")
        return 0

    held = ""
    amended = ""
    released = ""
    archived = False
    partition_changed = False
    job_released = False
    try:
        command(["scontrol", "hold", str(JOB_ID)])
        held = scheduler_record()
        require_record(held, partition=OLD_PARTITION, held=True)
        run_dir.rename(archive)
        archived = True
        update_partition(NEW_PARTITION)
        partition_changed = True
        amended = scheduler_record()
        require_record(amended, partition=NEW_PARTITION, held=True)
        payload = {
            "schema": "e106_qwen3_preemption_clean_restart_all_partition_amendment_v1",
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": digest(PROTOCOL),
            "script": str(Path(__file__).relative_to(ROOT)),
            "script_sha256": digest(Path(__file__)),
            "e106_ledger": str(LEDGER.relative_to(ROOT)),
            "e106_ledger_sha256": digest(LEDGER),
            "prior_placement": str(PRIOR_PLACEMENT.relative_to(ROOT)),
            "prior_placement_sha256": digest(PRIOR_PLACEMENT),
            "time_limit_amendment": str(TIME_LIMIT_AMENDMENT.relative_to(ROOT)),
            "time_limit_amendment_sha256": digest(TIME_LIMIT_AMENDMENT),
            "resume_identity_evidence": str(RESUME_EVIDENCE.relative_to(ROOT)),
            "resume_identity_evidence_sha256": digest(RESUME_EVIDENCE),
            "job_id": JOB_ID,
            "scale": "qwen3b",
            "domain": "python_factors",
            "seed": 70,
            "snapshot_root": ledger["snapshot_root"],
            "snapshot_sha256": e106.SNAPSHOT_SHA256,
            "scheduler_only": True,
            "clean_restart_required": True,
            "interrupted_step": INTERRUPTED_STEP,
            "checkpoint_present": False,
            "old_partition": OLD_PARTITION,
            "new_partition": NEW_PARTITION,
            "node_list": NODE_LIST,
            "gpu_type_changed": False,
            "environment_changed": False,
            "scientific_configuration_changed": False,
            "post_e104_or_e106_update_outcomes_inspected": False,
            "pointmaze": "excluded",
            "run_dir": str(run_dir),
            "archive_dir": str(archive),
            "archived_prefix_manifest": manifest,
            "authorized_nodes": list(AUTHORIZED_NODES),
            "inventory": inventory,
            "before": before,
            "held": held,
            "amended_held": amended,
            "released": False,
        }
        e104.e81.atomic_json(OUT, payload)
        command(["scontrol", "release", str(JOB_ID)])
        job_released = True
        released = scheduler_record()
        require_record(released, partition=NEW_PARTITION, held=False)
        payload["released"] = True
        payload["released_scheduler_record"] = released
        payload["released_at"] = datetime.now(timezone.utc).isoformat()
        e104.e81.atomic_json(OUT, payload)
    except Exception:
        if not job_released:
            if partition_changed:
                update_partition(OLD_PARTITION)
            if archived and archive.exists() and not run_dir.exists():
                archive.rename(run_dir)
            command(["scontrol", "release", str(JOB_ID)])
        raise

    print(f"[e106-q3-clean-restart] job={JOB_ID} artifact={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
