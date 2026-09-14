#!/usr/bin/env python3
"""Record and later finalize the E111 Pantry partial-checkpoint quarantine."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import subprocess
import zipfile
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "paper/preregistration/e111_qwen3_pantry_partial_checkpoint_quarantine_20260818.md"
RECORD = ROOT / "var/artifacts/e111_qwen3_pantry_partial_checkpoint_quarantine.json"
JOB_ID = 30674762
RUN = ROOT / "var/data/xdr_qwen25_3b_instruct_verified_replay_semantic_maxent_verified_support_discovery_e111_qwen3b_pantry_verified_support_discovery_s70/debug_job30674762"
CHECKPOINTS = RUN / "checkpoints"
VALID = CHECKPOINTS / "step_00002"
QUARANTINED = RUN / "quarantine_partial_checkpoints/step_00004_badzip_20260818T1729"
STDOUT = ROOT / "var/artifacts/logs/e111-q3-pantry-30674762.out"
CORRUPTION = "PytorchStreamReader failed reading zip archive: failed finding central directory"
MODEL = "mp_rank_00_model_states.pt"
OPTIMIZER = "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zip_directory_status(path: Path) -> dict[str, object]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile as exc:
        return {"valid": False, "error": type(exc).__name__, "detail": str(exc)}
    return {"valid": True, "members": len(names), "last_member": names[-1]}


def file_evidence(directory: Path) -> dict[str, dict[str, object]]:
    return {
        name: {
            "path": str(directory / name),
            "size": (directory / name).stat().st_size,
            "zip_directory": zip_directory_status(directory / name),
        }
        for name in (MODEL, OPTIMIZER)
    }


def scheduler_record() -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(JOB_ID)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or f"JobId={JOB_ID}" not in result.stdout:
        raise RuntimeError("cannot inspect Pantry recovery job")
    return result.stdout.strip()


def now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("record", "finalize"))
    args = parser.parse_args()
    text = STDOUT.read_text(encoding="utf-8", errors="replace")
    if args.mode == "record":
        if RECORD.exists():
            raise SystemExit(f"refusing duplicate quarantine record: {RECORD}")
        valid = file_evidence(VALID)
        quarantined = file_evidence(QUARANTINED)
        if not all(row["zip_directory"]["valid"] for row in valid.values()):
            raise RuntimeError("step_00002 is not a valid recovery checkpoint")
        if quarantined[MODEL]["zip_directory"]["valid"] is not True:
            raise RuntimeError("quarantined step model state is unexpectedly invalid")
        if quarantined[OPTIMIZER]["zip_directory"]["valid"] is not False:
            raise RuntimeError("quarantined optimizer archive is unexpectedly valid")
        if (CHECKPOINTS / "latest").read_text().strip() != "step_00002":
            raise RuntimeError("latest marker does not select step_00002")
        payload = {
            "schema": "e111_qwen3_pantry_partial_checkpoint_quarantine_v1",
            "recorded_at": now(),
            "protocol": str(PROTOCOL),
            "protocol_sha256": digest(PROTOCOL),
            "job_id": JOB_ID,
            "run_root": str(RUN),
            "valid_checkpoint": str(VALID),
            "quarantined_checkpoint": str(QUARANTINED),
            "valid_checkpoint_evidence": valid,
            "quarantined_checkpoint_evidence": quarantined,
            "latest_marker": "step_00002",
            "pre_recovery_failure_counts": {
                "tracebacks": text.count("Traceback (most recent call last)"),
                "partial_checkpoint": text.count(CORRUPTION),
            },
            "scheduler_record_after_timeout_requeue": scheduler_record(),
            "checkpoint_storage_only": True,
            "quarantined_bytes_recoverable": True,
            "bytes_deleted": False,
            "job_signaled_for_quarantine": False,
            "optimizer_update_changed": False,
            "treatment_changed": False,
            "outcomes_inspected": False,
            "pointmaze": "excluded",
            "verified_after_requeue": False,
        }
    else:
        payload = json.loads(RECORD.read_text(encoding="utf-8"))
        if payload.get("verified_after_requeue") is not False:
            raise RuntimeError("quarantine record was already finalized")
        expected = payload["pre_recovery_failure_counts"]
        if expected != {"tracebacks": 1, "partial_checkpoint": 1}:
            raise RuntimeError("archived pre-recovery failure evidence drifted")
        observed = {
            "tracebacks": text.count("Traceback (most recent call last)"),
            "partial_checkpoint": text.count(CORRUPTION),
        }
        if observed != {"tracebacks": 0, "partial_checkpoint": 0}:
            raise RuntimeError(f"partial-checkpoint failure recurred: {observed}")
        if f"auto_resume={VALID}" not in text or (
            f"Loaded checkpoint from {VALID / OPTIMIZER}" not in text
        ):
            raise RuntimeError("Pantry lacks successful step-2 optimizer restore evidence")
        active_steps = sorted(
            path for path in CHECKPOINTS.glob("step_*") if path.is_dir()
        )
        valid_after = [
            path for path in active_steps
            if int(path.name.removeprefix("step_")) > 2
            and all(
                zip_directory_status(path / name).get("valid") is True
                for name in (MODEL, OPTIMIZER)
            )
        ]
        if not valid_after:
            raise RuntimeError("Pantry has not written a valid post-recovery checkpoint")
        payload.update(
            {
                "verified_after_requeue_at": now(),
                "post_recovery_failure_counts": observed,
                "stdout_reinitialized_on_requeue": True,
                "post_recovery_valid_checkpoints": [
                    {"path": str(path), "evidence": file_evidence(path)}
                    for path in valid_after
                ],
                "scheduler_record_after_verified_progress": scheduler_record(),
                "verified_after_requeue": True,
            }
        )
    RECORD.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "[e111-pantry-quarantine] "
        f"verified_after_requeue={payload['verified_after_requeue']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
