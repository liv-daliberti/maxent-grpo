#!/usr/bin/env python3
"""Replace E91's broken resume jobs with state-preserving repaired jobs.

The original runtime restores the semantic tracker before the adaptive RMS
controller. Because the tracker checkpoint contains the evolved coefficient
and a fresh process starts at the registered base coefficient, every resume
fails its strict contract check. This repair reverses only those two restore
operations, preserves the strict cross-state check, and resumes each old
debug_job directory from its retained checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_LEDGER = ROOT / "var/artifacts/e91_falcon_adaptive_semantic_maxent_jobs.json"
REPAIR_LEDGER = ROOT / "var/artifacts/e91_adaptive_semantic_resume_repair_jobs.json"
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e91_adaptive_semantic_resume_repair_20260814.md"
)
ORIGINAL_SNAPSHOT = (
    ROOT
    / "var/artifacts/source_snapshots/"
    "e91_falcon_adaptive_semantic_c34bc3a1141e1631"
)
PATCHED_RELATIVE = Path("src/oat_drgrpo/learner/run.py")
TARGET_STEPS = 3072
REPAIR_TIME_LIMIT = "08:00:00"

OLD_BLOCK = """\
            semantic_shannon_tracker.load_state_dict(saved_semantic_shannon)
            rms_controller = getattr(self, "_semantic_rms_controller", None)
            saved_rms = resume_states.get("semantic_rms_controller_state")
            if rms_controller is not None:
                if not isinstance(saved_rms, dict):
                    raise ValueError(
                        "adaptive semantic run cannot resume without controller state"
                    )
                rms_controller.load_state_dict(saved_rms)
                semantic_shannon_tracker.coefficient = float(
                    rms_controller.current_coefficient
                )
            elif saved_rms is not None:
                raise ValueError(
                    "fixed-coefficient run cannot resume an adaptive checkpoint"
                )
"""

NEW_BLOCK = """\
            rms_controller = getattr(self, "_semantic_rms_controller", None)
            saved_rms = resume_states.get("semantic_rms_controller_state")
            if rms_controller is not None:
                if not isinstance(saved_rms, dict):
                    raise ValueError(
                        "adaptive semantic run cannot resume without controller state"
                    )
                # The tracker stores the coefficient that was active when the
                # checkpoint was written, while a fresh adaptive run starts at
                # the registered base coefficient. Restore the controller
                # first so the tracker's strict contract check compares the
                # two checkpoint states rather than comparing evolved state to
                # the base configuration.
                rms_controller.load_state_dict(saved_rms)
                semantic_shannon_tracker.coefficient = float(
                    rms_controller.current_coefficient
                )
            elif saved_rms is not None:
                raise ValueError(
                    "fixed-coefficient run cannot resume an adaptive checkpoint"
                )
            semantic_shannon_tracker.load_state_dict(saved_semantic_shannon)
"""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def repaired_source() -> tuple[bytes, str, str]:
    source_path = ORIGINAL_SNAPSHOT / PATCHED_RELATIVE
    original = source_path.read_text(encoding="utf-8")
    if original.count(OLD_BLOCK) != 1:
        raise RuntimeError("E91 frozen runtime does not contain the exact broken block")
    patched = original.replace(OLD_BLOCK, NEW_BLOCK, 1).encode("utf-8")
    return patched, sha256_bytes(original.encode("utf-8")), sha256_bytes(patched)


def repaired_snapshot_path(patched_sha256: str) -> Path:
    return (
        ORIGINAL_SNAPSHOT.parent
        / f"e91_falcon_adaptive_semantic_resume_{patched_sha256[:16]}"
    )


def relative_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }


def verify_snapshot(target: Path, patched: bytes) -> None:
    expected_files = relative_files(ORIGINAL_SNAPSHOT)
    actual_files = relative_files(target)
    if actual_files != expected_files:
        raise RuntimeError("repaired snapshot file surface differs from E91")
    for relative in sorted(expected_files):
        old_path = ORIGINAL_SNAPSHOT / relative
        new_path = target / relative
        if relative == PATCHED_RELATIVE:
            if new_path.read_bytes() != patched:
                raise RuntimeError("repaired snapshot has the wrong learner/run.py")
        elif sha256_file(old_path) != sha256_file(new_path):
            raise RuntimeError(f"unexpected repaired snapshot change: {relative}")


def materialize_snapshot(patched: bytes, patched_sha256: str) -> Path:
    target = repaired_snapshot_path(patched_sha256)
    if target.exists():
        verify_snapshot(target, patched)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent))
    )
    try:
        shutil.copytree(ORIGINAL_SNAPSHOT, temporary, dirs_exist_ok=True)
        (temporary / PATCHED_RELATIVE).write_bytes(patched)
        verify_snapshot(temporary, patched)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    verify_snapshot(target, patched)
    return target


def scontrol_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect E91 job {job_id}: {result.stderr.strip()}")
    return result.stdout.strip()


def field(record: str, name: str) -> str:
    marker = f"{name}="
    for token in record.split():
        if token.startswith(marker):
            return token[len(marker) :]
    raise RuntimeError(f"Slurm record lacks {name}")


def submit_command(record: str) -> list[str]:
    marker = " SubmitLine="
    end_marker = " WorkDir="
    if marker not in record or end_marker not in record:
        raise RuntimeError("Slurm record lacks an extractable SubmitLine")
    line = record.split(marker, 1)[1].split(end_marker, 1)[0]
    command = shlex.split(line)
    if not command or command[0] != "sbatch":
        raise RuntimeError("E91 SubmitLine is not an sbatch command")
    return command


def patch_export(
    token: str, *, old_job_id: int, repaired_snapshot: Path
) -> str:
    if not token.startswith("--export="):
        raise RuntimeError("not an sbatch export token")
    parts = token.split("=", 1)[1].split(",")
    old_source = f"OAT_ZERO_SOURCE_ROOT={ORIGINAL_SNAPSHOT / 'src'}"
    old_ops = f"OAT_ZERO_OPS_SNAPSHOT_ROOT={ORIGINAL_SNAPSHOT / 'ops'}"
    new_source = f"OAT_ZERO_SOURCE_ROOT={repaired_snapshot / 'src'}"
    new_ops = f"OAT_ZERO_OPS_SNAPSHOT_ROOT={repaired_snapshot / 'ops'}"
    if old_source not in parts or old_ops not in parts:
        raise RuntimeError("E91 export does not point at its frozen snapshot")
    parts = [
        new_source if part == old_source else new_ops if part == old_ops else part
        for part in parts
        if not part.startswith("OAT_ZERO_FIXED_EXP_SUFFIX=")
    ]
    parts.append(f"OAT_ZERO_FIXED_EXP_SUFFIX=job{old_job_id}")
    return "--export=" + ",".join(parts)


def replacement_command(
    record: str, *, old_job_id: int, repaired_snapshot: Path
) -> list[str]:
    command = submit_command(record)
    output: list[str] = []
    saw_export = saw_time = saw_nice = saw_hold = False
    for token in command:
        if token == "--hold":
            saw_hold = True
            output.append(token)
        elif token.startswith("--job-name="):
            output.append(token + "-rr")
        elif token.startswith("--export="):
            output.append(
                patch_export(
                    token,
                    old_job_id=old_job_id,
                    repaired_snapshot=repaired_snapshot,
                )
            )
            saw_export = True
        elif token.startswith("--time="):
            output.append(f"--time={REPAIR_TIME_LIMIT}")
            saw_time = True
        elif token.startswith("--nice="):
            output.append("--nice=0")
            saw_nice = True
        else:
            output.append(token)
    if not (saw_hold and saw_export and saw_time and saw_nice):
        raise RuntimeError("E91 command lacks hold/export/time/nice safeguards")
    return output


def audit_replacement(
    new_job_id: int,
    *,
    old_job_id: int,
    old_record: str,
    repaired_snapshot: Path,
) -> None:
    record = scontrol_record(new_job_id)
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Nice=0",
        f"ReqNodeList={field(old_record, 'ReqNodeList')}",
        f"TimeLimit={REPAIR_TIME_LIMIT}",
        f"OAT_ZERO_SOURCE_ROOT={repaired_snapshot / 'src'}",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={repaired_snapshot / 'ops'}",
        f"OAT_ZERO_FIXED_EXP_SUFFIX=job{old_job_id}",
    )
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"replacement job {new_job_id} lacks {missing}")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not PRIMARY_LEDGER.is_file() or not PROTOCOL.is_file():
        raise SystemExit("E91 primary ledger or resume-repair protocol is missing")
    if REPAIR_LEDGER.exists():
        raise SystemExit(f"refusing duplicate E91 resume repair: {REPAIR_LEDGER}")

    primary = json.loads(PRIMARY_LEDGER.read_text(encoding="utf-8"))
    if Path(primary["snapshot_root"]) != ORIGINAL_SNAPSHOT:
        raise SystemExit("E91 primary ledger no longer points at the original snapshot")
    patched, original_sha256, patched_sha256 = repaired_source()
    target = repaired_snapshot_path(patched_sha256)

    candidates: list[tuple[dict[str, Any], str, list[str]]] = []
    for run in primary["runs"]:
        old_job_id = int(run["job_id"])
        try:
            record = scontrol_record(old_job_id)
        except RuntimeError:
            continue
        if "JobState=PENDING" not in record:
            continue
        if "Reason=JobHeldUser" not in record:
            raise SystemExit(f"E91 job {old_job_id} is not safely held")
        command = replacement_command(
            record,
            old_job_id=old_job_id,
            repaired_snapshot=target,
        )
        candidates.append((run, record, command))
    if len(candidates) != 15:
        raise SystemExit(f"expected 15 held E91 repair candidates, found {len(candidates)}")

    if not args.submit:
        for run, record, command in candidates:
            print(
                f"{run['domain']:<16} s{run['seed']} old={run['job_id']} "
                f"node={field(record, 'ReqNodeList')} "
                f"checkpoint_suffix=job{run['job_id']} "
                f"command={shlex.join(command[:7])}"
            )
        print(
            f"[e91-resume-repair] dry_run=True cells={len(candidates)} "
            f"snapshot={target} patched_sha256={patched_sha256}"
        )
        return 0

    target = materialize_snapshot(patched, patched_sha256)
    replacements: list[dict[str, Any]] = []
    new_job_ids: list[int] = []
    try:
        for run, old_record, command in candidates:
            result = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            raw_job_id = result.stdout.strip().split(";", 1)[0]
            if not raw_job_id.isdigit():
                raise RuntimeError(f"invalid replacement job id: {result.stdout!r}")
            new_job_id = int(raw_job_id)
            new_job_ids.append(new_job_id)
            old_job_id = int(run["job_id"])
            audit_replacement(
                new_job_id,
                old_job_id=old_job_id,
                old_record=old_record,
                repaired_snapshot=target,
            )
            replacements.append(
                {
                    "domain": run["domain"],
                    "seed": int(run["seed"]),
                    "old_job_id": old_job_id,
                    "new_job_id": new_job_id,
                    "run_dir": run["run_dir"],
                    "resume_fixed_exp_suffix": f"job{old_job_id}",
                    "node": field(old_record, "ReqNodeList"),
                }
            )
    except Exception:
        if new_job_ids:
            subprocess.run(
                ["scancel", ",".join(str(job_id) for job_id in new_job_ids)],
                check=False,
            )
        raise

    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    repair = {
        "schema": "e91_adaptive_semantic_resume_repair_jobs_v1",
        "created_at": timestamp,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "original_snapshot_root": str(ORIGINAL_SNAPSHOT),
        "repaired_snapshot_root": str(target),
        "patched_file": str(PATCHED_RELATIVE),
        "original_file_sha256": original_sha256,
        "repaired_file_sha256": patched_sha256,
        "scientific_change": "none",
        "runtime_change": "restore RMS controller before strict tracker state check",
        "target_steps": TARGET_STEPS,
        "time_limit": REPAIR_TIME_LIMIT,
        "replacements": replacements,
    }
    atomic_json(REPAIR_LEDGER, repair)

    backup = PRIMARY_LEDGER.with_name(
        "e91_falcon_adaptive_semantic_maxent_jobs.pre_resume_repair.json"
    )
    if backup.exists():
        raise RuntimeError(f"refusing to overwrite E91 ledger backup: {backup}")
    shutil.copy2(PRIMARY_LEDGER, backup)
    updated = copy.deepcopy(primary)
    by_old = {item["old_job_id"]: item for item in replacements}
    for run in updated["runs"]:
        old_job_id = int(run["job_id"])
        if old_job_id not in by_old:
            continue
        replacement = by_old[old_job_id]
        run["supersedes_job_id"] = old_job_id
        run["job_id"] = replacement["new_job_id"]
        run["resume_fixed_exp_suffix"] = replacement["resume_fixed_exp_suffix"]
    updated["snapshot_root"] = str(target)
    updated["resume_repair"] = {
        "ledger": str(REPAIR_LEDGER.relative_to(ROOT)),
        "repaired_snapshot_root": str(target),
        "patched_file": str(PATCHED_RELATIVE),
        "repaired_file_sha256": patched_sha256,
        "created_at": timestamp,
    }
    atomic_json(PRIMARY_LEDGER, updated)

    old_ids = [item["old_job_id"] for item in replacements]
    subprocess.run(
        ["scancel", ",".join(str(job_id) for job_id in old_ids)], check=True
    )
    subprocess.run(
        ["scontrol", "release", ",".join(str(job_id) for job_id in new_job_ids)],
        check=True,
    )
    print(
        f"[e91-resume-repair] replacements={len(replacements)} "
        f"released={len(new_job_ids)} snapshot={target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
