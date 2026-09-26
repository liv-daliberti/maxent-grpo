#!/usr/bin/env python3
"""Replace E98-R1 Pantry jobs with action-surface-compatible frozen jobs."""

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
PRIMARY_LEDGER = ROOT / "var/artifacts/e98r1_sparse_rlep_dr_05b_jobs.json"
REPAIR_LEDGER = (
    ROOT / "var/artifacts/e98r1_pantry_action_surface_repair_jobs.json"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/e98r1_pantry_action_surface_repair_20260814.md"
)
SMOKE_AUDIT = "ops/exp_scaling/audit_e98r1_sparse_rlep_smoke.py"
PATCHED_FILES = (
    Path("src/oat_drgrpo/pantry_support_action.py"),
    Path("src/oat_drgrpo/canonical_actions.py"),
    Path("src/oat_drgrpo/learner/grpo.py"),
)
PATCH_NEEDLES = {
    PATCHED_FILES[0]: "pantry_support_mask_from_allocation",
    PATCHED_FILES[1]: "canonical_action_code_from_verified_response",
    PATCHED_FILES[2]: "canonical_action_code_token_ids",
}
SMOKE_ROOT = (
    ROOT / "var/data/e98r1_sparse_rlep_pantry_surface_repair_smoke_s44"
)
SEEDS = (43, 44, 45, 46, 47)
OLD_FAILED = {43: 30538126, 44: 30538127}
OLD_HELD = {45: 30538128, 46: 30538129, 47: 30538130}
TARGET_STEPS = 3072


def sha256_file(path: Path) -> str:
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


def relative_files(root: Path) -> set[Path]:
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }


def snapshot_path(original: Path) -> Path:
    manifest = "\n".join(
        f"{relative}:{sha256_file(ROOT / relative)}" for relative in PATCHED_FILES
    )
    digest = hashlib.sha256(manifest.encode("utf-8")).hexdigest()[:16]
    return original.parent / f"e98r1_pantry_action_surface_{digest}"


def verify_snapshot(original: Path, repaired: Path) -> None:
    expected = relative_files(original)
    if relative_files(repaired) != expected:
        raise RuntimeError("E98-R1 repaired snapshot file surface drifted")
    for relative in expected:
        old_path, new_path = original / relative, repaired / relative
        if relative in PATCHED_FILES:
            if new_path.read_bytes() != (ROOT / relative).read_bytes():
                raise RuntimeError(f"repaired snapshot has wrong {relative}")
            needle = PATCH_NEEDLES[relative]
            if needle not in new_path.read_text(encoding="utf-8"):
                raise RuntimeError(f"repaired snapshot lacks {needle}")
        elif sha256_file(old_path) != sha256_file(new_path):
            raise RuntimeError(f"unexpected E98-R1 snapshot change: {relative}")


def materialize_snapshot(original: Path) -> tuple[Path, dict[str, Any]]:
    repaired = snapshot_path(original)
    if not repaired.exists():
        repaired.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{repaired.name}.", dir=str(repaired.parent))
        )
        try:
            shutil.copytree(original, temporary, dirs_exist_ok=True)
            for relative in PATCHED_FILES:
                shutil.copy2(ROOT / relative, temporary / relative)
            verify_snapshot(original, temporary)
            os.replace(temporary, repaired)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    verify_snapshot(original, repaired)
    manifest = {
        str(relative): {
            "original_sha256": sha256_file(original / relative),
            "repaired_sha256": sha256_file(repaired / relative),
        }
        for relative in PATCHED_FILES
    }
    if any(row["original_sha256"] == row["repaired_sha256"] for row in manifest.values()):
        raise RuntimeError("an intended E98-R1 repair file did not change")
    return repaired, manifest


def submit_line(record: str) -> list[str]:
    marker, end_marker = " SubmitLine=", " WorkDir="
    if marker not in record or end_marker not in record:
        raise RuntimeError("E98-R1 scheduler record lacks its SubmitLine")
    command = shlex.split(record.split(marker, 1)[1].split(end_marker, 1)[0])
    if not command or command[0] != "sbatch":
        raise RuntimeError("E98-R1 SubmitLine is not an sbatch command")
    return command


def patch_export(token: str, updates: dict[str, str]) -> str:
    if not token.startswith("--export="):
        raise RuntimeError("not an sbatch export token")
    fields = token.split("=", 1)[1].split(",")
    output: list[str] = []
    remaining = dict(updates)
    for field in fields:
        name = field.split("=", 1)[0]
        if name in remaining:
            output.append(f"{name}={remaining.pop(name)}")
        else:
            output.append(field)
    if remaining:
        raise RuntimeError(f"E98-R1 export lacks keys {sorted(remaining)}")
    return "--export=" + ",".join(output)


def placement(seed: int) -> tuple[str, str, str, str]:
    if seed in (43, 44):
        return "cs", "allcs", "gpu:a5000:1", "node202,node203,node204"
    return "mltheory", "mltheory", "gpu:a100:1", "node302"


def replacement_command(
    run: dict[str, Any],
    *,
    repaired_snapshot: Path,
    dependency: str,
) -> list[str]:
    seed = int(run["seed"])
    partition, account, gres, nodes = placement(seed)
    source = submit_line(str(run["held_scheduler_record"]))
    updates = {
        "OAT_ZERO_SOURCE_ROOT": str(repaired_snapshot / "src"),
        "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(repaired_snapshot / "ops"),
    }
    command: list[str] = []
    saw = {"name": False, "export": False, "partition": False, "account": False, "gres": False}
    for token in source:
        if token == "--hold" or token.startswith("--dependency="):
            continue
        if token.startswith("--job-name="):
            command.append(token + "-rr")
            saw["name"] = True
        elif token.startswith("--export="):
            command.append(patch_export(token, updates))
            saw["export"] = True
        elif token.startswith("--partition="):
            command.append(f"--partition={partition}")
            saw["partition"] = True
        elif token.startswith("--account="):
            command.append(f"--account={account}")
            saw["account"] = True
        elif token.startswith("--gres="):
            command.append(f"--gres={gres}")
            saw["gres"] = True
        elif token.startswith("--nodelist="):
            continue
        else:
            command.append(token)
    if not all(saw.values()) or "--parsable" not in command:
        raise RuntimeError(f"E98-R1 replacement command lacks safeguards: {saw}")
    command.insert(command.index("--parsable") + 1, "--hold")
    command.insert(-1, f"--dependency=afterok:{dependency}")
    command.insert(-1, f"--nodelist={nodes}")
    return command


def smoke_command(run: dict[str, Any], *, repaired_snapshot: Path) -> list[str]:
    source = submit_line(str(run["held_scheduler_record"]))
    updates = {
        "OAT_ZERO_SOURCE_ROOT": str(repaired_snapshot / "src"),
        "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(repaired_snapshot / "ops"),
        "SAVE_PATH": str(SMOKE_ROOT),
        "RUN_STAMP": "e98r1_sparse_rlep_pantry_surface_repair_smoke_s44",
        "OAT_ZERO_MAX_TRAIN": "32",
        "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
        "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL": "32",
        "OAT_ZERO_SAVE_STEPS": "32",
        "OAT_ZERO_SAVE_FROM": "32",
        "OAT_ZERO_AUTO_RESUME": "0",
        "OAT_ZERO_WATCHDOG_REQUEUE": "0",
    }
    command: list[str] = []
    for token in source:
        if token == "--hold" or token.startswith("--dependency="):
            continue
        if token.startswith("--job-name="):
            command.append("--job-name=e98r1-pantry-surface-smoke")
        elif token.startswith("--export="):
            command.append(patch_export(token, updates))
        elif token.startswith("--partition="):
            command.append("--partition=all")
        elif token.startswith("--account="):
            command.append("--account=allcs")
        elif token.startswith("--gres="):
            command.append("--gres=gpu:a6000:1")
        elif token.startswith("--time="):
            command.append("--time=01:00:00")
        elif token.startswith("--mem="):
            command.append("--mem=32G")
        elif token.startswith("--nodelist="):
            continue
        else:
            command.append(token)
    command.insert(command.index("--parsable") + 1, "--hold")
    command.insert(
        -1, "--nodelist=node103,node104,node205,node206,node207,node805"
    )
    return command


def smoke_audit_command(
    *, repaired_snapshot: Path, smoke_job_id: int
) -> list[str]:
    python = ROOT / "var/seed_paper_eval/paper310/bin/python"
    script = repaired_snapshot / SMOKE_AUDIT
    wrapped = shlex.join(
        [
            str(python),
            str(script),
            "--run-root",
            str(SMOKE_ROOT),
            "--expected-terminal-step",
            "32",
        ]
    )
    return [
        "sbatch",
        "--parsable",
        "--hold",
        "--job-name=e98r1-pantry-surface-audit",
        f"--dependency=afterok:{smoke_job_id}",
        f"--export=ALL,PYTHONPATH={repaired_snapshot / 'src'}",
        "--partition=all",
        "--account=allcs",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:15:00",
        "--nice=0",
        f"--output={ROOT / 'var/artifacts/logs'}/%x-%j.out",
        f"--error={ROOT / 'var/artifacts/logs'}/%x-%j.err",
        "--wrap",
        wrapped,
    ]


def scontrol_record(job_id: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", str(job_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"cannot inspect job {job_id}")
    return result.stdout.strip()


def submit_held(command: list[str]) -> int:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "E98-R1 repair sbatch failed")
    return int(result.stdout.strip().split(";", 1)[0])


def audit_held(job_id: int, expected: tuple[str, ...]) -> str:
    record = scontrol_record(job_id)
    required = ("JobState=PENDING", "Reason=JobHeldUser", *expected)
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"E98-R1 repair job {job_id} lacks {missing}")
    return record


def scheduler_state(job_id: int) -> str:
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", str(job_id), "--format=State", "-P"],
        check=False,
        capture_output=True,
        text=True,
    )
    values = [line.strip().split("+", 1)[0] for line in result.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError(f"cannot resolve old E98-R1 job {job_id}")
    return values[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not PRIMARY_LEDGER.is_file() or not PROTOCOL.is_file():
        raise SystemExit("E98-R1 primary ledger or repair protocol is missing")
    if REPAIR_LEDGER.exists():
        raise SystemExit(f"refusing duplicate E98-R1 repair: {REPAIR_LEDGER}")
    if SMOKE_ROOT.exists():
        raise SystemExit(f"refusing existing repair smoke root: {SMOKE_ROOT}")

    primary = json.loads(PRIMARY_LEDGER.read_text(encoding="utf-8"))
    original_primary = copy.deepcopy(primary)
    original_snapshot = Path(primary["snapshot_root"]).resolve()
    repaired_snapshot, patch_manifest = materialize_snapshot(original_snapshot)
    runs = {
        int(row["seed"]): row
        for row in primary["runs"]
        if row["domain"] == "pantry_plan"
    }
    if set(runs) != set(SEEDS):
        raise SystemExit("E98-R1 primary ledger lacks the five Pantry cells")
    for seed, expected_job in (OLD_FAILED | OLD_HELD).items():
        if int(runs[seed]["job_id"]) != expected_job:
            raise SystemExit(f"E98-R1 Pantry s{seed} job identity drifted")
        expected_state = "FAILED" if seed in OLD_FAILED else "PENDING"
        state = scheduler_state(expected_job)
        if state != expected_state:
            raise SystemExit(
                f"E98-R1 Pantry s{seed} is {state}, expected {expected_state}"
            )
        if list(Path(runs[seed]["run_dir"]).glob("**/checkpoints/step_*")):
            raise SystemExit(f"E98-R1 Pantry s{seed} unexpectedly has a checkpoint")
        if seed in OLD_HELD:
            record = scontrol_record(expected_job)
            if "Reason=JobHeldUser" not in record:
                raise SystemExit(f"old E98-R1 Pantry s{seed} is not safely held")

    smoke = smoke_command(runs[44], repaired_snapshot=repaired_snapshot)
    if not args.submit:
        print(shlex.join(smoke))
        print("<audit depends afterok on repair smoke>")
        for seed in SEEDS:
            print(f"<Pantry s{seed} depends afterok on repair smoke audit>")
        print(f"snapshot={repaired_snapshot}")
        return 0

    submitted: list[int] = []
    committed = False
    try:
        smoke_id = submit_held(smoke)
        submitted.append(smoke_id)
        smoke_record = audit_held(
            smoke_id,
            (
                "JobName=e98r1-pantry-surface-smoke",
                "ReqNodeList=node103,node104,node205,node206,node207,node805",
                "gres/gpu:a6000=1",
                "OAT_ZERO_CANONICAL_ACTION_TASK=pantry_support_mask",
                "OAT_ZERO_MAX_TRAIN=32",
                f"OAT_ZERO_SOURCE_ROOT={repaired_snapshot / 'src'}",
            ),
        )
        audit_id = submit_held(
            smoke_audit_command(
                repaired_snapshot=repaired_snapshot,
                smoke_job_id=smoke_id,
            )
        )
        submitted.append(audit_id)
        audit_record = audit_held(
            audit_id,
            (
                "JobName=e98r1-pantry-surface-audit",
                f"Dependency=afterok:{smoke_id}",
                "--expected-terminal-step",
                str(SMOKE_ROOT),
            ),
        )

        replacements: list[dict[str, Any]] = []
        for seed in SEEDS:
            command = replacement_command(
                runs[seed],
                repaired_snapshot=repaired_snapshot,
                dependency=str(audit_id),
            )
            job_id = submit_held(command)
            submitted.append(job_id)
            _partition, _account, gres, nodes = placement(seed)
            normalized_nodes = "node[202-204]" if seed in (43, 44) else nodes
            gres_resource, gres_count = gres.rsplit(":", 1)
            normalized_gres = (
                f"gres/{gres_resource}={gres_count}"
                if seed in (43, 44)
                else "gres/gpu=1"
            )
            record = audit_held(
                job_id,
                (
                    f"JobName=e98r1-pantry-s{seed}-rr",
                    f"Dependency=afterok:{audit_id}",
                    f"ReqNodeList={normalized_nodes}",
                    normalized_gres,
                    f"OAT_ZERO_SEED={seed}",
                    "OAT_ZERO_RLEP_REPLAY_COUNT=2",
                    "OAT_ZERO_RLEP_SPARSE_FALLBACK=1",
                    f"OAT_ZERO_SOURCE_ROOT={repaired_snapshot / 'src'}",
                    f"OAT_ZERO_OPS_SNAPSHOT_ROOT={repaired_snapshot / 'ops'}",
                ),
            )
            replacements.append(
                {
                    "seed": seed,
                    "old_job_id": int(runs[seed]["job_id"]),
                    "old_state": scheduler_state(int(runs[seed]["job_id"])),
                    "new_job_id": job_id,
                    "placement": {
                        "partition": placement(seed)[0],
                        "account": placement(seed)[1],
                        "gres": placement(seed)[2],
                        "nodes": placement(seed)[3],
                    },
                    "held_scheduler_record": record,
                }
            )

        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        for replacement in replacements:
            run = runs[int(replacement["seed"])]
            repair = {
                "reason": "offline Pantry witnesses lacked canonical action serialization",
                "protocol": str(PROTOCOL),
                "submitted_at": timestamp,
                "old_job_id": replacement["old_job_id"],
                "old_state": replacement["old_state"],
                "new_job_id": replacement["new_job_id"],
                "snapshot_root": str(repaired_snapshot),
                "smoke_job_id": smoke_id,
                "smoke_audit_job_id": audit_id,
                "placement": replacement["placement"],
                "held_scheduler_record": replacement["held_scheduler_record"],
            }
            run.setdefault("repair_history", []).append(repair)
            run["replaced_job_ids"] = [
                *run.get("replaced_job_ids", []),
                replacement["old_job_id"],
            ]
            run["job_id"] = replacement["new_job_id"]
            run["held_scheduler_record"] = replacement["held_scheduler_record"]
            run["snapshot_root"] = str(repaired_snapshot)
            run["smoke_audit_dependency_job_id"] = audit_id

        payload = {
            "schema": "e98r1_pantry_action_surface_repair_jobs_v1",
            "released": False,
            "protocol": str(PROTOCOL),
            "primary_ledger": str(PRIMARY_LEDGER),
            "original_snapshot": str(original_snapshot),
            "repaired_snapshot": str(repaired_snapshot),
            "patch_manifest": patch_manifest,
            "smoke": {
                "job_id": smoke_id,
                "audit_job_id": audit_id,
                "run_root": str(SMOKE_ROOT),
                "seed": 44,
                "target_steps": 32,
                "held_scheduler_record": smoke_record,
                "held_audit_scheduler_record": audit_record,
            },
            "replacements": replacements,
            "old_pending_jobs_to_cancel": sorted(OLD_HELD.values()),
        }
        primary.setdefault("repair_amendments", []).append(
            {
                "protocol": str(PROTOCOL),
                "repair_ledger": str(REPAIR_LEDGER),
                "snapshot_root": str(repaired_snapshot),
                "domains": ["pantry_plan"],
                "seeds": list(SEEDS),
            }
        )
        atomic_json(PRIMARY_LEDGER, primary)
        atomic_json(REPAIR_LEDGER, payload)
        for old_job_id in OLD_HELD.values():
            subprocess.run(["scancel", str(old_job_id)], check=True)
        committed = True
        for job_id in submitted:
            subprocess.run(["scontrol", "release", str(job_id)], check=True)
        payload["released"] = True
        payload["cancelled_old_jobs"] = sorted(OLD_HELD.values())
        atomic_json(REPAIR_LEDGER, payload)
    except Exception:
        if not committed:
            for job_id in submitted:
                subprocess.run(["scancel", str(job_id)], check=False)
            atomic_json(PRIMARY_LEDGER, original_primary)
        raise

    print(
        f"released E98-R1 Pantry smoke {smoke_id}, audit {audit_id}, "
        + "replacements "
        + " ".join(str(row["new_job_id"]) for row in replacements)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
