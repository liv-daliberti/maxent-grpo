#!/usr/bin/env python3
"""Launch identity-bound ConstructiveCode v5 smoke or final cells."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "ops"
SRC = ROOT / "src"
for path in (OPS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_constructive_code_v5 as train  # noqa: E402


PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
TRAINER = ROOT / "ops/train_constructive_code_v5.py"
AUDITOR = ROOT / "ops/audit_constructive_code_v5.py"
TRAIN_BATCH = ROOT / "ops/slurm/train_constructive_code_v5.slurm"
AUDIT_BATCH = ROOT / "ops/slurm/audit_constructive_code_v5.slurm"
STAGE_DEP_BATCH = (
    ROOT / "ops/slurm/launch_constructive_code_v5_stage_b_after_smoke.slurm"
)
V5_EVALUATOR = ROOT / "ops/evaluate_constructive_code_v5_coder_viability.py"
V4_EVALUATOR = ROOT / "ops/evaluate_constructive_code_v4_coder_viability.py"
V3_EVALUATOR = ROOT / "ops/evaluate_constructive_code_v3_coder_viability.py"
GATE_IDENTITY = ROOT / "var/artifacts/constructive_code_v5_gate_identity.json"
GATE_AUDIT = ROOT / "var/artifacts/constructive_code_v5_gate_audit.json"
VIABILITY = ROOT / "var/artifacts/constructive_code_v5_coder_05b_viability.json"
PAIRED_PROTOCOL = (
    ROOT
    / "paper/preregistration/constructive_code_v5_paired_online_smoke_20260730.md"
)
STAGE_PROTOCOL = (
    ROOT
    / "paper/preregistration/constructive_code_v5_stage_b_05b_12pass_20260730.md"
)
PAIRED_IDENTITY = ROOT / "var/artifacts/constructive_code_v5_paired_smoke_identity.json"
PAIRED_SUBMISSION = ROOT / "var/artifacts/constructive_code_v5_paired_smoke_submission.json"
PAIRED_AUDIT = ROOT / "var/artifacts/constructive_code_v5_paired_smoke_audit.json"
STAGE_IDENTITY = ROOT / "var/artifacts/constructive_code_v5_stage_b_identity.json"
STAGE_SUBMISSION = ROOT / "var/artifacts/constructive_code_v5_stage_b_submission.json"
STAGE_AUDIT = ROOT / "var/artifacts/constructive_code_v5_stage_b_audit.json"
STAGE_MANIFEST = (
    ROOT
    / "var/artifacts/cce70_clean_stage_b_05b_12pass_comparative_jobs.tsv"
)
MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/"
    "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
)
IMAGE = ROOT / "var/images/python-3.10-slim-c1e4e6c01eb4.sqsh"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shell_tree_sha256(root: Path) -> str:
    lines = bytearray()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = "./" + path.relative_to(root).as_posix()
        lines.extend(sha256_file(path).encode("ascii"))
        lines.extend(b"  ")
        lines.extend(relative.encode("utf-8"))
        lines.extend(b"\n")
    if not lines:
        raise ValueError(f"empty execution tree: {root}")
    return hashlib.sha256(lines).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str], *, timeout: int = 120, capture: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )
    return result.stdout.strip() if capture else ""


def sbatch(*arguments: str) -> int:
    raw = run(["sbatch", "--parsable", *arguments])
    value = raw.split(";", 1)[0]
    if not value.isdigit():
        raise RuntimeError(f"invalid sbatch job id: {raw!r}")
    return int(value)


def scheduler_record(job_id: int) -> str:
    return run(["scontrol", "show", "job", str(job_id), "-o"])


def cancel(job_ids: list[int]) -> None:
    if job_ids:
        subprocess.run(
            ["scancel", *map(str, job_ids)],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=60,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=train.MODES)
    parser.add_argument("phase", choices=("config", "run"))
    return parser.parse_args()


def mode_paths(mode: str) -> tuple[Path, Path, Path, Path]:
    if mode == train.PAIRED_MODE:
        return PAIRED_PROTOCOL, PAIRED_IDENTITY, PAIRED_SUBMISSION, PAIRED_AUDIT
    return STAGE_PROTOCOL, STAGE_IDENTITY, STAGE_SUBMISSION, STAGE_AUDIT


def check_prerequisites(mode: str, *, outcomes: bool) -> tuple[dict[str, Any], Path, Path]:
    protocol, identity, submission, audit = mode_paths(mode)
    required = (
        PYTHON,
        TRAINER,
        AUDITOR,
        TRAIN_BATCH,
        AUDIT_BATCH,
        STAGE_DEP_BATCH,
        V5_EVALUATOR,
        V4_EVALUATOR,
        V3_EVALUATOR,
        GATE_IDENTITY,
        protocol,
        MODEL / "config.json",
        MODEL / "model.safetensors",
        MODEL / "tokenizer.json",
        IMAGE,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    gate_identity = read_json(GATE_IDENTITY)
    source_hash = str(gate_identity.get("source_hash") or "")
    gate_execution_hash = str(gate_identity.get("execution_hash") or "")
    if len(source_hash) != 64 or len(gate_execution_hash) != 64:
        raise ValueError("ConstructiveCode gate identity lacks snapshot hashes")
    source_root = (
        ROOT / f"var/artifacts/source_snapshots/constructive_code_v5_source_{source_hash}/src"
    )
    gate_ops_root = (
        ROOT
        / f"var/artifacts/source_snapshots/constructive_code_v5_gate_{gate_execution_hash}"
    )
    if shell_tree_sha256(source_root) != source_hash:
        raise ValueError("ConstructiveCode gate source snapshot hash drift")
    if shell_tree_sha256(gate_ops_root) != gate_execution_hash:
        raise ValueError("ConstructiveCode gate ops snapshot hash drift")
    if outcomes:
        for path in (GATE_AUDIT, VIABILITY):
            if not path.is_file():
                raise FileNotFoundError(path)
        gate = read_json(GATE_AUDIT)
        viability = read_json(VIABILITY)
        if (
            gate.get("status") != "pass"
            or gate.get("expected_replay_count") != 2304
            or gate.get("observed_replay_count") != 2304
            or gate.get("violations") not in ([], None)
            or viability.get("status") != "pass"
            or viability.get("decision")
            != "eligible_for_paired_online_training_smoke"
            or viability.get("summary", {}).get("terminal_worker_records") != 256
            or viability.get("summary", {}).get("prefix_success_tasks", 0) < 1
            or viability.get("summary", {}).get("multimode_tasks", 0) < 1
            or viability.get("hard_violations") not in ([], None)
        ):
            raise ValueError("ConstructiveCode gate/viability antecedent is not pass")
        if mode == train.STAGE_B_MODE:
            if not PAIRED_AUDIT.is_file() or not PAIRED_IDENTITY.is_file():
                raise FileNotFoundError("ConstructiveCode paired audit/identity missing")
            paired_audit = read_json(PAIRED_AUDIT)
            if (
                paired_audit.get("status") != "pass"
                or paired_audit.get("decision")
                != "eligible_for_ten_constructive_code_stage_b_jobs"
            ):
                raise ValueError("ConstructiveCode paired audit did not authorize Stage B")
            paired_identity = read_json(PAIRED_IDENTITY)
            for name, path in (("trainer", TRAINER), ("auditor", AUDITOR)):
                if paired_identity.get("core_sha256", {}).get(name) != sha256_file(path):
                    raise ValueError(
                        f"ConstructiveCode Stage B {name} differs from qualified smoke"
                    )
    fresh_launch = [identity, submission, audit]
    if mode == train.STAGE_B_MODE:
        fresh_launch.append(STAGE_MANIFEST)
    if outcomes and any(path.exists() for path in fresh_launch):
        raise FileExistsError("fresh ConstructiveCode launch artifacts are required")
    return gate_identity, source_root, gate_ops_root


def execution_snapshot(mode: str) -> tuple[str, Path, dict[str, str]]:
    protocol, _identity, _submission, _audit = mode_paths(mode)
    files = {
        "train_constructive_code_v5.py": TRAINER,
        "audit_constructive_code_v5.py": AUDITOR,
        "train_constructive_code_v5.slurm": TRAIN_BATCH,
        "audit_constructive_code_v5.slurm": AUDIT_BATCH,
        "launch_constructive_code_v5_stage_b_after_smoke.slurm": STAGE_DEP_BATCH,
        "evaluate_constructive_code_v5_coder_viability.py": V5_EVALUATOR,
        "evaluate_constructive_code_v4_coder_viability.py": V4_EVALUATOR,
        "evaluate_constructive_code_v3_coder_viability.py": V3_EVALUATOR,
        protocol.name: protocol,
        "launch_constructive_code_v5_experiment.py": Path(__file__).resolve(),
    }
    snapshots = ROOT / "var/artifacts/source_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".constructive-v5-experiment.", dir=snapshots)
    )
    for name, source in files.items():
        shutil.copy2(source, temporary / name)
    execution_hash = shell_tree_sha256(temporary)
    destination = snapshots / f"constructive_code_v5_experiment_{execution_hash}"
    if destination.exists():
        if shell_tree_sha256(destination) != execution_hash:
            raise ValueError("ConstructiveCode execution snapshot collision")
        shutil.rmtree(temporary)
    else:
        temporary.replace(destination)
    hashes = {
        "trainer": sha256_file(TRAINER),
        "auditor": sha256_file(AUDITOR),
        "train_batch": sha256_file(TRAIN_BATCH),
        "audit_batch": sha256_file(AUDIT_BATCH),
    }
    return execution_hash, destination, hashes


def export_string(values: Mapping[str, Any]) -> str:
    for key, value in values.items():
        if "," in str(value) or "\n" in str(value):
            raise ValueError(f"unsafe Slurm export value: {key}")
    return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def launch() -> None:
    args = parse_args()
    protocol, identity_path, submission_path, audit_path = mode_paths(args.mode)
    gate_identity, source_root, gate_ops_root = check_prerequisites(
        args.mode, outcomes=args.phase == "run"
    )
    run([str(PYTHON), "-m", "py_compile", str(TRAINER), str(AUDITOR), str(Path(__file__).resolve())])
    run(["bash", "-n", str(TRAIN_BATCH), str(AUDIT_BATCH)])
    if args.phase == "config":
        print(f"[constructive-v5-launch] {args.mode} configuration passed; no jobs submitted")
        return

    execution_hash, execution_root, core_hashes = execution_snapshot(args.mode)
    source_hash = str(gate_identity["source_hash"])
    common = {
        "ROOT_DIR": ROOT,
        "OAT_ZERO_SOURCE_ROOT": source_root,
        "OAT_ZERO_GATE_OPS_ROOT": gate_ops_root,
        "OAT_ZERO_EXECUTION_ROOT": execution_root,
        "OAT_ZERO_SOURCE_HASH": source_hash,
        "OAT_ZERO_EXECUTION_HASH": execution_hash,
        "OAT_ZERO_CONSTRUCTIVE_MODE": args.mode,
        "OAT_ZERO_CONSTRUCTIVE_PROTOCOL": execution_root / protocol.name,
        "OAT_ZERO_CONSTRUCTIVE_IDENTITY": identity_path,
    }
    specs = []
    if args.mode == train.PAIRED_MODE:
        for arm in train.ARMS:
            stem = f"constructive_code_v5_paired_smoke_{arm}"
            specs.append((arm, arm, 78101, stem))
    else:
        for arm in train.ARMS:
            for seed in (43, 44, 45, 46, 47):
                label = f"{arm}/s{seed}"
                stem = f"constructive_code_v5_stage_b_{arm}_s{seed}"
                specs.append((label, arm, seed, stem))

    submitted: list[int] = []
    records: dict[str, dict[str, Any]] = {}
    try:
        for label, arm, seed, stem in specs:
            receipt = ROOT / f"var/artifacts/{stem}_receipt.json"
            candidates = ROOT / f"var/artifacts/{stem}_candidates.jsonl"
            evaluations = ROOT / f"var/artifacts/{stem}_evaluations.jsonl"
            if args.mode == train.PAIRED_MODE:
                metrics = ROOT / f"var/artifacts/{stem}_metrics.jsonl"
                run_dir = None
            else:
                run_stamp = f"cce70_clean_stage_b_05b_12pass_{arm}_s{seed}"
                run_dir = ROOT / f"var/data/xdr_constructive_code_{run_stamp}"
                metrics = None
            fresh = [receipt, candidates]
            if metrics is not None:
                fresh.append(metrics)
            if run_dir is not None:
                fresh.append(run_dir)
            if args.mode == train.STAGE_B_MODE:
                fresh.append(evaluations)
            if any(path.exists() for path in fresh):
                raise FileExistsError(f"fresh ConstructiveCode cell outputs required: {label}")
            cell_export = {
                **common,
                "OAT_ZERO_CONSTRUCTIVE_ARM": arm,
                "OAT_ZERO_CONSTRUCTIVE_SEED": seed,
                "OAT_ZERO_CONSTRUCTIVE_RECEIPT": receipt,
                "OAT_ZERO_CONSTRUCTIVE_CANDIDATES": candidates,
            }
            if metrics is not None:
                cell_export["OAT_ZERO_CONSTRUCTIVE_METRICS"] = metrics
            if run_dir is not None:
                cell_export["OAT_ZERO_CONSTRUCTIVE_RUN_DIR"] = run_dir
            if args.mode == train.STAGE_B_MODE:
                cell_export.update(
                    {
                        "OAT_ZERO_CONSTRUCTIVE_PAIRED_AUDIT": PAIRED_AUDIT,
                        "OAT_ZERO_CONSTRUCTIVE_EVALUATIONS": evaluations,
                    }
                )
            job_id = sbatch(
                "--hold",
                f"--job-name=constructive-v5-{args.mode}-{arm}-s{seed}",
                "--time=12:00:00" if args.mode == train.PAIRED_MODE else "--time=7-00:00:00",
                f"--export={export_string(cell_export)}",
                str(execution_root / TRAIN_BATCH.name),
            )
            submitted.append(job_id)
            if run_dir is not None:
                metrics = run_dir / f"debug_job{job_id}/train_metrics.jsonl"
            assert metrics is not None
            records[label] = {
                "job_id": job_id,
                "source_hash": source_hash,
                "execution_hash": execution_hash,
                "receipt": relative(receipt),
                "metrics": relative(metrics),
                "candidates": relative(candidates),
                **(
                    {"evaluations": relative(evaluations)}
                    if args.mode == train.STAGE_B_MODE
                    else {}
                ),
            }
        manifest_sha256 = None
        if args.mode == train.STAGE_B_MODE:
            lines = ["arm\tseed\tjob_id\trun_stamp"]
            for label, arm, seed, _stem in specs:
                record = records[label]
                lines.append(
                    f"{arm}\t{seed}\t{record['job_id']}\t"
                    f"cce70_clean_stage_b_05b_12pass_{arm}_s{seed}"
                )
            STAGE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_manifest = tempfile.mkstemp(
                prefix=f".{STAGE_MANIFEST.name}.", dir=STAGE_MANIFEST.parent
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
            os.replace(temporary_manifest, STAGE_MANIFEST)
            manifest_sha256 = sha256_file(STAGE_MANIFEST)
        dependency = "afterok:" + ":".join(str(record["job_id"]) for record in records.values())
        audit_export = {
            **common,
            "OAT_ZERO_CONSTRUCTIVE_AUDIT": audit_path,
            "OAT_ZERO_CONSTRUCTIVE_SUBMISSION": submission_path,
        }
        if args.mode == train.STAGE_B_MODE:
            audit_export["OAT_ZERO_CONSTRUCTIVE_PAIRED_AUDIT"] = PAIRED_AUDIT
        audit_job = sbatch(
            f"--dependency={dependency}",
            f"--job-name=audit-constructive-v5-{args.mode}",
            f"--export={export_string(audit_export)}",
            str(execution_root / AUDIT_BATCH.name),
        )
        submitted.append(audit_job)
        stage_dependency_job = None
        if args.mode == train.PAIRED_MODE:
            stage_dependency_job = sbatch(
                f"--dependency=afterok:{audit_job}",
                f"--export={export_string({'ROOT_DIR': ROOT, 'OAT_ZERO_EXPECTED_LAUNCHER_SHA256': sha256_file(Path(__file__).resolve())})}",
                str(STAGE_DEP_BATCH),
            )
            submitted.append(stage_dependency_job)
        identity = {
            "schema": (
                "constructive-code-v5-paired-smoke-identity-v1"
                if args.mode == train.PAIRED_MODE
                else "constructive-code-v5-stage-b-identity-v1"
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "source_hash": source_hash,
            "gate_execution_hash": str(gate_identity["execution_hash"]),
            "execution_hash": execution_hash,
            "protocol_sha256": sha256_file(protocol),
            "gate_audit_sha256": sha256_file(GATE_AUDIT),
            "gate_identity_sha256": sha256_file(GATE_IDENTITY),
            "viability_receipt_sha256": sha256_file(VIABILITY),
            "model_revision": train.MODEL_REVISION,
            "model_config_sha256": sha256_file(MODEL / "config.json"),
            "runtime_image_sha256": sha256_file(IMAGE),
            "core_sha256": core_hashes,
            "audit_job_id": audit_job,
            "audit_dependency": dependency,
            "stage_b_dependency_job_id": stage_dependency_job,
            "jobs": (
                {label: record["job_id"] for label, record in records.items()}
                if args.mode == train.PAIRED_MODE
                else None
            ),
            "runs": records if args.mode == train.PAIRED_MODE else None,
            "cells": records if args.mode == train.STAGE_B_MODE else None,
            "paired_audit_sha256": (
                None
                if args.mode == train.PAIRED_MODE
                else sha256_file(PAIRED_AUDIT)
            ),
            "comparative_jobs_manifest": (
                None if args.mode == train.PAIRED_MODE else relative(STAGE_MANIFEST)
            ),
            "comparative_jobs_manifest_sha256": manifest_sha256,
            "evaluation_rows_loaded": args.mode == train.STAGE_B_MODE,
            "fail_closed": True,
        }
        identity = {key: value for key, value in identity.items() if value is not None}
        atomic_json(identity_path, identity)
        held_records = {}
        for label, record in records.items():
            scheduler = scheduler_record(int(record["job_id"]))
            for required in (
                "JobState=PENDING",
                "Reason=JobHeldUser",
                "gres/gpu:a5000:1",
                "NumCPUs=16",
                "MinMemoryNode=64G",
            ):
                if required not in scheduler:
                    raise RuntimeError(f"held ConstructiveCode job lacks {required}: {label}")
            held_records[label] = scheduler
        for record in records.values():
            run(["scontrol", "update", f"JobId={record['job_id']}", "Requeue=0"])
        for record in records.values():
            run(["scontrol", "release", str(record["job_id"])])
        submission = {
            "schema": (
                "constructive-code-v5-paired-smoke-submission-v1"
                if args.mode == train.PAIRED_MODE
                else "constructive-code-v5-stage-b-submission-v1"
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "identity_sha256": sha256_file(identity_path),
            "held_job_audit": "pass",
            "released": True,
            "scheduler_records": held_records,
            "audit_job_id": audit_job,
            "audit_dependency": dependency,
            "stage_b_dependency_job_id": stage_dependency_job,
        }
        atomic_json(submission_path, submission)
    except BaseException:
        cancel(submitted)
        raise
    print(
        f"[constructive-v5-launch] mode={args.mode} cells={len(records)} "
        f"audit_job={audit_job} identity={identity_path}"
    )


if __name__ == "__main__":
    launch()
