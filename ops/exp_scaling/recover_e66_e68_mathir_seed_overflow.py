#!/usr/bin/env python3
"""Recover the six paired E66/E68 MathIR jobs from step 4224."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e66_e68_mathir_seed_overflow_recovery_amendment_20260728.md"
)
RECORD = (
    ROOT / "var/artifacts/e66_e68_mathir_seed_overflow_recovery.json"
)
MANIFEST = (
    ROOT / "var/artifacts/e66_e68_mathir_seed_overflow_recovery_jobs.tsv"
)
SNAPSHOT_ROOT = ROOT / "var/artifacts/source_snapshots"
PATCHED_FILES = (
    "oat_drgrpo/learner/grpo.py",
    "oat_drgrpo/replicated_group.py",
)
PATCHED_FILE_HASHES = {
    "oat_drgrpo/learner/grpo.py": (
        "57d6308c573751cdb6867f9552dcb93e3ffd65943d324cfbed6ab6f97bd90373"
    ),
    "oat_drgrpo/replicated_group.py": (
        "5799ccc587ab6485ed4a2254f18b531adfad5b968d4c033fb5c2101842b253d9"
    ),
}
FAILURE = b"ValueError: Seed must be between 0 and 2**32 - 1"
CHECKPOINT_TAG = "step_04224"

ARMS = {
    "e66": {
        "variant": "verified_first_global_replay_canonical",
        "identity": ROOT
        / "var/artifacts/e66_same_plumbing_actuator_ablation_identity.json",
        "original_source": SNAPSHOT_ROOT
        / (
            "e65_entropy_gate_"
            "f6147daacbfdde22e0e9d5fab6fc45b41017d4dbf5e2848f827923ddf7828a7f"
        )
        / "src",
        "original_source_hash": (
            "f6147daacbfdde22e0e9d5fab6fc45b41017d4dbf5e2848f827923ddf7828a7f"
        ),
        "jobs": {43: 30128403, 44: 30128404, 45: 30128405},
    },
    "e68": {
        "variant": "verified_entropy_gated_singleton_escape_canonical",
        "identity": ROOT
        / "var/artifacts/e68_separated_support_actuator_ablation_identity.json",
        "original_source": SNAPSHOT_ROOT
        / (
            "e68_separated_support_"
            "4972c4816de2776c377a20d4e9704d9492535291aab8c85d7677e020fe2346a1"
        )
        / "src",
        "original_source_hash": (
            "4972c4816de2776c377a20d4e9704d9492535291aab8c85d7677e020fe2346a1"
        ),
        "jobs": {43: 30130478, 44: 30130479, 45: 30130480},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    """Match the campaign launchers' sorted sha256sum tree hash."""

    outer = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        outer.update(f"{_sha256(path)}  ./{relative}\n".encode())
    return outer.hexdigest()


def _tree_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP)
    root.chmod(root.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP)


def _materialize_patched_source(label: str, config: dict[str, Any]) -> dict[str, Any]:
    original = Path(config["original_source"])
    original_hash = _tree_hash(original)
    if original_hash != config["original_source_hash"]:
        raise RuntimeError(
            f"{label}: original source hash {original_hash} is not frozen hash "
            f"{config['original_source_hash']}"
        )
    for relative, expected in PATCHED_FILE_HASHES.items():
        if _sha256(ROOT / "src" / relative) != expected:
            raise RuntimeError(f"{label}: active repair file drifted: {relative}")

    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{label}-seedwrap.", dir=SNAPSHOT_ROOT)
    )
    staging = staging_parent / "src"
    try:
        shutil.copytree(original, staging)
        for relative in PATCHED_FILES:
            shutil.copy2(ROOT / "src" / relative, staging / relative)
        original_files = _tree_file_hashes(original)
        patched_files = _tree_file_hashes(staging)
        changed = sorted(
            relative
            for relative in set(original_files) | set(patched_files)
            if original_files.get(relative) != patched_files.get(relative)
        )
        if changed != sorted(PATCHED_FILES):
            raise RuntimeError(
                f"{label}: recovery source changed unexpected files: {changed}"
            )
        patched_hash = _tree_hash(staging)
        target = SNAPSHOT_ROOT / f"{label}_mathir_seedwrap_{patched_hash}" / "src"
        if target.exists():
            if _tree_hash(target) != patched_hash:
                raise RuntimeError(f"{label}: existing patched snapshot drifted")
            shutil.rmtree(staging_parent)
        else:
            target.parent.mkdir(parents=True, exist_ok=False)
            os.replace(staging, target)
            staging_parent.rmdir()
            _make_read_only(target)
        return {
            "original_source": str(original.relative_to(ROOT)),
            "original_source_hash": original_hash,
            "patched_source": str(target.relative_to(ROOT)),
            "patched_source_hash": patched_hash,
            "modified_files": changed,
            "patched_file_sha256": {
                relative: patched_files[relative] for relative in changed
            },
        }
    except Exception:
        if staging_parent.exists():
            shutil.rmtree(staging_parent)
        raise


def _archived_submission(job_id: int) -> dict[str, str]:
    command = [
        "sacct",
        "-X",
        "-j",
        str(job_id),
        "--starttime",
        "2026-07-26",
        "-o",
        "JobIDRaw,State,ExitCode,SubmitLine%10000",
        "-n",
        "-P",
    ]
    lines = [
        line
        for line in subprocess.check_output(command, text=True).splitlines()
        if line.strip()
    ]
    rows = [line.split("|", 3) for line in lines]
    matches = [row for row in rows if row[0] == str(job_id)]
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve archived Slurm job {job_id}")
    _, state, exit_code, submit_line = matches[0]
    if state != "FAILED" or exit_code != "75:0":
        raise RuntimeError(
            f"job {job_id} has unexpected terminal state {state}/{exit_code}"
        )
    return {
        "state": state,
        "exit_code": exit_code,
        "submit_line": submit_line,
        "submit_line_sha256": hashlib.sha256(submit_line.encode()).hexdigest(),
    }


def _split_submit_line(submit_line: str) -> tuple[list[str], list[str]]:
    argv = shlex.split(submit_line)
    if not argv or argv[0] != "sbatch" or "--parsable" not in argv:
        raise RuntimeError("archived submission is not a parsable sbatch command")
    export_index = next(
        (
            index
            for index, value in enumerate(argv)
            if value.startswith("--export=")
        ),
        None,
    )
    if export_index is None:
        raise RuntimeError("archived submission does not freeze its environment")
    export_value = argv[export_index].removeprefix("--export=")
    exports = export_value.split(",")
    if not exports or exports[0] != "ALL":
        raise RuntimeError("archived submission does not begin with --export=ALL")
    return argv, exports


def _export_dict(exports: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in exports[1:]:
        name, separator, value = item.partition("=")
        if not separator or not name:
            raise RuntimeError(f"malformed archived export: {item!r}")
        if name in result and result[name] != value:
            raise RuntimeError(
                f"conflicting duplicate archived export: {name}"
            )
        result[name] = value
    return result


def _run_dir(run_stamp: str, job_id: int) -> Path:
    candidates = sorted(
        (ROOT / "var/data").glob(f"*_{run_stamp}/debug_job{job_id}")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one run directory for {run_stamp}/job{job_id}"
        )
    return candidates[0]


def _checkpoint_metadata(run_dir: Path) -> dict[str, Any]:
    checkpoint_root = run_dir / "checkpoints"
    checkpoint = checkpoint_root / CHECKPOINT_TAG
    if not checkpoint.is_dir():
        raise RuntimeError(f"missing recovery checkpoint: {checkpoint}")
    latest = (checkpoint_root / "latest").read_text(encoding="utf-8").strip()
    if latest != CHECKPOINT_TAG:
        raise RuntimeError(f"{checkpoint_root}: latest is {latest!r}")
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"empty recovery checkpoint: {checkpoint}")
    return {
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_tag": CHECKPOINT_TAG,
        "files": [
            {
                "path": path.relative_to(checkpoint).as_posix(),
                "size": path.stat().st_size,
            }
            for path in files
        ],
    }


def _hash_checkpoint(metadata: dict[str, Any]) -> dict[str, Any]:
    root = Path(metadata["checkpoint_root"]) / metadata["checkpoint_tag"]
    files = []
    for item in metadata["files"]:
        path = root / item["path"]
        if path.stat().st_size != item["size"]:
            raise RuntimeError(f"checkpoint changed during hashing: {path}")
        files.append({**item, "sha256": _sha256(path)})
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        **metadata,
        "files": files,
        "tree_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _failure_log_metadata(job_id: int) -> dict[str, Any]:
    path = ROOT / f"var/artifacts/logs/xdr_train-{job_id}.out"
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - 65536))
        tail = handle.read()
    if FAILURE not in tail:
        raise RuntimeError(f"job {job_id}: registered seed failure absent from tail")
    return {
        "path": str(path.relative_to(ROOT)),
        "size": size,
        "tail_bytes": len(tail),
        "tail_sha256": hashlib.sha256(tail).hexdigest(),
        "registered_failure": FAILURE.decode(),
    }


def _prepare_jobs(source_records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for label, config in ARMS.items():
        for seed, job_id in config["jobs"].items():
            archived = _archived_submission(job_id)
            argv, export_items = _split_submit_line(archived["submit_line"])
            exports = _export_dict(export_items)
            if (
                exports.get("OAT_ZERO_SEED") != str(seed)
                or exports.get("OAT_ZERO_VARIANT") != config["variant"]
                or Path(exports.get("OAT_ZERO_PROTOCOL_IDENTITY", ""))
                != config["identity"]
                or Path(exports.get("OAT_ZERO_SOURCE_ROOT", ""))
                != config["original_source"]
                or exports.get("OAT_ZERO_MAX_PROMPT_EPOCHS") != "12"
            ):
                raise RuntimeError(f"job {job_id}: archived scientific contract drift")
            run_stamp = exports["RUN_STAMP"]
            run_dir = _run_dir(run_stamp, job_id)
            checkpoint = _checkpoint_metadata(run_dir)
            jobs.append(
                {
                    "cohort": label,
                    "arm": config["variant"],
                    "seed": seed,
                    "original_job_id": job_id,
                    "run_stamp": run_stamp,
                    "original_run_dir": str(run_dir.relative_to(ROOT)),
                    "checkpoint": checkpoint,
                    "failure_log": _failure_log_metadata(job_id),
                    "archived_submission": archived,
                    "argv": argv,
                    "export_items": export_items,
                    "exports": exports,
                    "patched_source": str(
                        ROOT / source_records[label]["patched_source"]
                    ),
                }
            )
    return jobs


def _recovery_argv(job: dict[str, Any]) -> list[str]:
    argv = list(job["argv"])
    export_index = next(
        index for index, value in enumerate(argv) if value.startswith("--export=")
    )
    replacements = {
        "OAT_ZERO_SOURCE_ROOT": job["patched_source"],
        "OAT_ZERO_LOCAL_ROOT": (
            job["exports"]["OAT_ZERO_LOCAL_ROOT"] + "_seedwrap_recovery"
        ),
    }
    items = ["ALL"]
    seen: set[str] = set()
    for raw in job["export_items"][1:]:
        name, _, value = raw.partition("=")
        items.append(f"{name}={replacements.get(name, value)}")
        seen.add(name)
    additions = {
        "OAT_ZERO_INITIAL_RESUME_DIR": job["checkpoint"]["checkpoint_root"],
        "OAT_ZERO_INITIAL_RESUME_TAG": CHECKPOINT_TAG,
        "OAT_ZERO_SEED_OVERFLOW_RECOVERY_RECORD": str(RECORD),
    }
    if set(additions) & seen:
        raise RuntimeError("archived job already contains recovery-only exports")
    items.extend(f"{name}={value}" for name, value in additions.items())
    argv[export_index] = "--export=" + ",".join(items)
    return argv


def _audit_held_job(job: dict[str, Any]) -> None:
    record = subprocess.check_output(
        ["scontrol", "show", "job", str(job["recovery_job_id"]), "-o"],
        text=True,
    )
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"OAT_ZERO_VARIANT={job['arm']}",
        f"OAT_ZERO_SEED={job['seed']}",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=12",
        f"OAT_ZERO_SOURCE_ROOT={job['patched_source']}",
        (
            "OAT_ZERO_INITIAL_RESUME_DIR="
            f"{job['checkpoint']['checkpoint_root']}"
        ),
        f"OAT_ZERO_INITIAL_RESUME_TAG={CHECKPOINT_TAG}",
        f"OAT_ZERO_SEED_OVERFLOW_RECOVERY_RECORD={RECORD}",
        "TresPerNode=gres/gpu:a100:1",
        "ReqNodeList=node302",
    )
    missing = [value for value in required if value not in record]
    if missing:
        raise RuntimeError(
            f"held recovery job {job['recovery_job_id']} missing {missing}"
        )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _write_manifest(jobs: list[dict[str, Any]]) -> None:
    fields = (
        "cohort",
        "arm",
        "seed",
        "original_job_id",
        "recovery_job_id",
        "run_stamp",
        "checkpoint_root",
        "checkpoint_tag",
        "patched_source",
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{MANIFEST.name}.", dir=MANIFEST.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    **{name: job[name] for name in fields if name in job},
                    "checkpoint_root": job["checkpoint"]["checkpoint_root"],
                    "checkpoint_tag": CHECKPOINT_TAG,
                }
            )
    os.replace(temporary, MANIFEST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("config", "full"))
    args = parser.parse_args()
    if args.phase == "full" and (RECORD.exists() or MANIFEST.exists()):
        raise SystemExit("fresh recovery record and manifest required")

    source_records = {
        label: _materialize_patched_source(label, config)
        for label, config in ARMS.items()
    }
    jobs = _prepare_jobs(source_records)
    print(
        "[seedwrap-recovery] validated "
        f"{len(jobs)} failed jobs and step-4224 checkpoints"
    )
    if args.phase == "config":
        for label, source in source_records.items():
            print(
                f"[seedwrap-recovery] {label} patched_source="
                f"{source['patched_source_hash']}"
            )
        return

    print("[seedwrap-recovery] hashing six complete checkpoints before submission")
    with ThreadPoolExecutor(max_workers=6) as executor:
        hashed = list(
            executor.map(_hash_checkpoint, [job["checkpoint"] for job in jobs])
        )
    for job, checkpoint in zip(jobs, hashed, strict=True):
        job["checkpoint"] = checkpoint

    submitted: list[int] = []
    try:
        for job in jobs:
            output = subprocess.check_output(_recovery_argv(job), text=True).strip()
            recovery_job_id = int(output.split(";", 1)[0])
            job["recovery_job_id"] = recovery_job_id
            submitted.append(recovery_job_id)
        for job in jobs:
            _audit_held_job(job)

        _write_manifest(jobs)
        payload = {
            "schema": "e66_e68_mathir_seed_overflow_recovery_v1",
            "amendment_sha256": _sha256(AMENDMENT),
            "launcher_sha256": _sha256(Path(__file__).resolve()),
            "seed_fix_commit": "8191c16",
            "scientific_settings_changed": False,
            "resume_tag": CHECKPOINT_TAG,
            "source_snapshots": source_records,
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "manifest_sha256": _sha256(MANIFEST),
            "jobs": [
                {
                    key: value
                    for key, value in job.items()
                    if key
                    not in {
                        "argv",
                        "export_items",
                        "exports",
                        "patched_source",
                    }
                }
                | {
                    "patched_source": str(
                        Path(job["patched_source"]).relative_to(ROOT)
                    )
                }
                for job in jobs
            ],
        }
        _atomic_text(RECORD, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        subprocess.run(
            ["scontrol", "release", *map(str, submitted)],
            check=True,
        )
    except Exception:
        if submitted:
            subprocess.run(
                ["scancel", *map(str, submitted)],
                check=False,
            )
        raise
    print(
        "[seedwrap-recovery] released paired jobs: "
        + " ".join(map(str, submitted))
    )


if __name__ == "__main__":
    main()
