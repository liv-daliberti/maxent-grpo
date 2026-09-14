#!/usr/bin/env python3
"""Shared fail-closed helpers for E114--E116 comparator completion.

The helpers clone the immutable scheduler argv recorded for a paired Dr.GRPO
control.  Callers may change only explicitly supplied environment keys,
runtime roots, output identity, dependency, and job name.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing immutable parent ledger: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("released"):
        raise SystemExit(f"parent ledger was not released: {path}")
    return payload


def controls(
    payload: dict[str, Any], *, domains: Iterable[str], seeds: Iterable[int]
) -> list[dict[str, Any]]:
    domain_order = tuple(domains)
    seed_order = tuple(int(seed) for seed in seeds)
    expected = {(domain, seed) for domain in domain_order for seed in seed_order}
    rows = [
        row
        for row in payload.get("runs", [])
        if str(row.get("arm")) == "control"
        and (str(row.get("domain")), int(row.get("seed", -1))) in expected
    ]
    found = {(str(row["domain"]), int(row["seed"])) for row in rows}
    if found != expected or len(rows) != len(expected):
        raise SystemExit(
            f"parent controls cover {len(found)} cells, expected {len(expected)}"
        )
    rank = {domain: index for index, domain in enumerate(domain_order)}
    return sorted(rows, key=lambda row: (rank[str(row["domain"])], int(row["seed"])))


def submit_line(run: dict[str, Any]) -> list[str]:
    record = str(run.get("held_scheduler_record", ""))
    if "SubmitLine=" not in record:
        raise RuntimeError(
            f"{run.get('domain')}/s{run.get('seed')}: parent lacks SubmitLine"
        )
    line = record.split("SubmitLine=", 1)[1]
    for stop in (" WorkDir=", " StdErr=", " StdOut=", " StdIn=", " TresPer"):
        if stop in line:
            line = line.split(stop, 1)[0]
    argv = shlex.split(line.strip())
    if not argv or argv[0] != "sbatch":
        raise RuntimeError("recorded parent SubmitLine is not sbatch")
    return argv


def export_pairs(argv: list[str]) -> dict[str, str]:
    token = next((item for item in argv if item.startswith("--export=")), "")
    if not token:
        raise RuntimeError("submission has no --export token")
    pairs: dict[str, str] = {}
    for item in token[len("--export=") :].split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            pairs[key] = value
    return pairs


def base_snapshot(run: dict[str, Any]) -> Path:
    exports = export_pairs(submit_line(run))
    source = Path(exports.get("OAT_ZERO_SOURCE_ROOT", "")).resolve()
    ops = Path(exports.get("OAT_ZERO_OPS_SNAPSHOT_ROOT", "")).resolve()
    if source.name != "src" or ops.name != "ops" or source.parent != ops.parent:
        raise RuntimeError("parent runtime roots do not identify one snapshot")
    if not (source / "oat_drgrpo/__init__.py").is_file():
        raise RuntimeError(f"invalid parent snapshot: {source.parent}")
    return source.parent
def ensure_overlay_snapshot(
    *,
    base: Path,
    patch_source: Path,
    prefix: str,
    patched_files: Iterable[str],
) -> tuple[Path, list[str]]:
    """Copy a parent runtime and overlay files from an immutable baseline."""

    files = sorted(set(str(name) for name in patched_files))
    identity = hashlib.sha256(str(base.resolve()).encode("utf-8"))
    identity.update(str(patch_source.resolve()).encode("utf-8"))
    for relative in files:
        source = patch_source / relative
        if not source.is_file():
            raise RuntimeError(f"immutable overlay lacks {relative}: {source}")
        identity.update(relative.encode("utf-8"))
        identity.update(digest(source).encode("ascii"))
    target = (
        ROOT
        / "var/artifacts/source_snapshots"
        / f"{prefix}_{identity.hexdigest()[:16]}"
    )
    if not target.is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            for entry in ("src", "ops"):
                shutil.copytree(
                    base / entry,
                    staging / entry,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            for relative in files:
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(patch_source / relative, destination)
                shutil.copystat(patch_source / relative, destination)
            (staging / "SNAPSHOT_IDENTITY.json").write_text(
                json.dumps(
                    {
                        "schema": "direct_comparator_immutable_overlay_v1",
                        "derived_from": str(base.resolve()),
                        "patch_source": str(patch_source.resolve()),
                        "patched_files": files,
                        "sha256": identity.hexdigest(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return target, files


def clone_command(
    run: dict[str, Any],
    *,
    name: str,
    overrides: dict[str, str],
    dependency: str = "",
    time_limit: str = "",
) -> list[str]:
    """Clone one parent command with a closed set of explicit changes."""

    argv = submit_line(run)
    parent_exports = export_pairs(argv)
    parent_exports.update({key: str(value) for key, value in overrides.items()})
    out: list[str] = []
    saw_hold = False
    saw_time = False
    for token in argv:
        if token.startswith("--export="):
            out.append(
                "--export=ALL,"
                + ",".join(f"{key}={value}" for key, value in parent_exports.items())
            )
        elif token.startswith("--job-name="):
            out.append(f"--job-name={name}")
        elif token.startswith("--dependency="):
            continue
        elif token == "--hold":
            saw_hold = True
            out.append(token)
        elif token.startswith("--time=") and time_limit:
            out.append(f"--time={time_limit}")
            saw_time = True
        else:
            out.append(token)
    if not saw_hold:
        out.insert(2, "--hold")
    if time_limit and not saw_time:
        out.insert(-1, f"--time={time_limit}")
    if dependency:
        out.insert(-1, f"--dependency=afterok:{dependency}")
    return out


def submit_held(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid sbatch response: {result.stdout!r}")
    return job_id


def audit_held(job_id: str, *, name: str, expected: Iterable[str]) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"cannot inspect held comparator job {job_id}: {result.stderr.strip()}"
        )
    record = result.stdout
    required = ("JobState=PENDING", "Reason=JobHeldUser", f"JobName={name}", *expected)
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"held comparator job {job_id} lacks {missing}")
    return record


def cancel(job_ids: Iterable[str]) -> None:
    ids = list(job_ids)
    if ids:
        subprocess.run(["scancel", *ids], check=False)


def release(job_ids: Iterable[str]) -> None:
    for job_id in job_ids:
        subprocess.run(["scontrol", "release", str(job_id)], check=True)


def cpu_audit_command(
    *,
    name: str,
    dependency: str,
    snapshot: Path,
    command: str,
    nice: int = 100,
) -> list[str]:
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--dependency=afterok:{dependency}",
        f"--export=ALL,PYTHONPATH={snapshot / 'src'}",
        "--partition=all",
        "--account=allcs",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:15:00",
        f"--nice={nice}",
        f"--output={ROOT / 'var/artifacts/logs'}/%x-%j.out",
        f"--error={ROOT / 'var/artifacts/logs'}/%x-%j.err",
        "--wrap",
        command,
    ]

