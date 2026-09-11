"""Validation helpers for the registered E66/E68 MathIR recovery."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e66_e68_mathir_seed_overflow_recovery_amendment_20260728.md"
)
LAUNCHER = (
    ROOT / "ops/exp_scaling/recover_e66_e68_mathir_seed_overflow.py"
)
RECORD = (
    ROOT / "var/artifacts/e66_e68_mathir_seed_overflow_recovery.json"
)
EXPECTED = {
    "e66": {
        "arm": "verified_first_global_replay_canonical",
        "identity": ROOT
        / "var/artifacts/e66_same_plumbing_actuator_ablation_identity.json",
        "original_jobs": {43: 30128403, 44: 30128404, 45: 30128405},
        "original_source_hash": (
            "f6147daacbfdde22e0e9d5fab6fc45b41017d4dbf5e2848f827923ddf7828a7f"
        ),
    },
    "e68": {
        "arm": "verified_entropy_gated_singleton_escape_canonical",
        "identity": ROOT
        / "var/artifacts/e68_separated_support_actuator_ablation_identity.json",
        "original_jobs": {43: 30130478, 44: 30130479, 45: 30130480},
        "original_source_hash": (
            "4972c4816de2776c377a20d4e9704d9492535291aab8c85d7677e020fe2346a1"
        ),
    },
}
PATCHED_FILES = {
    "oat_drgrpo/learner/grpo.py",
    "oat_drgrpo/replicated_group.py",
}
REGISTERED_FAILURE = "ValueError: Seed must be between 0 and 2**32 - 1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    outer = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        outer.update(f"{_sha256(path)}  ./{relative}\n".encode())
    return outer.hexdigest()


def _checkpoint_manifest_digest(files: list[dict[str, Any]]) -> str:
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_recovery_contract() -> tuple[dict[str, dict[int, dict[str, Any]]], list[str]]:
    violations: list[str] = []
    mappings: dict[str, dict[int, dict[str, Any]]] = {
        "e66": {},
        "e68": {},
    }
    if not RECORD.is_file():
        return mappings, ["MathIR seed-overflow recovery record missing"]
    try:
        payload = json.loads(RECORD.read_text(encoding="utf-8"))
    except Exception as error:
        return mappings, [f"cannot load MathIR recovery record: {error}"]
    if (
        payload.get("schema")
        != "e66_e68_mathir_seed_overflow_recovery_v1"
        or payload.get("amendment_sha256") != _sha256(AMENDMENT)
        or payload.get("launcher_sha256") != _sha256(LAUNCHER)
        or payload.get("seed_fix_commit") != "8191c16"
        or payload.get("scientific_settings_changed") is not False
        or payload.get("resume_tag") != "step_04224"
    ):
        violations.append("MathIR seed-overflow recovery header mismatch")

    source_snapshots = payload.get("source_snapshots", {})
    for cohort, expected in EXPECTED.items():
        source = source_snapshots.get(cohort, {})
        patched_path = ROOT / str(source.get("patched_source", ""))
        if (
            source.get("original_source_hash")
            != expected["original_source_hash"]
            or set(source.get("modified_files", [])) != PATCHED_FILES
            or set(source.get("patched_file_sha256", {})) != PATCHED_FILES
            or not patched_path.is_dir()
            or source.get("patched_source_hash") != _tree_hash(patched_path)
        ):
            violations.append(
                f"{cohort}: MathIR recovery source contract mismatch"
            )

    identities: dict[str, dict[int, dict[str, Any]]] = {}
    for cohort, expected in EXPECTED.items():
        identity = json.loads(expected["identity"].read_text(encoding="utf-8"))
        identities[cohort] = {
            int(job["job_id"]): job
            for job in identity["jobs"]["mathir"]
        }

    jobs = payload.get("jobs", [])
    if len(jobs) != 6:
        violations.append("MathIR recovery does not contain six jobs")
    recovery_ids: set[int] = set()
    for job in jobs:
        cohort = str(job.get("cohort", ""))
        if cohort not in EXPECTED:
            violations.append(f"unknown MathIR recovery cohort {cohort!r}")
            continue
        expected = EXPECTED[cohort]
        seed = int(job.get("seed", -1))
        original_job_id = int(job.get("original_job_id", -1))
        recovery_job_id = int(job.get("recovery_job_id", -1))
        original = identities[cohort].get(original_job_id, {})
        if (
            expected["original_jobs"].get(seed) != original_job_id
            or job.get("arm") != expected["arm"]
            or original.get("seed") != seed
            or original.get("arm") != job.get("arm")
            or original.get("run_stamp") != job.get("run_stamp")
            or recovery_job_id <= 0
            or recovery_job_id in recovery_ids
        ):
            violations.append(
                f"{cohort}/s{seed}: MathIR recovery job binding mismatch"
            )
        recovery_ids.add(recovery_job_id)

        failure = job.get("failure_log", {})
        failure_path = ROOT / str(failure.get("path", ""))
        if failure_path.is_file():
            size = failure_path.stat().st_size
            with failure_path.open("rb") as handle:
                handle.seek(max(0, size - int(failure.get("tail_bytes", 0))))
                tail = handle.read()
            valid_failure = (
                size == failure.get("size")
                and hashlib.sha256(tail).hexdigest()
                == failure.get("tail_sha256")
                and REGISTERED_FAILURE.encode() in tail
                and failure.get("registered_failure") == REGISTERED_FAILURE
            )
        else:
            valid_failure = False
        if not valid_failure:
            violations.append(
                f"{cohort}/s{seed}: registered failure log mismatch"
            )

        checkpoint = job.get("checkpoint", {})
        checkpoint_root = Path(str(checkpoint.get("checkpoint_root", "")))
        checkpoint_tag = str(checkpoint.get("checkpoint_tag", ""))
        files = checkpoint.get("files", [])
        valid_checkpoint = (
            checkpoint_tag == "step_04224"
            and checkpoint_root.is_dir()
            and _checkpoint_manifest_digest(files)
            == checkpoint.get("tree_manifest_sha256")
        )
        for item in files:
            path = checkpoint_root / checkpoint_tag / str(item.get("path", ""))
            valid_checkpoint = (
                valid_checkpoint
                and path.is_file()
                and path.stat().st_size == item.get("size")
                and isinstance(item.get("sha256"), str)
                and len(item["sha256"]) == 64
            )
        if not valid_checkpoint:
            violations.append(
                f"{cohort}/s{seed}: recovery checkpoint contract mismatch"
            )
        mappings[cohort][original_job_id] = job

    manifest_path = ROOT / str(payload.get("manifest", ""))
    try:
        rows = list(csv.DictReader(manifest_path.open(), delimiter="\t"))
    except OSError:
        rows = []
    manifest_bindings = {
        (
            row.get("cohort"),
            int(row.get("seed", -1)),
            int(row.get("original_job_id", -1)),
            int(row.get("recovery_job_id", -1)),
            row.get("run_stamp"),
        )
        for row in rows
    }
    record_bindings = {
        (
            job.get("cohort"),
            int(job.get("seed", -1)),
            int(job.get("original_job_id", -1)),
            int(job.get("recovery_job_id", -1)),
            job.get("run_stamp"),
        )
        for job in jobs
    }
    if (
        not manifest_path.is_file()
        or payload.get("manifest_sha256") != _sha256(manifest_path)
        or manifest_bindings != record_bindings
    ):
        violations.append("MathIR recovery manifest mismatch")
    return mappings, sorted(set(violations))


def is_registered_seed_overflow(text: str, match_end: int) -> bool:
    return REGISTERED_FAILURE in text[match_end : match_end + 12000]

