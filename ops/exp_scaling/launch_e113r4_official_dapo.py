#!/usr/bin/env python3
"""Audit, snapshot, and submit pinned official-verl E113-R4 DAPO."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Iterable

import e113r4_official_dapo_common as common


LOCAL_RUNTIME_FILES = (
    "ops/run_e113r4_official_dapo.sh",
    "ops/exp_scaling/e113r4_verl_reward.py",
    "ops/exp_scaling/e113r4_write_receipt.py",
)
SLURM_SCRIPT = common.ROOT / "ops/slurm/e113r4_official_dapo.slurm"
UPSTREAM_HASHES = {
    "recipe/dapo/run_dapo_qwen2.5_32b.sh": (
        "0b4e03024dfb790a949657e33411b788d139b7e8faa689c8a9a082aa00c1919a"
    ),
    "recipe/dapo/src/dapo_ray_trainer.py": (
        "ead0bcc8503a55a6958468fcf5889da164750db4fb264c145abb3972f28491c3"
    ),
    "recipe/dapo/src/main_dapo.py": (
        "a3364eb91d4fe25c706f866e8b5f7caf58f0e3129132840f5cba0f14fec11a87"
    ),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def tree_hash(paths: Iterable[Path]) -> str:
    value = hashlib.sha256()
    for base in paths:
        if base.is_file():
            value.update(str(base.relative_to(common.ROOT)).encode("utf-8"))
            value.update(b"\0")
            value.update(base.read_bytes())
            value.update(b"\0")
            continue
        for path in sorted(candidate for candidate in base.rglob("*") if candidate.is_file()):
            if ".git" in path.parts or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            value.update(str(path.relative_to(base.parent)).encode("utf-8"))
            value.update(b"\0")
            value.update(path.read_bytes())
            value.update(b"\0")
    return value.hexdigest()


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


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=common.ROOT,
        capture_output=True,
        text=True,
        check=check,
    )


def validate_upstream() -> dict[str, Any]:
    if not common.VERL_ROOT.is_dir():
        raise SystemExit(f"missing upstream checkout: {common.VERL_ROOT}")
    head = run(["git", "-C", str(common.VERL_ROOT), "rev-parse", "HEAD"]).stdout.strip()
    if head != common.VERL_COMMIT:
        raise SystemExit(f"upstream commit drift: {head}")
    status = run(
        ["git", "-C", str(common.VERL_ROOT), "status", "--porcelain", "--untracked-files=no"]
    ).stdout.strip()
    if status:
        raise SystemExit("upstream tracked files are dirty")
    observed: dict[str, str] = {}
    for relative, expected in UPSTREAM_HASHES.items():
        path = common.VERL_ROOT / relative
        actual = digest(path)
        if actual != expected:
            raise SystemExit(f"upstream source hash drift: {relative}: {actual}")
        observed[relative] = actual
    return {"commit": head, "source_sha256": observed}


def validate_data() -> tuple[dict[str, Any], dict[tuple[str, str], str]]:
    if not common.DATA_MANIFEST.is_file():
        raise SystemExit(f"missing data manifest: {common.DATA_MANIFEST}")
    payload = json.loads(common.DATA_MANIFEST.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "e113r4_official_verl_dapo_data_v1"
        or payload.get("train_rows") != common.GEN_PROMPT_BATCH
        or payload.get("prompt_rendering_exactly_matches_frozen_comparators") is not True
    ):
        raise SystemExit("E113-R4 data manifest contract failed")
    hashes: dict[tuple[str, str], str] = {}
    for record in payload.get("outputs", []):
        key = (str(record["domain"]), str(record["split"]))
        path = Path(str(record["path"]))
        actual = digest(path)
        if actual != record.get("sha256"):
            raise SystemExit(f"converted data hash drift: {path}")
        hashes[key] = actual
    expected = {(domain, split) for domain in common.DOMAINS for split in ("train", "eval")}
    if set(hashes) != expected:
        raise SystemExit("converted data manifest does not cover ten exact splits")
    return payload, hashes


def validate_r3_retired() -> dict[str, Any]:
    payload = json.loads(common.R3_LEDGER.read_text(encoding="utf-8"))
    job_ids = sorted(int(item["job_id"]) for item in payload.get("runs", []))
    if job_ids != list(range(30790925, 30790975)):
        raise SystemExit("E113-R3 ledger is not the exact retired 50-cell cohort")
    queued = run(
        ["squeue", "-h", "-j", ",".join(map(str, job_ids)), "-o", "%A"],
        check=False,
    ).stdout.split()
    if queued:
        raise SystemExit(f"E113-R3 still has queued jobs: {sorted(set(queued))}")
    accounting_text = run(
        [
            "sacct", "-X", "-j", ",".join(map(str, job_ids)),
            "--starttime", "2026-08-19", "-n", "-P",
            "-o", "JobIDRaw,State,ExitCode,Start,End,NodeList",
        ]
    ).stdout
    accounting: dict[int, dict[str, str]] = {}
    for line in accounting_text.splitlines():
        fields = line.split("|")
        if len(fields) < 6 or not fields[0].isdigit():
            continue
        accounting[int(fields[0])] = {
            "state": fields[1],
            "exit_code": fields[2],
            "start": fields[3],
            "end": fields[4],
            "node_list": fields[5],
        }
    if set(accounting) != set(job_ids):
        raise SystemExit("Slurm accounting does not cover all 50 retired E113-R3 jobs")
    noncanceled = {
        job_id: value["state"]
        for job_id, value in accounting.items()
        if not value["state"].startswith("CANCELLED")
    }
    if noncanceled:
        raise SystemExit(f"retired E113-R3 jobs are not all canceled: {noncanceled}")
    reaper = run(
        [
            "sacct", "-X", "-j", "30791185", "--starttime", "2026-08-19",
            "-n", "-P", "-o", "JobIDRaw,State,ExitCode,Start,End",
        ]
    ).stdout.strip()
    record = {
        "schema": "e113r3_retirement_for_official_dapo_v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "retired custom one-prompt DAPO adapter; replaced prospectively by "
            "pinned upstream multi-prompt verl-recipe/dapo"
        ),
        "r3_ledger": str(common.R3_LEDGER),
        "r3_ledger_sha256": digest(common.R3_LEDGER),
        "registered_job_ids": job_ids,
        "status_at_user_request": {"running": 11, "pending": 31, "terminal": 8},
        "exact_jobs_remaining_in_queue": [],
        "all_registered_jobs_canceled": True,
        "final_accounting": {
            str(job_id): accounting[job_id] for job_id in sorted(accounting)
        },
        "failure_reaper_job_id": 30791185,
        "failure_reaper_accounting": reaper,
        "r3_eligible_for_named_dapo_efficacy": False,
    }
    atomic_json(common.RETIREMENT, record)
    return record


def make_snapshot() -> tuple[Path, str]:
    sources = [
        common.ROOT / "src/oat_drgrpo",
        common.VERL_ROOT,
        *(common.ROOT / relative for relative in LOCAL_RUNTIME_FILES),
    ]
    identity = tree_hash(sources)
    snapshot = (
        common.ROOT
        / "var/artifacts/source_snapshots"
        / f"e113r4_official_verl_dapo_{identity[:16]}"
    )
    if snapshot.is_dir():
        metadata = json.loads((snapshot / "SNAPSHOT_IDENTITY.json").read_text())
        if metadata.get("sha256") != identity:
            raise SystemExit(f"existing runtime snapshot identity drift: {snapshot}")
        return snapshot, identity
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot.name}.", dir=snapshot.parent))
    try:
        shutil.copytree(
            common.ROOT / "src/oat_drgrpo",
            temporary / "src/oat_drgrpo",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(
            common.VERL_ROOT,
            temporary / "verl",
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )
        for relative in LOCAL_RUNTIME_FILES:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(common.ROOT / relative, target)
        atomic_json(
            temporary / "SNAPSHOT_IDENTITY.json",
            {
                "schema": "e113r4_official_verl_dapo_runtime_snapshot_v1",
                "sha256": identity,
                "upstream_commit": common.VERL_COMMIT,
            },
        )
        os.replace(temporary, snapshot)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return snapshot, identity


def cells() -> list[dict[str, Any]]:
    result = []
    for family in common.FAMILIES:
        for domain in common.DOMAINS:
            for seed in common.SEEDS[family]:
                result.append({"family": family, "domain": domain, "seed": seed})
    return result


def export_env(
    snapshot: Path,
    image_size: int,
    data_hashes: dict[tuple[str, str], str],
    family: str,
    domain: str,
    seed: int,
    *,
    smoke: bool,
) -> dict[str, str]:
    steps = 1 if smoke else common.TOTAL_TRAINING_STEPS
    max_epochs = common.MAX_GENERATION_BATCHES if smoke else common.MAX_TOTAL_EPOCHS
    return {
        "E113R4_ROOT": str(common.ROOT),
        "E113R4_RUNTIME_SNAPSHOT": str(snapshot),
        "E113R4_VERL_ROOT": str(snapshot / "verl"),
        "E113R4_VERIFIER_SITE": str(common.VERIFIER_SITE),
        "E113R4_IMAGE": str(common.IMAGE),
        "E113R4_IMAGE_SIZE": str(image_size),
        "E113R4_FAMILY": family,
        "E113R4_DOMAIN": domain,
        "E113R4_SEED": str(seed),
        "E113R4_OUTPUT": str(common.run_dir(family, domain, seed, smoke=smoke)),
        "E113R4_TRAIN_FILE": str(common.parquet_path(domain, "train")),
        "E113R4_TRAIN_SHA256": data_hashes[(domain, "train")],
        "E113R4_VAL_FILE": str(common.parquet_path(domain, "eval")),
        "E113R4_VAL_SHA256": data_hashes[(domain, "eval")],
        "E113R4_MODEL": str(common.MODEL_ROOTS[family]),
        "E113R4_PROMPT_LENGTH": str(common.PROMPT_LENGTHS[domain]),
        "E113R4_RESPONSE_LENGTH": str(common.response_length(family, domain)),
        "E113R4_OVERLONG_BUFFER": str(common.overlong_buffer(family, domain)),
        "E113R4_TOTAL_STEPS": str(steps),
        "E113R4_MAX_EPOCHS": str(max_epochs),
        "E113R4_RUN_SCRIPT": str(snapshot / "ops/run_e113r4_official_dapo.sh"),
    }


def job_name(family: str, domain: str, seed: int, *, smoke: bool) -> str:
    family_tag = "q05" if family == "qwen05b" else "f1"
    stage = "s0" if smoke else ""
    return f"e113r4{stage}-{family_tag}-{common.DOMAIN_TAGS[domain][:5]}-s{seed}"


def sbatch_command(
    env: dict[str, str],
    family: str,
    domain: str,
    seed: int,
    *,
    smoke: bool,
    dependency: str | None,
) -> list[str]:
    name = job_name(family, domain, seed, smoke=smoke)
    command = [
        "sbatch",
        "--parsable",
        "--hold",
        "--requeue",
        f"--job-name={name}",
        "--partition=all",
        "--account=allcs",
        "--gres=gpu:a6000:1",
        "--cpus-per-task=16",
        "--mem=128G",
        "--time=7-00:00:00",
        "--nice=100",
        f"--output={common.ROOT}/var/artifacts/logs/{name}-%j.out",
        f"--error={common.ROOT}/var/artifacts/logs/{name}-%j.err",
        "--export=ALL," + ",".join(f"{key}={value}" for key, value in env.items()),
    ]
    if dependency:
        command.extend(
            [f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"]
        )
    command.append(str(SLURM_SCRIPT))
    return command


def submit(command: list[str]) -> str:
    response = run(command).stdout.strip().split(";", maxsplit=1)[0]
    if not response.isdigit():
        raise RuntimeError(f"invalid sbatch response: {response!r}")
    return response


def audit_held(job_id: str, *, dependency: str | None) -> str:
    record = run(["scontrol", "show", "job", "-dd", "-o", job_id]).stdout.strip()
    required = [
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Requeue=1",
        "Account=allcs",
        "TRES=cpu=16,mem=128G",
        "gres/gpu:a6000=1",
        "TimeLimit=7-00:00:00",
    ]
    if dependency:
        required.append("Dependency=afterok:")
    missing = [value for value in required if value not in record]
    # Slurm preserves the broad `all` request on in-place account amendments,
    # while fresh allcs submissions normalize to the eligible `cs` partition.
    if "Partition=all" not in record and "Partition=cs" not in record:
        missing.append("Partition in {all,cs}")
    if missing:
        raise RuntimeError(f"held job {job_id} failed scheduler audit: {missing}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    if common.LEDGER.exists():
        raise SystemExit(f"refusing duplicate E113-R4 cohort: {common.LEDGER}")
    if not common.PROTOCOL.is_file():
        raise SystemExit(f"missing frozen protocol: {common.PROTOCOL}")
    if not common.IMAGE.is_file():
        raise SystemExit(f"missing pinned official image: {common.IMAGE}")
    if not common.IMAGE_OCI_MANIFEST.is_file():
        raise SystemExit(
            f"missing source OCI manifest: {common.IMAGE_OCI_MANIFEST}"
        )
    manifest_hash = digest(common.IMAGE_OCI_MANIFEST)
    if f"sha256:{manifest_hash}" != common.IMAGE_OCI_MANIFEST_DIGEST:
        raise SystemExit(f"source OCI manifest hash drift: {manifest_hash}")
    if not common.IMAGE_BUILDER_SOURCE.is_file():
        raise SystemExit(
            f"missing official image-builder source: {common.IMAGE_BUILDER_SOURCE}"
        )
    builder_hash = digest(common.IMAGE_BUILDER_SOURCE)
    if builder_hash != common.IMAGE_BUILDER_SOURCE_SHA256:
        raise SystemExit(f"official image-builder source hash drift: {builder_hash}")

    verifier_wheel_hashes: dict[str, str] = {}
    for wheel, expected in common.VERIFIER_WHEELS.items():
        if not wheel.is_file():
            raise SystemExit(f"missing E113-R4 verifier wheel: {wheel}")
        actual = digest(wheel)
        if actual != expected:
            raise SystemExit(f"E113-R4 verifier wheel hash drift: {wheel}: {actual}")
        verifier_wheel_hashes[str(wheel)] = actual
    verifier_files = (
        common.VERIFIER_SITE / "latex2sympy2_extended/__init__.py",
        common.VERIFIER_SITE / "math_verify/__init__.py",
    )
    missing_verifier_files = [
        str(path) for path in verifier_files if not path.is_file()
    ]
    if missing_verifier_files:
        raise SystemExit(
            f"missing E113-R4 project-local verifier files: {missing_verifier_files}"
        )
    verifier_site_hash = tree_hash([common.VERIFIER_SITE])

    for model in common.MODEL_ROOTS.values():
        if not (model / "config.json").is_file():
            raise SystemExit(f"missing frozen model: {model}")
    for relative in (*LOCAL_RUNTIME_FILES, "ops/slurm/e113r4_official_dapo.slurm"):
        if not (common.ROOT / relative).is_file():
            raise SystemExit(f"missing E113-R4 runtime input: {relative}")

    upstream = validate_upstream()
    data_manifest, data_hashes = validate_data()
    retirement = validate_r3_retired()
    image_hash = digest(common.IMAGE)
    image_size = common.IMAGE.stat().st_size
    snapshot, snapshot_identity = make_snapshot()
    validation = {
        "protocol_sha256": digest(common.PROTOCOL),
        "launcher_sha256": digest(Path(__file__)),
        "image_sha256": image_hash,
        "image_size": image_size,
        "image_source_oci_manifest_digest": common.IMAGE_OCI_MANIFEST_DIGEST,
        "image_source_oci_manifest_sha256": manifest_hash,
        "image_source_oci_manifest": str(common.IMAGE_OCI_MANIFEST),
        "image_builder": f"apptainer-{common.IMAGE_BUILDER_VERSION}",
        "image_builder_source_sha256": builder_hash,
        "image_builder_source": str(common.IMAGE_BUILDER_SOURCE),
        "verifier_site": str(common.VERIFIER_SITE),
        "verifier_site_sha256": verifier_site_hash,
        "verifier_wheels_sha256": verifier_wheel_hashes,
        "upstream": upstream,
        "data_manifest_sha256": digest(common.DATA_MANIFEST),
        "data_schema": data_manifest["schema"],
        "retirement_sha256": digest(common.RETIREMENT),
        "runtime_snapshot": str(snapshot),
        "runtime_snapshot_sha256": snapshot_identity,
        "slurm_script_sha256": digest(SLURM_SCRIPT),
    }
    if not args.launch:
        print(json.dumps(validation, indent=2, sort_keys=True))
        print("dry-run only; pass --launch to submit two smokes and 50 dependent cells")
        return 0

    submissions: list[dict[str, Any]] = []
    submitted_ids: list[str] = []
    common.LEDGER.parent.mkdir(parents=True, exist_ok=True)
    (common.ROOT / "var/artifacts/logs").mkdir(parents=True, exist_ok=True)
    try:
        smoke_specs = (
            ("qwen05b", "graph_coloring", 43),
            ("falcon1b", "graph_coloring", 55),
        )
        smoke_ids: list[str] = []
        for family, domain, seed in smoke_specs:
            output = common.run_dir(family, domain, seed, smoke=True)
            if output.exists():
                raise RuntimeError(f"refusing existing smoke output: {output}")
            env = export_env(
                snapshot, image_size, data_hashes, family, domain, seed, smoke=True
            )
            command = sbatch_command(
                env, family, domain, seed, smoke=True, dependency=None
            )
            job_id = submit(command)
            submitted_ids.append(job_id)
            smoke_ids.append(job_id)
            submissions.append(
                {
                    "scientific_cell": False,
                    "stage": "operational_smoke",
                    "max_train": 1,
                    "family": family,
                    "domain": domain,
                    "seed": seed,
                    "job_id": int(job_id),
                    "run_dir": str(output),
                    "log_path": str(common.ROOT / "var/artifacts/logs" / f"{job_name(family, domain, seed, smoke=True)}-{job_id}.out"),
                    "command": command,
                    "environment": env,
                    "held_scheduler_record": audit_held(job_id, dependency=None),
                }
            )
        dependency = ":".join(smoke_ids)
        r3 = json.loads(common.R3_LEDGER.read_text(encoding="utf-8"))
        controls = {
            (str(row["model_family"]), str(row["domain"]), int(row["seed"])): row[
                "paired_control"
            ]
            for row in r3["runs"]
        }
        for cell in cells():
            family = str(cell["family"])
            domain = str(cell["domain"])
            seed = int(cell["seed"])
            output = common.run_dir(family, domain, seed)
            if output.exists():
                raise RuntimeError(f"refusing existing scientific output: {output}")
            env = export_env(
                snapshot, image_size, data_hashes, family, domain, seed, smoke=False
            )
            command = sbatch_command(
                env, family, domain, seed, smoke=False, dependency=dependency
            )
            job_id = submit(command)
            submitted_ids.append(job_id)
            submissions.append(
                {
                    "scientific_cell": True,
                    "stage": "full",
                    "arm": "dapo",
                    **cell,
                    "job_id": int(job_id),
                    "run_dir": str(output),
                    "log_path": str(common.ROOT / "var/artifacts/logs" / f"{job_name(family, domain, seed, smoke=False)}-{job_id}.out"),
                    "paired_control": controls[(family, domain, seed)],
                    "dependency_smoke_job_ids": [int(value) for value in smoke_ids],
                    "command": command,
                    "environment": env,
                    "held_scheduler_record": audit_held(job_id, dependency=dependency),
                }
            )
        ledger = {
            "schema": "e113r4_official_verl_dapo_jobs_v1",
            "released": False,
            "scientific_cells": 50,
            "operational_smokes": 2,
            "target_steps": common.TOTAL_TRAINING_STEPS,
            "train_rows": common.TOTAL_TRAINING_STEPS,
            "passes": 1,
            "checkpoint_interval_steps": 5,
            "arms": ["dapo"],
            "domains": list(common.DOMAINS),
            "official_upstream_unmodified": True,
            "accepted_prompt_groups_per_cell": common.ACCEPTED_PROMPT_GROUPS,
            "accepted_training_steps_per_cell": common.TOTAL_TRAINING_STEPS,
            "maximum_sampled_responses_per_cell": common.MAX_SAMPLED_RESPONSES,
            "validation": validation,
            "r3_retirement": retirement,
            "smokes": {
                str(item["family"]): item
                for item in submissions
                if not item["scientific_cell"]
            },
            "runs": [item for item in submissions if item["scientific_cell"]],
        }
        atomic_json(common.LEDGER, ledger)
        run(["scontrol", "release", *submitted_ids])
        ledger["released"] = True
        ledger["released_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(common.LEDGER, ledger)
    except BaseException:
        if submitted_ids:
            run(["scancel", *submitted_ids], check=False)
        raise

    smoke_ids = [item["job_id"] for item in submissions if not item["scientific_cell"]]
    science_ids = [item["job_id"] for item in submissions if item["scientific_cell"]]
    print(f"released E113-R4 smokes: {smoke_ids}")
    print(f"released 50 dependency-gated scientific jobs: {science_ids[0]}--{science_ids[-1]}")
    print(f"ledger: {common.LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

