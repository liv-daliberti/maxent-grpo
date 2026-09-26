#!/usr/bin/env python3
"""Freeze and transactionally launch the 36-cell E117 Stage 1 development grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e117_same_plumbing_component_preflight as r2audit  # noqa: E402
import launch_e76_tuned_scale as snapshot_util  # noqa: E402


PROTOCOL = "paper/preregistration/e117_stage1_development_execution_20260830.md"
SCHEDULER_AMENDMENT = "paper/preregistration/e117_stage1s1_effective_cs_route_20260830.md"
AUDIT_SCHEDULER_AMENDMENT = "paper/preregistration/e117_stage1s2_cpu_audit_cs_route_20260830.md"
STORAGE_AMENDMENT = "paper/preregistration/e117_stage1s3_checkpoint_space_safety_20260830.md"
SUPERSEDED_LEDGER = "var/artifacts/e117_stage1s3_superseded_storage_unsafe_jobs.json"
SUPERSEDED_LEDGER_SHA256 = "bc4e276f9215000ad3f16cbe88593aa311a396fb98db09186c33dfe35fa9845e"
SUPERSEDED_AUDIT_JOB = "var/artifacts/e117_stage1s3_superseded_storage_unsafe_audit_job.json"
SUPERSEDED_AUDIT_JOB_SHA256 = "64882cab729180034b888418c128d8cc3ca7018a3f25209f167963a94eb3933c"
MANIFEST = "var/artifacts/e117_successor_effective_contract_v11.json"
R2_LEDGER = "var/artifacts/e117r2_same_plumbing_component_preflight_jobs.json"
R2_AUDIT = "var/artifacts/e117r2_same_plumbing_component_preflight_audit.json"
LEDGER = "var/artifacts/e117_stage1_development_jobs.json"
AUDIT_JOB = "var/artifacts/e117_stage1_development_audit_job.json"
AUDIT_OUTPUT = "var/artifacts/e117_stage1_development_audit.json"
RESULTS_OUTPUT = "paper/results/e117_stage1_development.json"
TABLE_OUTPUT = "var/artifacts/e117_stage1_development_table.jsonl"
BASE_SNAPSHOT = "var/artifacts/source_snapshots/e76_tuned_scale_aced1ee8338a0cf1"
OFFICIAL_R2_AUDIT_JOB = 30978841
PARENT_ACTOR_SHA256 = "b9f52af1b185e05b8723d97f64e2d168cb2512bb42284b8f1dc3255f7c38e185"

SENTINELS = (
    ("qwen05b", "countdown"),
    ("qwen05b", "graph_coloring"),
    ("qwen05b", "python_factors"),
    ("falcon1b", "mathir"),
)
ARMS = ("c", "p", "f")
ARM_LABELS = {
    "c": "compute_only_control",
    "p": "proposal_replay",
    "f": "full_verified_support",
}
SEEDS = (201, 202, 203)
START_ORDERS = {201: ("c", "p", "f"), 202: ("p", "f", "c"), 203: ("f", "c", "p")}
NODES = {
    ("qwen05b", "countdown", 201): "node202",
    ("qwen05b", "countdown", 202): "node203",
    ("qwen05b", "countdown", 203): "node202",
    ("qwen05b", "graph_coloring", 201): "node203",
    ("qwen05b", "graph_coloring", 202): "node202",
    ("qwen05b", "graph_coloring", 203): "node203",
    ("qwen05b", "python_factors", 201): "node202",
    ("qwen05b", "python_factors", 202): "node203",
    ("qwen05b", "python_factors", 203): "node202",
    ("falcon1b", "mathir", 201): "node203",
    ("falcon1b", "mathir", 202): "node202",
    ("falcon1b", "mathir", 203): "node203",
}
TRAIN_ROWS = 384
PASSES = 8
TARGET_STEPS = 3072
EVAL_INTERVAL = 192
CHECKPOINTS = tuple(range(0, TARGET_STEPS + 1, EVAL_INTERVAL))
EVAL_DRAWS = 16
EVAL_SEED_BASE = 117900
RESUME_INTERVAL = 64
MAX_RESUME_CHECKPOINTS = 1

SNAPSHOT_OVERRIDES = (
    "src/oat_drgrpo/actor.py",
    "ops/exp_scaling/launch_e117_stage1_development.py",
    "ops/exp_scaling/audit_e117_stage1_development.py",
    "ops/exp_scaling/build_e117_stage1_development_results.py",
    "ops/exp_scaling/e117_stage1_statistics.py",
)


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def command(argv: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(argv)}\n"
            f"{result.stderr.strip()}"
        )
    return result


def show(job_id: int) -> str:
    return command(["scontrol", "show", "job", "-dd", "-o", str(job_id)]).stdout.strip()


def field(record: str, name: str) -> str:
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]*)", record)
    if match is None:
        raise RuntimeError(f"scheduler record lacks {name}")
    return match.group(1)


def validate_v11(repo: Path) -> dict[str, Any]:
    path = repo / MANIFEST
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "effective_contract": "effective_contract_sha256",
        "latest_chained_amendment_evidence": "latest_chained_amendment_evidence_sha256",
        "analysis_implementation": "analysis_implementation_sha256",
        "analysis_tests": "analysis_tests_sha256",
        "reserve_identity": "reserve_identity_sha256",
        "reserve_materializer": "reserve_materializer_sha256",
        "reserve_tests": "reserve_tests_sha256",
        "successor_ladder": "successor_ladder_sha256",
        "successor_ladder_evidence": "successor_ladder_evidence_sha256",
    }
    for path_key, digest_key in checks.items():
        candidate = repo / str(payload[path_key])
        if not candidate.is_file() or sha256(candidate) != payload[digest_key]:
            raise RuntimeError(f"v11 binding drifted: {path_key}")
    if payload.get("stage1_grid") != {
        "training_seeds": [201, 202, 203],
        "evaluation_draw_count": 16,
        "checkpoint_count": 17,
        "terminal_step": 3072,
        "passes": 8,
    }:
        raise RuntimeError("v11 Stage 1 grid drifted")
    identity = repo / str(payload["reserve_identity"])
    if sha256(identity) != "a5b9eb4289cca8f78d90249fabbd85343c1c3eb3b5c3df6a2235bbd124446289":
        raise RuntimeError("development reserve identity drifted")
    return payload


def validate_r2(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger_path = repo / R2_LEDGER
    audit_path = repo / R2_AUDIT
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        ledger.get("audit_job_id") != OFFICIAL_R2_AUDIT_JOB
        or audit.get("passed") is not True
        or audit.get("failures") != []
        or audit.get("stage1_execution_readiness", {}).get("ready") is not True
        or audit.get("stage1_execution_readiness", {}).get("blockers") != []
    ):
        raise RuntimeError("official E117-R2 audit does not authorize Stage 1")
    accounting = command(
        [
            "sacct", "-X", "-n", "-P", "-j", str(OFFICIAL_R2_AUDIT_JOB),
            "--format=JobIDRaw,State,ExitCode",
        ]
    ).stdout
    if f"{OFFICIAL_R2_AUDIT_JOB}|COMPLETED|0:0" not in accounting:
        raise RuntimeError("official E117-R2 audit job is not completed 0:0")
    return ledger, audit


def _files(path: Path) -> dict[str, str]:
    result = {}
    for candidate in sorted(value for value in path.rglob("*") if value.is_file()):
        if "__pycache__" in candidate.parts or candidate.suffix == ".pyc":
            continue
        relative = str(candidate.relative_to(path))
        if relative == "SNAPSHOT_IDENTITY.json":
            continue
        result[relative] = sha256(candidate)
    return result


def ensure_stage1_snapshot(repo: Path) -> tuple[Path, dict[str, Any]]:
    base = repo / BASE_SNAPSHOT
    if not base.is_dir() or sha256(base / "src/oat_drgrpo/actor.py") != PARENT_ACTOR_SHA256:
        raise RuntimeError("validated R2 base snapshot or parent actor drifted")
    for relative in SNAPSHOT_OVERRIDES:
        if not (repo / relative).is_file():
            raise RuntimeError(f"Stage 1 snapshot override is absent: {relative}")
    parent_files = _files(base)
    temporary = Path(tempfile.mkdtemp(prefix=".e117_stage1_", dir=base.parent))
    try:
        shutil.copytree(base, temporary, dirs_exist_ok=True)
        for relative in SNAPSHOT_OVERRIDES:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / relative, target)
        derived_files = _files(temporary)
        changed = sorted(
            relative
            for relative in set(parent_files) | set(derived_files)
            if parent_files.get(relative) != derived_files.get(relative)
        )
        if set(changed) != {
            relative
            for relative in SNAPSHOT_OVERRIDES
            if parent_files.get(relative) != derived_files.get(relative)
        }:
            raise RuntimeError(f"unregistered Stage 1 snapshot change: {changed}")
        if "src/oat_drgrpo/actor.py" not in changed:
            raise RuntimeError("Stage 1 telemetry repair did not change actor.py")
        identity = snapshot_util.tree_hash(
            (temporary / "src/oat_drgrpo", temporary / "ops")
        )
        destination = base.parent / f"e117_stage1_{identity[:16]}"
        identity_payload = {
            "schema": "e117_stage1_runtime_snapshot_v1",
            "sha256": identity,
            "parent_snapshot": str(base.resolve()),
            "parent_snapshot_identity_sha256": sha256(base / "SNAPSHOT_IDENTITY.json"),
            "parent_actor_sha256": PARENT_ACTOR_SHA256,
            "changed_files": [
                {
                    "path": relative,
                    "parent_sha256": parent_files.get(relative),
                    "derived_sha256": derived_files[relative],
                }
                for relative in changed
            ],
            "telemetry_only_actor_repair": True,
        }
        (temporary / "SNAPSHOT_IDENTITY.json").write_text(
            json.dumps(identity_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            existing = json.loads(
                (destination / "SNAPSHOT_IDENTITY.json").read_text(encoding="utf-8")
            )
            if existing != identity_payload:
                raise RuntimeError(f"existing Stage 1 snapshot conflicts: {destination}")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        return destination, identity_payload
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def r2_templates(ledger: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result = {}
    for run in ledger.get("runs", []):
        key = (str(run["scale"]), str(run["domain"]), str(run["arm"]))
        result[key] = run | {"environment": r2audit.exported_environment(str(run["held_scheduler_record"]))}
    expected = {(scale, domain, arm) for scale, domain in SENTINELS for arm in ARMS}
    if set(result) != expected:
        raise RuntimeError("R2 template ledger lacks the exact four C/P/F blocks")
    return result


def run_stamp(scale: str, domain: str, seed: int, arm: str) -> str:
    domain_tag = {"countdown": "countdown", "graph_coloring": "graph", "python_factors": "python", "mathir": "mathir"}[domain]
    return f"e117s1_{scale}_{domain_tag}_{ARM_LABELS[arm]}_s{seed}"


def build_environment(
    repo: Path,
    snapshot: Path,
    template: dict[str, Any],
    *,
    scale: str,
    domain: str,
    seed: int,
    arm: str,
) -> tuple[dict[str, str], Path]:
    environment = dict(template["environment"])
    stamp = run_stamp(scale, domain, seed, arm)
    model_tag = str(template["model_tag"])
    target = repo / "var/data" / f"xdr_{model_tag}_verified_replay_semantic_maxent_verified_support_discovery_{stamp}"
    environment.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": stamp,
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_SEED": str(seed),
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_DATA": str(repo / "var/data/e117_evaluation_reserve_v1/development" / domain / "eval"),
            "OAT_ZERO_TEST_SPLIT": "multi_answer",
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(EVAL_INTERVAL),
            "OAT_ZERO_EVAL_MODE_COVERAGE_K": "8",
            "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": str(EVAL_DRAWS),
            "OAT_ZERO_EVAL_MODE_COVERAGE_SEED": str(EVAL_SEED_BASE),
            "OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE": "1.0",
            "OAT_ZERO_SAVE_STEPS": str(EVAL_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(EVAL_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(RESUME_INTERVAL),
            "OAT_ZERO_RESUME_FROM": str(RESUME_INTERVAL),
            "OAT_ZERO_EXPORT_STEPS": "-1",
            "OAT_ZERO_MAX_SAVE_NUM": "2",
            "OAT_ZERO_MAX_RESUME_NUM": str(MAX_RESUME_CHECKPOINTS),
            "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS": "1",
            "OAT_ZERO_AUTO_RESUME": "1",
            "OAT_ZERO_WATCHDOG_REQUEUE": "1",
            "OAT_ZERO_SAVE_CKPT": "1",
            "OAT_ZERO_ALLOW_SPARSE_EVAL": "1",
            "OAT_ZERO_EVAL_BATCH_SIZE": "8",
            "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING": "1",
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY": "1" if arm == "c" else "0",
            "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.1" if arm == "f" else "0.0",
        }
    )
    if "confirmation" in environment["OAT_ZERO_EVAL_DATA"]:
        raise RuntimeError("Stage 1 attempted to bind the sealed confirmation reserve")
    return environment, target


def planned_cells(repo: Path, snapshot: Path, r2_ledger: dict[str, Any]) -> list[dict[str, Any]]:
    templates = r2_templates(r2_ledger)
    cells = []
    for scale, domain in SENTINELS:
        for seed in SEEDS:
            environments = {}
            for position, arm in enumerate(START_ORDERS[seed]):
                template = templates[(scale, domain, arm)]
                environment, target = build_environment(
                    repo, snapshot, template, scale=scale, domain=domain, seed=seed, arm=arm
                )
                environments[arm] = environment
                cells.append(
                    {
                        "scale": scale,
                        "model_tag": str(template["model_tag"]),
                        "domain": domain,
                        "seed": seed,
                        "arm": arm,
                        "arm_label": ARM_LABELS[arm],
                        "start_order": "-".join(value.upper() for value in START_ORDERS[seed]),
                        "start_position": position,
                        "node": NODES[(scale, domain, seed)],
                        "run_stamp": run_stamp(scale, domain, seed, arm),
                        "run_dir": str(target),
                        "environment": environment,
                    }
                )
            varying = {
                "SAVE_PATH", "RUN_STAMP", "OAT_ZERO_SEMANTIC_SHANNON_COEF",
                "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY",
            }
            common = {
                arm: {key: value for key, value in env.items() if key not in varying}
                for arm, env in environments.items()
            }
            if len({json.dumps(value, sort_keys=True) for value in common.values()}) != 1:
                raise RuntimeError(f"same-plumbing export drifted: {scale}/{domain}/{seed}")
    node_positions: dict[str, int] = {}
    for cell in cells:
        node = str(cell["node"])
        cell["node_lane_position"] = node_positions.get(node, 0)
        node_positions[node] = int(cell["node_lane_position"]) + 1
    if len(cells) != 36:
        raise RuntimeError(f"Stage 1 planned {len(cells)} cells, expected 36")
    return cells


def job_name(cell: dict[str, Any]) -> str:
    scale = "q05" if cell["scale"] == "qwen05b" else "f1"
    domain = {"countdown": "count", "graph_coloring": "graph", "python_factors": "py", "mathir": "math"}[str(cell["domain"])]
    return f"e117s1-{scale}-{domain}-s{cell['seed']}-{cell['arm']}"


def sbatch_command(repo: Path, snapshot: Path, cell: dict[str, Any], dependency: int | None) -> list[str]:
    export = "ALL," + ",".join(f"{key}={value}" for key, value in cell["environment"].items())
    argv = [
        "sbatch", "--parsable", "--hold", f"--job-name={job_name(cell)}",
        f"--export={export}", "--partition=all", "--account=allcs",
        f"--nodelist={cell['node']}", "--gres=gpu:a5000:1", "--cpus-per-task=8",
        "--mem=64G", "--time=7-00:00:00", "--nice=0", "--requeue",
        f"--chdir={repo}", str(snapshot / "ops/slurm/train_node302.slurm"),
    ]
    if dependency is not None:
        argv.insert(3, f"--dependency=afterok:{dependency}")
    return argv


def validate_held(record: str, cell: dict[str, Any], snapshot: Path, dependency: int | None) -> str:
    expected = {
        "JobState": "PENDING", "Reason": "JobHeldUser", "Account": "allcs",
        "Partition": "cs", "ReqNodeList": str(cell["node"]),
        "TresPerNode": "gres/gpu:a5000:1", "TimeLimit": "7-00:00:00", "Requeue": "1",
    }
    failures = {key: (field(record, key), value) for key, value in expected.items() if field(record, key) != value}
    actual_environment = r2audit.exported_environment(record)
    if actual_environment != cell["environment"]:
        failures["environment"] = ("scheduler export", "planned export")
    scheduler_dependency = field(record, "Dependency")
    if dependency is None:
        if scheduler_dependency not in ("", "(null)"):
            failures["dependency"] = (scheduler_dependency, "")
    elif f"afterok:{dependency}" not in scheduler_dependency:
        failures["dependency"] = (scheduler_dependency, f"afterok:{dependency}")
    if str(snapshot / "ops/slurm/train_node302.slurm") not in record:
        failures["command"] = ("mutable/unknown", "frozen wrapper")
    if failures:
        raise RuntimeError(f"held Stage 1 scheduler identity drifted: {failures}")
    return r2audit.exported_environment_text(record)


def schedule_audit(repo: Path, snapshot: Path, ledger: Path, job_ids: list[int]) -> tuple[int, str]:
    dependency = "afterany:" + ":".join(str(value) for value in job_ids)
    script = snapshot / "ops/exp_scaling/audit_e117_stage1_development.py"
    argv = [
        "sbatch", "--parsable", "--hold", f"--dependency={dependency}",
        "--job-name=e117s1-audit", "--partition=all", "--account=allcs",
        "--cpus-per-task=2", "--mem=16G", "--time=04:00:00", "--nice=0",
        "--no-requeue", f"--chdir={repo}",
        f"--output={repo / 'var/artifacts/logs/e117s1-audit-%j.out'}",
        f"--error={repo / 'var/artifacts/logs/e117s1-audit-%j.err'}",
        f"--wrap=python {script} --ledger {ledger} --output {repo / AUDIT_OUTPUT} --table {repo / TABLE_OUTPUT} --results {repo / RESULTS_OUTPUT}",
    ]
    result = command(argv)
    raw = result.stdout.strip().split(";", 1)[0]
    if not raw.isdigit():
        raise RuntimeError(f"invalid Stage 1 audit job id: {result.stdout!r}")
    job_id = int(raw)
    record = show(job_id)
    expected = {
        "JobState": "PENDING", "Reason": "JobHeldUser", "Account": "allcs",
        "Partition": "cs", "Requeue": "0", "TimeLimit": "04:00:00",
    }
    failures = {
        key: (field(record, key), value)
        for key, value in expected.items()
        if field(record, key) != value
    }
    dependency_record = field(record, "Dependency")
    missing_dependencies = [value for value in job_ids if str(value) not in dependency_record]
    if missing_dependencies:
        failures["dependency"] = (missing_dependencies, job_ids)
    if str(script) not in record:
        failures["command"] = ("unknown/mutable", str(script))
    if failures:
        raise RuntimeError(f"held Stage 1 terminal audit identity drifted: {failures}")
    return job_id, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit == args.dry_run:
        raise SystemExit("choose exactly one of --submit or --dry-run")
    repo = root()
    required = [
        repo / PROTOCOL, repo / SCHEDULER_AMENDMENT,
        repo / AUDIT_SCHEDULER_AMENDMENT, repo / STORAGE_AMENDMENT,
        repo / SUPERSEDED_LEDGER, repo / SUPERSEDED_AUDIT_JOB,
        repo / MANIFEST, repo / R2_LEDGER, repo / R2_AUDIT,
    ]
    if any(not value.is_file() for value in required):
        raise SystemExit(f"required Stage 1 provenance is absent: {[str(v) for v in required if not v.is_file()]}")
    if sha256(repo / SUPERSEDED_LEDGER) != SUPERSEDED_LEDGER_SHA256:
        raise SystemExit("superseded storage-unsafe Stage 1 ledger drifted")
    if sha256(repo / SUPERSEDED_AUDIT_JOB) != SUPERSEDED_AUDIT_JOB_SHA256:
        raise SystemExit("superseded storage-unsafe Stage 1 audit receipt drifted")
    v11 = validate_v11(repo)
    r2_ledger, r2_audit = validate_r2(repo)
    snapshot, snapshot_identity = ensure_stage1_snapshot(repo)
    cells = planned_cells(repo, snapshot, r2_ledger)
    existing_runs = [cell["run_dir"] for cell in cells if Path(cell["run_dir"]).exists()]
    if existing_runs:
        raise SystemExit(f"refusing to overwrite Stage 1 runs: {existing_runs[:3]}")
    ledger_path = repo / LEDGER
    audit_job_path = repo / AUDIT_JOB
    if args.submit and (ledger_path.exists() or audit_job_path.exists()):
        raise SystemExit("refusing duplicate E117 Stage 1 installation")
    if args.dry_run:
        previous: dict[str, int] = {}
        for index, cell in enumerate(cells, start=1):
            node = str(cell["node"])
            dependency = previous.get(node)
            print(shlex.join(sbatch_command(repo, snapshot, cell, dependency)))
            previous[node] = -index
        print(f"[e117s1] dry_run=True cells=36 snapshot={snapshot}")
        return 0

    submitted: list[int] = []
    records: list[dict[str, Any]] = []
    previous: dict[str, int] = {}
    audit_job_id = 0
    released = False
    try:
        for cell in cells:
            node = str(cell["node"])
            dependency = previous.get(node)
            result = command(sbatch_command(repo, snapshot, cell, dependency))
            raw = result.stdout.strip().split(";", 1)[0]
            if not raw.isdigit():
                raise RuntimeError(f"invalid Stage 1 job id: {result.stdout!r}")
            job_id = int(raw)
            submitted.append(job_id)
            held = show(job_id)
            export_text = validate_held(held, cell, snapshot, dependency)
            records.append(
                {key: cell[key] for key in (
                    "scale", "model_tag", "domain", "seed", "arm", "arm_label",
                    "start_order", "start_position", "node", "node_lane_position", "run_stamp", "run_dir",
                )}
                | {
                    "job_id": job_id,
                    "start_dependency_job_id": dependency,
                    "start_dependency_type": "afterok",
                    "held_scheduler_record": held,
                    "scientific_environment_sha256": hashlib.sha256(export_text.encode()).hexdigest(),
                }
            )
            previous[node] = job_id
        if len(records) != 36:
            raise RuntimeError("Stage 1 did not hold exactly 36 jobs")
        payload = {
            "schema": "e117_stage1_development_jobs_v1",
            "protocol": str((repo / PROTOCOL).resolve()),
            "protocol_sha256": sha256(repo / PROTOCOL),
            "scheduler_amendment": str((repo / SCHEDULER_AMENDMENT).resolve()),
            "scheduler_amendment_sha256": sha256(repo / SCHEDULER_AMENDMENT),
            "audit_scheduler_amendment": str((repo / AUDIT_SCHEDULER_AMENDMENT).resolve()),
            "audit_scheduler_amendment_sha256": sha256(repo / AUDIT_SCHEDULER_AMENDMENT),
            "launcher": str(snapshot / "ops/exp_scaling/launch_e117_stage1_development.py"),
            "storage_amendment": str((repo / STORAGE_AMENDMENT).resolve()),
            "storage_amendment_sha256": sha256(repo / STORAGE_AMENDMENT),
            "superseded_storage_unsafe_ledger": str((repo / SUPERSEDED_LEDGER).resolve()),
            "superseded_storage_unsafe_ledger_sha256": SUPERSEDED_LEDGER_SHA256,
            "superseded_storage_unsafe_audit_job": str((repo / SUPERSEDED_AUDIT_JOB).resolve()),
            "superseded_storage_unsafe_audit_job_sha256": SUPERSEDED_AUDIT_JOB_SHA256,
            "superseded_storage_unsafe_job_ids": list(range(30980336, 30980373)),
            "superseded_storage_unsafe_realized_runtime_seconds": 0,
            "launcher_sha256": sha256(snapshot / "ops/exp_scaling/launch_e117_stage1_development.py"),
            "terminal_auditor": str(snapshot / "ops/exp_scaling/audit_e117_stage1_development.py"),
            "terminal_auditor_sha256": sha256(snapshot / "ops/exp_scaling/audit_e117_stage1_development.py"),
            "table_builder": str(snapshot / "ops/exp_scaling/build_e117_stage1_development_results.py"),
            "table_builder_sha256": sha256(snapshot / "ops/exp_scaling/build_e117_stage1_development_results.py"),
            "effective_contract_manifest": str((repo / MANIFEST).resolve()),
            "effective_contract_manifest_sha256": sha256(repo / MANIFEST),
            "effective_contract_bindings": v11,
            "r2_ledger": str((repo / R2_LEDGER).resolve()),
            "r2_ledger_sha256": sha256(repo / R2_LEDGER),
            "r2_audit": str((repo / R2_AUDIT).resolve()),
            "r2_audit_sha256": sha256(repo / R2_AUDIT),
            "r2_official_audit_job_id": OFFICIAL_R2_AUDIT_JOB,
            "r2_stage1_readiness": r2_audit["stage1_execution_readiness"],
            "base_snapshot": str((repo / BASE_SNAPSHOT).resolve()),
            "base_snapshot_identity_sha256": sha256(repo / BASE_SNAPSHOT / "SNAPSHOT_IDENTITY.json"),
            "snapshot_root": str(snapshot),
            "snapshot_identity_sha256": sha256(snapshot / "SNAPSHOT_IDENTITY.json"),
            "snapshot_identity": snapshot_identity,
            "development_only": True,
            "confirmation_reserve_read": False,
            "pointmaze": "excluded",
            "sentinels": [f"{scale}/{domain}" for scale, domain in SENTINELS],
            "arms": list(ARMS),
            "training_seeds": list(SEEDS),
            "actual_start_orders": {str(seed): "-".join(v.upper() for v in order) for seed, order in START_ORDERS.items()},
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": EVAL_INTERVAL,
            "checkpoints": list(CHECKPOINTS),
            "evaluation_draws": list(range(EVAL_DRAWS)),
            "evaluation_request_seeds": [EVAL_SEED_BASE + draw for draw in range(EVAL_DRAWS)],
            "request_seed_projection": "each of 128 prompts records [117900 + draw] for n=8 fixed-seed sampling",
            "storage_policy": {
                "resume_interval_steps": RESUME_INTERVAL,
                "maximum_resume_checkpoints": MAX_RESUME_CHECKPOINTS,
                "prune_resume_on_success": True,
                "terminal_model_export": False,
            },
            "concurrency_policy": {
                "kind": "one completion-serialized lane per physical node",
                "dependency": "afterok",
                "nodes": ["node202", "node203"],
                "maximum_active_jobs": 2,
                "maximum_active_jobs_per_node": 1,
                "measured_worst_case_atomic_checkpoint_gib": 104,
            },
            "resource_envelope": {
                "gpu_class": "a5000", "gpus": 1, "cpus": 8, "memory": "64G",
                "requested_partition": "all", "effective_partition": "cs",
                "account": "allcs", "time_limit": "7-00:00:00",
            },
            "results_output": str((repo / RESULTS_OUTPUT).resolve()),
            "table_output": str((repo / TABLE_OUTPUT).resolve()),
            "runs": records,
            "released": False,
        }
        atomic_json(ledger_path, payload)
        audit_job_id, audit_record = schedule_audit(repo, snapshot, ledger_path, submitted)
        payload["audit_job_id"] = audit_job_id
        payload["audit_script"] = str(snapshot / "ops/exp_scaling/audit_e117_stage1_development.py")
        payload["audit_script_sha256"] = sha256(Path(payload["audit_script"]))
        audit_payload = {
            "schema": "e117_stage1_development_audit_job_v1",
            "audit_job_id": audit_job_id,
            "dependency_job_ids": submitted,
            "ledger": str(ledger_path),
            "audit_output": str((repo / AUDIT_OUTPUT).resolve()),
            "results_output": str((repo / RESULTS_OUTPUT).resolve()),
            "table_output": str((repo / TABLE_OUTPUT).resolve()),
            "scheduler_record": audit_record,
            "released": False,
        }
        atomic_json(ledger_path, payload)
        atomic_json(audit_job_path, audit_payload)
        for job_id in submitted:
            command(["scontrol", "release", str(job_id)])
        command(["scontrol", "release", str(audit_job_id)])
        released = True
        payload["released"] = True
        audit_payload["released"] = True
        atomic_json(ledger_path, payload)
        atomic_json(audit_job_path, audit_payload)
    except Exception:
        if not released:
            candidates = submitted + ([audit_job_id] if audit_job_id else [])
            if candidates:
                subprocess.run(["scancel", *[str(value) for value in candidates]], check=False)
            for path in (ledger_path, audit_job_path):
                if path.exists():
                    path.unlink()
        raise
    print(
        f"[e117s1] released=36 audit={audit_job_id} "
        f"jobs={','.join(str(value) for value in submitted)} snapshot={snapshot}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
