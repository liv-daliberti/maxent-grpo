#!/usr/bin/env python3
"""Submit E121 fixed-bank survival telemetry after all E120-R1 science jobs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e78_verified_replay_only_05b as e78  # noqa: E402


PROTOCOL = "paper/preregistration/e121_fixed_bank_survival_telemetry_20260903.md"
LEDGER = "var/artifacts/e121_fixed_bank_survival_telemetry_jobs.json"
E120_LEDGER = "var/artifacts/e120r1_frequency_weighted_replay_jobs.json"
DOMAIN = "graph_coloring"
SEEDS = (43, 44, 45, 46, 47)
PASSES = 8
TRAIN_ROWS = 384
TARGET_STEPS = PASSES * TRAIN_ROWS
FREEZE_STEP = 384
NODES = ("node202", "node203", "node204")


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def refuse_pvl(value: object, *, label: str) -> None:
    if "pvl" in str(value).lower():
        raise RuntimeError(f"E121 refuses PVL in {label}: {value}")


def source_gate(repo: Path) -> dict[str, object]:
    python = repo / "var/seed_paper_eval/paper310/bin/python"
    library = repo / "var/seed_paper_eval/paper310/lib"
    sources = [
        "src/oat_drgrpo/online_canonical_bank.py",
        "src/oat_drgrpo/args.py",
        "src/oat_drgrpo/learner/grpo.py",
        "ops/exp_scaling/launch_e121_fixed_bank_survival_telemetry.py",
        "ops/exp_scaling/audit_e121_fixed_bank_survival.py",
        "tests/test_online_canonical_bank.py",
        "tests/test_args.py",
        "tests/test_e121_fixed_bank_survival.py",
    ]
    compiled = subprocess.run(
        [str(python), "-m", "py_compile", *sources],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if compiled.returncode:
        raise RuntimeError(compiled.stderr)
    shell_checked = subprocess.run(
        [
            "bash",
            "-n",
            "ops/slurm/e121_dependency_barrier.slurm",
            "ops/slurm/e121_resource_fenced_train.slurm",
            "ops/train.sh",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if shell_checked.returncode:
        raise RuntimeError(shell_checked.stderr)
    env = dict(os.environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["LD_LIBRARY_PATH"] = str(library)
    tested = subprocess.run(
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "tests/test_online_canonical_bank.py",
            "tests/test_args.py",
            "tests/test_e121_fixed_bank_survival.py",
            "tests/test_cohort_registry.py",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if tested.returncode:
        raise RuntimeError(tested.stdout + tested.stderr)
    return {
        "compile": "passed",
        "shell_syntax": "passed",
        "pytest": "passed",
        "pytest_tail": tested.stdout.strip().splitlines()[-1],
        "e121_outcomes_inspected": False,
    }


def e120_prerequisites(repo: Path) -> tuple[dict[str, object], list[str]]:
    path = repo / E120_LEDGER
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("runs")
    if payload.get("released") is not True or not isinstance(runs, list) or len(runs) != 45:
        raise RuntimeError("E120-R1 prerequisite ledger is not its released 45-cell form")
    ids = [str(int(run["job_id"])) for run in runs]
    if len(set(ids)) != 45:
        raise RuntimeError("E120-R1 prerequisite job IDs are not unique")
    return payload, ids


def live_and_completed_prerequisites(
    prerequisite_ids: list[str],
) -> tuple[list[str], list[str]]:
    queue = subprocess.run(
        ["squeue", "-h", "-j", ",".join(prerequisite_ids), "-o", "%A"],
        capture_output=True,
        text=True,
        check=False,
    )
    if queue.returncode:
        raise RuntimeError(queue.stderr.strip())
    live = sorted(
        {line.strip() for line in queue.stdout.splitlines() if line.strip()},
        key=int,
    )
    live_set = set(live)
    finished = [job_id for job_id in prerequisite_ids if job_id not in live_set]
    if finished:
        accounting = subprocess.run(
            [
                "sacct",
                "-n",
                "-X",
                "-j",
                ",".join(finished),
                "--format=JobIDRaw,State",
                "-P",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if accounting.returncode:
            raise RuntimeError(accounting.stderr.strip())
        states = {}
        for line in accounting.stdout.splitlines():
            if not line.strip():
                continue
            job_id, state = line.split("|", 1)
            states[job_id] = state.split()[0].split("+")[0]
        invalid = {job_id: states.get(job_id) for job_id in finished if states.get(job_id) != "COMPLETED"}
        if invalid:
            raise RuntimeError(
                f"E120-R1 prerequisite is neither live nor completed: {invalid}"
            )
    if len(live) + len(finished) != 45:
        raise RuntimeError("E120-R1 prerequisite accounting lost a job")
    return live, finished


def chunks(values: list[str], size: int = 10) -> list[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [values[start : start + size] for start in range(0, len(values), size)]


def cells(repo: Path) -> list[dict[str, Any]]:
    selected = [
        run
        for run in e78.references(repo)
        if str(run["domain"]) == DOMAIN and int(run["seed"]) in SEEDS
    ]
    if {int(run["seed"]) for run in selected} != set(SEEDS) or len(selected) != 5:
        raise RuntimeError("E121 template grid is not exactly Graph seeds 43--47")
    return sorted(selected, key=lambda run: int(run["seed"]))


def run_stamp(seed: int) -> str:
    return f"e121_qwen05b_graph_fixed_bank_survival_s{seed}"


def run_dir(repo: Path, seed: int) -> Path:
    return repo / "var/data" / run_stamp(seed)


def environment(
    repo: Path, template: dict[str, Any], snapshot: Path
) -> tuple[dict[str, str], Path]:
    seed = int(template["seed"])
    env, _ = e78.build_env(repo, template, "replay", snapshot)
    target = run_dir(repo, seed)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(seed),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING": "uniform",
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_FREEZE_STEP": str(FREEZE_STEP),
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "0",
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS": "0",
            "E121_FIXED_BANK_TELEMETRY": "1",
        }
    )
    return env, target


def placement(seed: int) -> dict[str, str]:
    return {
        "partition": "cs",
        "account": "allcs",
        "node": NODES[SEEDS.index(seed) % len(NODES)],
        "gpu": "a5000",
        "cpus": "8",
        "mem": "64G",
        "time": "1-12:00:00",
    }


def barrier_command(
    repo: Path, snapshot: Path, index: int, prerequisite_ids: list[str]
) -> list[str]:
    if not prerequisite_ids or len(prerequisite_ids) > 10:
        raise ValueError("E121 barrier requires one to ten prerequisites")
    dependency = "afterok:" + ":".join(prerequisite_ids)
    result = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name=e121-barrier-{index:02d}",
        "--partition=cs",
        "--account=allcs",
        "--nodelist=node202",
        "--cpus-per-task=1",
        "--mem=256M",
        "--time=00:05:00",
        "--nice=100",
        f"--dependency={dependency}",
        "--chdir=" + str(repo),
        str(snapshot / "ops/slurm/e121_dependency_barrier.slurm"),
    ]
    refuse_pvl(" ".join(result), label="barrier submission command")
    return result


def command(
    repo: Path,
    seed: int,
    env: dict[str, str],
    snapshot: Path,
    dependency_ids: list[str],
) -> list[str]:
    place = placement(seed)
    if not dependency_ids or len(dependency_ids) > 10:
        raise ValueError("E121 science requires one to ten fan-in barriers")
    dependency = "afterok:" + ":".join(dependency_ids)
    exports = ",".join(f"{key}={value}" for key, value in env.items())
    result = [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name=e121-graph-s{seed}",
        f"--export=ALL,{exports}",
        f"--partition={place['partition']}",
        f"--account={place['account']}",
        f"--nodelist={place['node']}",
        f"--gres=gpu:{place['gpu']}:1",
        f"--cpus-per-task={place['cpus']}",
        f"--mem={place['mem']}",
        f"--time={place['time']}",
        "--nice=100",
        "--requeue",
        f"--dependency={dependency}",
        "--chdir=" + str(repo),
        str(snapshot / "ops/slurm/e121_resource_fenced_train.slurm"),
    ]
    refuse_pvl(" ".join(result), label="submission command")
    return result


def submit(parts: list[str]) -> str:
    result = subprocess.run(parts, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid Slurm job id: {result.stdout!r}")
    return job_id


def audit_barrier(job_id: str, index: int) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    record = result.stdout
    refuse_pvl(record, label=f"held barrier scheduler record {job_id}")
    required = [
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName=e121-barrier-{index:02d}",
        "Account=allcs",
        "ReqNodeList=node202",
        "Dependency=afterok:",
    ]
    missing = [value for value in required if value not in record]
    if not any(value in record for value in ("Partition=cs", "Partition=all")):
        missing.append("effective Partition in {cs,all}")
    if missing:
        raise RuntimeError(f"held E121 barrier {job_id} lacks {missing}")
    return record


def audit_held(job_id: str, seed: int, snapshot: Path) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    record = result.stdout
    refuse_pvl(record, label=f"held scheduler record {job_id}")
    place = placement(seed)
    required = [
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName=e121-graph-s{seed}",
        f"Account={place['account']}",
        f"Partition={place['partition']}",
        f"ReqNodeList={place['node']}",
        f"gres/gpu:{place['gpu']}:1",
        "Dependency=afterok:",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING=uniform",
        f"OAT_ZERO_ONLINE_CANONICAL_REPLAY_BANK_FREEZE_STEP={FREEZE_STEP}",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "E121_FIXED_BANK_TELEMETRY=1",
    ]
    missing = [value for value in required if value not in record]
    if not any(value in record for value in ("Partition=cs", "Partition=all")):
        missing.append("effective Partition in {cs,all}")
    if missing:
        raise RuntimeError(f"held E121 job {job_id} lacks {missing}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run")

    repo = root()
    protocol = repo / PROTOCOL
    ledger = repo / LEDGER
    if not protocol.is_file():
        raise SystemExit(f"missing E121 preregistration: {protocol}")
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E121 submission: {ledger}")

    e120, prerequisite_ids = e120_prerequisites(repo)
    live_prerequisites, completed_prerequisites = live_and_completed_prerequisites(
        prerequisite_ids
    )
    if not live_prerequisites:
        raise RuntimeError(
            "E120-R1 is already complete; submit E121 without obsolete barriers "
            "under a preregistered operational amendment"
        )
    prerequisite_chunks = chunks(live_prerequisites)
    gate = source_gate(repo)
    snapshot = e78.snapshot_util.ensure_snapshot(repo, args.snapshot_root)
    for required in (
        snapshot / "src/oat_drgrpo/online_canonical_bank.py",
        snapshot / "ops/slurm/e121_resource_fenced_train.slurm",
        snapshot / "ops/slurm/e121_dependency_barrier.slurm",
        snapshot / "ops/exp_scaling/audit_e121_fixed_bank_survival.py",
    ):
        if not required.is_file():
            raise RuntimeError(f"E121 snapshot lacks {required}")

    planned: list[dict[str, Any]] = []
    for template in cells(repo):
        seed = int(template["seed"])
        env, target = environment(repo, template, snapshot)
        if target.exists():
            raise SystemExit(f"refusing existing E121 run directory: {target}")
        planned.append(
            {
                "model_key": "qwen05b",
                "model": "Qwen2.5-0.5B-Instruct",
                "domain": DOMAIN,
                "seed": seed,
                "run_stamp": run_stamp(seed),
                "run_dir": str(target),
                "placement": placement(seed),
                "environment": env,
            }
        )

    if args.dry_run or not args.submit:
        for index, group in enumerate(prerequisite_chunks):
            preview = barrier_command(repo, snapshot, index, group)
            print(" ".join(shlex.quote(part) for part in preview))
        placeholders = [f"<barrier_{index:02d}>" for index in range(len(prerequisite_chunks))]
        for item in planned:
            preview = command(
                repo, int(item["seed"]), item["environment"], snapshot, placeholders
            )
            print(" ".join(shlex.quote(part) for part in preview))
        print(
            f"[e121] dry_run=True cells=5 freeze_step={FREEZE_STEP} "
            f"e120_live_dependencies={len(live_prerequisites)} "
            f"e120_completed={len(completed_prerequisites)} "
            f"barriers={len(prerequisite_chunks)} pvl=forbidden"
        )
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    barrier_records: list[dict[str, Any]] = []
    try:
        barrier_ids: list[str] = []
        for index, group in enumerate(prerequisite_chunks):
            job_id = submit(barrier_command(repo, snapshot, index, group))
            submitted.append(job_id)
            barrier_ids.append(job_id)
            barrier_records.append(
                {
                    "index": index,
                    "job_id": int(job_id),
                    "e120_job_ids": [int(value) for value in group],
                    "held_scheduler_record": audit_barrier(job_id, index),
                }
            )
        for item in planned:
            science_command = command(
                repo,
                int(item["seed"]),
                item["environment"],
                snapshot,
                barrier_ids,
            )
            job_id = submit(science_command)
            submitted.append(job_id)
            held = audit_held(job_id, int(item["seed"]), snapshot)
            records.append(
                {
                    key: item[key]
                    for key in (
                        "model_key",
                        "model",
                        "domain",
                        "seed",
                        "run_stamp",
                        "run_dir",
                        "placement",
                    )
                }
                | {"job_id": int(job_id), "held_scheduler_record": held}
            )
        payload = {
            "schema": "e121_fixed_bank_survival_telemetry_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e78.digest(protocol),
            "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": e78.digest(Path(__file__)),
            "snapshot_root": str(snapshot),
            "snapshot_identity_sha256": json.loads(
                (snapshot / "SNAPSHOT_IDENTITY.json").read_text(encoding="utf-8")
            )["sha256"],
            "model": "Qwen2.5-0.5B-Instruct",
            "domains": [DOMAIN],
            "seeds": list(SEEDS),
            "passes": PASSES,
            "train_rows": TRAIN_ROWS,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": 192,
            "bank_freeze_step": FREEZE_STEP,
            "identity_telemetry": [
                "prompt_fingerprint",
                "outcome_fingerprint",
                "membership_fingerprint",
                "mean_logprob",
                "sequence_logprob",
            ],
            "source_gate": gate,
            "e120_prerequisite": {
                "ledger": str(repo / E120_LEDGER),
                "ledger_sha256": e78.digest(repo / E120_LEDGER),
                "schema": e120["schema"],
                "job_ids": [int(value) for value in prerequisite_ids],
                "completed_before_submission_job_ids": [
                    int(value) for value in completed_prerequisites
                ],
                "live_dependency_job_ids": [
                    int(value) for value in live_prerequisites
                ],
                "dependency": "transitive_afterok_all_45",
                "fan_in_maximum": 10,
                "barriers": barrier_records,
            },
            "pvl_compute_allowed": False,
            "scheduler_record_pvl_substring_audit": "passed",
            "runs": records,
            "released": False,
            "e121_outcomes_inspected_before_release": False,
        }
        e78.atomic_json(ledger, payload)
        for job_id in submitted:
            released = subprocess.run(
                ["scontrol", "release", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if released.returncode:
                raise RuntimeError(
                    f"failed to release E121 job {job_id}: {released.stderr.strip()}"
                )
        payload["released"] = True
        e78.atomic_json(ledger, payload)
    except Exception:
        e78.cancel(submitted)
        if ledger.exists():
            ledger.unlink()
        raise

    print(
        f"[e121] science={len(records)} released={len(submitted)} "
        f"barriers={len(barrier_records)} "
        f"dependency=transitive_afterok_all_{len(prerequisite_ids)} snapshot={snapshot} "
        f"ledger={ledger} pvl=forbidden"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
