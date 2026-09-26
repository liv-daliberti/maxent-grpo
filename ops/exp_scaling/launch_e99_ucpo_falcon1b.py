#!/usr/bin/env python3
"""Submit E99: UCPO on all five Falcon3-1B static domains."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402
import launch_e82_falcon_semantic_maxent_verified_replay as e82  # noqa: E402
import launch_e97_ucpo_05b as e97  # noqa: E402


DOMAINS = e79.DOMAINS
SEEDS = e79.SEEDS
PASSES = e79.PASSES
TRAIN_ROWS = e79.TRAIN_ROWS
CHECKPOINT_INTERVAL = e79.CHECKPOINT_INTERVAL
TARGET_STEPS = e79.TARGET_STEPS
ARM = "ucpo"
VARIANT = "ucpo"
TAU = e97.TAU
LEDGER = "var/artifacts/e99_ucpo_falcon1b_jobs.json"
PROTOCOL = "paper/preregistration/e99_ucpo_falcon1b_20260814.md"
PAIR_LEDGER = e82.PAIR_LEDGER
SOURCE_MANIFEST = e79.SOURCE_MANIFEST
SNAPSHOT_PREFIX = "e99_ucpo_falcon1b"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_stamp(domain: str, seed: int) -> str:
    return f"e99_ucpo_falcon_{e79.DOMAIN_TAGS[domain]}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data" / f"xdr_{e79.MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"


def build_env(
    root: Path,
    run: dict[str, Any],
    snapshot: Path,
    falcon_root: Path,
) -> tuple[dict[str, str], Path]:
    env, _ = e79.build_env(root, run, "control", snapshot, falcon_root)
    domain, seed = str(run["domain"]), int(run["seed"])
    target = save_path(root, domain, seed)
    env.update({"SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain, seed)})
    env.update(e97.objective())
    return env, target


def smoke_env(
    root: Path,
    run: dict[str, Any],
    snapshot: Path,
    falcon_root: Path,
) -> tuple[dict[str, str], Path]:
    env, _ = build_env(root, run, snapshot, falcon_root)
    target = root / "var/data/e99_ucpo_falcon1b_smoke_graph_s55"
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": "e99_ucpo_falcon1b_smoke_graph_s55",
            "OAT_ZERO_MAX_TRAIN": "32",
            "OAT_ZERO_MAX_QUERIES": "32",
            "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
            "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": "32",
            "OAT_ZERO_SAVE_STEPS": "32",
            "OAT_ZERO_SAVE_FROM": "32",
            "OAT_ZERO_AUTO_RESUME": "0",
            "OAT_ZERO_WATCHDOG_REQUEUE": "0",
        }
    )
    return env, target


def sbatch_command(
    root: Path,
    run: dict[str, Any],
    env: dict[str, str],
    *,
    smoke: bool = False,
    dependency: str = "",
) -> list[str]:
    command = e79.sbatch_command(root, run, "control", env)
    domain, seed = str(run["domain"]), int(run["seed"])
    name = "e99-ucpo-smoke" if smoke else f"e99-{e79.DOMAIN_TAGS[domain][:6]}-ucpo-s{seed}"
    command = [
        f"--job-name={name}" if item.startswith("--job-name=") else item
        for item in command
    ]
    if dependency:
        command.insert(-1, f"--dependency=afterok:{dependency}")
    return command


def audit_held(
    job_id: str,
    *,
    name: str,
    run: dict[str, Any],
    falcon_root: Path,
    expected: tuple[str, ...],
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect held E99 job {job_id}: {result.stderr.strip()}")
    domain, seed = str(run["domain"]), int(run["seed"])
    node, gpu = e79.placement(domain, seed)
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName={name}",
        f"ReqNodeList={node}",
        f"gres/gpu:{gpu}=1",
        f"OAT_ZERO_PRETRAIN={falcon_root}",
        f"OAT_ZERO_SEED={seed}",
    ) + expected
    missing = [item for item in required if item not in result.stdout]
    if missing:
        raise RuntimeError(f"held E99 job {job_id} lacks {missing}")
    return result.stdout


def submit_held(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid sbatch response: {result.stdout!r}")
    return job_id


def cancel(job_ids: list[str]) -> None:
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=False)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    protocol, ledger = root / PROTOCOL, root / LEDGER
    for required in (protocol, root / SOURCE_MANIFEST, root / PAIR_LEDGER):
        if not required.is_file():
            raise SystemExit(f"required E99 input is absent: {required}")
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E99 submission: {ledger}")

    pair_payload = e82.pair_ledger(root)
    pairs = e82.pair_index(pair_payload)
    runs = e79.references(root)
    falcon_root = e79.model_root(root)
    base_snapshot = (
        args.snapshot_root.resolve()
        if args.snapshot_root
        else Path(str(pair_payload["snapshot_root"])).resolve()
    )
    snapshot, patched = e81.ensure_paired_snapshot(
        root,
        base_snapshot,
        prefix=SNAPSHOT_PREFIX,
        patched_files=e97.PATCHED_FILES,
    )
    e97.verify_snapshot(snapshot)

    cells: list[dict[str, Any]] = []
    for run in runs:
        domain, seed = str(run["domain"]), int(run["seed"])
        control = pairs[(domain, seed)]["control"]
        node, gpu = e79.placement(domain, seed)
        if (str(control["node"]), str(control["gpu"])) != (node, gpu):
            raise SystemExit(f"{domain}/s{seed}: E79 control placement drift")
        if not (Path(str(control["run_dir"])) / "TRAINING_COMPLETE.json").is_file():
            raise SystemExit(f"{domain}/s{seed}: E79 control is not terminal")
        env, target = build_env(root, run, snapshot, falcon_root)
        if target.exists():
            raise SystemExit(f"refusing to overwrite E99 output: {target}")
        cells.append({"run": run, "env": env, "target": target, "control": control})

    smoke_run = next(
        run for run in runs
        if str(run["domain"]) == "graph_coloring" and int(run["seed"]) == 55
    )
    smoke_vars, smoke_target = smoke_env(root, smoke_run, snapshot, falcon_root)
    if smoke_target.exists():
        raise SystemExit(f"refusing to overwrite E99 smoke: {smoke_target}")

    if args.dry_run or not args.submit:
        print(shlex.join(sbatch_command(root, smoke_run, smoke_vars, smoke=True)))
        for cell in cells:
            print(shlex.join(sbatch_command(root, cell["run"], cell["env"], dependency="SMOKE_JOB_ID")))
        print(f"[e99] smoke=1 scientific={len(cells)} snapshot={snapshot} patched={patched}")
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        smoke_id = submit_held(sbatch_command(root, smoke_run, smoke_vars, smoke=True))
        submitted.append(smoke_id)
        smoke_record = audit_held(
            smoke_id,
            name="e99-ucpo-smoke",
            run=smoke_run,
            falcon_root=falcon_root,
            expected=("OAT_ZERO_VARIANT=ucpo", "OAT_ZERO_UCPO_TAU=0.2", "OAT_ZERO_MAX_QUERIES=32"),
        )
        for cell in cells:
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            name = f"e99-{e79.DOMAIN_TAGS[domain][:6]}-ucpo-s{seed}"
            job_id = submit_held(sbatch_command(root, run, cell["env"], dependency=smoke_id))
            submitted.append(job_id)
            held = audit_held(
                job_id,
                name=name,
                run=run,
                falcon_root=falcon_root,
                expected=(
                    f"Dependency=afterok:{smoke_id}",
                    "OAT_ZERO_VARIANT=ucpo",
                    "OAT_ZERO_UCPO_TAU=0.2",
                    "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
                    "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1",
                    "OAT_ZERO_NUM_PROMPT_EPOCH=8",
                    "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
                ),
            )
            control = cell["control"]
            records.append(
                {
                    "domain": domain,
                    "arm": ARM,
                    "seed": seed,
                    "node": e79.placement(domain, seed)[0],
                    "gpu": e79.placement(domain, seed)[1],
                    "run_stamp": run_stamp(domain, seed),
                    "run_dir": str(cell["target"]),
                    "job_id": int(job_id),
                    "smoke_dependency_job_id": int(smoke_id),
                    "paired_e79_control": {
                        "job_id": int(control["job_id"]),
                        "run_stamp": str(control["run_stamp"]),
                        "run_dir": str(control["run_dir"]),
                    },
                    "held_scheduler_record": held,
                }
            )

        payload = {
            "schema": "e99_ucpo_falcon1b_jobs_v1",
            "cohort": "e99",
            "released": False,
            "model": "Falcon3-1B-Instruct",
            "model_revision": e79.MODEL_REVISION,
            "protocol": str(protocol),
            "protocol_sha256": e81.digest(protocol),
            "launcher_sha256": e81.digest(Path(__file__)),
            "source_manifest": str(root / SOURCE_MANIFEST),
            "source_manifest_sha256": e81.digest(root / SOURCE_MANIFEST),
            "pair_ledger": str(root / PAIR_LEDGER),
            "pair_ledger_sha256": e81.digest(root / PAIR_LEDGER),
            "snapshot_root": str(snapshot),
            "snapshot_patched_files": patched,
            "domains": list(DOMAINS),
            "seeds": list(SEEDS),
            "arms": [ARM],
            "inherited_arms": ["control"],
            "passes": PASSES,
            "train_rows": TRAIN_ROWS,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [i / 2 for i in range(2 * PASSES + 1)],
            "variant": VARIANT,
            "ucpo_tau": TAU,
            "objective": "DrGRPO_with_uniform_correct_advantage_redistribution",
            "scientific_difference": "against E79 control: UCPO tau=.2 advantage redistribution only",
            "smoke": {
                "scientific": False,
                "job_id": int(smoke_id),
                "run_dir": str(smoke_target),
                "max_queries": 32,
                "held_scheduler_record": smoke_record,
            },
            "runs": records,
        }
        e81.atomic_json(ledger, payload)
        for job_id in submitted:
            subprocess.run(["scontrol", "release", job_id], check=True)
        payload["released"] = True
        e81.atomic_json(ledger, payload)
    except Exception:
        cancel(submitted)
        raise

    print(f"[e99] released smoke {smoke_id} and {len(records)} dependent scientific cells")
    print(f"[e99] ledger {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
