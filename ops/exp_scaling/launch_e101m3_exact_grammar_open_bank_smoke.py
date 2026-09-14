#!/usr/bin/env python3
"""Submit E101m3: a 32-update exact-grammar open-bank mechanism smoke."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e101_open_bank_countdown_pilot as e101  # noqa: E402


SEED = 102
TRAIN_ROWS = 32
PASSES = 1
TARGET_STEPS = 32
MAX_ATTEMPTS = 1
PROTOCOL = "paper/preregistration/e101m3_exact_grammar_open_bank_smoke_20260814.md"
LEDGER = "var/artifacts/e101m3_exact_grammar_open_bank_smoke_job.json"
RUN_STAMP = f"e101m3_exact_grammar_open_bank_s{SEED}"
NODES = "node[007,020,022-023,101,103,202,204-206,302,403,805]"
PARTITION = "all"
ACCOUNT = "mltheory"
GRES = "gpu:1"
MEMORY = "22G"
TIME_LIMIT = "00:30:00"


def exact_env(root: Path, snapshot_root: Path) -> tuple[dict[str, str], Path]:
    env, _ = e101.build_env(root, e101.reference(root), "open", snapshot_root)
    target = root / "var/data" / (
        f"xdr_{e101.MODEL_TAG}_open_bank_maxent_replay_{RUN_STAMP}"
    )
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": RUN_STAMP,
            "OAT_ZERO_SEED": str(SEED),
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(TARGET_STEPS),
            "OAT_ZERO_EVAL_BATCH_SIZE": "8",
            "OAT_ZERO_EVAL_MODE_COVERAGE_K": "4",
            "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": "1",
            "OAT_ZERO_EVAL_MODE_COVERAGE_SEED": "830102",
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS": "0",
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS": "1",
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS": str(MAX_ATTEMPTS),
        }
    )
    return env, target


def sbatch_command(root: Path, env: dict[str, str]) -> list[str]:
    exports = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch", "--parsable", "--hold",
        f"--job-name=e101m3-exact-s{SEED}",
        f"--export=ALL,{exports}",
        f"--partition={PARTITION}", f"--account={ACCOUNT}",
        f"--nodelist={NODES}", f"--gres={GRES}",
        "--cpus-per-task=8", f"--mem={MEMORY}", f"--time={TIME_LIMIT}",
        "--nice=100", str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_audit(job_id: str) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E101m3 job: {result.stderr.strip()}")
    record = result.stdout
    required = (
        "JobState=PENDING", "Reason=JobHeldUser",
        f"JobName=e101m3-exact-s{SEED}", f"Account={ACCOUNT}",
        f"Partition={PARTITION}", f"ReqNodeList={NODES}", "gres/gpu=1",
        f"OAT_ZERO_SEED={SEED}", f"OAT_ZERO_MAX_TRAIN={TRAIN_ROWS}",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS=1",
        f"OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS={MAX_ATTEMPTS}",
        "OAT_ZERO_NUM_GPUS_PER_ACTOR=1", "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1",
        "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1", "OAT_ZERO_VLLM_SLEEP_LEVEL=1",
    )
    missing = [value for value in required if value not in record]
    if missing:
        raise RuntimeError(f"held E101m3 job {job_id} lacks {missing}")
    return record


def main() -> int:
    root = e101.repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    protocol = root / PROTOCOL
    ledger = root / LEDGER
    if not protocol.is_file():
        raise SystemExit(f"missing E101m3 protocol: {protocol}")
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E101m3 submission: {ledger}")

    snapshot_root = e101.snapshot_util.ensure_snapshot(root, None)
    env, target = exact_env(root, snapshot_root)
    if target.exists():
        raise SystemExit(f"refusing to overwrite E101m3 run: {target}")
    command = sbatch_command(root, env)
    if args.dry_run or not args.submit:
        print(" ".join(shlex.quote(part) for part in command))
        print(f"[e101m3] dry_run=True snapshot={snapshot_root}")
        return 0

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"E101m3 submission failed: {result.stderr.strip()}")
    job_id_text = result.stdout.strip().split(";", 1)[0]
    if not job_id_text.isdigit():
        raise RuntimeError(f"invalid E101m3 job id: {result.stdout!r}")
    job_id = int(job_id_text)
    try:
        held_record = held_audit(job_id_text)
        payload: dict[str, Any] = {
            "schema": "e101m3_exact_grammar_open_bank_smoke_job_v1",
            "protocol": str(protocol),
            "protocol_sha256": e101.digest(protocol),
            "launcher_sha256": e101.digest(Path(__file__)),
            "snapshot_root": str(snapshot_root),
            "snapshot_identity_sha256": e101.digest(snapshot_root / "SNAPSHOT_IDENTITY.json"),
            "data_root": str(root / e101.DATA_ROOT),
            "data_tree_sha256": e101.tree_digest(root / e101.DATA_ROOT),
            "seed": SEED, "train_rows": TRAIN_ROWS, "passes": PASSES,
            "target_steps": TARGET_STEPS, "max_attempts": MAX_ATTEMPTS,
            "registered_evaluation_steps": [0, TARGET_STEPS],
            "run_stamp": RUN_STAMP, "run_dir": str(target), "job_id": job_id,
            "stdout": str(root / f"var/artifacts/logs/e101m3-exact-s{SEED}-{job_id}.out"),
            "stderr": str(root / f"var/artifacts/logs/e101m3-exact-s{SEED}-{job_id}.err"),
            "command": command, "held_scheduler_record": held_record,
            "placement": {"partition": PARTITION, "account": ACCOUNT,
                          "nodes": [NODES], "gres": GRES, "memory": MEMORY,
                          "time_limit": TIME_LIMIT},
            "interpretation": "exact_grammar_mechanism_only_no_performance_comparison",
            "released": False,
        }
        e101.atomic_json(ledger, payload)
        subprocess.run(["scontrol", "release", job_id_text], check=True)
        payload["released"] = True
        e101.atomic_json(ledger, payload)
    except Exception:
        subprocess.run(["scancel", job_id_text], check=False)
        if ledger.exists():
            ledger.unlink()
        raise
    print(f"[e101m3] job={job_id} released=True run={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
