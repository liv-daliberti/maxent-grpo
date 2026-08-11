#!/usr/bin/env python3
"""Submit E88: adaptive semantic MaxEnt on verified replay, Qwen2.5-0.5B.

E88 is E81 with one thing changed: the semantic coefficient is chosen by a
controller targeting a registered ratio of semantic-advantage RMS to
task-advantage RMS, instead of being held at .10. Its primary comparator is
E81 itself, which isolates the adaptation.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402


DOMAINS = e81.DOMAINS
ARM = "adaptive_semantic"
SEEDS = e81.SEEDS
PASSES = e81.PASSES
TRAIN_ROWS = e81.TRAIN_ROWS
CHECKPOINT_INTERVAL = e81.CHECKPOINT_INTERVAL
TARGET_STEPS = e81.TARGET_STEPS
REPLAY_WEIGHT = e81.REPLAY_WEIGHT
MODEL_TAG = e81.MODEL_TAG
VARIANT = "adaptive_semantic_maxent_replay"

# Registered controller settings. The target ratio is read from E81's own
# mechanism telemetry, never from any E81 outcome: at a fixed eta = .10 the
# realized ratio measured 1.0%, 1.6%, 1.9%, and 3.7% across domains, so .05
# sits just above the top of the observed band and is a genuine intervention
# in every domain rather than a no-op in some and a large change in others.
START_COEFFICIENT = 0.10
TARGET_RATIO = 0.05
MIN_COEFFICIENT = 0.02
MAX_COEFFICIENT = 0.40
EMA_DECAY = 0.98
GAIN = 0.5
MAX_STEP_RATIO = 1.1
WARMUP_STEPS = 64
MIN_ELIGIBLE_FRACTION = 0.05

LEDGER = "var/artifacts/e88_adaptive_semantic_maxent_05b_jobs.json"
PROTOCOL = "paper/preregistration/e88_adaptive_semantic_maxent_05b_20260810.md"
PAIR_LEDGER = "var/artifacts/e81_semantic_maxent_verified_replay_05b_jobs.json"
SNAPSHOT_PREFIX = "e88_adaptive_semantic_maxent"

# Wider than E81's patch set: the controller is new code, and the semantic key
# repair rides along so PantryPlan is live from the first update.
PATCHED_FILES = (
    "src/oat_drgrpo/args.py",
    "ops/run_experiment.sh",
    "ops/train.sh",
    "src/oat_drgrpo/learner/grpo.py",
    "src/oat_drgrpo/learner/init.py",
    "src/oat_drgrpo/learner/run.py",
    "src/oat_drgrpo/semantic_rms_controller.py",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_stamp(domain: str, seed: int) -> str:
    return f"e88_adaptive_semantic_{e81.DOMAIN_TAGS[domain]}_{ARM}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data" / f"xdr_{MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"


def fixed_objective() -> dict[str, str]:
    """E81's objective plus the controller; the fixed dose becomes the start."""

    objective = dict(e81.fixed_objective())
    objective["OAT_ZERO_VARIANT"] = VARIANT
    objective["OAT_ZERO_SEMANTIC_SHANNON_COEF"] = repr(START_COEFFICIENT)
    objective.update(
        {
            "OAT_ZERO_SEMANTIC_RMS_CONTROL": "1",
            "OAT_ZERO_SEMANTIC_RMS_TARGET_RATIO": repr(TARGET_RATIO),
            "OAT_ZERO_SEMANTIC_RMS_MIN_COEFFICIENT": repr(MIN_COEFFICIENT),
            "OAT_ZERO_SEMANTIC_RMS_MAX_COEFFICIENT": repr(MAX_COEFFICIENT),
            "OAT_ZERO_SEMANTIC_RMS_EMA_DECAY": repr(EMA_DECAY),
            "OAT_ZERO_SEMANTIC_RMS_GAIN": repr(GAIN),
            "OAT_ZERO_SEMANTIC_RMS_MAX_STEP_RATIO": repr(MAX_STEP_RATIO),
            "OAT_ZERO_SEMANTIC_RMS_WARMUP_STEPS": str(WARMUP_STEPS),
            "OAT_ZERO_SEMANTIC_RMS_MIN_ELIGIBLE_FRACTION": repr(MIN_ELIGIBLE_FRACTION),
        }
    )
    if float(objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) != REPLAY_WEIGHT:
        raise RuntimeError("E88 replay weight drifted from E81")
    return objective


def build_env(
    root: Path, run: dict[str, Any], snapshot: Path
) -> tuple[dict[str, str], Path]:
    domain, seed = str(run["domain"]), int(run["seed"])
    env, _ = e81.build_env(root, run, snapshot)
    target = save_path(root, domain, seed)
    env.update({"SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain, seed)})
    env.update(fixed_objective())
    return env, target


def sbatch_command(root: Path, run: dict[str, Any], env: dict[str, str]) -> list[str]:
    node = str(run["source_node"])
    place = e81.placement(node)
    name = f"e88-{e81.DOMAIN_TAGS[str(run['domain'])][:6]}-adp-s{run['seed']}"
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch", "--parsable", "--hold",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        f"--partition={place['partition']}",
        f"--account={place['account']}",
        f"--nodelist={node}",
        f"--gres={place['gres']}",
        "--cpus-per-task=8", "--mem=64G", "--time=1-12:00:00", "--nice=100",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(job_id: str, run: dict[str, Any]) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E88 job {job_id}: {result.stderr.strip()}")
    record = result.stdout
    required = (
        "JobState=PENDING", "Reason=JobHeldUser",
        f"JobName=e88-{e81.DOMAIN_TAGS[str(run['domain'])][:6]}-adp-s{run['seed']}",
        f"ReqNodeList={run['source_node']}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={run['seed']}",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
        "OAT_ZERO_SEMANTIC_RMS_CONTROL=1",
        "OAT_ZERO_SEMANTIC_RMS_TARGET_RATIO=0.05",
        "OAT_ZERO_SEMANTIC_RMS_MIN_COEFFICIENT=0.02",
        "OAT_ZERO_SEMANTIC_RMS_MAX_COEFFICIENT=0.4",
        "OAT_ZERO_SEMANTIC_RMS_WARMUP_STEPS=64",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_MAXENT_ALPHA=0.0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
    )
    missing = [text for text in required if text not in record]
    if missing:
        raise RuntimeError(f"held E88 job {job_id} lacks {missing}")
    return record


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    protocol = root / PROTOCOL
    if not protocol.is_file():
        raise SystemExit(f"required frozen input is absent: {protocol}")
    ledger_path = root / LEDGER
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate E88 submission: {ledger_path}")

    parent = json.loads((root / PAIR_LEDGER).read_text(encoding="utf-8"))
    if not parent.get("released"):
        raise SystemExit("E81 comparator ledger is not released")
    fixed_by_cell = {
        (str(r["domain"]), int(r["seed"])): r for r in parent["runs"]
    }
    snapshot, patched = e81.ensure_paired_snapshot(
        root, Path(parent["e78_snapshot_root"]),
        prefix=SNAPSHOT_PREFIX, patched_files=PATCHED_FILES,
    )

    planned = []
    for run in e81.references(root):
        domain, seed = str(run["domain"]), int(run["seed"])
        env, target = build_env(root, run, snapshot)
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing E88 run: {target}")
        planned.append({
            "domain": domain, "arm": ARM, "seed": seed,
            "source_node": str(run["source_node"]),
            "run_stamp": run_stamp(domain, seed), "run_dir": str(target),
            "paired_fixed_run": {
                "run_stamp": str(fixed_by_cell[(domain, seed)]["run_stamp"]),
                "run_dir": str(fixed_by_cell[(domain, seed)]["run_dir"]),
                "job_id": int(fixed_by_cell[(domain, seed)]["job_id"]),
            },
            "command": sbatch_command(root, run, env), "template": run,
        })

    if len(planned) != 25:
        raise SystemExit(f"E88 expected 25 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(p) for p in cell["command"]))
        print(f"[e88] dry_run=True cells={len(planned)} snapshot={snapshot} "
              f"patched={len(patched)} files")
        return 0

    submitted, records = [], []
    try:
        for cell in planned:
            result = subprocess.run(cell["command"], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"submission failed for {cell['run_stamp']}: {result.stderr.strip()}")
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"invalid job id: {result.stdout!r}")
            submitted.append(job_id)
            held = held_job_audit(job_id, cell["template"])
            records.append({k: cell[k] for k in
                            ("domain", "arm", "seed", "source_node", "run_stamp",
                             "run_dir", "paired_fixed_run")}
                           | {"job_id": int(job_id), "held_scheduler_record": held})
        payload = {
            "schema": "e88_adaptive_semantic_maxent_05b_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e81.digest(protocol),
            "launcher_sha256": e81.digest(Path(__file__)),
            "pair_ledger": str(root / PAIR_LEDGER),
            "pair_ledger_sha256": e81.digest(root / PAIR_LEDGER),
            "snapshot_root": str(snapshot),
            "snapshot_patched_files": patched,
            "model": "Qwen2.5-0.5B-Instruct",
            "domains": list(DOMAINS), "arms": [ARM], "seeds": list(SEEDS),
            "inherited_arms": ["control", "replay", "semantic"],
            "train_rows": TRAIN_ROWS, "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [i / 2 for i in range(17)],
            "replay_weight": REPLAY_WEIGHT,
            "semantic_coefficient": START_COEFFICIENT,
            "controller": {
                "kind": "semantic_task_advantage_rms_ratio_v1",
                "target_ratio": TARGET_RATIO,
                "min_coefficient": MIN_COEFFICIENT,
                "max_coefficient": MAX_COEFFICIENT,
                "ema_decay": EMA_DECAY, "gain": GAIN,
                "max_step_ratio": MAX_STEP_RATIO,
                "warmup_steps": WARMUP_STEPS,
                "min_eligible_fraction": MIN_ELIGIBLE_FRACTION,
            },
            "objective": "adaptive_open_set_semantic_maxent_on_verified_replay",
            "scientific_difference": "against E81: how eta is chosen, nothing else",
            "runs": records, "released": False,
        }
        e81.atomic_json(ledger_path, payload)
        for job_id in submitted:
            release = subprocess.run(["scontrol", "release", job_id],
                                     capture_output=True, text=True, check=False)
            if release.returncode != 0:
                raise RuntimeError(f"release failed for {job_id}: {release.stderr.strip()}")
        payload["released"] = True
        e81.atomic_json(ledger_path, payload)
    except Exception:
        e81.cancel(submitted)
        if ledger_path.exists():
            ledger_path.unlink()
        raise

    print(f"[e88] cells={len(records)} released={len(submitted)} "
          f"snapshot={snapshot} ledger={ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
