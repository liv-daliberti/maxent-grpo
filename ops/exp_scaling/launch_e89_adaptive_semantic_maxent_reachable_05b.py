#!/usr/bin/env python3
"""Submit E89: adaptive semantic MaxEnt at a globally reachable target ratio.

E88 targeted rho = .05 and two of five domains welded to the coefficient
ceiling without reaching it. The ceiling is not adjustable: |A_sem| <= eta
against a unit reward gap means eta = .40 already narrows the correctness
margin to .60, and widening it would erode the guarantee that makes the term
safe. The registered response was to re-derive the target, not the bound.

E89 re-derives it from what the saturated cells actually achieved at the
ceiling. Everything else is inherited from E88 unchanged.
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
import launch_e88_adaptive_semantic_maxent_05b as e88  # noqa: E402


DOMAINS = e88.DOMAINS
ARM = "adaptive_semantic_reachable"
SEEDS = e88.SEEDS
PASSES = e88.PASSES
TRAIN_ROWS = e88.TRAIN_ROWS
CHECKPOINT_INTERVAL = e88.CHECKPOINT_INTERVAL
TARGET_STEPS = e88.TARGET_STEPS
REPLAY_WEIGHT = e88.REPLAY_WEIGHT
MODEL_TAG = e88.MODEL_TAG
VARIANT = e88.VARIANT

# Re-derived from E88's saturated cells. At the eta = .40 ceiling the lowest
# realized ratio any cell sustained was .0170 (python_factors s44); MathIR s47
# sustained .0276 and s46 .0361. A globally reachable target must sit below the
# binding cell with margin, so rho = .015 (88% of .0170).
#
# The cost is explicit and is the point of the experiment: at rho = .015 the
# common dose is set by the weakest domain, so Countdown -- which realized 3.7%
# at the fixed eta = .10 and showed the largest fixed-arm gain -- is dosed
# *lower* here than in E81. E89 therefore tests uniformity, not strength.
START_COEFFICIENT = 0.10
TARGET_RATIO = 0.015
MIN_COEFFICIENT = e88.MIN_COEFFICIENT
MAX_COEFFICIENT = e88.MAX_COEFFICIENT
EMA_DECAY = e88.EMA_DECAY
GAIN = e88.GAIN
MAX_STEP_RATIO = e88.MAX_STEP_RATIO
WARMUP_STEPS = e88.WARMUP_STEPS
MIN_ELIGIBLE_FRACTION = e88.MIN_ELIGIBLE_FRACTION

LEDGER = "var/artifacts/e89_adaptive_semantic_maxent_reachable_05b_jobs.json"
PROTOCOL = (
    "paper/preregistration/e89_adaptive_semantic_maxent_reachable_05b_20260810.md"
)
PAIR_LEDGER = e88.PAIR_LEDGER
SNAPSHOT_PREFIX = "e89_adaptive_semantic_reachable"
PATCHED_FILES = e88.PATCHED_FILES


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_stamp(domain: str, seed: int) -> str:
    return f"e89_adaptive_reachable_{e81.DOMAIN_TAGS[domain]}_{ARM}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return root / "var/data" / f"xdr_{MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"


def fixed_objective() -> dict[str, str]:
    """E88's objective with the re-derived target ratio; nothing else moves."""

    objective = dict(e88.fixed_objective())
    objective["OAT_ZERO_SEMANTIC_RMS_TARGET_RATIO"] = repr(TARGET_RATIO)
    moved = {
        key
        for key in set(objective) | set(e88.fixed_objective())
        if objective.get(key) != e88.fixed_objective().get(key)
    }
    if moved != {"OAT_ZERO_SEMANTIC_RMS_TARGET_RATIO"}:
        raise RuntimeError(f"E89 differs from E88 in more than the target: {moved}")
    return objective


def build_env(root: Path, run: dict[str, Any], snapshot: Path):
    domain, seed = str(run["domain"]), int(run["seed"])
    env, _ = e81.build_env(root, run, snapshot)
    target = save_path(root, domain, seed)
    env.update({"SAVE_PATH": str(target), "RUN_STAMP": run_stamp(domain, seed)})
    env.update(fixed_objective())
    return env, target


def sbatch_command(root: Path, run: dict[str, Any], env: dict[str, str]) -> list[str]:
    node = str(run["source_node"])
    place = e81.placement(node)
    name = f"e89-{e81.DOMAIN_TAGS[str(run['domain'])][:6]}-rch-s{run['seed']}"
    export_pairs = ",".join(f"{k}={v}" for k, v in env.items())
    return [
        "sbatch", "--parsable", "--hold",
        f"--job-name={name}", f"--export=ALL,{export_pairs}",
        f"--partition={place['partition']}", f"--account={place['account']}",
        f"--nodelist={node}", f"--gres={place['gres']}",
        "--cpus-per-task=8", "--mem=64G", "--time=1-12:00:00", "--nice=100",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(job_id: str, run: dict[str, Any]) -> str:
    result = subprocess.run(["scontrol", "show", "job", "-dd", "-o", job_id],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E89 job {job_id}: {result.stderr.strip()}")
    record = result.stdout
    required = (
        "JobState=PENDING", "Reason=JobHeldUser",
        f"JobName=e89-{e81.DOMAIN_TAGS[str(run['domain'])][:6]}-rch-s{run['seed']}",
        f"ReqNodeList={run['source_node']}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={run['seed']}",
        "OAT_ZERO_SEMANTIC_RMS_CONTROL=1",
        "OAT_ZERO_SEMANTIC_RMS_TARGET_RATIO=0.015",
        "OAT_ZERO_SEMANTIC_RMS_MAX_COEFFICIENT=0.4",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
    )
    missing = [t for t in required if t not in record]
    if missing:
        raise RuntimeError(f"held E89 job {job_id} lacks {missing}")
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
        raise SystemExit(f"refusing duplicate E89 submission: {ledger_path}")

    parent = json.loads((root / PAIR_LEDGER).read_text(encoding="utf-8"))
    fixed_by_cell = {(str(r["domain"]), int(r["seed"])): r for r in parent["runs"]}
    snapshot, patched = e81.ensure_paired_snapshot(
        root, Path(parent["e78_snapshot_root"]),
        prefix=SNAPSHOT_PREFIX, patched_files=PATCHED_FILES,
    )

    planned = []
    for run in e81.references(root):
        domain, seed = str(run["domain"]), int(run["seed"])
        env, target = build_env(root, run, snapshot)
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing E89 run: {target}")
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
        raise SystemExit(f"E89 expected 25 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(p) for p in cell["command"]))
        print(f"[e89] dry_run=True cells={len(planned)} rho={TARGET_RATIO} "
              f"snapshot={snapshot}")
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
            "schema": "e89_adaptive_semantic_maxent_reachable_05b_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e81.digest(protocol),
            "launcher_sha256": e81.digest(Path(__file__)),
            "pair_ledger": str(root / PAIR_LEDGER),
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
            "supersedes_target_ratio_of": "e88_adaptive_semantic_maxent_05b_jobs_v1",
            "target_ratio_derivation": (
                "lowest sustained realized ratio at the eta=.40 ceiling across "
                "E88's saturated cells was .0170 (python_factors s44); rho set "
                "to .015, 88% of that binding constraint"
            ),
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
            "scientific_difference": "against E88: the target ratio only",
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

    print(f"[e89] cells={len(records)} released={len(submitted)} rho={TARGET_RATIO} "
          f"snapshot={snapshot} ledger={ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
