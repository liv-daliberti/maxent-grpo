#!/usr/bin/env python3
"""Submit E108: paired passive/adaptive admission-retention mechanism gate."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e102_full_open_bank_maxent_replay_05b as e102  # noqa: E402


ARMS = ("retention_tracking", "adaptive_retention")
VARIANTS = {
    "retention_tracking": "open_bank_retention_tracking_maxent_replay",
    "adaptive_retention": "open_bank_adaptive_retention_maxent_replay",
}
DOMAINS = e102.DOMAINS
SEED = 43
TRAIN_ROWS = 8
PASSES = 8
TARGET_STEPS = TRAIN_ROWS * PASSES
CHECKPOINT_INTERVAL = 32
MAX_MISSED_ROLLOUT_OPPORTUNITIES = 2
MAX_MEAN_LOGPROB_DROP = 0.5
REFRESH_VISITS = 4
SCORE_COOLDOWN_OBSERVATIONS = 2

LEDGER = "var/artifacts/e108_admission_retention_mechanism_gate_jobs.json"
PROTOCOL = (
    "paper/preregistration/" "e108_admission_retention_mechanism_gate_20260817.md"
)
E102_LEDGER = e102.LEDGER
SOURCE_MANIFEST = e102.SOURCE_MANIFEST
MODEL_TAG = e102.MODEL_TAG

# Live placement excludes drained node020/node023 and node105, whose completed
# workload was followed by an NHC over-temperature drain. node302 remains
# excluded. The remaining list is the healthy all-partition one-GPU pool.
NODES = "node[021-022,024-025,101,103-104,202-205,207-208,403,805]"
PARTITION = "all"
ACCOUNT = "mltheory"
GRES = "gpu:1"
MEMORY = "36G"
TIME_LIMIT = "04:00:00"
BATCH_SCRIPT = "ops/slurm/train_all_partition.slurm"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_e102(root: Path) -> Path:
    path = root / E102_LEDGER
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("released") is not True or len(payload.get("runs", [])) != 25:
        raise SystemExit("E108 requires the released 25-cell E102 ledger")
    if any(
        not (Path(str(run["run_dir"])) / "TRAINING_COMPLETE.json").is_file()
        for run in payload["runs"]
    ):
        raise SystemExit("E108 requires terminal E102 source cells")
    return path


def fixed_objective(arm: str) -> dict[str, str]:
    if arm not in ARMS:
        raise ValueError(f"unknown E108 arm: {arm}")
    objective = e102.fixed_objective()
    objective.update(
        {
            "OAT_ZERO_VARIANT": VARIANTS[arm],
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK": "0",
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING": "1",
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY": (
                "1" if arm == "adaptive_retention" else "0"
            ),
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MISSED_ROLLOUT_OPPORTUNITIES": str(
                MAX_MISSED_ROLLOUT_OPPORTUNITIES
            ),
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MEAN_LOGPROB_DROP": repr(
                MAX_MEAN_LOGPROB_DROP
            ),
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_REFRESH_VISITS": str(
                REFRESH_VISITS
            ),
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_SCORE_COOLDOWN_OBSERVATIONS": str(
                SCORE_COOLDOWN_OBSERVATIONS
            ),
        }
    )
    return objective


def run_stamp(arm: str, domain: str) -> str:
    return f"e108_{arm}_{e102.e78.DOMAIN_TAGS[domain]}_s{SEED}"


def save_path(root: Path, arm: str, domain: str) -> Path:
    return (
        root / "var/data" / f"xdr_{MODEL_TAG}_{VARIANTS[arm]}_{run_stamp(arm, domain)}"
    )


def build_env(
    root: Path,
    source: dict[str, Any],
    snapshot_root: Path,
    arm: str,
) -> tuple[dict[str, str], Path]:
    env, _ = e102.build_env(root, source, snapshot_root, smoke=False)
    domain = str(source["domain"])
    target = save_path(root, arm, domain)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(arm, domain),
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(TARGET_STEPS),
            "OAT_ZERO_ALLOW_SPARSE_EVAL": "1",
            "OAT_ZERO_EVAL_BATCH_SIZE": "8",
            "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": "1",
            "OAT_ZERO_SAVE_CKPT": "1",
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_MAX_SAVE_NUM": "2",
            "OAT_ZERO_MAX_RESUME_NUM": "2",
            "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS": "0",
            "OAT_ZERO_AUTO_RESUME": "1",
            "OAT_ZERO_WATCHDOG_REQUEUE": "0",
        }
    )
    env.update(fixed_objective(arm))
    return env, target


def sbatch_command(
    root: Path,
    source: dict[str, Any],
    env: dict[str, str],
    arm: str,
) -> list[str]:
    domain = str(source["domain"])
    name = f"e108-{arm[:3]}-{e102.e78.DOMAIN_TAGS[domain][:6]}"
    exports = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{exports}",
        f"--partition={PARTITION}",
        f"--account={ACCOUNT}",
        f"--nodelist={NODES}",
        f"--gres={GRES}",
        "--cpus-per-task=8",
        f"--mem={MEMORY}",
        f"--time={TIME_LIMIT}",
        "--nice=0",
        str(root / BATCH_SCRIPT),
    ]


def held_job_audit(job_id: str, source: dict[str, Any], arm: str) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E108 job {job_id}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"Account={ACCOUNT}",
        f"Partition={PARTITION}",
        f"OAT_ZERO_VARIANT={VARIANTS[arm]}",
        f"OAT_ZERO_SEED={source['seed']}",
        f"OAT_ZERO_MAX_TRAIN={TRAIN_ROWS}",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={PASSES}",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK=0",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING=1",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY="
        + ("1" if arm == "adaptive_retention" else "0"),
    )
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"held E108 job {job_id} lacks {missing}")
    node_match = re.search(r"(?:^| )ReqNodeList=([^ ]+)", record)
    if node_match is None or any(
        excluded in node_match.group(1)
        for excluded in ("node020", "node023", "node105", "node302")
    ):
        raise RuntimeError(f"held E108 job {job_id} has invalid placement")
    return record


def cancel(job_ids: list[str]) -> None:
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=False)


def release(job_ids: list[str]) -> None:
    for job_id in job_ids:
        result = subprocess.run(
            ["scontrol", "release", job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"release failed for E108 job {job_id}")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    protocol = root / PROTOCOL
    source_manifest = root / SOURCE_MANIFEST
    for required in (protocol, source_manifest):
        if not required.is_file():
            raise SystemExit(f"required frozen E108 input is absent: {required}")
    e102_ledger = check_e102(root)
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E108 submission: {ledger}")

    sources = [
        run
        for run in e102.e78.references(root)
        if int(run["seed"]) == SEED and str(run["domain"]) in DOMAINS
    ]
    if {str(run["domain"]) for run in sources} != set(DOMAINS):
        raise SystemExit("E108 lacks one or more seed-43 source domains")
    snapshot_root = e102.snapshot_util.ensure_snapshot(root, args.snapshot_root)
    planned: list[dict[str, Any]] = []
    for arm in ARMS:
        for source in sources:
            domain = str(source["domain"])
            env, target = build_env(root, source, snapshot_root, arm)
            if target.exists():
                raise SystemExit(f"refusing to overwrite E108 run: {target}")
            planned.append(
                {
                    "arm": arm,
                    "domain": domain,
                    "seed": SEED,
                    "run_stamp": run_stamp(arm, domain),
                    "run_dir": str(target),
                    "objective": fixed_objective(arm),
                    "template": source,
                    "command": sbatch_command(root, source, env, arm),
                }
            )
    if len(planned) != 10:
        raise SystemExit(f"E108 expected 10 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(
            f"[e108] dry_run=True cells=10 snapshot={snapshot_root} "
            f"campaign_stats_ledger={ledger}"
        )
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in planned:
            result = subprocess.run(
                cell["command"], capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"submission failed for {cell['run_stamp']}: "
                    f"{result.stderr.strip()}"
                )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"invalid E108 job id: {result.stdout!r}")
            submitted.append(job_id)
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "arm",
                        "domain",
                        "seed",
                        "run_stamp",
                        "run_dir",
                        "objective",
                    )
                }
                | {
                    "job_id": int(job_id),
                    "held_scheduler_record": held_job_audit(
                        job_id, cell["template"], cell["arm"]
                    ),
                }
            )
        payload = {
            "schema": "e108_admission_retention_mechanism_gate_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e102.e78.digest(protocol),
            "launcher_sha256": e102.e78.digest(Path(__file__)),
            "audit": str(
                root / "ops/exp_scaling/"
                "audit_e108_admission_retention_mechanism_gate.py"
            ),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": e102.e78.digest(source_manifest),
            "e102_context_ledger": str(e102_ledger),
            "e102_context_ledger_sha256": e102.e78.digest(e102_ledger),
            "snapshot_root": str(snapshot_root),
            "model": "Qwen2.5-0.5B-Instruct",
            "domains": list(DOMAINS),
            "arms": list(ARMS),
            "seeds": [SEED],
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "max_missed_rollout_opportunities": (MAX_MISSED_ROLLOUT_OPPORTUNITIES),
            "max_mean_logprob_drop": MAX_MEAN_LOGPROB_DROP,
            "refresh_visits": REFRESH_VISITS,
            "score_cooldown_observations": SCORE_COOLDOWN_OBSERVATIONS,
            "outcomes_are_release_gate": False,
            "placement": {
                "partition": PARTITION,
                "account": ACCOUNT,
                "nodes": NODES,
                "batch_script": BATCH_SCRIPT,
                "excluded_nodes": ["node020", "node023", "node105", "node302"],
                "gres": GRES,
                "memory": MEMORY,
            },
            "runs": records,
            "released": False,
        }
        e102.e78.atomic_json(ledger, payload)
        release(submitted)
        payload["released"] = True
        e102.e78.atomic_json(ledger, payload)
    except Exception:
        cancel(submitted)
        if ledger.exists():
            ledger.unlink()
        raise

    print(
        f"[e108] cells={len(records)} released={len(submitted)} "
        f"snapshot={snapshot_root} ledger={ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
