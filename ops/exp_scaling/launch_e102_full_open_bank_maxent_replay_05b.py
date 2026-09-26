#!/usr/bin/env python3
"""Submit E102: full replay-side MaxEnt plus open-bank exploration at 0.5B."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_b3a_replay_ablation as base  # noqa: E402
import launch_e76_tuned_scale as snapshot_util  # noqa: E402
import launch_e78_verified_replay_only_05b as e78  # noqa: E402


ARM = "full_open_bank"
VARIANT = "open_bank_full_maxent_replay"
DOMAINS = e78.DOMAINS
SEEDS = e78.SEEDS
PASSES = e78.PASSES
TRAIN_ROWS = e78.TRAIN_ROWS
CHECKPOINT_INTERVAL = e78.CHECKPOINT_INTERVAL
TARGET_STEPS = e78.TARGET_STEPS
REPLAY_MASS_WEIGHT = 0.10
BANK_BALANCE_WEIGHT = 0.10
PROPOSAL_TEMPERATURE = 1.20
PROPOSAL_MAX_ATTEMPTS = 1
PRIORITY_VISITS = 4
PRIORITY_MULTIPLIER = 4.0
SMOKE_TRAIN_ROWS = 4
SMOKE_PASSES = 8
SMOKE_TARGET_STEPS = SMOKE_TRAIN_ROWS * SMOKE_PASSES
SMOKE_SUFFIX = "smoke_r1"
LEDGER = "var/artifacts/e102_full_open_bank_maxent_replay_05b_jobs.json"
PROTOCOL = "paper/preregistration/e102_full_open_bank_maxent_replay_05b_20260814.md"
E78_LEDGER = "var/artifacts/e78_verified_replay_only_05b_jobs.json"
SMOKE_AUDIT = "var/artifacts/e102_full_open_bank_maxent_replay_05b_smoke_gate.json"
SOURCE_MANIFEST = e78.SOURCE_MANIFEST
MODEL_TAG = e78.MODEL_TAG

# Explicitly excludes node302 (the congested A100 host) and node105 (the other
# E78 source host). Generic one-GPU requests let Slurm use the healthy 24--48 GB
# cards in this pool; optimizer steps, not wall time, define the comparison.
NODES = "node[007,020,022-023,101,103,202,204-206,403,805]"
PARTITION = "all"
ACCOUNT = "mltheory"
GRES = "gpu:1"
MEMORY = "22G"
TIME_LIMIT = "2-00:00:00"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def check_e78_comparators(root: Path) -> Path:
    path = root / E78_LEDGER
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("released") is not True:
        raise SystemExit("E102 requires the released E78 comparator ledger")
    for key, expected in (
        ("train_rows", TRAIN_ROWS),
        ("passes", PASSES),
        ("target_steps", TARGET_STEPS),
        ("checkpoint_interval_steps", CHECKPOINT_INTERVAL),
    ):
        if int(payload.get(key, -1)) != expected:
            raise SystemExit(f"E78 comparator {key} drifted from {expected}")
    runs = payload.get("runs", [])
    if Counter(str(run.get("arm")) for run in runs) != {
        "control": 25,
        "replay": 25,
    }:
        raise SystemExit("E78 comparator ledger must contain 25 control and 25 replay cells")
    expected_pairs = {(domain, seed) for domain in DOMAINS for seed in SEEDS}
    for arm in ("control", "replay"):
        pairs = {
            (str(run.get("domain")), int(run.get("seed", -1)))
            for run in runs
            if run.get("arm") == arm
        }
        if pairs != expected_pairs:
            raise SystemExit(f"E78 {arm} domain/seed cells do not match E102")
    return path


def check_smoke_gate(root: Path) -> Path:
    path = root / SMOKE_AUDIT
    if not path.is_file():
        raise SystemExit(f"E102 smoke gate audit is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "e102-full-open-bank-smoke-audit-v1":
        raise SystemExit("E102 smoke gate has an unknown schema")
    if payload.get("passed") is not True:
        raise SystemExit("E102 full release requires a passing smoke gate")
    run_dir = str(payload.get("run_dir", ""))
    if not run_dir.endswith("e102_full_open_bank_graph_s43_smoke_r1"):
        raise SystemExit("E102 smoke gate does not identify the repaired Graph run")
    report = payload.get("report", {})
    required_positive = (
        "proposal_cumulative_admissions",
        "priority_replay_groups_cumulative",
        "mass_weight_max",
    )
    if int(report.get("last_step", -1)) < SMOKE_TARGET_STEPS or any(
        float(report.get(key, 0.0)) <= (1.0 if key == "mass_weight_max" else 0.0)
        for key in required_positive
    ):
        raise SystemExit("E102 smoke gate lacks discovery-to-priority actuation")
    if float(report.get("applied_positive_gradient_max", math.inf)) > 1e-7:
        raise SystemExit("E102 smoke gate violates retention safety")
    return path


def run_stamp(domain: str, seed: int) -> str:
    return f"e102_full_open_bank_{e78.DOMAIN_TAGS[domain]}_s{seed}"


def save_path(root: Path, domain: str, seed: int) -> Path:
    return (
        root
        / "var/data"
        / f"xdr_{MODEL_TAG}_{VARIANT}_{run_stamp(domain, seed)}"
    )


def fixed_objective() -> dict[str, str]:
    return {
        "OAT_ZERO_VARIANT": VARIANT,
        "OAT_ZERO_POLICY_ENTROPY_COEF": "0.0",
        "OAT_ZERO_XDR_TAU": "inf",
        "OAT_ZERO_SEED_ENTROPY_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_DUAL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_INVERSE_ADAPTATION": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA": "0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE": (
            "split_mass_balance_per_rollout"
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA": repr(BANK_BALANCE_WEIGHT),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA": repr(REPLAY_MASS_WEIGHT),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY": "16",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS": "1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT": "1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ANCHOR_MAX_TOKENS": "256",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS": str(
            PROPOSAL_MAX_ATTEMPTS
        ),
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE": repr(
            PROPOSAL_TEMPERATURE
        ),
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS": str(
            PRIORITY_VISITS
        ),
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER": repr(
            PRIORITY_MULTIPLIER
        ),
        "OAT_ZERO_BETA": "0.0",
    }


def build_env(
    root: Path,
    run: dict[str, Any],
    snapshot_root: Path,
    *,
    smoke: bool = False,
) -> tuple[dict[str, str], Path]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    target = save_path(root, domain, seed)
    if smoke:
        target = target.with_name(target.name + f"_{SMOKE_SUFFIX}")
    env = base.build_export_vars(root, run, target, "b1b")
    canonical_learner = str(
        run["inherited_eval_config"]["canonical_action_task"]
    ) != "none"
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(domain, seed)
            + (f"_{SMOKE_SUFFIX}" if smoke else ""),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot_root / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot_root / "ops"),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(SMOKE_PASSES if smoke else PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(
                SMOKE_PASSES if smoke else PASSES
            ),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(
                SMOKE_TARGET_STEPS if smoke else CHECKPOINT_INTERVAL
            ),
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_MAX_SAVE_NUM": "1",
            "OAT_ZERO_MAX_RESUME_NUM": "1",
            "OAT_ZERO_SAVE_CKPT": "0" if smoke else "1",
            "OAT_ZERO_EXPORT_STEPS": "0",
            "OAT_ZERO_AUTO_RESUME": "0" if smoke else "1",
            "OAT_ZERO_WATCHDOG_REQUEUE": "0" if smoke else "1",
            "OAT_ZERO_USE_WB": "0",
            "OAT_ZERO_NUM_GPUS_PER_ACTOR": "1",
            "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING": (
                "0" if canonical_learner else "1"
            ),
            "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC": "0" if canonical_learner else "1",
            "OAT_ZERO_VLLM_SLEEP_LEVEL": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if smoke:
        env.update(
            {
                "OAT_ZERO_MAX_TRAIN": str(SMOKE_TRAIN_ROWS),
                "OAT_ZERO_ALLOW_SPARSE_EVAL": "1",
                "OAT_ZERO_EVAL_BATCH_SIZE": "8",
                "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": "1",
            }
        )
    env.update(fixed_objective())
    return env, target


def sbatch_command(
    root: Path,
    run: dict[str, Any],
    env: dict[str, str],
    *,
    smoke: bool = False,
) -> list[str]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    name = f"e102{'m' if smoke else ''}-{e78.DOMAIN_TAGS[domain][:6]}-s{seed}"
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        f"--partition={PARTITION}",
        f"--account={ACCOUNT}",
        f"--nodelist={NODES}",
        f"--gres={GRES}",
        "--cpus-per-task=8",
        f"--mem={MEMORY}",
        f"--time={'01:00:00' if smoke else TIME_LIMIT}",
        "--nice=100",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(job_id: str, run: dict[str, Any], *, smoke: bool = False) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held job {job_id}: {result.stderr.strip()}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"Account={ACCOUNT}",
        f"Partition={PARTITION}",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={run['seed']}",
        f"OAT_ZERO_MAX_TRAIN={SMOKE_TRAIN_ROWS if smoke else TRAIN_ROWS}",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={SMOKE_PASSES if smoke else PASSES}",
        f"OAT_ZERO_EVAL_PROMPT_INTERVAL={SMOKE_TARGET_STEPS if smoke else CHECKPOINT_INTERVAL}",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.2",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS=4",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER=4.0",
        "OAT_ZERO_NUM_GPUS_PER_ACTOR=1",
        (
            "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=0"
            if str(run["domain"]) == "pantry_plan"
            else "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1"
        ),
        (
            "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=0"
            if str(run["domain"]) == "pantry_plan"
            else "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1"
        ),
    )
    missing = [item for item in required if item not in record]
    if missing:
        raise RuntimeError(f"held job {job_id} lacks {missing}")
    node_match = re.search(r"(?:^| )ReqNodeList=([^ ]+)", record)
    if node_match is None:
        raise RuntimeError(f"held E102 job {job_id} lacks ReqNodeList")
    requested_nodes = node_match.group(1)
    if "node302" in requested_nodes or "node105" in requested_nodes:
        raise RuntimeError(f"held E102 job {job_id} permits an excluded node")
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
            raise RuntimeError(f"release failed for {job_id}: {result.stderr.strip()}")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-domain", choices=DOMAINS, default="graph_coloring")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")
    if args.smoke and not args.submit and not args.dry_run:
        args.dry_run = True

    protocol = root / PROTOCOL
    source_manifest = root / SOURCE_MANIFEST
    for required in (protocol, source_manifest):
        if not required.is_file():
            raise SystemExit(f"required frozen input is absent: {required}")
    e78_ledger = check_e78_comparators(root)
    smoke_audit = (
        check_smoke_gate(root) if args.submit and not args.smoke else None
    )
    ledger = root / LEDGER
    if args.submit and not args.smoke and ledger.exists():
        raise SystemExit(f"refusing duplicate E102 submission: {ledger}")

    source_runs = e78.references(root)
    if args.smoke:
        source_runs = [
            run
            for run in source_runs
            if str(run["domain"]) == args.smoke_domain and int(run["seed"]) == 43
        ]
    snapshot_root = snapshot_util.ensure_snapshot(root, args.snapshot_root)
    planned: list[dict[str, Any]] = []
    for run in source_runs:
        env, target = build_env(root, run, snapshot_root, smoke=args.smoke)
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing E102 run: {target}")
        planned.append(
            {
                "arm": ARM,
                "domain": str(run["domain"]),
                "seed": int(run["seed"]),
                "run_stamp": run_stamp(str(run["domain"]), int(run["seed"]))
                + (f"_{SMOKE_SUFFIX}" if args.smoke else ""),
                "run_dir": str(target),
                "command": sbatch_command(root, run, env, smoke=args.smoke),
                "template": run,
                "objective": fixed_objective(),
            }
        )
    expected_cells = 1 if args.smoke else 25
    if len(planned) != expected_cells:
        raise SystemExit(f"E102 expected {expected_cells} cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(
            f"[e102] dry_run=True smoke={args.smoke} cells={len(planned)} "
            f"snapshot={snapshot_root}"
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
                    f"submission failed for {cell['run_stamp']}: {result.stderr.strip()}"
                )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"invalid job id: {result.stdout!r}")
            submitted.append(job_id)
            held_record = held_job_audit(
                job_id, cell["template"], smoke=args.smoke
            )
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
                    "held_scheduler_record": held_record,
                }
            )

        if args.smoke:
            release(submitted)
            print(
                f"[e102] smoke_job={submitted[0]} run_dir={records[0]['run_dir']} "
                f"snapshot={snapshot_root}"
            )
            return 0

        payload = {
            "schema": "e102_full_open_bank_maxent_replay_05b_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e78.digest(protocol),
            "launcher_sha256": e78.digest(Path(__file__)),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": e78.digest(source_manifest),
            "comparator_ledger": str(e78_ledger),
            "comparator_ledger_sha256": e78.digest(e78_ledger),
            "smoke_gate_audit": str(smoke_audit),
            "smoke_gate_audit_sha256": e78.digest(smoke_audit),
            "snapshot_root": str(snapshot_root),
            "model": "Qwen2.5-0.5B-Instruct",
            "domains": list(DOMAINS),
            "arms": [ARM],
            "comparators": ["e78/replay", "e78/control"],
            "seeds": list(SEEDS),
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [index / 2 for index in range(17)],
            "replay_mass_weight": REPLAY_MASS_WEIGHT,
            "bank_balance_weight_requested": BANK_BALANCE_WEIGHT,
            "retention_safe_balance": True,
            "proposal_temperature": PROPOSAL_TEMPERATURE,
            "proposal_max_attempts": PROPOSAL_MAX_ATTEMPTS,
            "priority_visits": PRIORITY_VISITS,
            "priority_multiplier": PRIORITY_MULTIPLIER,
            "placement": {
                "partition": PARTITION,
                "account": ACCOUNT,
                "nodes": NODES,
                "excluded_nodes": ["node302", "node105"],
                "gres": GRES,
                "memory": MEMORY,
            },
            "scientific_difference": (
                "E78 replay plus direct whole-bank balance, retention-safe capping, "
                "target-free original-prompt discovery, and fresh-mode mass priority"
            ),
            "runs": records,
            "released": False,
        }
        e78.atomic_json(ledger, payload)
        release(submitted)
        payload["released"] = True
        e78.atomic_json(ledger, payload)
    except Exception:
        cancel(submitted)
        if not args.smoke and ledger.exists():
            ledger.unlink()
        raise

    print(
        f"[e102] cells={len(records)} released={len(submitted)} "
        f"snapshot={snapshot_root} ledger={ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
