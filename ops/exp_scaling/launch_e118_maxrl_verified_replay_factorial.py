#!/usr/bin/env python3
"""Submit E118: matched MaxRL versus Re:Max extension to E78."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import launch_e78_verified_replay_only_05b as e78
import launch_e79_falcon1b_aligned_verified_replay as e79
import status_e78 as status


DOMAINS = e79.DOMAINS
ARMS = ("maxrl", "replay_maxrl")
SEEDS = e79.SEEDS
PASSES = 8
TRAIN_ROWS = 384
TARGET_STEPS = PASSES * TRAIN_ROWS
CHECKPOINT_INTERVAL = 192
REPLAY_WEIGHT = 0.10
LEDGER = "var/artifacts/e118f1_maxrl_verified_replay_extension_jobs.json"
PROTOCOL = "paper/preregistration/e118f1_falcon_five_seed_extension_20260901.md"
E78_LEDGER = "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json"
MODEL_TAG = e79.MODEL_TAG

VARIANTS = {
    "maxrl": "maxrl_compute_matched",
    "replay_maxrl": "maxrl_verified_replay",
}


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_runs(repo: Path) -> list[dict[str, Any]]:
    runs = e79.references(repo)
    expected = len(DOMAINS) * len(SEEDS)
    if len(runs) != expected:
        raise SystemExit(f"E118 did not resolve exactly {expected} source templates")
    return runs


def completed_comparators(repo: Path) -> list[dict[str, Any]]:
    path = repo / E78_LEDGER
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = [
        run
        for run in payload["runs"]
        if int(run["seed"]) in SEEDS
        and str(run["domain"]) in DOMAINS
        and str(run["arm"]) in {"control", "replay"}
    ]
    expected = len(DOMAINS) * len(SEEDS) * 2
    if len(selected) != expected:
        raise SystemExit(f"E118 requires exactly {expected} selected E78 comparator cells")
    target = int(payload["target_steps"])
    for run in selected:
        run_dir = Path(str(run["run_dir"]))
        if not status.is_complete(
            run_dir,
            max(
                status.run_step(run_dir),
                status.receipt_step(run_dir),
            ),
            target,
        ):
            raise SystemExit(
                f"E118 comparator is not terminal: {run['domain']}/"
                f"{run['arm']}/s{run['seed']}"
            )
    return selected


def run_stamp(domain: str, arm: str, seed: int) -> str:
    return f"e118f1_{e79.DOMAIN_TAGS[domain]}_{arm}_s{seed}"


def save_path(repo: Path, domain: str, arm: str, seed: int) -> Path:
    return (
        repo
        / "var/data"
        / f"xdr_{MODEL_TAG}_{VARIANTS[arm]}_{run_stamp(domain, arm, seed)}"
    )


def objective(arm: str) -> dict[str, str]:
    return {
        "OAT_ZERO_VARIANT": VARIANTS[arm],
        "OAT_ZERO_MAXRL_TASK_OBJECTIVE": "1",
        "OAT_ZERO_CRITIC_TYPE": "drgrpo",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA": "0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE": (
            "verified_likelihood_per_rollout"
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA": repr(REPLAY_WEIGHT),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA": repr(REPLAY_WEIGHT),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY": "16",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": (
            "1" if arm == "maxrl" else "0"
        ),
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY": "0",
        "OAT_ZERO_MAXENT_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_DUAL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_INVERSE_ADAPTATION": "0",
        "OAT_ZERO_POLICY_ENTROPY_COEF": "0.0",
        "OAT_ZERO_SEED_ENTROPY_ALPHA": "0.0",
        "OAT_ZERO_BETA": "0.0",
        "OAT_ZERO_DAPO_ENABLED": "0",
        "OAT_ZERO_RLEP_REPLAY_COUNT": "0",
    }


def environment(
    repo: Path,
    template: dict[str, Any],
    arm: str,
    snapshot: Path,
) -> tuple[dict[str, str], Path]:
    domain = str(template["domain"])
    seed = int(template["seed"])
    target = save_path(repo, domain, arm, seed)
    env = e79.base.build_export_vars(repo, template, target, "b1b")
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(domain, arm, seed),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_PRETRAIN": str(e79.model_root(repo)),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_MAX_SAVE_NUM": "1",
            "OAT_ZERO_MAX_RESUME_NUM": "1",
            "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS": "1",
            "OAT_ZERO_AUTO_RESUME": "1",
            "OAT_ZERO_WATCHDOG_REQUEUE": "1",
            "OAT_ZERO_USE_WB": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(
        e79.task_surface(
            domain, str(template["inherited_eval_config"]["prompt_template"])
        )
    )
    env.update(e79.optimizer_env())
    env.update(objective(arm))
    return env, target


def command(
    repo: Path,
    template: dict[str, Any],
    arm: str,
    env: dict[str, str],
) -> list[str]:
    name = (
        f"e118f1-{e79.DOMAIN_TAGS[str(template['domain'])][:6]}-"
        f"{'m' if arm == 'maxrl' else 'rm'}-s{template['seed']}"
    )
    exports = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{exports}",
        "--partition=mltheory",
        "--account=mltheory",
        "--nodelist=node302",
        "--gres=gpu:a100:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=12:00:00",
        "--nice=0",
        "--requeue",
        "--chdir=" + str(repo),
        str(snapshot_script(env)),
    ]


def snapshot_script(env: dict[str, str]) -> Path:
    return Path(env["OAT_ZERO_OPS_SNAPSHOT_ROOT"]) / "slurm/train_node302.slurm"


def audit(job_id: str, template: dict[str, Any], arm: str) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Account=mltheory",
        "Partition=mltheory",
        "ReqNodeList=node302",
        "OAT_ZERO_MAXRL_TASK_OBJECTIVE=1",
        "OAT_ZERO_CRITIC_TYPE=drgrpo",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_NUM_PPO_EPOCHS=1",
        "OAT_ZERO_BETA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY="
        + ("1" if arm == "maxrl" else "0"),
        f"OAT_ZERO_SEED={template['seed']}",
    )
    missing = [value for value in required if value not in record]
    if missing:
        raise RuntimeError(f"held E118 job {job_id} lacks {missing}")
    return record


def main() -> int:
    repo = root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run")

    protocol = repo / PROTOCOL
    ledger_path = repo / LEDGER
    for path in (protocol, repo / E78_LEDGER):
        if not path.is_file():
            raise SystemExit(f"required input is absent: {path}")
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate E118 submission: {ledger_path}")

    templates = source_runs(repo)
    comparators = completed_comparators(repo)
    snapshot = e78.snapshot_util.ensure_snapshot(repo, args.snapshot_root)
    planned: list[dict[str, Any]] = []
    for template in templates:
        for arm in ARMS:
            env, target = environment(repo, template, arm, snapshot)
            if target.exists():
                raise SystemExit(f"refusing existing E118 run directory: {target}")
            planned.append(
                {
                    "domain": str(template["domain"]),
                    "arm": arm,
                    "seed": int(template["seed"]),
                    "run_stamp": run_stamp(
                        str(template["domain"]), arm, int(template["seed"])
                    ),
                    "run_dir": str(target),
                    "template": template,
                    "command": command(repo, template, arm, env),
                }
            )
    expected_cells = len(DOMAINS) * len(SEEDS) * len(ARMS)
    if len(planned) != expected_cells:
        raise SystemExit(f"E118 expected {expected_cells} new cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(f"[e118] dry_run=True cells={expected_cells} snapshot={snapshot}")
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
                raise RuntimeError(f"invalid job id: {result.stdout!r}")
            submitted.append(job_id)
            held = audit(job_id, cell["template"], str(cell["arm"]))
            records.append(
                {
                    key: cell[key]
                    for key in ("domain", "arm", "seed", "run_stamp", "run_dir")
                }
                | {"job_id": int(job_id), "held_scheduler_record": held}
            )

        payload = {
            "schema": "e118_maxrl_verified_replay_factorial_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e78.digest(protocol),
            "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": e78.digest(Path(__file__)),
            "e78_ledger": str(repo / E78_LEDGER),
            "e78_ledger_sha256": e78.digest(repo / E78_LEDGER),
            "snapshot_root": str(snapshot),
            "model": "Falcon3-1B-Instruct",
            "domains": list(DOMAINS),
            "new_arms": list(ARMS),
            "effective_arms": [
                "drgrpo",
                "replay_grpo",
                "maxrl",
                "replay_maxrl",
            ],
            "seeds": list(SEEDS),
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "replay_weight": REPLAY_WEIGHT,
            "comparators": comparators,
            "runs": records,
            "released": False,
            "outcomes_inspected_before_release": False,
        }
        e78.atomic_json(ledger_path, payload)
        for job_id in submitted:
            release = subprocess.run(
                ["scontrol", "release", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if release.returncode != 0:
                raise RuntimeError(
                    f"release failed for {job_id}: {release.stderr.strip()}"
                )
        payload["released"] = True
        e78.atomic_json(ledger_path, payload)
    except Exception:
        e78.cancel(submitted)
        if ledger_path.exists():
            ledger_path.unlink()
        raise

    print(
        f"[e118] cells={len(records)} released={len(submitted)} "
        f"snapshot={snapshot} ledger={ledger_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
