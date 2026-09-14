#!/usr/bin/env python3
"""Submit the frozen E104 score-function mechanism cohort at three scales."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e76_tuned_scale as snapshot_util  # noqa: E402
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402
import launch_e80r1_qwen3b_aligned_verified_replay as e80  # noqa: E402
import launch_e81_semantic_maxent_verified_replay_05b as e81  # noqa: E402


DOMAINS = e81.DOMAINS
SCALE_SEEDS = {"qwen05b": 43, "falcon1b": 55, "qwen3b": 70}
MODEL_TAGS = {
    "qwen05b": e81.MODEL_TAG,
    "falcon1b": e79.MODEL_TAG,
    "qwen3b": e80.MODEL_TAG,
}
VARIANT = "verified_replay_semantic_maxent_group_centered"
ARM = "semantic_group_centered"
SMOKE_TRAIN_ROWS = 64
SMOKE_PASSES = 1
SMOKE_TARGET_STEPS = SMOKE_TRAIN_ROWS * SMOKE_PASSES
CHECKPOINT_INTERVAL = 32
SEMANTIC_COEFFICIENT = 0.10
REPLAY_WEIGHT = 0.10
LEDGER = "var/artifacts/e104_group_centered_semantic_repair_three_scale_jobs.json"
PROTOCOL = (
    "paper/preregistration/"
    "e104_group_centered_semantic_repair_three_scale_20260817.md"
)
SOURCE_MANIFEST = e81.SOURCE_MANIFEST
AUDIT = "var/artifacts/e104_group_centered_semantic_repair_gate.json"

SNAPSHOT_REQUIREMENTS = {
    "src/oat_drgrpo/semantic_shannon.py": (
        "semantic_shannon_tracker_v6_group_centered",
        "success_conditioned_group_centered_advantage",
    ),
    "src/oat_drgrpo/args.py": (
        "semantic_shannon_success_conditioned_group_centered_advantage",
    ),
    "src/oat_drgrpo/learner/init.py": (
        "success_conditioned_group_centered_advantage",
    ),
    "src/oat_drgrpo/learner/grpo.py": (
        "semantic_shannon_success_conditioned_group_centered",
    ),
    "ops/train.sh": (
        "--semantic-shannon-success-conditioned-group-centered-advantage",
    ),
    "ops/run_experiment.sh": (
        "verified_replay_semantic_maxent_group_centered)",
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_objective() -> dict[str, str]:
    objective = dict(e81.fixed_objective())
    objective.update(
        {
            "OAT_ZERO_VARIANT": VARIANT,
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE": "1",
            "OAT_ZERO_SEMANTIC_RMS_CONTROL": "0",
        }
    )
    assert float(objective["OAT_ZERO_SEMANTIC_SHANNON_COEF"]) == SEMANTIC_COEFFICIENT
    assert float(objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) == REPLAY_WEIGHT
    return objective


def verify_snapshot(snapshot: Path) -> None:
    missing: list[str] = []
    for relative, needles in SNAPSHOT_REQUIREMENTS.items():
        path = snapshot / relative
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        for needle in needles:
            if needle not in text:
                missing.append(f"{relative}: {needle}")
    if missing:
        raise SystemExit("E104 snapshot is incomplete:\n  " + "\n  ".join(missing))


def references(root: Path, scale: str) -> list[dict[str, Any]]:
    seed = SCALE_SEEDS[scale]
    if scale == "qwen05b":
        runs = e81.references(root)
    elif scale == "falcon1b":
        runs = e79.references(root)
    else:
        runs = e80.references(root)
    selected = [
        run
        for run in runs
        if int(run["seed"]) == seed and str(run["domain"]) in DOMAINS
    ]
    if len(selected) != len(DOMAINS):
        raise SystemExit(f"E104 {scale} expected five templates, found {len(selected)}")
    return selected


def run_stamp(scale: str, domain: str) -> str:
    seed = SCALE_SEEDS[scale]
    return f"e104_{scale}_{e81.DOMAIN_TAGS[domain]}_{ARM}_s{seed}"


def save_path(root: Path, scale: str, domain: str) -> Path:
    return root / "var/data" / (
        f"xdr_{MODEL_TAGS[scale]}_{VARIANT}_{run_stamp(scale, domain)}"
    )


def build_env(
    root: Path,
    scale: str,
    run: dict[str, Any],
    snapshot: Path,
) -> tuple[dict[str, str], Path]:
    domain = str(run["domain"])
    if scale == "qwen05b":
        env, _ = e81.build_env(root, run, snapshot)
    elif scale == "falcon1b":
        env, _ = e79.build_env(
            root, run, "replay", snapshot, e79.model_root(root)
        )
    else:
        env, _ = e80.build_env(
            root, run, "replay", snapshot, e80.model_root(root)
        )
    target = save_path(root, scale, domain)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(scale, domain),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_MAX_TRAIN": str(SMOKE_TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(SMOKE_PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(SMOKE_PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_MAX_SAVE_NUM": "1",
            "OAT_ZERO_MAX_RESUME_NUM": "1",
            "OAT_ZERO_AUTO_RESUME": "1",
            "OAT_ZERO_WATCHDOG_REQUEUE": "1",
            "OAT_ZERO_USE_WB": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(fixed_objective())
    return env, target


def base_command(
    root: Path, scale: str, run: dict[str, Any], env: dict[str, str]
) -> list[str]:
    if scale == "qwen05b":
        return e81.sbatch_command(root, run, env)
    if scale == "falcon1b":
        return e79.sbatch_command(root, run, "replay", env)
    return e80.sbatch_command(root, run, "replay", env)


def sbatch_command(
    root: Path, scale: str, run: dict[str, Any], env: dict[str, str]
) -> list[str]:
    domain = str(run["domain"])
    command = base_command(root, scale, run, env)
    name = job_name(scale, domain)
    output: list[str] = []
    for token in command:
        if token.startswith("--job-name="):
            output.append(f"--job-name={name}")
        elif token.startswith("--time="):
            output.append("--time=08:00:00")
        elif token.startswith("--nice="):
            output.append("--nice=0")
        else:
            output.append(token)
    if not any(token.startswith("--nice=") for token in output):
        output.insert(-1, "--nice=0")
    return output


def job_name(scale: str, domain: str) -> str:
    scale_tag = {"qwen05b": "q05", "falcon1b": "f1", "qwen3b": "q3"}[scale]
    return f"e104-{scale_tag}-{e81.DOMAIN_TAGS[domain][:6]}"


def held_job_audit(job_id: str, scale: str, run: dict[str, Any]) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E104 job {job_id}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={SCALE_SEEDS[scale]}",
        f"OAT_ZERO_MAX_TRAIN={SMOKE_TRAIN_ROWS}",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={SMOKE_PASSES}",
        f"OAT_ZERO_EVAL_PROMPT_INTERVAL={CHECKPOINT_INTERVAL}",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_RMS_CONTROL=0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0",
        "OAT_ZERO_MAXENT_ALPHA=0.0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
    )
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"held E104 job {job_id} lacks {missing}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    root = repo_root()
    protocol = root / PROTOCOL
    manifest = root / SOURCE_MANIFEST
    for required in (protocol, manifest):
        if not required.is_file():
            raise SystemExit(f"required frozen input is absent: {required}")
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E104 submission: {ledger}")
    snapshot = snapshot_util.ensure_snapshot(root, args.snapshot_root)
    verify_snapshot(snapshot)

    planned: list[dict[str, Any]] = []
    for scale in SCALE_SEEDS:
        for run in references(root, scale):
            domain = str(run["domain"])
            env, target = build_env(root, scale, run, snapshot)
            if args.submit and target.exists():
                raise SystemExit(f"refusing to overwrite E104 run: {target}")
            planned.append(
                {
                    "scale": scale,
                    "model_tag": MODEL_TAGS[scale],
                    "domain": domain,
                    "seed": SCALE_SEEDS[scale],
                    "run_stamp": run_stamp(scale, domain),
                    "run_dir": str(target),
                    "command": sbatch_command(root, scale, run, env),
                    "template": run,
                }
            )
    if len(planned) != 15:
        raise SystemExit(f"E104 expected 15 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(token) for token in cell["command"]))
        print(f"[e104] dry_run=True cells=15 snapshot={snapshot}")
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
            held = held_job_audit(
                job_id, str(cell["scale"]), cell["template"]
            )
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "scale",
                        "model_tag",
                        "domain",
                        "seed",
                        "run_stamp",
                        "run_dir",
                    )
                }
                | {
                    "job_id": int(job_id),
                    "stdout": str(
                        root
                        / "var/artifacts/logs"
                        / (
                            f"{job_name(str(cell['scale']), str(cell['domain']))}"
                            f"-{job_id}.out"
                        )
                    ),
                    "stderr": str(
                        root
                        / "var/artifacts/logs"
                        / (
                            f"{job_name(str(cell['scale']), str(cell['domain']))}"
                            f"-{job_id}.err"
                        )
                    ),
                    "held_scheduler_record": held,
                }
            )
        payload = {
            "schema": "e104_group_centered_semantic_repair_three_scale_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": digest(protocol),
            "launcher_sha256": digest(Path(__file__)),
            "source_manifest": str(manifest),
            "source_manifest_sha256": digest(manifest),
            "snapshot_root": str(snapshot),
            "models": list(SCALE_SEEDS),
            "domains": list(DOMAINS),
            "seeds": SCALE_SEEDS,
            "arms": [ARM],
            "train_rows": SMOKE_TRAIN_ROWS,
            "passes": SMOKE_PASSES,
            "target_steps": SMOKE_TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "semantic_coefficient": SEMANTIC_COEFFICIENT,
            "replay_weight": REPLAY_WEIGHT,
            "objective": "sampled_group_centered_semantic_score_plus_verified_replay",
            "pointmaze": "excluded",
            "audit": str(root / AUDIT),
            "runs": records,
            "released": False,
        }
        e81.atomic_json(ledger, payload)
        for job_id in submitted:
            released = subprocess.run(
                ["scontrol", "release", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if released.returncode != 0:
                raise RuntimeError(f"release failed for {job_id}")
        payload["released"] = True
        e81.atomic_json(ledger, payload)
    except Exception:
        e81.cancel(submitted)
        if ledger.exists():
            ledger.unlink()
        raise
    print(f"[e104] cells=15 released=15 snapshot={snapshot} ledger={ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
