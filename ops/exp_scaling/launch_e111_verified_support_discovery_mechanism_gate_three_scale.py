#!/usr/bin/env python3
"""Submit E111: v7 verified-support MaxEnt plus discovery at three scales."""

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
VARIANT = "verified_replay_semantic_maxent_verified_support_discovery"
ARM = "verified_support_discovery"
SMOKE_TRAIN_ROWS = 8
SMOKE_PASSES = 8
SMOKE_TARGET_STEPS = SMOKE_TRAIN_ROWS * SMOKE_PASSES
CHECKPOINT_INTERVAL = 32
SEMANTIC_COEFFICIENT = 0.10
REPLAY_WEIGHT = 0.10
LEDGER = "var/artifacts/e111_verified_support_discovery_mechanism_gate_jobs.json"
PROTOCOL = (
    "paper/preregistration/"
    "e111_verified_support_discovery_mechanism_gate_three_scale_20260818.md"
)
SOURCE_MANIFEST = e81.SOURCE_MANIFEST
AUDIT = "var/artifacts/e111_verified_support_discovery_mechanism_gate_audit.json"
UNIT_EVIDENCE = "var/artifacts/e111_verified_support_discovery_unit_tests.json"
PROPOSAL_TEMPERATURE = 1.20
PROPOSAL_MAX_ATTEMPTS = 1
GENERAL_GPU_NODES = "node[020-022,025,103-104,202-208,403]"
QWEN3_A6000_NODES = "node[103-104,205-208]"

SNAPSHOT_REQUIREMENTS = {
    "src/oat_drgrpo/semantic_shannon.py": (
        "semantic_shannon_tracker_v7_verified_support",
        "verified_support_keys_by_group",
        "structural_unseen_bucket",
    ),
    "src/oat_drgrpo/online_canonical_bank.py": (
        "verified_replay_support",
        "Proposal rows never become on-policy",
    ),
    "src/oat_drgrpo/args.py": (
        "semantic_shannon_verified_support_include_replay_bank",
    ),
    "src/oat_drgrpo/learner/init.py": (
        "verified_support_include_replay_bank",
    ),
    "src/oat_drgrpo/learner/grpo.py": (
        "semantic_shannon_verified_support_include_replay_bank_active",
        "external_verified_support_nonempty_group_fraction",
    ),
    "ops/train.sh": (
        "--semantic-shannon-verified-support-include-replay-bank",
    ),
    "ops/run_experiment.sh": (
        "verified_replay_semantic_maxent_verified_support_discovery)",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS=0",
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
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE": "0",
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE": "1",
            "OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK": "1",
            "OAT_ZERO_SEMANTIC_RMS_CONTROL": "0",
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
            "OAT_ZERO_ONLINE_CANONICAL_REPLAY_RETENTION_SAFE_BALANCE": "0",
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
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK": "0",
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS": "0",
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER": "1.0",
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING": "1",
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY": "0",
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MISSED_ROLLOUT_OPPORTUNITIES": "2",
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_MAX_MEAN_LOGPROB_DROP": "0.5",
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_REFRESH_VISITS": "4",
            "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_SCORE_COOLDOWN_OBSERVATIONS": "2",
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
        raise SystemExit("E111 snapshot is incomplete:\n  " + "\n  ".join(missing))


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
        raise SystemExit(f"E111 {scale} expected five templates, found {len(selected)}")
    return selected


def run_stamp(scale: str, domain: str) -> str:
    seed = SCALE_SEEDS[scale]
    return f"e111_{scale}_{e81.DOMAIN_TAGS[domain]}_{ARM}_s{seed}"


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
    canonical_learner = str(
        run["inherited_eval_config"]["canonical_action_task"]
    ) != "none"
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
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(SMOKE_TARGET_STEPS),
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_MAX_SAVE_NUM": "2",
            "OAT_ZERO_MAX_RESUME_NUM": "2",
            "OAT_ZERO_AUTO_RESUME": "1",
            "OAT_ZERO_WATCHDOG_REQUEUE": "1",
            "OAT_ZERO_USE_WB": "0",
            "OAT_ZERO_SAVE_CKPT": "1",
            "OAT_ZERO_ALLOW_SPARSE_EVAL": "1",
            "OAT_ZERO_EVAL_BATCH_SIZE": "8",
            "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": "1",
            "OAT_ZERO_NUM_GPUS_PER_ACTOR": "1",
            "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING": (
                "0" if canonical_learner else "1"
            ),
            "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC": (
                "0" if canonical_learner else "1"
            ),
            "OAT_ZERO_VLLM_SLEEP_LEVEL": "1",
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
        elif token.startswith("--partition="):
            output.append("--partition=all")
        elif token.startswith("--account="):
            output.append("--account=mltheory")
        elif token.startswith("--nodelist="):
            nodes = (
                QWEN3_A6000_NODES
                if scale == "qwen3b"
                else GENERAL_GPU_NODES
            )
            output.append(f"--nodelist={nodes}")
        elif token.startswith("--gres="):
            output.append(
                "--gres=gpu:a6000:1"
                if scale == "qwen3b"
                else "--gres=gpu:1"
            )
        elif token.startswith("--time="):
            output.append(
                "--time=12:00:00" if scale == "qwen3b" else "--time=08:00:00"
            )
        elif token.startswith("--nice="):
            output.append("--nice=0")
        else:
            output.append(token)
    if not any(token.startswith("--nice=") for token in output):
        output.insert(-1, "--nice=0")
    return output


def job_name(scale: str, domain: str) -> str:
    scale_tag = {"qwen05b": "q05", "falcon1b": "f1", "qwen3b": "q3"}[scale]
    return f"e111-{scale_tag}-{e81.DOMAIN_TAGS[domain][:6]}"


def held_job_audit(job_id: str, scale: str, run: dict[str, Any]) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E111 job {job_id}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Account=mltheory",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={SCALE_SEEDS[scale]}",
        f"OAT_ZERO_MAX_TRAIN={SMOKE_TRAIN_ROWS}",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={SMOKE_PASSES}",
        f"OAT_ZERO_EVAL_PROMPT_INTERVAL={SMOKE_TARGET_STEPS}",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE=1",
        "OAT_ZERO_SEMANTIC_SHANNON_VERIFIED_SUPPORT_INCLUDE_REPLAY_BANK=1",
        "OAT_ZERO_SEMANTIC_RMS_CONTROL=0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_EXACT_GRAMMAR_TRANSFORMS=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.2",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS=0",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_MULTIPLIER=1.0",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_RETENTION_TRACKING=1",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY=0",
        "OAT_ZERO_MAXENT_ALPHA=0.0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
    )
    missing = [needle for needle in required if needle not in record]
    if scale == "qwen3b":
        placement = (
            "Partition=mltheory",
            f"ReqNodeList={QWEN3_A6000_NODES}",
            "TresPerNode=gres/gpu:a6000:1",
            "TimeLimit=12:00:00",
        )
    else:
        placement = (
            "Partition=mltheory",
            f"ReqNodeList={GENERAL_GPU_NODES}",
            "TresPerNode=gres/gpu:1",
            "TimeLimit=08:00:00",
        )
    missing.extend(needle for needle in placement if needle not in record)
    if missing:
        raise RuntimeError(f"held E111 job {job_id} lacks {missing}")
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
    unit_evidence = root / UNIT_EVIDENCE
    for required in (protocol, manifest, unit_evidence):
        if not required.is_file():
            raise SystemExit(f"required frozen input is absent: {required}")
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E111 submission: {ledger}")
    snapshot = snapshot_util.ensure_snapshot(root, args.snapshot_root)
    verify_snapshot(snapshot)

    planned: list[dict[str, Any]] = []
    for scale in SCALE_SEEDS:
        for run in references(root, scale):
            domain = str(run["domain"])
            env, target = build_env(root, scale, run, snapshot)
            if args.submit and target.exists():
                raise SystemExit(f"refusing to overwrite E111 run: {target}")
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
        raise SystemExit(f"E111 expected 15 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(token) for token in cell["command"]))
        print(f"[e111] dry_run=True cells=15 snapshot={snapshot}")
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
            "schema": "e111_verified_support_discovery_mechanism_gate_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": digest(protocol),
            "launcher_sha256": digest(Path(__file__)),
            "source_manifest": str(manifest),
            "source_manifest_sha256": digest(manifest),
            "unit_evidence": str(unit_evidence),
            "unit_evidence_sha256": digest(unit_evidence),
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
            "proposal_temperature": PROPOSAL_TEMPERATURE,
            "proposal_max_attempts": PROPOSAL_MAX_ATTEMPTS,
            "proposal_priority_visits": 0,
            "proposal_priority_multiplier": 1.0,
            "objective": "verified_support_v7_plus_uniform_replaydr_plus_support_discovery",
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
    print(f"[e111] cells=15 released=15 snapshot={snapshot} ledger={ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
