#!/usr/bin/env python3
"""Submit E117: the 12-cell same-plumbing C/P/F mechanism preflight."""

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
import launch_e111_verified_support_discovery_mechanism_gate_three_scale as e111  # noqa: E402


VARIANT = e111.VARIANT
SEED = 117
TRAIN_ROWS = 64
PASSES = 1
TARGET_STEPS = TRAIN_ROWS * PASSES
CHECKPOINT_INTERVAL = 32
EVAL_DRAWS = 1
PROPOSAL_GROUPS = 1
SEMANTIC_COEFFICIENT = e111.SEMANTIC_COEFFICIENT
CAMPAIGN_TAG = "e117"
JOB_NAME_PREFIX = "e117"
LEDGER = "var/artifacts/e117_same_plumbing_component_preflight_jobs.json"
PROTOCOL = "paper/preregistration/" "e117_same_plumbing_component_preflight_20260824.md"

ARMS = ("c", "p", "f")
ARM_LABELS = {
    "c": "compute_only_control",
    "p": "proposal_replay",
    "f": "full_verified_support",
}
SENTINELS = (
    ("qwen05b", "countdown"),
    ("qwen05b", "graph_coloring"),
    ("qwen05b", "python_factors"),
    ("falcon1b", "mathir"),
)

# Each complete C/P/F block is pinned to one physical node. This makes the
# preflight a hardware-class block without sending more work to node302.
SENTINEL_NODES = {
    ("qwen05b", "countdown"): "node021",
    ("qwen05b", "graph_coloring"): "node021",
    ("qwen05b", "python_factors"): "node022",
    ("falcon1b", "mathir"): "node022",
}

SNAPSHOT_REQUIREMENTS = {
    "src/oat_drgrpo/args.py": (
        "semantic_shannon_allow_zero_coefficient_control",
        "online_canonical_counterfactual_admission_compute_only",
        "online_canonical_counterfactual_fixed_control_groups",
    ),
    "src/oat_drgrpo/learner/init.py": (
        "build_semantic_shannon_tracker",
        "semantic_shannon_allow_zero_coefficient_control",
    ),
    "src/oat_drgrpo/semantic_shannon.py": (
        "success_conditioned_semantic_metric_values",
        "if coefficient_used == 0.0",
    ),
    "src/oat_drgrpo/learner/grpo.py": ("success_conditioned_semantic_metric_values",),
    "src/oat_drgrpo/learner/run.py": (
        "counterfactual_proposal_admission_compute_only",
        "counterfactual_proposal_candidates_discarded",
        "_generate_counterfactual_fixed_control_groups",
    ),
    "ops/train.sh": (
        "--semantic-shannon-allow-zero-coefficient-control",
        "--online-canonical-counterfactual-admission-compute-only",
        "--online-canonical-counterfactual-fixed-control-groups",
    ),
    "ops/run_experiment.sh": (
        "verified_replay_semantic_maxent_verified_support_discovery)",
    ),
    "ops/exp_scaling/audit_e117_same_plumbing_component_preflight.py": (
        "_validate_terminal_training_sentinel",
        "duplicate canonical E117 optimizer row",
    ),
    "ops/exp_scaling/smoke_e117_same_plumbing_runtime.py": (
        "metric_key_parity",
        "positive bitwise zero",
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_snapshot(snapshot: Path) -> None:
    missing: list[str] = []
    for relative, needles in SNAPSHOT_REQUIREMENTS.items():
        path = snapshot / relative
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        for needle in needles:
            if needle not in text:
                missing.append(f"{relative}: {needle}")
    if missing:
        raise SystemExit("E117 snapshot is incomplete:\n  " + "\n  ".join(missing))


def objective_for_arm(arm: str) -> dict[str, str]:
    if arm not in ARMS:
        raise ValueError(f"unknown E117 arm: {arm}")
    objective = dict(e111.fixed_objective())
    objective.update(
        {
            "OAT_ZERO_SEMANTIC_SHANNON_ALLOW_ZERO_COEFFICIENT_CONTROL": "1",
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS": str(
                PROPOSAL_GROUPS
            ),
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY": (
                "1" if arm == "c" else "0"
            ),
            "OAT_ZERO_SEMANTIC_SHANNON_COEF": (
                repr(SEMANTIC_COEFFICIENT) if arm == "f" else "0.0"
            ),
        }
    )
    return objective


def assert_factorization() -> None:
    objectives = {arm: objective_for_arm(arm) for arm in ARMS}
    keys = set().union(*(objective.keys() for objective in objectives.values()))
    varying = {
        key for key in keys if len({objectives[arm].get(key) for arm in ARMS}) > 1
    }
    expected = {
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF",
    }
    if varying != expected:
        raise SystemExit(f"E117 arm factorization drifted: {sorted(varying)}")
    if (
        objectives["c"][
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY"
        ]
        != "1"
    ):
        raise SystemExit("E117 C no longer discards at the admission boundary")
    if any(
        objectives[arm]["OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS"]
        != str(PROPOSAL_GROUPS)
        for arm in ARMS
    ):
        raise SystemExit("E117 proposal compute is not fixed across arms")


def template(root: Path, scale: str, domain: str) -> dict[str, Any]:
    matches = [
        run for run in e111.references(root, scale) if str(run["domain"]) == domain
    ]
    if len(matches) != 1:
        raise SystemExit(f"E117 expected one {scale}/{domain} template")
    return e111.e81.base.reseed(matches[0], SEED)


def run_stamp(scale: str, domain: str, arm: str) -> str:
    domain_tag = e111.e81.DOMAIN_TAGS[domain]
    return f"{CAMPAIGN_TAG}_{scale}_{domain_tag}_{ARM_LABELS[arm]}_s{SEED}"


def save_path(root: Path, scale: str, domain: str, arm: str) -> Path:
    return (
        root
        / "var/data"
        / (
            f"xdr_{e111.MODEL_TAGS[scale]}_{VARIANT}_"
            f"{run_stamp(scale, domain, arm)}"
        )
    )


def build_env(
    root: Path,
    scale: str,
    domain: str,
    arm: str,
    run: dict[str, Any],
    snapshot: Path,
) -> tuple[dict[str, str], Path]:
    env, _ = e111.build_env(root, scale, run, snapshot)
    target = save_path(root, scale, domain, arm)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(scale, domain, arm),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(TARGET_STEPS),
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
            "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": str(EVAL_DRAWS),
            "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING": "1",
            "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC": "1",
            "OAT_ZERO_NUM_GPUS_PER_ACTOR": "1",
            "OAT_ZERO_VLLM_SLEEP_LEVEL": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(objective_for_arm(arm))
    return env, target


def job_name(scale: str, domain: str, arm: str) -> str:
    scale_tag = {"qwen05b": "q05", "falcon1b": "f1"}[scale]
    domain_tag = e111.e81.DOMAIN_TAGS[domain][:5]
    return f"{JOB_NAME_PREFIX}-{scale_tag}-{domain_tag}-{arm}"


def sbatch_command(
    root: Path,
    scale: str,
    domain: str,
    arm: str,
    run: dict[str, Any],
    env: dict[str, str],
) -> list[str]:
    del run  # Scientific identity is carried in the frozen export surface.
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={job_name(scale, domain, arm)}",
        f"--export=ALL,{export_pairs}",
        "--partition=all",
        "--account=mltheory",
        f"--nodelist={SENTINEL_NODES[(scale, domain)]}",
        "--gres=gpu:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=08:00:00",
        "--nice=0",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(
    job_id: str,
    *,
    scale: str,
    domain: str,
    arm: str,
    snapshot: Path,
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E117 job {job_id}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "Account=mltheory",
        f"JobName={job_name(scale, domain, arm)}",
        f"ReqNodeList={SENTINEL_NODES[(scale, domain)]}",
        "TresPerNode=gres/gpu:1",
        "TimeLimit=08:00:00",
        f"OAT_ZERO_VARIANT={VARIANT}",
        f"OAT_ZERO_SEED={SEED}",
        f"OAT_ZERO_MAX_TRAIN={TRAIN_ROWS}",
        f"OAT_ZERO_NUM_PROMPT_EPOCH={PASSES}",
        f"OAT_ZERO_EVAL_PROMPT_INTERVAL={TARGET_STEPS}",
        f"OAT_ZERO_SAVE_STEPS={CHECKPOINT_INTERVAL}",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
        "OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1",
        "OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.2",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_TRANSFORM_PROPOSALS=0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_STARVATION_FALLBACK=0",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_REPLAY_PRIORITY_VISITS=0",
        "OAT_ZERO_ONLINE_CANONICAL_PROPOSAL_ADAPTIVE_RETENTION_PRIORITY=0",
        "OAT_ZERO_SEMANTIC_RMS_CONTROL=0",
        "OAT_ZERO_SEMANTIC_SHANNON_ALLOW_ZERO_COEFFICIENT_CONTROL=1",
        "OAT_ZERO_MAXENT_ALPHA=0.0",
        "OAT_ZERO_POLICY_ENTROPY_COEF=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY="
        + ("1" if arm == "c" else "0"),
        "OAT_ZERO_SEMANTIC_SHANNON_COEF="
        + (repr(SEMANTIC_COEFFICIENT) if arm == "f" else "0.0"),
    )
    missing = [needle for needle in required if needle not in record]
    if missing:
        raise RuntimeError(f"held E117 job {job_id} lacks {missing}")
    return record


def planned_cells(root: Path, snapshot: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for scale, domain in SENTINELS:
        run = template(root, scale, domain)
        arm_envs: dict[str, dict[str, str]] = {}
        for arm in ARMS:
            env, target = build_env(root, scale, domain, arm, run, snapshot)
            arm_envs[arm] = env
            cells.append(
                {
                    "scale": scale,
                    "model_tag": e111.MODEL_TAGS[scale],
                    "domain": domain,
                    "seed": SEED,
                    "arm": arm,
                    "arm_label": ARM_LABELS[arm],
                    "node": SENTINEL_NODES[(scale, domain)],
                    "run_stamp": run_stamp(scale, domain, arm),
                    "run_dir": str(target),
                    "command": sbatch_command(root, scale, domain, arm, run, env),
                }
            )
        ignored = {
            "SAVE_PATH",
            "RUN_STAMP",
            "OAT_ZERO_SEMANTIC_SHANNON_COEF",
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY",
        }
        comparable = {
            arm: {key: value for key, value in env.items() if key not in ignored}
            for arm, env in arm_envs.items()
        }
        if comparable["c"] != comparable["p"] or comparable["p"] != comparable["f"]:
            raise SystemExit(
                f"E117 same-plumbing environment drifted: {scale}/{domain}"
            )
    if len(cells) != len(SENTINELS) * len(ARMS):
        raise SystemExit(f"E117 expected 12 cells, found {len(cells)}")
    return cells


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
    if not protocol.is_file():
        raise SystemExit(f"E117 protocol is absent: {protocol}")
    assert_factorization()
    snapshot = snapshot_util.ensure_snapshot(root, args.snapshot_root)
    verify_snapshot(snapshot)
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E117 submission: {ledger}")
    cells = planned_cells(root, snapshot)
    if args.submit:
        existing = [cell["run_dir"] for cell in cells if Path(cell["run_dir"]).exists()]
        if existing:
            raise SystemExit(f"refusing to overwrite E117 runs: {existing}")
    if args.dry_run or not args.submit:
        for cell in cells:
            print(" ".join(shlex.quote(token) for token in cell["command"]))
        print(f"[e117] dry_run=True cells={len(cells)} snapshot={snapshot}")
        return 0

    submitted: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        for cell in cells:
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
                job_id,
                scale=str(cell["scale"]),
                domain=str(cell["domain"]),
                arm=str(cell["arm"]),
                snapshot=snapshot,
            )
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "scale",
                        "model_tag",
                        "domain",
                        "seed",
                        "arm",
                        "arm_label",
                        "node",
                        "run_stamp",
                        "run_dir",
                    )
                }
                | {"job_id": int(job_id), "held_scheduler_record": held}
            )
        payload = {
            "schema": "e117_same_plumbing_component_preflight_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": digest(protocol),
            "launcher": str(Path(__file__).resolve()),
            "launcher_sha256": digest(Path(__file__)),
            "snapshot_root": str(snapshot),
            "snapshot_identity_sha256": digest(snapshot / "SNAPSHOT_IDENTITY.json"),
            "variant": VARIANT,
            "seed": SEED,
            "sentinels": [
                {
                    "scale": scale,
                    "domain": domain,
                    "node": SENTINEL_NODES[(scale, domain)],
                }
                for scale, domain in SENTINELS
            ],
            "arms": list(ARMS),
            "arm_labels": ARM_LABELS,
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "evaluation_draws": EVAL_DRAWS,
            "proposal_fixed_control_groups": PROPOSAL_GROUPS,
            "proposal_max_attempts": e111.PROPOSAL_MAX_ATTEMPTS,
            "proposal_temperature": e111.PROPOSAL_TEMPERATURE,
            "replay_weight": e111.REPLAY_WEIGHT,
            "semantic_coefficient_f": SEMANTIC_COEFFICIENT,
            "efficacy_gate": False,
            "outcomes_inspected_for_release": False,
            "pointmaze": "excluded",
            "runs": records,
            "released": False,
        }
        e111.e81.atomic_json(ledger, payload)
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
        e111.e81.atomic_json(ledger, payload)
    except Exception:
        e111.e81.cancel(submitted)
        if ledger.exists():
            ledger.unlink()
        raise
    print(f"[e117] cells=12 released=12 snapshot={snapshot} ledger={ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
