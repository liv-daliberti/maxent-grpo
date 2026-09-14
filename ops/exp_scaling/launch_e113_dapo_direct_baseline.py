#!/usr/bin/env python3
"""Submit the registered two-family DAPO direct-comparator cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e76_tuned_scale as snapshot_util  # noqa: E402
import launch_e78_verified_replay_only_05b as e78  # noqa: E402
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402


COHORT = "e113"
DOMAINS = e78.DOMAINS
PASSES = 8
TRAIN_ROWS = 384
TARGET_STEPS = TRAIN_ROWS * PASSES
CHECKPOINT_INTERVAL = 192
MAX_GENERATION_BATCHES = 10
MAX_QUERIES = TARGET_STEPS * 16 * MAX_GENERATION_BATCHES
SMOKE_MAX_TRAIN = 32
SMOKE_MAX_QUERIES = 640
LEDGER = "var/artifacts/e113_dapo_direct_baseline_jobs.json"
PROTOCOL = "paper/preregistration/e113_dapo_direct_baseline_20260818.md"
QWEN_CONTROL_LEDGER = e78.LEDGER
FALCON_CONTROL_LEDGER = e79.LEDGER
SOURCE_MANIFEST = e78.SOURCE_MANIFEST
FAMILY_SEEDS = {"qwen05b": e78.SEEDS, "falcon1b": e79.SEEDS}
MODEL_NAMES = {
    "qwen05b": "Qwen2.5-0.5B-Instruct",
    "falcon1b": "Falcon3-1B-Instruct",
}
MODEL_TAGS = {
    "qwen05b": e78.MODEL_TAG,
    "falcon1b": e79.MODEL_TAG,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def control_index(root: Path, family: str) -> dict[tuple[str, int], dict[str, Any]]:
    relative = QWEN_CONTROL_LEDGER if family == "qwen05b" else FALCON_CONTROL_LEDGER
    payload = load_json(root / relative)
    if payload.get("released") is not True:
        raise SystemExit(f"{family} paired-control ledger is not released")
    controls = [run for run in payload.get("runs", []) if run.get("arm") == "control"]
    index = {(str(run["domain"]), int(run["seed"])): run for run in controls}
    expected = {
        (domain, int(seed)) for domain in DOMAINS for seed in FAMILY_SEEDS[family]
    }
    if set(index) != expected or len(controls) != 25:
        raise SystemExit(f"{family} paired-control ledger does not contain 25 cells")
    for key, run in index.items():
        if not (Path(str(run["run_dir"])) / "TRAINING_COMPLETE.json").is_file():
            raise SystemExit(f"{family} paired control is not terminal: {key}")
    return index


def references(root: Path, family: str) -> list[dict[str, Any]]:
    runs = e78.references(root) if family == "qwen05b" else e79.references(root)
    expected = {
        (domain, int(seed)) for domain in DOMAINS for seed in FAMILY_SEEDS[family]
    }
    found = {(str(run["domain"]), int(run["seed"])) for run in runs}
    if found != expected or len(runs) != 25:
        raise SystemExit(f"{family} source templates do not cover exactly 25 cells")
    return sorted(runs, key=lambda run: (str(run["domain"]), int(run["seed"])))


def objective() -> dict[str, str]:
    """Return a complete isolated DAPO objective surface."""

    return {
        "OAT_ZERO_VARIANT": "dapo",
        "OAT_ZERO_CRITIC_TYPE": "grpo",
        "OAT_ZERO_DAPO_ENABLED": "1",
        "OAT_ZERO_DAPO_CLIP_LOW": "0.20",
        "OAT_ZERO_DAPO_CLIP_HIGH": "0.28",
        "OAT_ZERO_DAPO_MAX_NUM_GEN_BATCHES": str(MAX_GENERATION_BATCHES),
        "OAT_ZERO_DAPO_OVERLONG_BUFFER_RATIO": "0.20",
        "OAT_ZERO_DAPO_OVERLONG_PENALTY_FACTOR": "1.0",
        "OAT_ZERO_XDR_TAU": "inf",
        "OAT_ZERO_UCPO_TAU": "0.0",
        "OAT_ZERO_RLEP_EXPERIENCE_ROOT": "",
        "OAT_ZERO_RLEP_REPLAY_COUNT": "0",
        "OAT_ZERO_RLEP_SPARSE_FALLBACK": "0",
        "OAT_ZERO_POLICY_ENTROPY_COEF": "0.0",
        "OAT_ZERO_SEED_ENTROPY_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_DUAL_TARGET_RATIO": "0.0",
        "OAT_ZERO_MAXENT_INVERSE_ADAPTATION": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE": "0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_VERIFIED_SUPPORT_ADVANTAGE": "0",
        "OAT_ZERO_OUTCOME_COLLISION_COEF": "0.0",
        "OAT_ZERO_DIAYN_NUM_OPTIONS": "0",
        "OAT_ZERO_DIAYN_MI_BETA": "0.0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA": "0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS": "0",
        "OAT_ZERO_VERIFIED_DISCOVERY_TRACKING": "0",
        "OAT_ZERO_CANONICAL_GRAPH_ACTIONS": "0",
        "OAT_ZERO_CANONICAL_ACTION_TASK": "none",
        "OAT_ZERO_BETA": "0.0",
        "OAT_ZERO_TEMPERATURE": "1.0",
        "OAT_ZERO_TOP_P": "1.0",
        "OAT_ZERO_NUM_SAMPLES": "16",
        "OAT_ZERO_TRAIN_BATCH_SIZE": "16",
        "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE": "16",
        "OAT_ZERO_ROLLOUT_BATCH_SIZE": "1",
        "OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE": "1",
        "OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE": "16",
    }


def run_stamp(family: str, domain: str, seed: int) -> str:
    return f"e113_{family}_{e78.DOMAIN_TAGS[domain]}_dapo_s{seed}"


def save_path(root: Path, family: str, domain: str, seed: int) -> Path:
    return (
        root
        / "var/data"
        / (f"xdr_{MODEL_TAGS[family]}_dapo_{run_stamp(family, domain, seed)}")
    )


def build_env(
    root: Path,
    family: str,
    run: dict[str, Any],
    snapshot: Path,
    falcon_root: Path | None,
) -> tuple[dict[str, str], Path]:
    if family == "qwen05b":
        env, _ = e78.build_env(root, run, "control", snapshot)
    else:
        if falcon_root is None:
            raise RuntimeError("Falcon model root was not resolved")
        env, _ = e79.build_env(root, run, "control", snapshot, falcon_root)
    domain, seed = str(run["domain"]), int(run["seed"])
    target = save_path(root, family, domain, seed)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(family, domain, seed),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_MAX_QUERIES": str(MAX_QUERIES),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_AUTO_RESUME": "1",
            "OAT_ZERO_WATCHDOG_REQUEUE": "1",
            "OAT_ZERO_SAVE_CKPT": "1",
            "OAT_ZERO_USE_WB": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(objective())
    return env, target


def smoke_env(
    root: Path,
    family: str,
    run: dict[str, Any],
    snapshot: Path,
    falcon_root: Path | None,
) -> tuple[dict[str, str], Path]:
    env, _ = build_env(root, family, run, snapshot, falcon_root)
    target = root / "var/data" / f"e113_{family}_dapo_smoke"
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": f"e113_{family}_dapo_smoke",
            "OAT_ZERO_MAX_TRAIN": str(SMOKE_MAX_TRAIN),
            "OAT_ZERO_MAX_QUERIES": str(SMOKE_MAX_QUERIES),
            "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
            "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": "32",
            "OAT_ZERO_SAVE_STEPS": "32",
            "OAT_ZERO_SAVE_FROM": "32",
            "OAT_ZERO_RESUME_STEPS": "32",
            "OAT_ZERO_AUTO_RESUME": "0",
            "OAT_ZERO_WATCHDOG_REQUEUE": "0",
        }
    )
    return env, target


def job_name(family: str, domain: str, seed: int, *, smoke: bool = False) -> str:
    if smoke:
        return f"e113-{'q05' if family == 'qwen05b' else 'f1'}-dapo-smoke"
    family_tag = "q05" if family == "qwen05b" else "f1"
    return f"e113-{family_tag}-{e78.DOMAIN_TAGS[domain][:5]}-s{seed}"


def sbatch_command(
    root: Path,
    family: str,
    run: dict[str, Any],
    env: dict[str, str],
    *,
    smoke: bool = False,
    dependency: str = "",
) -> list[str]:
    if family == "qwen05b":
        command = e78.sbatch_command(root, run, "control", env)
    else:
        command = e79.sbatch_command(root, run, "control", env)
    name = job_name(family, str(run["domain"]), int(run["seed"]), smoke=smoke)
    command = [
        f"--job-name={name}" if token.startswith("--job-name=") else token
        for token in command
    ]
    if smoke:
        command = [
            "--time=08:00:00" if token.startswith("--time=") else token
            for token in command
        ]
    elif family == "qwen05b":
        command = [
            "--time=3-00:00:00" if token.startswith("--time=") else token
            for token in command
        ]
    if dependency:
        command.insert(-1, f"--dependency=afterok:{dependency}")
    return command


def submit_held(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "sbatch failed")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid sbatch response: {result.stdout!r}")
    return job_id


def audit_held(
    job_id: str,
    family: str,
    run: dict[str, Any],
    snapshot: Path,
    *,
    smoke: bool,
    dependency: str = "",
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot inspect held E113 job {job_id}")
    domain, seed = str(run["domain"]), int(run["seed"])
    required = [
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName={job_name(family, domain, seed, smoke=smoke)}",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        "OAT_ZERO_VARIANT=dapo",
        "OAT_ZERO_CRITIC_TYPE=grpo",
        "OAT_ZERO_DAPO_ENABLED=1",
        "OAT_ZERO_DAPO_CLIP_LOW=0.20",
        "OAT_ZERO_DAPO_CLIP_HIGH=0.28",
        "OAT_ZERO_DAPO_MAX_NUM_GEN_BATCHES=10",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=0",
        "OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=0",
        f"OAT_ZERO_MAX_QUERIES={SMOKE_MAX_QUERIES if smoke else MAX_QUERIES}",
    ]
    if dependency:
        required.append(f"Dependency=afterok:{dependency}")
    if family == "qwen05b":
        required.append(f"ReqNodeList={run['source_node']}")
    else:
        node, gpu = e79.placement(domain, seed)
        required.extend((f"ReqNodeList={node}", f"gres/gpu:{gpu}=1"))
    missing = [value for value in required if value not in result.stdout]
    if missing:
        raise RuntimeError(f"held E113 job {job_id} lacks {missing}")
    return result.stdout


def verify_snapshot(snapshot: Path) -> None:
    requirements = (
        ("src/oat_drgrpo/dapo.py", "dapo_token_level_policy_loss"),
        ("src/oat_drgrpo/args.py", "dapo_enabled:"),
        ("src/oat_drgrpo/learner/grpo.py", "dapo_clip_high"),
        ("src/oat_drgrpo/learner/run.py", "_collect_dapo_dynamic_feedback"),
        ("ops/run_experiment.sh", "  dapo)"),
        ("ops/train.sh", "--dapo-max-num-gen-batches"),
        ("ops/exp_scaling/launch_e113_dapo_direct_baseline.py", "MAX_QUERIES"),
    )
    missing = [
        f"{relative}: {needle}"
        for relative, needle in requirements
        if needle not in (snapshot / relative).read_text(encoding="utf-8")
    ]
    if missing:
        raise SystemExit("DAPO runtime snapshot is incomplete: " + "; ".join(missing))


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

    protocol = root / PROTOCOL
    ledger = root / LEDGER
    for required in (
        protocol,
        root / QWEN_CONTROL_LEDGER,
        root / FALCON_CONTROL_LEDGER,
        root / SOURCE_MANIFEST,
    ):
        if not required.is_file():
            raise SystemExit(f"required E113 input is absent: {required}")
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E113 submission: {ledger}")

    snapshot = snapshot_util.ensure_snapshot(root, args.snapshot_root)
    verify_snapshot(snapshot)
    falcon_root = e79.model_root(root)
    controls = {family: control_index(root, family) for family in FAMILY_SEEDS}
    cells: list[dict[str, Any]] = []
    family_runs: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILY_SEEDS:
        family_runs[family] = references(root, family)
        for run in family_runs[family]:
            domain, seed = str(run["domain"]), int(run["seed"])
            env, target = build_env(root, family, run, snapshot, falcon_root)
            if target.exists():
                raise SystemExit(f"refusing to overwrite E113 output: {target}")
            cells.append(
                {
                    "family": family,
                    "run": run,
                    "env": env,
                    "target": target,
                    "control": controls[family][(domain, seed)],
                }
            )

    smoke_cells: dict[str, dict[str, Any]] = {}
    for family in FAMILY_SEEDS:
        smoke_run = next(
            run
            for run in family_runs[family]
            if str(run["domain"]) == "countdown"
            and int(run["seed"]) == int(FAMILY_SEEDS[family][0])
        )
        env, target = smoke_env(root, family, smoke_run, snapshot, falcon_root)
        if target.exists():
            raise SystemExit(f"refusing to overwrite E113 smoke: {target}")
        smoke_cells[family] = {"run": smoke_run, "env": env, "target": target}

    if args.dry_run or not args.submit:
        for family, smoke in smoke_cells.items():
            print(
                shlex.join(
                    sbatch_command(root, family, smoke["run"], smoke["env"], smoke=True)
                )
            )
        for cell in cells:
            print(
                shlex.join(
                    sbatch_command(
                        root,
                        cell["family"],
                        cell["run"],
                        cell["env"],
                        dependency=f"{cell['family'].upper()}_SMOKE",
                    )
                )
            )
        print(f"[e113] smokes=2 scientific={len(cells)} snapshot={snapshot}")
        return 0

    submitted: list[str] = []
    smoke_records: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    try:
        for family, smoke in smoke_cells.items():
            job_id = submit_held(
                sbatch_command(root, family, smoke["run"], smoke["env"], smoke=True)
            )
            submitted.append(job_id)
            held = audit_held(job_id, family, smoke["run"], snapshot, smoke=True)
            smoke_records[family] = {
                "scientific": False,
                "job_id": int(job_id),
                "run_dir": str(smoke["target"]),
                "max_train": SMOKE_MAX_TRAIN,
                "max_queries": SMOKE_MAX_QUERIES,
                "held_scheduler_record": held,
            }

        for cell in cells:
            family = str(cell["family"])
            run = cell["run"]
            domain, seed = str(run["domain"]), int(run["seed"])
            dependency = str(smoke_records[family]["job_id"])
            job_id = submit_held(
                sbatch_command(root, family, run, cell["env"], dependency=dependency)
            )
            submitted.append(job_id)
            held = audit_held(
                job_id,
                family,
                run,
                snapshot,
                smoke=False,
                dependency=dependency,
            )
            control = cell["control"]
            records.append(
                {
                    "model_family": family,
                    "model": MODEL_NAMES[family],
                    "domain": domain,
                    "arm": "dapo",
                    "seed": seed,
                    "run_stamp": run_stamp(family, domain, seed),
                    "run_dir": str(cell["target"]),
                    "job_id": int(job_id),
                    "smoke_dependency_job_id": int(dependency),
                    "paired_control": {
                        "cohort": "e78" if family == "qwen05b" else "e79",
                        "job_id": int(control["job_id"]),
                        "run_stamp": str(control["run_stamp"]),
                        "run_dir": str(control["run_dir"]),
                    },
                    "held_scheduler_record": held,
                }
            )

        payload = {
            "schema": "e113_dapo_direct_baseline_jobs_v1",
            "cohort": COHORT,
            "released": False,
            "protocol": str(protocol),
            "protocol_sha256": e78.digest(protocol),
            "launcher_sha256": e78.digest(Path(__file__)),
            "source_manifest": str(root / SOURCE_MANIFEST),
            "source_manifest_sha256": e78.digest(root / SOURCE_MANIFEST),
            "paired_control_ledgers": {
                "qwen05b": {
                    "path": str(root / QWEN_CONTROL_LEDGER),
                    "sha256": e78.digest(root / QWEN_CONTROL_LEDGER),
                },
                "falcon1b": {
                    "path": str(root / FALCON_CONTROL_LEDGER),
                    "sha256": e78.digest(root / FALCON_CONTROL_LEDGER),
                },
            },
            "snapshot_root": str(snapshot),
            "snapshot_identity": load_json(snapshot / "SNAPSHOT_IDENTITY.json"),
            "models": MODEL_NAMES,
            "falcon_model_revision": e79.MODEL_REVISION,
            "domains": list(DOMAINS),
            "seeds": {key: list(value) for key, value in FAMILY_SEEDS.items()},
            "passes": PASSES,
            "train_rows": TRAIN_ROWS,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [index / 2 for index in range(2 * PASSES + 1)],
            "max_queries": MAX_QUERIES,
            "objective": objective(),
            "smokes": smoke_records,
            "runs": records,
        }
        e78.atomic_json(ledger, payload)
        for job_id in submitted:
            subprocess.run(["scontrol", "release", job_id], check=True)
        payload["released"] = True
        e78.atomic_json(ledger, payload)
    except Exception:
        cancel(submitted)
        raise

    print(f"[e113] released 2 smokes and {len(records)} dependent scientific cells")
    print(f"[e113] ledger {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
