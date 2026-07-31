#!/usr/bin/env python3
"""Configure or launch the prospectively repaired Pantry Dr.GRPO smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
MODEL = ROOT / (
    "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
DATA = ROOT / "var/data/pantry_plan_modebench_v2"
REPAIR_SUFFIX = os.environ.get("PANTRY_REPAIR_SUFFIX", "r1")
if REPAIR_SUFFIX not in {"r1", "r2"}:
    raise ValueError(f"unsupported Pantry repair suffix: {REPAIR_SUFFIX}")
IS_R2 = REPAIR_SUFFIX == "r2"
PROTOCOL = ROOT / (
    "paper/preregistration/"
    + (
        "pantry_support_mask_drgrpo_smoke_v1_r2_query_budget_repair_20260730.md"
        if IS_R2
        else "pantry_support_mask_drgrpo_smoke_v1_r1_horizon_repair_20260730.md"
    )
)
PREFIX = f"ppsmoke_support_mask_drgrpo_v1_{REPAIR_SUFFIX}"
IDENTITY = ROOT / f"var/artifacts/pantry_support_mask_drgrpo_smoke_v1_{REPAIR_SUFFIX}_identity.json"
SUBMISSION = ROOT / f"var/artifacts/pantry_support_mask_drgrpo_smoke_v1_{REPAIR_SUFFIX}_submission.json"
MANIFEST = ROOT / f"var/artifacts/{PREFIX}_comparative_jobs.tsv"
AUDIT_RECEIPT = ROOT / f"var/artifacts/pantry_support_mask_drgrpo_smoke_v1_{REPAIR_SUFFIX}_audit.json"
AUDIT_RUNNER = ROOT / f"var/artifacts/pantry_support_mask_drgrpo_smoke_v1_{REPAIR_SUFFIX}_audit_runner_identity.json"
AUDIT_SUBMISSION = ROOT / f"var/artifacts/pantry_support_mask_drgrpo_smoke_v1_{REPAIR_SUFFIX}_audit_submission.json"
FAILED_JOB_ID = 30199460 if IS_R2 else 30187473
FAILED_AUDIT = ROOT / (
    "var/artifacts/pantry_support_mask_drgrpo_smoke_v1_r1_audit.json"
    if IS_R2
    else "var/artifacts/pantry_support_mask_drgrpo_smoke_v1_audit.json"
)
FAILED_STDOUT = ROOT / f"var/artifacts/logs/xdr_train-{FAILED_JOB_ID}.out"
PLACEMENT = ROOT / "var/artifacts/pantry_support_mask_drgrpo_smoke_v1_placement_amendment.json"
AUDIT_WRAPPER_NAME = f"audit_pantry_support_mask_drgrpo_smoke_v1_{REPAIR_SUFFIX}.py"
AUDIT_SLURM_NAME = f"audit_pantry_support_mask_drgrpo_smoke_v1_{REPAIR_SUFFIX}.slurm"
TEST_NAME = f"test_pantry_support_mask_drgrpo_smoke_{REPAIR_SUFFIX}.py"
REPAIR_ATTEMPT = "query_budget_and_audit_contract_r2" if IS_R2 else "horizon_generic_r1"
MAX_QUERIES = "496" if IS_R2 else "32"
LAUNCHER_ENTRYPOINT = Path(
    os.environ.get("PANTRY_REPAIR_ENTRYPOINT", Path(__file__).resolve())
).resolve()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(root: Path) -> str:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        lines.append(f"{sha(path)}  ./{path.relative_to(root).as_posix()}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def snapshot_tree(source: Path, parent_prefix: str) -> tuple[Path, str]:
    digest = tree_hash(source)
    parent = ROOT / f"var/artifacts/source_snapshots/{parent_prefix}_{digest}"
    target = parent / source.name
    if not target.exists():
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".snapshot.", dir=parent)) / source.name
        shutil.copytree(source, staging)
        os.replace(staging, target)
        staging.parent.rmdir()
    if tree_hash(target) != digest:
        raise RuntimeError(f"snapshot mismatch: {target}")
    return target, digest


def snapshot_files(files: list[tuple[Path, Path]], prefix: str) -> tuple[Path, str]:
    staging = Path(tempfile.mkdtemp(prefix=f".{prefix}.", dir=ROOT / "var/artifacts/source_snapshots"))
    for source, relative in files:
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    digest = tree_hash(staging)
    target = ROOT / f"var/artifacts/source_snapshots/{prefix}_{digest}"
    if target.exists():
        shutil.rmtree(staging)
    else:
        os.replace(staging, target)
    if tree_hash(target) != digest:
        raise RuntimeError(f"execution snapshot mismatch: {target}")
    return target, digest


def scientific_environment() -> dict[str, str]:
    values = {
        "RUN_STAMP_PREFIX": PREFIX,
        "OAT_ZERO_REPO_ROOT": str(ROOT),
        "OAT_ZERO_PRETRAIN": str(MODEL),
        "OAT_ZERO_COMPARATIVE_TASK": "pantry_plan",
        "OAT_ZERO_COMPARATIVE_DATA_ROOT": str(DATA),
        "OAT_ZERO_COMPARATIVE_MODEL": "qwen2.5-0.5b-instruct",
        "OAT_ZERO_REQUIRE_EXISTING_DATA": "1",
        "OAT_ZERO_COMPARATIVE_REBUILD": "0",
        "OAT_ZERO_APPEND_MANIFEST": "0",
        "OAT_ZERO_TRAIN_SEEDS": "76201",
        "OAT_ZERO_ONLY_ARMS": "grpo",
        "OAT_ZERO_DRGRPO_VARIANT": "grpo",
        "OAT_ZERO_XDR_TAUS": "",
        "OAT_ZERO_VERIFIED_DISCOVERY_TRACKING": "1",
        "OAT_ZERO_NUM_SAMPLES": "16",
        "OAT_ZERO_LEARNING_RATE": "0.0000002",
        "OAT_ZERO_MAX_TRAIN": "32",
        "OAT_ZERO_MAX_QUERIES": MAX_QUERIES,
        "OAT_ZERO_MAX_PROMPT_EPOCHS": "1",
        "OAT_ZERO_NUM_PROMPT_EPOCH": "1",
        "OAT_ZERO_NUM_PPO_EPOCHS": "1",
        "OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES": "32",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL": "8",
        "OAT_ZERO_ALLOW_SPARSE_EVAL": "0",
        "OAT_ZERO_SAVE_STEPS": "32",
        "OAT_ZERO_SAVE_FROM": "32",
        "OAT_ZERO_SAVE_CKPT": "1",
        "OAT_ZERO_MAX_SAVE_NUM": "1",
        "OAT_ZERO_MAX_SAVE_MEM": "2000",
        "OAT_ZERO_TRAIN_BATCH_SIZE": "16",
        "OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE": "4",
        "OAT_ZERO_ROLLOUT_BATCH_SIZE": "1",
        "OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE": "1",
        "OAT_ZERO_N_GPU": "1",
        "OAT_ZERO_NUM_GPUS_PER_ACTOR": "1",
        "OAT_ZERO_PROMPT_TEMPLATE": "qwen_pantry_support_mask",
        "OAT_ZERO_CANONICAL_GRAPH_ACTIONS": "0",
        "OAT_ZERO_CANONICAL_ACTION_TASK": "pantry_support_mask",
        "OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT": "6",
        "OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING": "1",
        "OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING": "1",
        "OAT_ZERO_INPUT_KEY": "problem",
        "OAT_ZERO_OUTPUT_KEY": "answer",
        "OAT_ZERO_EVAL_INPUT_KEY": "problem",
        "OAT_ZERO_EVAL_OUTPUT_KEY": "answer",
        "OAT_ZERO_TEST_SPLIT": "multi_answer",
        "OAT_ZERO_VERIFIER_VERSION": "fast",
        "OAT_ZERO_PROMPT_MAX_LENGTH": "640",
        "OAT_ZERO_GENERATE_MAX_LENGTH": "8",
        "OAT_ZERO_EVAL_GENERATE_MAX_LENGTH": "8",
        "OAT_ZERO_MAX_MODEL_LEN": "704",
        "OAT_ZERO_TEMPERATURE": "1",
        "OAT_ZERO_TOP_P": "1",
        "OAT_ZERO_EVAL_TEMPERATURE": "0",
        "OAT_ZERO_EVAL_MODE_COVERAGE_K": "8",
        "OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE": "1",
        "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": "1",
        "OAT_ZERO_EVAL_MODE_COVERAGE_SEED": "76299",
        "OAT_ZERO_EVAL_BATCH_SIZE": "64",
        "OAT_ZERO_MAX_NORM": "1",
        "OAT_ZERO_BETA": "0",
        "OAT_ZERO_IGNORE_NO_EOS": "0",
        "OAT_ZERO_SYNC_PARAMS_EVERY": "1",
        "OAT_ZERO_ZERO_STAGE": "2",
        "OAT_ZERO_VLLM_GPU_RATIO": "0.25",
        "OAT_ZERO_ENABLE_FLASH_ATTN": "0",
        "OAT_ZERO_ADAM_OFFLOAD": "0",
        "OAT_ZERO_ACTIVATION_OFFLOADING": "0",
        "OAT_ZERO_COLLOCATE": "1",
        "OAT_ZERO_VLLM_SLEEP": "1",
        "OAT_ZERO_VLLM_SLEEP_LEVEL": "1",
        "VLLM_USE_V1": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OAT_ZERO_AUTO_RESUME": "0",
        "OAT_ZERO_WATCHDOG_REQUEUE": "0",
        "OAT_ZERO_WATCHDOG_MAX_RESTARTS": "0",
        "OAT_ZERO_RESUME_STEPS": "-1",
        "OAT_ZERO_RESUME_FROM": "0",
        "OAT_ZERO_MAX_RESUME_NUM": "1",
        "OAT_ZERO_PRUNE_RESUME_ON_SUCCESS": "1",
        "OAT_ZERO_TRAIN_CPUS_PER_TASK": "8",
        "OAT_ZERO_TRAIN_MEMORY": "64G",
        "OAT_ZERO_TRAIN_TIME_LIMIT": "04:00:00",
        "OAT_ZERO_TRAIN_NODELIST": "node302",
        "OAT_ZERO_TRAIN_GRES": "gpu:a100:1",
        "OAT_ZERO_TRAIN_PARTITION": "mltheory",
        "OAT_ZERO_TRAIN_ACCOUNT": "mltheory",
    }
    disabled = (
        "TOKEN_ENTROPY SEED XDR_ADAPT XDR_TAU_CONTROL XDR_SAC_DUAL MAXENT "
        "MAXENT_CONTROL MAXENT_DUAL MAXENT_INVERSE MAXENT_INVERSE_CANONICAL "
        "MAXENT_INVERSE_CANONICAL_REPLAY OPEN_SET_SPLIT_CANONICAL "
        "VERIFIED_FIRST_SPLIT_CANONICAL VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL "
        "VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL VERIFIED_COUNTERFACTUAL_CANONICAL "
        "VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL VERIFIED_ROUTE_SUCCESSOR "
        "MAXENT_LENGTH_DUAL DIAYN OUTCOME_COLLISION OUTCOME_COLLISION_OUTSIDE_CENTERING "
        "SEMANTIC_SHANNON SEMANTIC_SHANNON_ADVANTAGE QUALITY_GATED_SEMANTIC_NOVELTY "
        "SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON SIGNAL_FIRST_SEMANTIC_BALANCE "
        "ONLINE_CANONICAL_MAXENT ONLINE_CANONICAL_HAARNOJA ONLINE_CANONICAL_POLICY_ENTROPY"
    ).split()
    values.update({f"OAT_ZERO_INCLUDE_{name}_ARM": "0" for name in disabled})
    return values


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "command failed with status "
            f"{result.returncode}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def prerequisites() -> None:
    for path in (PYTHON, MODEL / "config.json", DATA / "identity.json", PROTOCOL, FAILED_AUDIT, FAILED_STDOUT, PLACEMENT):
        if not path.exists():
            raise FileNotFoundError(path)
    failed = json.loads(FAILED_AUDIT.read_text())
    if failed.get("status") != "fail" or failed.get("job_id") != FAILED_JOB_ID:
        raise RuntimeError(
            f"{REPAIR_SUFFIX} requires the immutable failed antecedent audit"
        )
    stdout = FAILED_STDOUT.read_text(errors="replace")
    if IS_R2:
        if (
            failed.get("checks", {}).get("learning_rounds_exact") is not False
            or failed.get("checks", {}).get("metrics_have_exact_order") is not False
            or "[train] target_optimizer_updates=32" not in stdout
        ):
            raise RuntimeError("failed-r1 evidence lacks the frozen query-budget mismatch")
    elif "canonical learner sampler requires three supports" not in stdout:
        raise RuntimeError("failed-v1 stdout lacks the frozen horizon error")


def validate_repair() -> None:
    run_command([
        str(PYTHON),
        "-m",
        "py_compile",
        str(Path(__file__)),
        str(ROOT / f"ops/{AUDIT_WRAPPER_NAME}"),
    ])
    run_command([
        "bash",
        "-n",
        str(ROOT / f"ops/slurm/{AUDIT_SLURM_NAME}"),
    ])
    test_environment = dict(os.environ)
    test_environment["PYTHONPATH"] = str(ROOT / "src")
    python_library = str(ROOT / "var/seed_paper_eval/paper310/lib")
    inherited_library = test_environment.get("LD_LIBRARY_PATH", "")
    test_environment["LD_LIBRARY_PATH"] = (
        python_library
        + ((":" + inherited_library) if inherited_library else "")
    )
    run_command(
        [
            str(PYTHON),
            "-m",
            "pytest",
            "-q",
            str(ROOT / f"tests/{TEST_NAME}"),
            (
                str(ROOT / "tests/test_pantry_support_mask.py")
                + "::test_pantry_learner_sampler_covers_all_six_steps_and_emits_task_identity"
            ),
            (
                str(ROOT / "tests/test_e16_canonical_countdown_core.py")
                + "::test_post_update_sensor_exactly_enumerates_the_finite_policy_tree"
            ),
        ],
        env=test_environment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "run"))
    args = parser.parse_args()
    prerequisites()
    validate_repair()
    environment = dict(os.environ)
    environment.update(scientific_environment())
    if args.phase == "config":
        environment.update(OAT_ZERO_COMPARATIVE_CONFIG_ONLY="1", OAT_ZERO_SBATCH_HOLD="0")
        run_command([str(ROOT / "ops/submit_countdown_comparative.sh")], env=environment)
        print(f"[pantry-{REPAIR_SUFFIX}] configuration passed; no job submitted")
        return

    for path in (IDENTITY, SUBMISSION, MANIFEST, AUDIT_RECEIPT, AUDIT_RUNNER, AUDIT_SUBMISSION):
        if path.exists():
            raise FileExistsError(f"fresh {REPAIR_SUFFIX} artifact required: {path}")
    snapshot_prefix = f"pantry_support_mask_drgrpo_smoke_v1_{REPAIR_SUFFIX}"
    source_root, source_hash = snapshot_tree(ROOT / "src", snapshot_prefix)
    ops_root, ops_hash = snapshot_files(
        [(ROOT / f"ops/{name}", Path(name)) for name in (
            "repo_env.sh", "run_experiment.sh", "train.sh", "resolve_eval_cadence.py", "submit_countdown_comparative.sh"
        )] + [(ROOT / "ops/slurm/train_node302.slurm", Path("slurm/train_node302.slurm"))],
        f"{snapshot_prefix}_ops",
    )
    audit_root, audit_hash = snapshot_files(
        [
            (ROOT / "ops/audit_pantry_support_mask_drgrpo_smoke_v1.py", Path("audit_pantry_support_mask_drgrpo_smoke_v1.py")),
            (ROOT / f"ops/{AUDIT_WRAPPER_NAME}", Path(AUDIT_WRAPPER_NAME)),
            (ROOT / f"ops/slurm/{AUDIT_SLURM_NAME}", Path(AUDIT_SLURM_NAME)),
        ],
        f"{snapshot_prefix}_audit",
    )
    identity = {
        "schema": "pantry-support-mask-drgrpo-smoke-identity-v1",
        "repair_attempt": REPAIR_ATTEMPT,
        "failed_job_id": FAILED_JOB_ID,
        "failed_audit_sha256": sha(FAILED_AUDIT),
        "failed_stdout_sha256": sha(FAILED_STDOUT),
        "placement_amendment_sha256": sha(PLACEMENT),
        "repair_protocol_sha256": sha(PROTOCOL),
        "protocol_sha256": sha(PROTOCOL),
        "launcher_sha256": sha(LAUNCHER_ENTRYPOINT),
        "source_hash": source_hash,
        "execution_hash": ops_hash,
        "audit_execution_hash": audit_hash,
        "data_tree_sha256": tree_hash(DATA),
        "model_config_sha256": sha(MODEL / "config.json"),
        "viability_receipt_sha256": sha(ROOT / "var/artifacts/pantry_support_mask_dev_v1.json"),
        "viability_identity_sha256": sha(ROOT / "var/artifacts/pantry_support_mask_dev_v1_identity.json"),
        "viability_audit_sha256": sha(ROOT / "var/artifacts/pantry_support_mask_dev_v1_audit.json"),
        "source_root": str(source_root),
        "ops_root": str(ops_root),
        "audit_root": str(audit_root),
        "development_only": True,
        "arm": "grpo",
        "seed": 76201,
        "optimizer_updates": 32,
        "train_rows": 32,
        "max_queries": int(MAX_QUERIES),
        "rollouts_per_prompt": 16,
        "canonical_action_task": "pantry_support_mask",
        "horizon": 6,
        "sequence_count": 64,
        "maxent_actuators_enabled": False,
        "recovery_enabled": False,
        "placement_only": True,
        "scientific_change": False,
        "required_regressions": [
            "pantry_learner_sampler_six_steps_and_task_identity",
            "pantry_exact_entropy_64_leaves_63_prefixes",
            *(
                [
                    "pantry_query_budget_496_for_32_updates",
                    "pantry_emitted_nonfinite_telemetry_contract",
                ]
                if IS_R2
                else []
            ),
        ],
    }
    atomic(IDENTITY, identity)
    environment.update(
        OAT_ZERO_CAMPAIGN_SOURCE_ROOT=str(source_root),
        OAT_ZERO_OPS_SNAPSHOT_ROOT=str(ops_root),
        OAT_ZERO_PROTOCOL_IDENTITY=str(IDENTITY),
        OAT_ZERO_COMPARATIVE_CONFIG_ONLY="0",
        OAT_ZERO_SBATCH_HOLD="1",
    )
    run_command([str(ops_root / "submit_countdown_comparative.sh")], env=environment)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1 or rows[0]["arm"] != "grpo" or rows[0]["seed"] != "76201":
        raise RuntimeError(f"{REPAIR_SUFFIX} manifest differs from the frozen cell")
    job_id = int(rows[0]["job_id"])
    before = run_command(["scontrol", "show", "job", "-o", str(job_id)])
    for text in ("JobState=PENDING", "Reason=JobHeldUser", "ReqNodeList=node302", "gres/gpu:a100:1", "RunTime=00:00:00"):
        if text not in before:
            raise RuntimeError(f"held {REPAIR_SUFFIX} job lacks {text}")
    run_command(["scontrol", "update", f"JobId={job_id}", "Partition=all", "Gres=gpu:a5000:1", "NodeList="])
    run_command(["scontrol", "update", f"JobId={job_id}", "Requeue=0"])
    after = run_command(["scontrol", "show", "job", "-o", str(job_id)])
    for text in ("JobState=PENDING", "Reason=JobHeldUser", "Partition=all", "ReqNodeList=(null)", "gres/gpu:a5000:1", "RunTime=00:00:00", "NumCPUs=8", "MinMemoryNode=64G", "TimeLimit=04:00:00"):
        if text not in after:
            raise RuntimeError(f"amended {REPAIR_SUFFIX} job lacks {text}")
    atomic(SUBMISSION, {
        "schema": f"pantry-support-mask-drgrpo-smoke-submission-{REPAIR_SUFFIX}",
        "identity_sha256": sha(IDENTITY),
        "manifest_sha256": sha(MANIFEST),
        "job_id": job_id,
        "held_job_audit": "pass",
        "released": True,
        "placement_only": True,
        "scientific_change": False,
        "before": before,
        "after": after,
    })
    run_command(["scontrol", "release", str(job_id)])
    audit_job_id = int(run_command([
        "sbatch", "--parsable", f"--dependency=afterany:{job_id}",
        # The r1 audit entrypoint reads PANTRY_R1_TRAIN_JOB_ID while the r2 one
        # reads PANTRY_TRAIN_JOB_ID; export both so either suffix resolves.
        f"--export=ALL,ROOT_DIR={ROOT},OAT_ZERO_EXECUTION_ROOT={audit_root}"
        f",PANTRY_TRAIN_JOB_ID={job_id}"
        f",PANTRY_R1_TRAIN_JOB_ID={job_id}",
        str(audit_root / AUDIT_SLURM_NAME),
    ]).split(";", 1)[0])
    atomic(AUDIT_RUNNER, {
        "schema": f"pantry-support-mask-drgrpo-smoke-{REPAIR_SUFFIX}-audit-runner-v1",
        "training_job_id": job_id,
        "audit_job_id": audit_job_id,
        "audit_execution_hash": audit_hash,
        "identity_sha256": sha(IDENTITY),
        "dependency": f"afterany:{job_id}",
    })
    atomic(AUDIT_SUBMISSION, {
        "schema": f"pantry-support-mask-drgrpo-smoke-{REPAIR_SUFFIX}-audit-submission-v1",
        "audit_runner_sha256": sha(AUDIT_RUNNER),
        "audit_job_id": audit_job_id,
        "released": True,
    })
    print(
        f"[pantry-{REPAIR_SUFFIX}] released training job {job_id}; "
        f"audit job {audit_job_id}"
    )


if __name__ == "__main__":
    main()
