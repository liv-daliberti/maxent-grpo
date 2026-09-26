#!/usr/bin/env python3
"""Submit E80: aligned Qwen2.5-3B Dr.GRPO versus verified replay only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e72_b3a_replay_ablation as base  # noqa: E402
import launch_e76_tuned_scale as snapshot_util  # noqa: E402
import launch_e78_verified_replay_only_05b as e78  # noqa: E402


DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
ARMS = ("control", "replay")
SEEDS = (65, 66, 67, 68, 69)
PASSES = 8
TRAIN_ROWS = 384
CHECKPOINT_INTERVAL = 192
TARGET_STEPS = TRAIN_ROWS * PASSES
REPLAY_WEIGHT = 0.10

# A common, scale-aware recipe fixed independently of replay-arm outcomes.
LEARNING_RATE = 1e-7
LR_SCHEDULER = "cosine_with_min_lr"
LR_WARMUP_RATIO = 0.10
MIN_LR_RATIO = 0.10
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.0
KL_BETA = 0.0
MAX_GRAD_NORM = 1.0
PPO_EPOCHS = 1
ROLLOUT_GROUP_SIZE = 16
TEMPERATURE = 1.0
TOP_P = 1.0

MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL_TAG = "qwen25_3b_instruct"
LEDGER = "var/artifacts/e80_qwen3b_aligned_verified_replay_jobs.json"
PROTOCOL = "paper/preregistration/e80_qwen3b_aligned_verified_replay_20260805.md"
SOURCE_MANIFEST = "var/artifacts/e72_frontier_source_runs.json"

VARIANTS = {
    "control": "grpo_compute_matched",
    "replay": "verified_first_replay_rehearsal_only",
}
DOMAIN_TAGS = {
    "graph_coloring": "graph",
    "countdown": "countdown",
    "python_factors": "python",
    "mathir": "mathir",
    "pantry_plan": "pantry",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def model_root(root: Path) -> Path:
    path = snapshot_util.model_root(root, "qwen3b")
    if path.name != MODEL_REVISION:
        raise SystemExit(
            f"E80 requires Qwen2.5-3B revision {MODEL_REVISION}, found {path.name}"
        )
    for required in ("config.json", "tokenizer_config.json", "generation_config.json"):
        if not (path / required).is_file():
            raise SystemExit(f"E80 model snapshot lacks {required}: {path}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads(
        (path / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    if config.get("model_type") != "qwen2":
        raise SystemExit(f"E80 model is not Qwen2: {path}")
    chat_template = str(tokenizer.get("chat_template", ""))
    for marker in ("<|im_start|>", "<|im_end|>"):
        if marker not in chat_template:
            raise SystemExit(f"E80 tokenizer lacks native Qwen marker {marker}")
    return path


def references(root: Path) -> list[dict[str, Any]]:
    payload = json.loads((root / SOURCE_MANIFEST).read_text(encoding="utf-8"))
    templates = {
        str(run["domain"]): run
        for run in payload["runs"]
        if run["arm"] == "xgrpo"
        and int(run["seed"]) == 43
        and str(run["domain"]) in DOMAINS
    }
    if set(templates) != set(DOMAINS):
        raise SystemExit("E80 lacks one canonical E72 template per domain")

    runs: list[dict[str, Any]] = []
    for domain in DOMAINS:
        template = templates[domain]
        evaluation = template["inherited_eval_config"]
        training = template["inherited_train_config"]
        if int(evaluation["max_train"]) != TRAIN_ROWS:
            raise SystemExit(f"{domain}: E80 expected 384 training prompts")
        if int(evaluation["num_samples"]) != ROLLOUT_GROUP_SIZE:
            raise SystemExit(f"{domain}: E80 expected rollout group size 16")
        if not str(evaluation["prompt_template"]).startswith("qwen_"):
            raise SystemExit(f"{domain}: E80 requires a native Qwen prompt surface")
        if float(training["temperature"]) != TEMPERATURE:
            raise SystemExit(f"{domain}: E80 expected rollout temperature 1")
        if float(training["top_p"]) != TOP_P:
            raise SystemExit(f"{domain}: E80 expected top-p 1")
        for key in ("prompt_data", "eval_data"):
            if not Path(evaluation[key]).exists():
                raise SystemExit(f"{domain}: missing {key}={evaluation[key]}")
        for seed in SEEDS:
            runs.append(base.reseed(template, seed))
    return runs


def run_stamp(domain: str, arm: str, seed: int) -> str:
    return f"e80_qwen3b_aligned_{DOMAIN_TAGS[domain]}_{arm}_s{seed}"


def save_path(root: Path, domain: str, arm: str, seed: int) -> Path:
    return (
        root
        / "var/data"
        / f"xdr_{MODEL_TAG}_{VARIANTS[arm]}_{run_stamp(domain, arm, seed)}"
    )


def optimizer_env() -> dict[str, str]:
    return {
        "OAT_ZERO_LEARNING_RATE": repr(LEARNING_RATE),
        "OAT_ZERO_LR_SCHEDULER": LR_SCHEDULER,
        "OAT_ZERO_LR_WARMUP_RATIO": repr(LR_WARMUP_RATIO),
        "OAT_ZERO_ADAM_BETA_1": repr(ADAM_BETAS[0]),
        "OAT_ZERO_ADAM_BETA_2": repr(ADAM_BETAS[1]),
        "OAT_ZERO_L2": repr(WEIGHT_DECAY),
        "OAT_ZERO_BETA": repr(KL_BETA),
        "OAT_ZERO_NUM_PPO_EPOCHS": str(PPO_EPOCHS),
        "OAT_ZERO_MAX_NORM": repr(MAX_GRAD_NORM),
        "OAT_ZERO_TEMPERATURE": repr(TEMPERATURE),
        "OAT_ZERO_TOP_P": repr(TOP_P),
        "OAT_ZERO_NUM_SAMPLES": str(ROLLOUT_GROUP_SIZE),
    }


def memory_env() -> dict[str, str]:
    return {
        "OAT_ZERO_N_GPU": "1",
        "OAT_ZERO_NUM_GPUS_PER_ACTOR": "1",
        "OAT_ZERO_ADAM_OFFLOAD": "1",
        "OAT_ZERO_ACTIVATION_OFFLOADING": "1",
        "OAT_ZERO_ZERO_STAGE": "2",
        "OAT_ZERO_VLLM_GPU_RATIO": "0.25",
        "OAT_ZERO_EVAL_BATCH_SIZE": "32",
    }


def fixed_objective(arm: str) -> dict[str, str]:
    objective = e78.fixed_objective(arm)
    if objective["OAT_ZERO_VARIANT"] != VARIANTS[arm]:
        raise RuntimeError("E80 objective variant drifted from E78")
    if float(objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) != REPLAY_WEIGHT:
        raise RuntimeError("E80 replay weight drifted from E78")
    return objective


def build_env(
    root: Path,
    run: dict[str, Any],
    arm: str,
    runtime_snapshot: Path,
    qwen_root: Path,
) -> tuple[dict[str, str], Path]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    target = save_path(root, domain, arm, seed)
    env = base.build_export_vars(root, run, target, "b1b")
    env.update(
        {
            "OAT_ZERO_PRETRAIN": str(qwen_root),
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(domain, arm, seed),
            "OAT_ZERO_SOURCE_ROOT": str(runtime_snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(runtime_snapshot / "ops"),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
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
    env.update(optimizer_env())
    env.update(memory_env())
    env.update(fixed_objective(arm))
    return env, target


def placement(_domain: str, _seed: int) -> tuple[str, str]:
    return "node302", "a100"


def sbatch_command(
    root: Path,
    run: dict[str, Any],
    arm: str,
    env: dict[str, str],
) -> list[str]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    node, gpu = placement(domain, seed)
    name = f"e80-{DOMAIN_TAGS[domain][:6]}-{arm[:3]}-s{seed}"
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        "--partition=mltheory",
        "--account=mltheory",
        f"--nodelist={node}",
        f"--gres=gpu:{gpu}:1",
        "--cpus-per-task=16",
        "--mem=128G",
        "--time=3-00:00:00",
        "--nice=100",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(
    job_id: str,
    run: dict[str, Any],
    arm: str,
    qwen_root: Path,
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E80 job {job_id}: {result.stderr.strip()}")
    domain = str(run["domain"])
    seed = int(run["seed"])
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        "ReqNodeList=node302",
        "TresPerNode=gres/gpu:a100:1",
        f"OAT_ZERO_PRETRAIN={qwen_root}",
        f"OAT_ZERO_VARIANT={VARIANTS[arm]}",
        f"OAT_ZERO_SEED={seed}",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
        "OAT_ZERO_SAVE_STEPS=192",
        "OAT_ZERO_LEARNING_RATE=1e-07",
        "OAT_ZERO_LR_SCHEDULER=cosine_with_min_lr",
        "OAT_ZERO_LR_WARMUP_RATIO=0.1",
        "OAT_ZERO_ADAM_BETA_1=0.9",
        "OAT_ZERO_ADAM_BETA_2=0.999",
        "OAT_ZERO_L2=0.0",
        "OAT_ZERO_BETA=0.0",
        "OAT_ZERO_NUM_PPO_EPOCHS=1",
        "OAT_ZERO_MAX_NORM=1.0",
        "OAT_ZERO_TEMPERATURE=1.0",
        "OAT_ZERO_TOP_P=1.0",
        "OAT_ZERO_NUM_SAMPLES=16",
        "OAT_ZERO_ADAM_OFFLOAD=1",
        "OAT_ZERO_ACTIVATION_OFFLOADING=1",
        "OAT_ZERO_ZERO_STAGE=2",
        "OAT_ZERO_VLLM_GPU_RATIO=0.25",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0",
        f"OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY={'1' if arm == 'control' else '0'}",
    )
    missing = [text for text in required if text not in record]
    if missing:
        raise RuntimeError(
            f"held E80 job {job_id} ({domain}, seed {seed}, {arm}) lacks {missing}"
        )
    return record


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
    source_manifest = root / SOURCE_MANIFEST
    for required in (protocol, source_manifest):
        if not required.is_file():
            raise SystemExit(f"required frozen E80 input is absent: {required}")
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E80 submission: {ledger}")

    qwen_root = model_root(root)
    source_runs = references(root)
    runtime_snapshot = snapshot_util.ensure_snapshot(root, args.snapshot_root)
    planned: list[dict[str, Any]] = []
    for run in source_runs:
        for arm in ARMS:
            env, target = build_env(root, run, arm, runtime_snapshot, qwen_root)
            if target.exists():
                raise SystemExit(f"refusing to overwrite existing E80 run: {target}")
            command = sbatch_command(root, run, arm, env)
            node, gpu = placement(str(run["domain"]), int(run["seed"]))
            planned.append(
                {
                    "domain": str(run["domain"]),
                    "arm": arm,
                    "seed": int(run["seed"]),
                    "node": node,
                    "gpu": gpu,
                    "run_stamp": run_stamp(
                        str(run["domain"]), arm, int(run["seed"])
                    ),
                    "run_dir": str(target),
                    "command": command,
                    "template": run,
                }
            )

    if len(planned) != 50:
        raise SystemExit(f"E80 expected 50 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(
            f"[e80] dry_run=True cells={len(planned)} "
            f"snapshot={runtime_snapshot} model={qwen_root}"
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
                raise RuntimeError(
                    f"invalid job id for {cell['run_stamp']}: {result.stdout!r}"
                )
            submitted.append(job_id)
            held_record = held_job_audit(
                job_id, cell["template"], str(cell["arm"]), qwen_root
            )
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "domain",
                        "arm",
                        "seed",
                        "node",
                        "gpu",
                        "run_stamp",
                        "run_dir",
                    )
                }
                | {"job_id": int(job_id), "held_scheduler_record": held_record}
            )

        payload = {
            "schema": "e80_qwen3b_aligned_verified_replay_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": digest(protocol),
            "launcher_sha256": digest(Path(__file__)),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": digest(source_manifest),
            "snapshot_root": str(runtime_snapshot),
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "model_revision": MODEL_REVISION,
            "model_root": str(qwen_root),
            "domains": list(DOMAINS),
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [index / 2 for index in range(17)],
            "replay_weight": REPLAY_WEIGHT,
            "objective": "uniform_verified_likelihood_only",
            "optimizer": {
                "name": "AdamW",
                "peak_learning_rate": LEARNING_RATE,
                "scheduler": LR_SCHEDULER,
                "warmup_ratio": LR_WARMUP_RATIO,
                "minimum_learning_rate": LEARNING_RATE * MIN_LR_RATIO,
                "adam_betas": list(ADAM_BETAS),
                "epsilon": ADAM_EPSILON,
                "weight_decay": WEIGHT_DECAY,
                "max_grad_norm": MAX_GRAD_NORM,
                "kl_beta": KL_BETA,
                "ppo_epochs": PPO_EPOCHS,
            },
            "sampling": {
                "group_size": ROLLOUT_GROUP_SIZE,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            },
            "scientific_difference": (
                "applied verified-replay derivative: exact zero versus live"
            ),
            "runs": records,
            "released": False,
        }
        atomic_json(ledger, payload)
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
        atomic_json(ledger, payload)
    except Exception:
        cancel(submitted)
        if ledger.exists():
            ledger.unlink()
        raise

    print(
        f"[e80] cells={len(records)} released={len(submitted)} "
        f"snapshot={runtime_snapshot} ledger={ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
