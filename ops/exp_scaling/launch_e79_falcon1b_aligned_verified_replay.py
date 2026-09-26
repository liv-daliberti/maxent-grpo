#!/usr/bin/env python3
"""Submit E79: aligned Falcon3-1B Dr.GRPO versus verified replay only."""

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
SEEDS = (55, 56, 57, 58, 59)
PASSES = 8
TRAIN_ROWS = 384
CHECKPOINT_INTERVAL = 192
TARGET_STEPS = TRAIN_ROWS * PASSES
REPLAY_WEIGHT = 0.10
LEARNING_RATE = 2e-7
LR_SCHEDULER = "constant"
LR_WARMUP_RATIO = 0.0
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.0
KL_BETA = 0.0
MODEL_REVISION = "28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"
MODEL_TAG = "falcon3_1b_instruct"
LEDGER = "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json"
PROTOCOL = "paper/preregistration/e79_falcon1b_aligned_verified_replay_20260804.md"
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
FALCON_PROMPT_TWINS = {
    "qwen_boxed": "falcon_boxed",
    "qwen_countdown_digits": "falcon_countdown_digits",
    "qwen_graph_digits": "falcon_graph_digits",
    "qwen_pantry_support_mask": "falcon_pantry_support_mask",
    "qwen_math": "falcon_math",
    "qwen_math_route": "falcon_math_route",
}
DOMAIN_BUDGETS = {
    "graph_coloring": (192, 512),
    "countdown": (192, 512),
    "python_factors": (512, 768),
    "mathir": (128, 384),
    "pantry_plan": (8, 704),
}
PLACEMENTS = {
    "graph_coloring": (("node202", "node203", "node204"), "a5000"),
    "countdown": (("node202", "node203", "node204"), "a5000"),
    "mathir": (("node202", "node203", "node204"), "a5000"),
    "python_factors": (("node205", "node206", "node207"), "a6000"),
    "pantry_plan": (("node205", "node206", "node207"), "a6000"),
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
    path = snapshot_util.model_root(root, "falcon1b")
    if path.name != MODEL_REVISION:
        raise SystemExit(
            f"E79 requires Falcon revision {MODEL_REVISION}, found {path.name}"
        )
    for required in ("config.json", "tokenizer_config.json"):
        if not (path / required).is_file():
            raise SystemExit(f"E79 model snapshot lacks {required}: {path}")
    return path


def references(root: Path) -> list[dict[str, Any]]:
    path = root / SOURCE_MANIFEST
    payload = json.loads(path.read_text(encoding="utf-8"))
    templates = {
        str(run["domain"]): run
        for run in payload["runs"]
        if run["arm"] == "xgrpo"
        and int(run["seed"]) == 43
        and str(run["domain"]) in DOMAINS
    }
    if set(templates) != set(DOMAINS):
        raise SystemExit("E79 lacks one canonical E72 domain template per domain")

    runs: list[dict[str, Any]] = []
    for domain in DOMAINS:
        template = templates[domain]
        evaluation = template["inherited_eval_config"]
        if int(evaluation["max_train"]) != TRAIN_ROWS:
            raise SystemExit(f"{domain}: E79 expected 384 training prompts")
        if int(evaluation["num_samples"]) != 16:
            raise SystemExit(f"{domain}: E79 expected rollout group size 16")
        if str(evaluation["prompt_template"]) not in FALCON_PROMPT_TWINS:
            raise SystemExit(f"{domain}: no Falcon prompt-surface twin")
        for key in ("prompt_data", "eval_data"):
            if not Path(evaluation[key]).exists():
                raise SystemExit(f"{domain}: missing {key}={evaluation[key]}")
        for seed in SEEDS:
            runs.append(base.reseed(template, seed))
    return runs


def run_stamp(domain: str, arm: str, seed: int) -> str:
    return f"e79_falcon_aligned_{DOMAIN_TAGS[domain]}_{arm}_s{seed}"


def save_path(root: Path, domain: str, arm: str, seed: int) -> Path:
    return (
        root
        / "var/data"
        / f"xdr_{MODEL_TAG}_{VARIANTS[arm]}_{run_stamp(domain, arm, seed)}"
    )


def task_surface(domain: str, source_template: str) -> dict[str, str]:
    generate_length, model_length = DOMAIN_BUDGETS[domain]
    return {
        "OAT_ZERO_PROMPT_TEMPLATE": FALCON_PROMPT_TWINS[source_template],
        "OAT_ZERO_GENERATE_MAX_LENGTH": str(generate_length),
        "OAT_ZERO_EVAL_GENERATE_MAX_LENGTH": str(generate_length),
        "OAT_ZERO_MAX_MODEL_LEN": str(model_length),
    }


def optimizer_env() -> dict[str, str]:
    return {
        "OAT_ZERO_LEARNING_RATE": repr(LEARNING_RATE),
        "OAT_ZERO_LR_SCHEDULER": LR_SCHEDULER,
        "OAT_ZERO_LR_WARMUP_RATIO": repr(LR_WARMUP_RATIO),
        "OAT_ZERO_ADAM_BETA_1": repr(ADAM_BETAS[0]),
        "OAT_ZERO_ADAM_BETA_2": repr(ADAM_BETAS[1]),
        "OAT_ZERO_L2": repr(WEIGHT_DECAY),
        "OAT_ZERO_BETA": repr(KL_BETA),
        "OAT_ZERO_NUM_PPO_EPOCHS": "1",
        "OAT_ZERO_MAX_NORM": "1.0",
    }


def fixed_objective(arm: str) -> dict[str, str]:
    objective = e78.fixed_objective(arm)
    if float(objective["OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA"]) != REPLAY_WEIGHT:
        raise RuntimeError("E79 replay weight drifted from the shared E78 objective")
    return objective


def build_env(
    root: Path,
    run: dict[str, Any],
    arm: str,
    runtime_snapshot: Path,
    falcon_root: Path,
) -> tuple[dict[str, str], Path]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    target = save_path(root, domain, arm, seed)
    env = base.build_export_vars(root, run, target, "b1b")
    env.update(
        {
            "OAT_ZERO_PRETRAIN": str(falcon_root),
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
            "OAT_ZERO_ADAM_OFFLOAD": "1",
            "OAT_ZERO_ACTIVATION_OFFLOADING": "0",
            "OAT_ZERO_EVAL_BATCH_SIZE": "64",
            "OAT_ZERO_USE_WB": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(
        task_surface(domain, str(run["inherited_eval_config"]["prompt_template"]))
    )
    env.update(optimizer_env())
    env.update(fixed_objective(arm))
    return env, target


def placement(domain: str, seed: int) -> tuple[str, str]:
    nodes, gpu = PLACEMENTS[domain]
    domain_offset = DOMAINS.index(domain)
    node = nodes[(domain_offset + SEEDS.index(seed)) % len(nodes)]
    return node, gpu


def sbatch_command(
    root: Path,
    run: dict[str, Any],
    arm: str,
    env: dict[str, str],
) -> list[str]:
    domain = str(run["domain"])
    seed = int(run["seed"])
    node, gpu = placement(domain, seed)
    name = f"e79-{DOMAIN_TAGS[domain][:6]}-{arm[:3]}-s{seed}"
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    return [
        "sbatch",
        "--parsable",
        "--hold",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        "--partition=cs",
        "--account=allcs",
        f"--nodelist={node}",
        f"--gres=gpu:{gpu}:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=3-00:00:00",
        "--nice=100",
        str(root / "ops/slurm/train_node302.slurm"),
    ]


def held_job_audit(
    job_id: str,
    run: dict[str, Any],
    arm: str,
    falcon_root: Path,
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E79 job {job_id}: {result.stderr.strip()}")
    domain = str(run["domain"])
    seed = int(run["seed"])
    node, gpu = placement(domain, seed)
    surface = task_surface(
        domain, str(run["inherited_eval_config"]["prompt_template"])
    )
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"ReqNodeList={node}",
        f"gres/gpu:{gpu}=1",
        f"OAT_ZERO_PRETRAIN={falcon_root}",
        f"OAT_ZERO_VARIANT={VARIANTS[arm]}",
        f"OAT_ZERO_SEED={seed}",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_MAX_PROMPT_EPOCHS=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
        "OAT_ZERO_SAVE_STEPS=192",
        "OAT_ZERO_LEARNING_RATE=2e-07",
        "OAT_ZERO_LR_SCHEDULER=constant",
        "OAT_ZERO_LR_WARMUP_RATIO=0.0",
        "OAT_ZERO_ADAM_BETA_1=0.9",
        "OAT_ZERO_ADAM_BETA_2=0.999",
        "OAT_ZERO_L2=0.0",
        "OAT_ZERO_BETA=0.0",
        f"OAT_ZERO_PROMPT_TEMPLATE={surface['OAT_ZERO_PROMPT_TEMPLATE']}",
        f"OAT_ZERO_GENERATE_MAX_LENGTH={surface['OAT_ZERO_GENERATE_MAX_LENGTH']}",
        f"OAT_ZERO_MAX_MODEL_LEN={surface['OAT_ZERO_MAX_MODEL_LEN']}",
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
        raise RuntimeError(f"held E79 job {job_id} lacks {missing}")
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
            raise SystemExit(f"required frozen E79 input is absent: {required}")
    ledger = root / LEDGER
    if args.submit and ledger.exists():
        raise SystemExit(f"refusing duplicate E79 submission: {ledger}")

    falcon_root = model_root(root)
    source_runs = references(root)
    runtime_snapshot = snapshot_util.ensure_snapshot(root, args.snapshot_root)
    planned: list[dict[str, Any]] = []
    for run in source_runs:
        for arm in ARMS:
            env, target = build_env(
                root, run, arm, runtime_snapshot, falcon_root
            )
            if target.exists():
                raise SystemExit(f"refusing to overwrite existing E79 run: {target}")
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
        raise SystemExit(f"E79 expected 50 cells, found {len(planned)}")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(part) for part in cell["command"]))
        print(
            f"[e79] dry_run=True cells={len(planned)} "
            f"snapshot={runtime_snapshot} model={falcon_root}"
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
                job_id, cell["template"], str(cell["arm"]), falcon_root
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
            "schema": "e79_falcon1b_aligned_verified_replay_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": digest(protocol),
            "launcher_sha256": digest(Path(__file__)),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": digest(source_manifest),
            "snapshot_root": str(runtime_snapshot),
            "model": "tiiuae/Falcon3-1B-Instruct",
            "model_revision": MODEL_REVISION,
            "model_root": str(falcon_root),
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
                "learning_rate": LEARNING_RATE,
                "scheduler": LR_SCHEDULER,
                "warmup_ratio": LR_WARMUP_RATIO,
                "adam_betas": list(ADAM_BETAS),
                "epsilon": 1e-8,
                "weight_decay": WEIGHT_DECAY,
                "max_grad_norm": 1.0,
                "kl_beta": KL_BETA,
                "ppo_epochs": 1,
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
        f"[e79] cells={len(records)} released={len(submitted)} "
        f"snapshot={runtime_snapshot} ledger={ledger}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
