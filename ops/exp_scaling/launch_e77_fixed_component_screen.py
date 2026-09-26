#!/usr/bin/env python3
"""Submit E77, a one-seed exact-zero screen of the three fixed components."""

from __future__ import annotations

import argparse
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
import launch_e76_tuned_scale as e76  # noqa: E402

DOMAINS = ("graph_coloring", "pantry_plan")
ARMS = (
    "none",
    "mass_only",
    "no_semantic",
    "no_mass",
    "no_balance",
    "full_fixed",
)
SEED = 58
TRAIN_ROWS = 320
VALIDATION_ROWS = 64
PASSES = 4
TARGET_STEPS = TRAIN_ROWS * PASSES
LEDGER = "var/artifacts/e77_fixed_component_screen_jobs.json"
PROTOCOL = "paper/preregistration/e77_fixed_component_screen_05b_20260804.md"
SPLIT_MANIFEST = "var/artifacts/e76_tuned_scale_splits.json"
MODEL_TAG = "qwen25_0p5b_instruct"

ARM_CONFIG: dict[str, dict[str, Any]] = {
    "none": {
        "variant": "grpo_compute_matched",
        "eta": 0.0,
        "mu": 0.0,
        "alpha": 0.0,
        "compute_only": True,
    },
    "mass_only": {
        "variant": "verified_first_replay_rehearsal_only",
        "eta": 0.0,
        "mu": 0.10,
        "alpha": 0.0,
        "compute_only": False,
    },
    "no_semantic": {
        "variant": "verified_first_replay_only_ablation",
        "eta": 0.0,
        "mu": 0.10,
        "alpha": 0.10,
        "compute_only": False,
    },
    "no_mass": {
        "variant": "verified_first_global_replay_canonical",
        "eta": 0.10,
        "mu": 0.0,
        "alpha": 0.10,
        "compute_only": False,
    },
    "no_balance": {
        "variant": "verified_first_global_replay_canonical",
        "eta": 0.10,
        "mu": 0.10,
        "alpha": 0.0,
        "compute_only": False,
    },
    "full_fixed": {
        "variant": "verified_first_global_replay_canonical",
        "eta": 0.10,
        "mu": 0.10,
        "alpha": 0.10,
        "compute_only": False,
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def references(root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(
        (root / "var/artifacts/e72_frontier_source_runs.json").read_text()
    )
    result = {
        str(run["domain"]): run
        for run in payload["runs"]
        if run["arm"] == "xgrpo"
        and int(run["seed"]) == 43
        and str(run["domain"]) in DOMAINS
    }
    if set(result) != set(DOMAINS):
        raise SystemExit(f"missing reference domains: {sorted(set(DOMAINS) - set(result))}")
    return result


def split_paths(root: Path, domain: str) -> tuple[Path, Path]:
    train = root / "var/data/e76_tuned_scale" / domain / "train"
    validation = root / "var/data/e76_tuned_scale" / domain / "validation"
    if not train.is_dir() or not validation.is_dir():
        raise SystemExit(f"{domain}: sealed E76 split is absent")
    return train, validation


def run_stamp(domain: str, arm: str) -> str:
    short = {"graph_coloring": "graph", "pantry_plan": "pantry"}[domain]
    return f"e77_fixed_components_{short}_{arm}_s{SEED}"


def save_path(root: Path, domain: str, arm: str) -> Path:
    variant = str(ARM_CONFIG[arm]["variant"])
    return (
        root
        / "var/data"
        / f"xdr_{MODEL_TAG}_{variant}_{run_stamp(domain, arm)}"
    )


def arm_overrides(arm: str) -> dict[str, str]:
    config = ARM_CONFIG[arm]
    eta = float(config["eta"])
    mu = float(config["mu"])
    alpha = float(config["alpha"])
    split = arm not in {"none", "mass_only"}
    return {
        "OAT_ZERO_VARIANT": str(config["variant"]),
        "OAT_ZERO_SEMANTIC_SHANNON_COEF": repr(eta),
        "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "1" if eta else "0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": (
            "1" if eta else "0"
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY": "1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE": (
            "split_mass_balance_per_rollout"
            if split
            else "verified_likelihood_per_rollout"
        ),
        # In the mass-only objective replay_alpha is the likelihood dose.
        # In the split objective replay_alpha is exactly the balance dose.
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA": repr(
            mu if arm == "mass_only" else (0.10 if arm == "none" else alpha)
        ),
        # Ignored by the non-split objectives; retained positive there because
        # argument validation treats it as a configured but inactive field.
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA": repr(
            mu if split else 0.10
        ),
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY": (
            "1" if bool(config["compute_only"]) else "0"
        ),
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS": "0",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SINGLETON_ONLY": "0",
        "OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_ALPHA": "0.0",
        "OAT_ZERO_MAXENT_INVERSE_ADAPTATION": "0",
        "OAT_ZERO_POLICY_ENTROPY_COEF": "0.0",
        "OAT_ZERO_SEED_ENTROPY_ALPHA": "0.0",
        "OAT_ZERO_BETA": "0.0",
    }


def build_env(
    root: Path,
    source: dict[str, Any],
    domain: str,
    arm: str,
    snapshot_root: Path,
) -> tuple[dict[str, str], Path]:
    clone = base.reseed(source, SEED)
    target = save_path(root, domain, arm)
    env = base.build_export_vars(root, clone, target, "xgrpo")
    train, validation = split_paths(root, domain)
    env.update(
        {
            "OAT_ZERO_PRETRAIN": str(source["inherited_eval_config"]["pretrain"]),
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(domain, arm),
            "OAT_ZERO_PROMPT_DATA": str(train),
            "OAT_ZERO_EVAL_DATA": str(validation),
            "OAT_ZERO_TEST_SPLIT": "multi_answer",
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_SEED": str(SEED),
            "OAT_ZERO_LEARNING_RATE": "2e-7",
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": "80",
            "OAT_ZERO_SAVE_STEPS": str(TRAIN_ROWS),
            "OAT_ZERO_SAVE_FROM": str(TRAIN_ROWS),
            "OAT_ZERO_RESUME_STEPS": str(TRAIN_ROWS),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot_root / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot_root / "ops"),
            "OAT_ZERO_N_GPU": "1",
            "OAT_ZERO_NUM_GPUS_PER_ACTOR": "1",
            "OAT_ZERO_ADAM_OFFLOAD": "1",
            "OAT_ZERO_USE_WB": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(arm_overrides(arm))
    return env, target


def placement(domain: str) -> dict[str, str]:
    if domain == "pantry_plan":
        return {
            "partition": "all",
            "account": "allcs",
            "nodelist": "node205,node206,node208",
            "gres": "gpu:a6000:1",
        }
    return {
        "partition": "all",
        "account": "allcs",
        "nodelist": "node105,node202,node203,node204",
        "gres": "gpu:a5000:1",
    }


def submit(
    root: Path,
    domain: str,
    arm: str,
    env: dict[str, str],
    dry_run: bool,
) -> str | None:
    place = placement(domain)
    name = f"e77-{domain[:3]}-{arm[:7]}-s{SEED}"
    export_pairs = ",".join(f"{key}={value}" for key, value in env.items())
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={name}",
        f"--export=ALL,{export_pairs}",
        f"--partition={place['partition']}",
        f"--account={place['account']}",
        f"--nodelist={place['nodelist']}",
        f"--gres={place['gres']}",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=12:00:00",
        "--nice=100",
        "--hold",
        str(root / "ops/slurm/train_node302.slurm"),
    ]
    if dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return None
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"submission failed for {name}: {result.stderr.strip()}")
    job_id = result.stdout.strip().split(";")[0]
    normalize = subprocess.run(
        [
            "scontrol",
            "update",
            f"JobId={job_id}",
            "Partition=all",
            f"NodeList={place['nodelist']}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if normalize.returncode != 0:
        subprocess.run(["scancel", job_id], check=False)
        raise SystemExit(
            f"placement normalization failed for {name}: {normalize.stderr.strip()}"
        )
    release = subprocess.run(
        ["scontrol", "release", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if release.returncode != 0:
        subprocess.run(["scancel", job_id], check=False)
        raise SystemExit(f"release failed for {name}: {release.stderr.strip()}")
    return job_id


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--snapshot-root", type=Path)
    args = parser.parse_args()
    if args.submit and args.dry_run:
        raise SystemExit("choose --submit or --dry-run, not both")

    for required in (root / PROTOCOL, root / SPLIT_MANIFEST):
        if not required.is_file():
            raise SystemExit(f"required frozen input is absent: {required}")
    ledger_path = root / LEDGER
    if args.submit and ledger_path.exists():
        existing = json.loads(ledger_path.read_text())
        if existing.get("runs"):
            raise SystemExit(f"refusing duplicate submission: {ledger_path}")

    snapshot_root = e76.ensure_snapshot(root, args.snapshot_root)
    source_runs = references(root)
    records: list[dict[str, Any]] = []
    submitted: list[str] = []
    for domain in DOMAINS:
        for arm in ARMS:
            env, target = build_env(
                root, source_runs[domain], domain, arm, snapshot_root
            )
            if target.exists():
                raise SystemExit(f"refusing to overwrite existing run: {target}")
            job_id = submit(
                root,
                domain,
                arm,
                env,
                dry_run=(args.dry_run or not args.submit),
            )
            if job_id:
                submitted.append(job_id)
            records.append(
                {
                    "domain": domain,
                    "arm": arm,
                    "job_id": job_id,
                    "run_stamp": run_stamp(domain, arm),
                    "run_dir": str(target),
                    "seed": SEED,
                    "target_steps": TARGET_STEPS,
                    "coefficients": {
                        "eta": ARM_CONFIG[arm]["eta"],
                        "mu": ARM_CONFIG[arm]["mu"],
                        "alpha": ARM_CONFIG[arm]["alpha"],
                    },
                    "variant": ARM_CONFIG[arm]["variant"],
                    "placement": placement(domain),
                }
            )

    payload = {
        "schema": "e77_fixed_component_screen_jobs_v1",
        "protocol": str(root / PROTOCOL),
        "split_manifest": str(root / SPLIT_MANIFEST),
        "snapshot_root": str(snapshot_root),
        "model": MODEL_TAG,
        "seed": SEED,
        "train_rows": TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "passes": PASSES,
        "target_steps": TARGET_STEPS,
        "runs": records,
    }
    if args.submit:
        atomic_json(ledger_path, payload)
    print(
        f"[e77] cells={len(records)} submitted={len(submitted)} "
        f"snapshot={snapshot_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
