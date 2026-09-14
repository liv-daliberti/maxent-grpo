#!/usr/bin/env python3
"""Submit 15 parser-matched Re:Dr.GRPO Python comparators for E105."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_e78_verified_replay_only_05b as e78  # noqa: E402
import launch_e79_falcon1b_aligned_verified_replay as e79  # noqa: E402
import launch_e80r1_qwen3b_aligned_verified_replay as e80  # noqa: E402
import launch_e106_python_lambda_normalization_three_scale as e106  # noqa: E402
import launch_e105_group_centered_semantic_repair_full_three_scale as e105  # noqa: E402


DOMAIN = "python_factors"
SCALE_SEEDS = {
    "qwen05b": (43, 44, 45, 46, 47),
    "falcon1b": (55, 56, 57, 58, 59),
    "qwen3b": (70, 71, 72, 73, 74),
}
MODEL_TAGS = {
    "qwen05b": e78.MODEL_TAG,
    "falcon1b": e79.MODEL_TAG,
    "qwen3b": e80.MODEL_TAG,
}
ARM = "replay"
TRAIN_ROWS = 384
PASSES = 8
TARGET_STEPS = TRAIN_ROWS * PASSES
CHECKPOINT_INTERVAL = 192
LEDGER = "var/artifacts/e109_repaired_python_replay_comparators_jobs.json"
PROTOCOL = (
    "paper/preregistration/e109_repaired_python_replay_comparators_20260817.md"
)
QWEN3_A6000_NODE_LIST = "node[103-104,205-208,805]"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def require_gate(root: Path) -> Path:
    snapshot, _audit_path, audit, _e104_audit = e105.check_gate(root)
    if audit.get("mechanism_gate_used_outcome_metrics") is not False:
        raise SystemExit("E109 gate used outcome metrics")
    if audit.get("post_update_outcome_metrics_inspected") is not False:
        raise SystemExit("E109 gate violated outcome blinding")
    return snapshot


def qwen3_a6000_assignment(root: Path) -> tuple[dict[str, Any], set[int]]:
    payload, paired = e105.require_qwen3_paired_placement(root)
    python_seeds = {seed for domain, seed in paired if domain == DOMAIN}
    if python_seeds != {73, 74}:
        raise SystemExit("E109 prospective Python placement drifted")
    return payload, python_seeds


def qwen3_a6000_seeds(root: Path) -> set[int]:
    return qwen3_a6000_assignment(root)[1]


def references(root: Path, scale: str) -> list[dict[str, Any]]:
    if scale == "qwen05b":
        runs = e78.references(root)
    elif scale == "falcon1b":
        runs = e79.references(root)
    elif scale == "qwen3b":
        runs = e80.references(root)
    else:
        raise SystemExit(f"unsupported E109 scale: {scale}")
    selected = [
        run
        for run in runs
        if str(run["domain"]) == DOMAIN
        and int(run["seed"]) in SCALE_SEEDS[scale]
    ]
    if {int(run["seed"]) for run in selected} != set(SCALE_SEEDS[scale]):
        raise SystemExit(f"E109 {scale} lacks five Python templates")
    return sorted(selected, key=lambda run: int(run["seed"]))


def run_stamp(scale: str, seed: int) -> str:
    return f"e109_{scale}_python_repaired_replay_s{seed}"


def save_path(root: Path, scale: str, seed: int) -> Path:
    return root / "var/data" / (
        f"xdr_{MODEL_TAGS[scale]}_verified_first_replay_rehearsal_only_"
        f"{run_stamp(scale, seed)}"
    )


def build_env(
    root: Path,
    scale: str,
    run: dict[str, Any],
    snapshot: Path,
) -> tuple[dict[str, str], Path]:
    seed = int(run["seed"])
    if scale == "qwen05b":
        env, _ = e78.build_env(root, run, ARM, snapshot)
    elif scale == "falcon1b":
        env, _ = e79.build_env(
            root, run, ARM, snapshot, e79.model_root(root)
        )
    else:
        env, _ = e80.build_env(
            root, run, ARM, snapshot, e80.model_root(root)
        )
    target = save_path(root, scale, seed)
    env.update(
        {
            "SAVE_PATH": str(target),
            "RUN_STAMP": run_stamp(scale, seed),
            "OAT_ZERO_SOURCE_ROOT": str(snapshot / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(snapshot / "ops"),
            "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
            "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
            "OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE": "0",
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
            "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE": "0",
            "OAT_ZERO_SEMANTIC_RMS_CONTROL": "0",
            "OAT_ZERO_MAX_TRAIN": str(TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_SAVE_FROM": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_RESUME_STEPS": str(CHECKPOINT_INTERVAL),
            "OAT_ZERO_AUTO_RESUME": "1",
            "OAT_ZERO_WATCHDOG_REQUEUE": "1",
            "OAT_ZERO_USE_WB": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if env.get("OAT_ZERO_ONLINE_CANONICAL_REPLAY") != "1":
        raise SystemExit("E109 inherited replay is inactive")
    if env.get("OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY") != "0":
        raise SystemExit("E109 inherited only a compute-only control")
    if env.get("OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA") != "0.1":
        raise SystemExit("E109 replay coefficient drifted")
    return env, target


def base_command(
    root: Path,
    scale: str,
    run: dict[str, Any],
    env: dict[str, str],
) -> list[str]:
    if scale == "qwen05b":
        return e78.sbatch_command(root, run, ARM, env)
    if scale == "falcon1b":
        return e79.sbatch_command(root, run, ARM, env)
    return e80.sbatch_command(root, run, ARM, env)


def job_name(scale: str, seed: int) -> str:
    tag = {"qwen05b": "q05", "falcon1b": "f1", "qwen3b": "q3"}[scale]
    return f"e109-{tag}-python-s{seed}"


def sbatch_command(
    root: Path,
    scale: str,
    run: dict[str, Any],
    env: dict[str, str],
    qwen3_a6000: set[int],
) -> list[str]:
    seed = int(run["seed"])
    command = base_command(root, scale, run, env)
    output: list[str] = []
    for token in command:
        if token.startswith("--job-name="):
            output.append(f"--job-name={job_name(scale, seed)}")
        elif scale == "qwen3b" and seed in qwen3_a6000 and token.startswith(
            "--partition="
        ):
            output.append("--partition=lowprio")
        elif scale == "qwen3b" and seed in qwen3_a6000 and token.startswith(
            "--nodelist="
        ):
            output.append(f"--nodelist={QWEN3_A6000_NODE_LIST}")
        elif scale == "qwen3b" and seed in qwen3_a6000 and token.startswith(
            "--gres="
        ):
            output.append("--gres=gpu:a6000:1")
        else:
            output.append(token)
    return output


def held_job_audit(
    job_id: str,
    scale: str,
    run: dict[str, Any],
    snapshot: Path,
    qwen3_a6000: set[int] | frozenset[int] = frozenset(),
) -> str:
    result = subprocess.run(
        ["scontrol", "show", "job", "-dd", "-o", job_id],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect held E109 job {job_id}")
    record = result.stdout
    required = (
        "JobState=PENDING",
        "Reason=JobHeldUser",
        f"JobName={job_name(scale, int(run['seed']))}",
        f"OAT_ZERO_SEED={run['seed']}",
        f"OAT_ZERO_SOURCE_ROOT={snapshot / 'src'}",
        f"OAT_ZERO_OPS_SNAPSHOT_ROOT={snapshot / 'ops'}",
        "OAT_ZERO_MAX_TRAIN=384",
        "OAT_ZERO_NUM_PROMPT_EPOCH=8",
        "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
        "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.0",
        "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_GROUP_CENTERED_ADVANTAGE=0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY=1",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=0",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood_per_rollout",
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.1",
    )
    missing = [needle for needle in required if needle not in record]
    if scale == "qwen3b":
        if int(run["seed"]) in qwen3_a6000:
            placement_required = (
                "Partition=lowprio",
                f"ReqNodeList={QWEN3_A6000_NODE_LIST}",
                "TresPerNode=gres/gpu:a6000:1",
            )
        else:
            placement_required = (
                "Partition=mltheory",
                "ReqNodeList=node302",
                "TresPerNode=gres/gpu:a100:1",
            )
        missing.extend(
            needle for needle in placement_required if needle not in record
        )
    if missing:
        raise RuntimeError(f"held E109 job {job_id} lacks {missing}")
    return record


def cancel(job_ids: list[str]) -> None:
    if job_ids:
        subprocess.run(["scancel", *job_ids], check=False)


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
        raise SystemExit(f"E109 protocol is absent: {protocol}")
    ledger_path = root / LEDGER
    if args.submit and ledger_path.exists():
        raise SystemExit(f"refusing duplicate E109 submission: {ledger_path}")
    if args.submit:
        snapshot = require_gate(root)
        qwen3_placement, qwen3_a6000 = qwen3_a6000_assignment(root)
    else:
        snapshot = Path(args.snapshot_root or (root / e106.SNAPSHOT)).resolve()
        e106.verify_snapshot(root, snapshot)
        qwen3_placement = {}
        qwen3_a6000 = set()

    planned: list[dict[str, Any]] = []
    for scale in SCALE_SEEDS:
        for run in references(root, scale):
            seed = int(run["seed"])
            env, target = build_env(root, scale, run, snapshot)
            if args.submit and target.exists():
                raise SystemExit(f"refusing to overwrite E109 run: {target}")
            planned.append(
                {
                    "scale": scale,
                    "model_tag": MODEL_TAGS[scale],
                    "arm": ARM,
                    "domain": DOMAIN,
                    "seed": seed,
                    "run_stamp": run_stamp(scale, seed),
                    "run_dir": str(target),
                    "command": sbatch_command(
                        root, scale, run, env, qwen3_a6000
                    ),
                    "template": run,
                }
            )
    if len(planned) != 15 or Counter(row["scale"] for row in planned) != {
        "qwen05b": 5,
        "falcon1b": 5,
        "qwen3b": 5,
    }:
        raise SystemExit("E109 did not materialize five Python cells per scale")
    if args.dry_run or not args.submit:
        for cell in planned:
            print(" ".join(shlex.quote(token) for token in cell["command"]))
        print(f"[e109] dry_run=True cells=15 snapshot={snapshot}")
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
                    f"E109 submission failed for {cell['run_stamp']}: "
                    f"{result.stderr.strip()}"
                )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"invalid E109 job id: {result.stdout!r}")
            submitted.append(job_id)
            held = held_job_audit(
                job_id,
                str(cell["scale"]),
                cell["template"],
                snapshot,
                qwen3_a6000,
            )
            records.append(
                {
                    key: cell[key]
                    for key in (
                        "scale",
                        "model_tag",
                        "arm",
                        "domain",
                        "seed",
                        "run_stamp",
                        "run_dir",
                    )
                }
                | {"job_id": int(job_id), "held_scheduler_record": held}
            )
        payload = {
            "schema": "e109_repaired_python_replay_comparators_jobs_v1",
            "protocol": str(protocol),
            "protocol_sha256": e78.digest(protocol),
            "launcher_sha256": e78.digest(Path(__file__)),
            "snapshot_root": str(snapshot),
            "snapshot_sha256": e106.SNAPSHOT_SHA256,
            "parser_surface_version": e106.SURFACE_VERSION,
            "domain": DOMAIN,
            "domains": [DOMAIN],
            "arms": [ARM],
            "scales": list(SCALE_SEEDS),
            "seeds": {scale: list(seeds) for scale, seeds in SCALE_SEEDS.items()},
            "train_rows": TRAIN_ROWS,
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "registered_passes": [index / 2 for index in range(17)],
            "replay_weight": 0.1,
            "semantic_coefficient": 0.0,
            "pointmaze": "excluded",
            "post_e104_or_e106_update_outcomes_inspected": False,
            "qwen3_a6000_seeds": sorted(qwen3_a6000),
            "qwen3_paired_placement_artifact": str(
                root / e105.QWEN3_PLACEMENT_ARTIFACT
            ),
            "qwen3_paired_placement_artifact_sha256": e78.digest(
                root / e105.QWEN3_PLACEMENT_ARTIFACT
            ),
            "qwen3_paired_placement_protocol_sha256": qwen3_placement.get(
                "protocol_sha256"
            ),
            "qwen3_paired_placement_script_sha256": qwen3_placement.get(
                "script_sha256"
            ),
            "runs": records,
            "released": False,
        }
        e78.atomic_json(ledger_path, payload)
        for job_id in submitted:
            result = subprocess.run(
                ["scontrol", "release", job_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"failed to release E109 job {job_id}")
        payload["released"] = True
        e78.atomic_json(ledger_path, payload)
    except Exception:
        cancel(submitted)
        if ledger_path.exists():
            ledger_path.unlink()
        raise

    print(
        f"[e109] cells={len(records)} released={len(submitted)} "
        f"snapshot={snapshot} ledger={ledger_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
