#!/usr/bin/env python3
"""Verify and pairwise-analyze the one-shot E75R3 untouched evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(os.environ.get("OAT_ZERO_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()
ARMS = (
    "grpo",
    "verified_first_global_replay_canonical",
    "verified_first_delayed_singleton_replay_canonical",
)
COMPARISONS = (
    ("current_minus_grpo", ARMS[1], ARMS[0]),
    ("delayed_minus_grpo", ARMS[2], ARMS[0]),
    ("delayed_minus_current", ARMS[2], ARMS[1]),
)
METRICS = ("mean8", "pass8", "distinct8", "modes_per_success")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_BASE_SEED = 756_400


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("frozen E75R3 final-evaluation plan hash mismatch")
    plan = json.loads(path.read_text())
    if plan.get("schema") != "e75r3-point-maze-waypoint-final-eval-plan-v1":
        raise RuntimeError("unexpected E75R3 final-evaluation plan schema")
    if tuple(plan.get("arms", ())) != ARMS:
        raise RuntimeError("frozen E75R3 arm order changed")
    if plan.get("exclusions") != []:
        raise RuntimeError("E75R3 final evaluation must have no exclusions")
    file_identities = (
        ("E75R3 identity", plan["e75r3_identity"]),
        ("development qualification", plan["dev_qualification"]),
        ("data identity", plan["data_identity"]),
        ("protocol", plan["protocol"]),
    )
    for label, identity in file_identities:
        if sha256_file(Path(identity["path"])) != identity["sha256"]:
            raise RuntimeError(f"frozen {label} hash mismatch")
    for label, identity in (
        ("source snapshot", plan["source_snapshot"]),
        ("execution snapshot", plan["execution_snapshot"]),
    ):
        if tree_sha256(Path(identity["path"])) != identity["tree_sha256"]:
            raise RuntimeError(f"frozen {label} tree hash mismatch")
    return plan


def verify_arm(plan: Mapping[str, Any], arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown E75R3 arm: {arm}")
    checkpoint = plan["checkpoints"][arm]
    model = Path(checkpoint["path"])
    receipt_path = Path(checkpoint["training_receipt"])
    if sha256_file(receipt_path) != checkpoint["training_receipt_sha256"]:
        raise RuntimeError(f"{arm}: training receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text())
    required = {
        "status": "complete",
        "arm": arm,
        "seed": 88504,
        "evaluation_only": False,
        "evaluation_split": "dev",
        "evaluation_prompt_count": 32,
        "optimizer_updates": 64,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"{arm}: training receipt {key} changed")
    if receipt.get("output_model_tree_sha256") != checkpoint["tree_sha256"]:
        raise RuntimeError(f"{arm}: receipt checkpoint hash changed")
    if tree_sha256(model) != checkpoint["tree_sha256"]:
        raise RuntimeError(f"{arm}: frozen checkpoint tree hash mismatch")
    data_identity = Path(plan["data_identity"]["path"])
    if sha256_file(data_identity) != plan["data_identity"]["sha256"]:
        raise RuntimeError("E75R3 data identity hash mismatch")
    if receipt.get("data_identity_sha256") != plan["data_identity"]["sha256"]:
        raise RuntimeError(f"{arm}: training data identity does not match plan")


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap(deltas: Sequence[float], *, seed: int) -> tuple[float, float]:
    if not deltas:
        raise ValueError("paired bootstrap requires map deltas")
    rng = random.Random(seed)
    count = len(deltas)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draws.append(sum(deltas[rng.randrange(count)] for _ in range(count)) / count)
    return percentile(draws, 0.025), percentile(draws, 0.975)


def eval_paths(plan: Mapping[str, Any], arm: str) -> dict[str, Path]:
    short = plan["arm_short"][arm]
    stem = ROOT / f"var/artifacts/e75r3_point_maze_waypoint_final_eval_{short}"
    return {
        "receipt": Path(str(stem) + ".json"),
        "metrics": Path(str(stem) + ".metrics.jsonl"),
        "replay": Path(str(stem) + ".replay.jsonl"),
    }


def load_evaluation(plan: Mapping[str, Any], arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_arm(plan, arm)
    paths = eval_paths(plan, arm)
    receipt = json.loads(paths["receipt"].read_text())
    required = {
        "status": "complete",
        "arm": arm,
        "seed": 88504,
        "evaluation_only": True,
        "evaluation_split": "eval",
        "evaluation_prompt_count": 64,
        "evaluation_coordinates": 1,
        "optimizer_updates": 0,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"{arm}: final evaluation receipt {key} invalid")
    if Path(receipt["model"]).resolve() != Path(plan["checkpoints"][arm]["path"]).resolve():
        raise RuntimeError(f"{arm}: final evaluator loaded the wrong checkpoint")
    if receipt.get("metrics_sha256") != sha256_file(paths["metrics"]):
        raise RuntimeError(f"{arm}: final metrics hash mismatch")
    if receipt.get("state_replay_sha256") != sha256_file(paths["replay"]):
        raise RuntimeError(f"{arm}: final replay hash mismatch")
    rows = [json.loads(line) for line in paths["metrics"].read_text().splitlines() if line]
    if len(rows) != 1:
        raise RuntimeError(f"{arm}: expected exactly one final evaluation coordinate")
    evaluation = rows[0]
    expected = {
        "schema": "point-maze-waypoint-pilot-evaluation-v1",
        "split": "eval",
        "arm": arm,
        "seed": 88504,
        "learning_round": 0,
        "evaluation_prompt_count": 64,
        "evaluation_trajectory_count": 512,
    }
    for key, value in expected.items():
        if evaluation.get(key) != value:
            raise RuntimeError(f"{arm}: final metric {key} invalid")
    maps = evaluation.get("per_map") or []
    if len(maps) != 64 or len({row.get("map_id") for row in maps}) != 64:
        raise RuntimeError(f"{arm}: final evaluation does not contain 64 unique maps")
    return receipt, evaluation


def analyze(plan: Mapping[str, Any]) -> dict[str, Any]:
    loaded = {arm: load_evaluation(plan, arm) for arm in ARMS}
    evaluations = {arm: loaded[arm][1] for arm in ARMS}
    reference = [(row["map_id"], row["family"]) for row in evaluations[ARMS[0]]["per_map"]]
    for arm in ARMS[1:]:
        observed = [(row["map_id"], row["family"]) for row in evaluations[arm]["per_map"]]
        if observed != reference:
            raise RuntimeError(f"{arm}: final map order/families are not paired")

    contrasts: dict[str, Any] = {}
    for comparison_index, (label, treatment, control) in enumerate(COMPARISONS):
        treatment_rows = evaluations[treatment]["per_map"]
        control_rows = evaluations[control]["per_map"]
        metrics: dict[str, Any] = {}
        for metric_index, metric in enumerate(METRICS):
            deltas = [float(left[metric]) - float(right[metric]) for left, right in zip(treatment_rows, control_rows)]
            seed = BOOTSTRAP_BASE_SEED + 100 * comparison_index + metric_index
            lower, upper = paired_bootstrap(deltas, seed=seed)
            metrics[metric] = {
                "mean_delta": sum(deltas) / len(deltas),
                "paired_map_bootstrap_95": [lower, upper],
                "bootstrap_seed": seed,
            }
        contrasts[label] = {"treatment": treatment, "control": control, "metrics": metrics}

    per_map = []
    for index, (map_id, family) in enumerate(reference):
        row: dict[str, Any] = {"map_id": map_id, "family": family, "arms": {}}
        for arm in ARMS:
            row["arms"][arm] = {metric: evaluations[arm]["per_map"][index][metric] for metric in METRICS}
        per_map.append(row)
    return {
        "schema": "e75r3-point-maze-waypoint-final-analysis-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "plan_sha256": plan["self_sha256_external"],
        "exclusions_applied": [],
        "map_count": 64,
        "trajectories_per_arm": 512,
        "arms": {
            arm: {
                "metrics": {metric: evaluations[arm][metric] for metric in METRICS},
                "receipt_sha256": sha256_file(eval_paths(plan, arm)["receipt"]),
                "metrics_sha256": sha256_file(eval_paths(plan, arm)["metrics"]),
            }
            for arm in ARMS
        },
        "comparisons": contrasts,
        "bootstrap": {
            "kind": "paired map percentile",
            "replicates": BOOTSTRAP_REPLICATES,
            "base_seed": BOOTSTRAP_BASE_SEED,
            "interval": [0.025, 0.975],
        },
        "per_map": per_map,
    }


def markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# E75R3 PointMaze untouched final evaluation",
        "",
        "All three frozen update-64 checkpoints were evaluated once on the same 64 maps (8 trajectories/map); no exclusions were applied.",
        "",
        "## Arm summaries",
        "",
        "| arm | mean8 | pass8 | distinct8 | modes/success |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = result["arms"][arm]["metrics"]
        lines.append(f"| {arm} | {values['mean8']:.6f} | {values['pass8']:.6f} | {values['distinct8']:.6f} | {values['modes_per_success']:.6f} |")
    lines += ["", "## Paired map contrasts", "", "| contrast | metric | mean delta | descriptive 95% CI |", "|---|---|---:|---:|"]
    for label, comparison in result["comparisons"].items():
        for metric in METRICS:
            row = comparison["metrics"][metric]
            lower, upper = row["paired_map_bootstrap_95"]
            lines.append(f"| {label} | {metric} | {row['mean_delta']:+.6f} | [{lower:+.6f}, {upper:+.6f}] |")
    lines += ["", "Intervals are descriptive paired-map bootstrap intervals for one training seed; they are not population-level hypothesis tests.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-arm")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--plan-sha256", required=True)
    verify.add_argument("--arm", choices=ARMS, required=True)
    report = subparsers.add_parser("analyze")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--plan-sha256", required=True)
    report.add_argument("--output-json", type=Path, required=True)
    report.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan, args.plan_sha256)
    plan["self_sha256_external"] = args.plan_sha256
    if args.command == "verify-arm":
        verify_arm(plan, args.arm)
        print(f"[e75r3-final] verified frozen arm {args.arm}")
        return 0
    if args.output_json.exists() or args.output_markdown.exists():
        raise FileExistsError("fresh E75R3 final-analysis outputs required")
    result = analyze(plan)
    atomic_json(args.output_json, result)
    args.output_markdown.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output_json)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
