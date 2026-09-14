#!/usr/bin/env python3
"""Render the PointMaze-v1 and PointMaze-Tour evidence boundary.

The static-domain figures deliberately exclude both interfaces because their
sequential actions have no meaningful greedy pass@1 endpoint.  This figure
keeps three boundaries explicit: PointMaze v1 and the redesigned Tour are not
pooled with one another; neither is pooled with the five static domains; and
Tour prefixes do not receive aggregates.  It shows every registered Tour
checkpoint, all terminal Tour pairs, and the completed five-seed Qwen/Falcon
v1 null blocks that motivated the redesign.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
import paper_method_style as method_visuals  # noqa: E402
import paper_style as style  # noqa: E402


DEFAULT_OUTPUT = ROOT / "paper/figures/interactive_domain_boundary"
TARGET_STEPS = 3072
CHECKPOINT_INTERVAL = 192
PASSES = 8
TRAIN_ROWS = 384
EVAL_ROWS = 128
T_CRIT_DF4 = 2.7764451051977987

COHORTS = {
    "qwen05b": {
        "model": "Qwen2.5-0.5B",
        "ledger": ROOT
        / "var/artifacts/e93pt_point_maze_tour_verified_replay_05b_jobs.json",
        "registered_seeds": [43, 44, 45, 46, 47],
        "arms": {"control": "drgrpo", "replay": "replay_grpo"},
    },
    "falcon1b": {
        "model": "Falcon3-1B",
        "ledger": ROOT / "var/artifacts/e94pt_falcon_point_maze_tour_jobs.json",
        "registered_seeds": [55, 56, 57, 58, 59],
        "arms": {"control": "drgrpo", "replay": "replay_grpo"},
    },
}
V1_COHORTS = {
    "qwen05b": {
        "model": "Qwen2.5-0.5B",
        "ledger": ROOT
        / "var/artifacts/e78pm_point_maze_verified_replay_only_05b_jobs.json",
        "registered_seeds": [43, 44, 45, 46, 47],
        "receipt_schema": "e78pm-point-maze-verified-replay-only-receipt-v1",
    },
    "falcon1b": {
        "model": "Falcon3-1B",
        "ledger": ROOT
        / "var/artifacts/e79pm_falcon_point_maze_verified_replay_jobs.json",
        "registered_seeds": [55, 56, 57, 58, 59],
        "receipt_schema": "e79pm-point-maze-verified-replay-only-receipt-v1",
    },
}
SEMANTIC_LEDGER = (
    ROOT / "var/artifacts/e96pt_point_maze_tour_semantic_maxent_jobs.json"
)
METHOD_ORDER = ("drgrpo", "replay_grpo", "replay_semantic_maxent")
METHOD_STYLE = {
    method: method_visuals.method_style(method) for method in METHOD_ORDER
}
METRICS = ("pass8", "mean8", "distinct8", "adjusted_breadth8")
TRAJECTORY_METRICS = ("mean8", "adjusted_breadth8")
TRAJECTORY_LABEL = {
    "mean8": "mean correctness@8",
    "adjusted_breadth8": r"adjusted breadth $D-P$",
}
COMPARISONS = (
    {
        "key": "qwen_replay_minus_control",
        "model": "qwen05b",
        "treatment": "replay_grpo",
        "control": "drgrpo",
        "label": "Qwen Replay − Dr",
    },
    {
        "key": "qwen_semantic_minus_replay",
        "model": "qwen05b",
        "treatment": "replay_semantic_maxent",
        "control": "replay_grpo",
        "label": "Qwen + Semantic − Replay",
    },
    {
        "key": "falcon_replay_minus_control",
        "model": "falcon1b",
        "treatment": "replay_grpo",
        "control": "drgrpo",
        "label": "Falcon Replay − Dr",
    },
)
V1_COMPARISONS = (
    {
        "key": "qwen_v1_replay_minus_control",
        "model": "qwen05b",
        "treatment": "replay_grpo",
        "control": "drgrpo",
        "label": "Qwen v1 Replay − Dr",
    },
    {
        "key": "falcon_v1_replay_minus_control",
        "model": "falcon1b",
        "treatment": "replay_grpo",
        "control": "drgrpo",
        "label": "Falcon v1 Replay − Dr",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, Any]:
    return {
        "byte_length": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _validate_ledger(payload: dict[str, Any], *, expected_arms: set[str]) -> None:
    if (
        payload.get("domains") != ["point_maze_tour"]
        or int(payload.get("checkpoint_interval_steps", -1))
        != CHECKPOINT_INTERVAL
        or int(payload.get("evaluation_interval_steps", -1))
        != CHECKPOINT_INTERVAL
        or int(payload.get("target_steps", -1)) != TARGET_STEPS
        or int(payload.get("passes", -1)) != PASSES
        or int(payload.get("train_rows", -1)) != TRAIN_ROWS
        or int(payload.get("eval_rows", -1)) != EVAL_ROWS
        or set(payload.get("arms", [])) != expected_arms
        or payload.get("evaluation_split") != "eval"
    ):
        raise RuntimeError("PointMaze Tour ledger contract drifted")


def _validate_v1_ledger(
    payload: dict[str, Any], *, schema: str, seeds: list[int]
) -> None:
    if (
        payload.get("schema") != schema
        or payload.get("domains") != ["point_maze"]
        or payload.get("arms") != ["control", "replay"]
        or payload.get("seeds") != seeds
        or int(payload.get("checkpoint_interval_steps", -1))
        != CHECKPOINT_INTERVAL
        or int(payload.get("target_steps", -1)) != TARGET_STEPS
        or int(payload.get("passes", -1)) != PASSES
        or int(payload.get("train_rows", -1)) != TRAIN_ROWS
        or int(payload.get("eval_rows", -1)) != EVAL_ROWS
        or payload.get("objective") != "uniform_verified_likelihood_only"
    ):
        raise RuntimeError("PointMaze v1 ledger contract drifted")


def _read_curve(path: Path, *, arm: str, seed: int) -> dict[str, dict[str, float]]:
    curve: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if '"schema": "point-maze-tour-evaluation-v1"' not in line:
                continue
            row = json.loads(line)
            step = int(row["learning_round"])
            if (
                row.get("split") != "eval"
                or str(row.get("arm")) != arm
                or int(row.get("seed")) != seed
                or step < 0
                or step > TARGET_STEPS
                or step % CHECKPOINT_INTERVAL
            ):
                raise RuntimeError(f"invalid evaluation row in {path}:{source_line}")
            key = f"{step / TRAIN_ROWS:.1f}"
            values = {
                "pass8": float(row["pass8"]),
                "mean8": float(row["mean8"]),
                "distinct8": float(row["distinct8"]),
            }
            values["adjusted_breadth8"] = (
                values["distinct8"] - values["pass8"]
            )
            if key in curve and any(
                not _same(curve[key][metric], values[metric])
                for metric in METRICS
            ):
                raise RuntimeError(f"conflicting duplicate checkpoint in {path}")
            curve[key] = values
    if not curve or "0.0" not in curve:
        raise RuntimeError(f"missing PointMaze Tour evaluation curve: {path}")
    ordered = sorted(float(value) for value in curve)
    if any(
        not _same(right - left, CHECKPOINT_INTERVAL / TRAIN_ROWS)
        for left, right in zip(ordered, ordered[1:])
    ):
        raise RuntimeError(f"non-contiguous PointMaze Tour checkpoints: {path}")
    return curve


def _read_v1_curve(
    path: Path, *, arm: str, seed: int
) -> dict[str, dict[str, float]]:
    curve: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if '"schema": "point-maze-waypoint-pilot-evaluation-v1"' not in line:
                continue
            row = json.loads(line)
            step = int(row["learning_round"])
            if (
                row.get("split") != "eval"
                or str(row.get("arm")) != arm
                or int(row.get("seed")) != seed
                or step < 0
                or step > TARGET_STEPS
                or step % CHECKPOINT_INTERVAL
                or int(row.get("evaluation_prompt_count", -1)) != EVAL_ROWS
            ):
                raise RuntimeError(
                    f"invalid PointMaze v1 evaluation row in {path}:{source_line}"
                )
            key = f"{step / TRAIN_ROWS:.1f}"
            values = {
                "pass8": float(row["pass8"]),
                "mean8": float(row["mean8"]),
                "distinct8": float(row["distinct8"]),
            }
            values["adjusted_breadth8"] = (
                values["distinct8"] - values["pass8"]
            )
            if key in curve and any(
                not _same(curve[key][metric], values[metric])
                for metric in METRICS
            ):
                raise RuntimeError(f"conflicting v1 checkpoint in {path}")
            curve[key] = values
    expected = {f"{index / 2:.1f}" for index in range(2 * PASSES + 1)}
    if set(curve) != expected:
        raise RuntimeError(f"incomplete PointMaze v1 evaluation curve: {path}")
    return curve


def _load_run(
    run: dict[str, Any],
    *,
    method: str,
    input_sha256: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    arm = str(run["arm"])
    seed = int(run["seed"])
    metrics_path = Path(run["metrics_path"])
    if not metrics_path.is_file():
        raise RuntimeError(f"missing PointMaze Tour metrics: {metrics_path}")
    curve = _read_curve(metrics_path, arm=arm, seed=seed)
    deepest_pass = max(float(key) for key in curve)
    deepest_step = int(round(deepest_pass * TRAIN_ROWS))
    receipt_path = Path(run["receipt_path"])
    terminal = deepest_step == TARGET_STEPS
    receipt: dict[str, Any] | None = None
    if terminal:
        if not receipt_path.is_file():
            raise RuntimeError(f"terminal PointMaze run lacks receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != "point-maze-tour-run-v1"
            or receipt.get("status") != "complete"
            or str(receipt.get("arm")) != arm
            or int(receipt.get("seed")) != seed
            or int(receipt.get("updates", -1)) != TARGET_STEPS
        ):
            raise RuntimeError(f"terminal PointMaze receipt drifted: {receipt_path}")
    for source in (metrics_path, receipt_path if terminal else None):
        if source is not None:
            input_sha256[_relative(source)] = _source_record(source)
    return {
        "method": method,
        "arm": arm,
        "seed": seed,
        "deepest_pass": deepest_pass,
        "terminal": terminal,
        "curve": curve,
        "metrics_path": _relative(metrics_path),
        "receipt_path": _relative(receipt_path) if terminal else None,
    }


def _load_v1_run(
    run: dict[str, Any],
    *,
    method: str,
    receipt_schema: str,
    input_sha256: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    arm = str(run["arm"])
    seed = int(run["seed"])
    metrics_path = Path(run["metrics_path"])
    receipt_path = Path(run["receipt_path"])
    if not metrics_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"missing PointMaze v1 run artifact for {arm}/{seed}")
    curve = _read_v1_curve(metrics_path, arm=arm, seed=seed)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != receipt_schema
        or receipt.get("status") != "complete"
        or receipt.get("arm") != arm
        or int(receipt.get("seed", -1)) != seed
        or int(receipt.get("optimizer_updates", -1)) != TARGET_STEPS
        or int(receipt.get("passes", -1)) != PASSES
        or int(receipt.get("evaluation_prompt_count", -1)) != EVAL_ROWS
        or receipt.get("evaluation_split") != "eval"
    ):
        raise RuntimeError(f"PointMaze v1 receipt drifted: {receipt_path}")
    for source in (metrics_path, receipt_path):
        input_sha256[_relative(source)] = _source_record(source)
    return {
        "method": method,
        "arm": arm,
        "seed": seed,
        "deepest_pass": float(PASSES),
        "terminal": True,
        "curve": curve,
        "metrics_path": _relative(metrics_path),
        "receipt_path": _relative(receipt_path),
        "data_identity_sha256": str(receipt["data_identity_sha256"]),
    }


def _load_primary(
    key: str,
    spec: dict[str, Any],
    input_sha256: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ledger_path = Path(spec["ledger"])
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    _validate_ledger(payload, expected_arms=set(spec["arms"]))
    if payload.get("seeds") != spec["registered_seeds"]:
        raise RuntimeError(f"{key}: registered seed block drifted")
    input_sha256[_relative(ledger_path)] = _source_record(ledger_path)
    runs: dict[str, dict[str, Any]] = {}
    for run in payload["runs"]:
        arm = str(run["arm"])
        seed = int(run["seed"])
        if arm not in spec["arms"] or seed not in spec["registered_seeds"]:
            raise RuntimeError(f"{key}: unexpected run {arm}/{seed}")
        method = spec["arms"][arm]
        run_key = f"{method}/{seed}"
        if run_key in runs:
            raise RuntimeError(f"{key}: duplicate run {run_key}")
        runs[run_key] = _load_run(
            run, method=method, input_sha256=input_sha256
        )
    expected = {
        f"{method}/{seed}"
        for method in spec["arms"].values()
        for seed in spec["registered_seeds"]
    }
    if set(runs) != expected:
        raise RuntimeError(f"{key}: incomplete registered run ledger")
    return {
        "model": spec["model"],
        "registered_seeds": spec["registered_seeds"],
        "data_identity_sha256": payload["data_identity_sha256"],
        "runs": runs,
    }


def _load_v1_primary(
    key: str,
    spec: dict[str, Any],
    input_sha256: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ledger_path = Path(spec["ledger"])
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    expected_schema = {
        "qwen05b": "e78pm_point_maze_verified_replay_only_jobs_v1",
        "falcon1b": "e79pm_falcon_point_maze_verified_replay_jobs_v1",
    }[key]
    _validate_v1_ledger(
        payload,
        schema=expected_schema,
        seeds=spec["registered_seeds"],
    )
    input_sha256[_relative(ledger_path)] = _source_record(ledger_path)
    runs: dict[str, dict[str, Any]] = {}
    arm_to_method = {"control": "drgrpo", "replay": "replay_grpo"}
    for run in payload["runs"]:
        arm = str(run["arm"])
        seed = int(run["seed"])
        if arm not in arm_to_method or seed not in spec["registered_seeds"]:
            raise RuntimeError(f"{key}: unexpected PointMaze v1 run {arm}/{seed}")
        method = arm_to_method[arm]
        run_key = f"{method}/{seed}"
        if run_key in runs:
            raise RuntimeError(f"{key}: duplicate PointMaze v1 run {run_key}")
        runs[run_key] = _load_v1_run(
            run,
            method=method,
            receipt_schema=spec["receipt_schema"],
            input_sha256=input_sha256,
        )
    expected = {
        f"{method}/{seed}"
        for method in arm_to_method.values()
        for seed in spec["registered_seeds"]
    }
    if set(runs) != expected:
        raise RuntimeError(f"{key}: incomplete PointMaze v1 registered block")
    identities = {run["data_identity_sha256"] for run in runs.values()}
    if len(identities) != 1:
        raise RuntimeError(f"{key}: PointMaze v1 data identity drifted")
    return {
        "model": spec["model"],
        "registered_seeds": spec["registered_seeds"],
        "data_identity_sha256": identities.pop(),
        "runs": runs,
    }


def _attach_semantic(
    model: dict[str, Any], input_sha256: dict[str, dict[str, Any]]
) -> None:
    payload = json.loads(SEMANTIC_LEDGER.read_text(encoding="utf-8"))
    _validate_ledger(payload, expected_arms={"semantic"})
    if payload.get("seeds") != model["registered_seeds"]:
        raise RuntimeError("PointMaze semantic seed block drifted")
    if payload.get("data_identity_sha256") != model["data_identity_sha256"]:
        raise RuntimeError("PointMaze semantic data identity drifted")
    input_sha256[_relative(SEMANTIC_LEDGER)] = _source_record(SEMANTIC_LEDGER)
    for run in payload["runs"]:
        seed = int(run["seed"])
        key = f"replay_semantic_maxent/{seed}"
        if key in model["runs"]:
            raise RuntimeError(f"duplicate PointMaze semantic run {seed}")
        model["runs"][key] = _load_run(
            run,
            method="replay_semantic_maxent",
            input_sha256=input_sha256,
        )


def _method_runs(model: dict[str, Any], method: str) -> list[dict[str, Any]]:
    return sorted(
        (
            run
            for run in model["runs"].values()
            if run["method"] == method
        ),
        key=lambda run: run["seed"],
    )


def _counts_by_pass(runs: list[dict[str, Any]]) -> dict[str, int]:
    checkpoints = sorted(
        {float(checkpoint) for run in runs for checkpoint in run["curve"]}
    )
    return {
        f"{checkpoint:.1f}": sum(
            f"{checkpoint:.1f}" in run["curve"] for run in runs
        )
        for checkpoint in checkpoints
    }


def _terminal_endpoint(run: dict[str, Any]) -> dict[str, float]:
    if not run["terminal"]:
        raise RuntimeError("requested terminal endpoint from a prefix")
    return run["curve"][f"{PASSES:.1f}"]


def _trapezoid_effect(
    treatment: dict[str, Any], control: dict[str, Any], metric: str
) -> float:
    checkpoints = [index / 2 for index in range(2 * PASSES + 1)]
    effects = []
    for checkpoint in checkpoints:
        key = f"{checkpoint:.1f}"
        effects.append(
            treatment["curve"][key][metric] - control["curve"][key][metric]
        )
    area = sum(
        (right - left) * (effects[index] + effects[index + 1]) / 2
        for index, (left, right) in enumerate(zip(checkpoints, checkpoints[1:]))
    )
    return area / PASSES


def _summary(per_seed: dict[str, float]) -> dict[str, Any]:
    values = list(per_seed.values())
    mean = statistics.fmean(values)
    half_width = T_CRIT_DF4 * statistics.stdev(values) / math.sqrt(5)
    return {
        "mean": mean,
        "student_t_95": [mean - half_width, mean + half_width],
        "range": [min(values), max(values)],
    }


def _comparison_record(
    spec: dict[str, str], model: dict[str, Any]
) -> dict[str, Any]:
    treatment = {
        run["seed"]: run
        for run in _method_runs(model, spec["treatment"])
        if run["terminal"]
    }
    control = {
        run["seed"]: run
        for run in _method_runs(model, spec["control"])
        if run["terminal"]
    }
    seeds = sorted(set(treatment) & set(control))
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        treatment_endpoint = _terminal_endpoint(treatment[seed])
        control_endpoint = _terminal_endpoint(control[seed])
        effects = {
            metric: treatment_endpoint[metric] - control_endpoint[metric]
            for metric in METRICS
        }
        auc = {
            f"normalized_auc_{metric}": _trapezoid_effect(
                treatment[seed], control[seed], metric
            )
            for metric in METRICS
        }
        per_seed[str(seed)] = {
            "control": {metric: control_endpoint[metric] for metric in METRICS},
            "treatment": {
                metric: treatment_endpoint[metric] for metric in METRICS
            },
            "effects": effects,
            "normalized_auc_effects": auc,
        }
    record: dict[str, Any] = {
        "key": spec["key"],
        "label": spec["label"],
        "model": spec["model"],
        "treatment": spec["treatment"],
        "control": spec["control"],
        "n": len(seeds),
        "seeds": seeds,
        "evidence": (
            "balanced_five_seed_terminal"
            if len(seeds) == 5
            else "exact_terminal_prefix"
        ),
        "per_seed": per_seed,
    }
    if len(seeds) == 5:
        record["summaries"] = {
            metric: _summary(
                {str(seed): per_seed[str(seed)]["effects"][metric] for seed in seeds}
            )
            for metric in METRICS
        }
        record["normalized_auc_summaries"] = {
            f"normalized_auc_{metric}": _summary(
                {
                    str(seed): per_seed[str(seed)]["normalized_auc_effects"][
                        f"normalized_auc_{metric}"
                    ]
                    for seed in seeds
                }
            )
            for metric in METRICS
        }
    return record


def build() -> dict[str, Any]:
    input_sha256: dict[str, dict[str, Any]] = {}
    models = {
        key: _load_primary(key, spec, input_sha256)
        for key, spec in COHORTS.items()
    }
    if (
        models["qwen05b"]["data_identity_sha256"]
        != models["falcon1b"]["data_identity_sha256"]
    ):
        raise RuntimeError("PointMaze model cohorts do not use identical data")
    _attach_semantic(models["qwen05b"], input_sha256)

    model_records = []
    for key in ("qwen05b", "falcon1b"):
        model = models[key]
        methods = [
            method
            for method in METHOD_ORDER
            if _method_runs(model, method)
        ]
        model_records.append(
            {
                "key": key,
                "model": model["model"],
                "registered_seeds": model["registered_seeds"],
                "methods": {
                    method: {
                        "terminal_seeds": [
                            run["seed"]
                            for run in _method_runs(model, method)
                            if run["terminal"]
                        ],
                        "deepest_pass_by_seed": {
                            str(run["seed"]): run["deepest_pass"]
                            for run in _method_runs(model, method)
                        },
                        "n_by_pass": _counts_by_pass(_method_runs(model, method)),
                        "runs": {
                            str(run["seed"]): run for run in _method_runs(model, method)
                        },
                    }
                    for method in methods
                },
            }
        )
    comparisons = [
        _comparison_record(spec, models[spec["model"]])
        for spec in COMPARISONS
    ]
    v1_models = {
        key: _load_v1_primary(key, spec, input_sha256)
        for key, spec in V1_COHORTS.items()
    }
    if (
        v1_models["qwen05b"]["data_identity_sha256"]
        != v1_models["falcon1b"]["data_identity_sha256"]
    ):
        raise RuntimeError("PointMaze v1 model cohorts do not use identical data")
    v1_model_records = []
    for key in ("qwen05b", "falcon1b"):
        model = v1_models[key]
        v1_model_records.append(
            {
                "key": key,
                "model": model["model"],
                "registered_seeds": model["registered_seeds"],
                "methods": {
                    method: {
                        "terminal_seeds": [
                            run["seed"] for run in _method_runs(model, method)
                        ],
                        "deepest_pass_by_seed": {
                            str(run["seed"]): run["deepest_pass"]
                            for run in _method_runs(model, method)
                        },
                        "n_by_pass": _counts_by_pass(_method_runs(model, method)),
                        "runs": {
                            str(run["seed"]): run
                            for run in _method_runs(model, method)
                        },
                    }
                    for method in ("drgrpo", "replay_grpo")
                },
            }
        )
    v1_comparisons = [
        _comparison_record(spec, v1_models[spec["model"]])
        for spec in V1_COMPARISONS
    ]
    return {
        "schema": "paper-interactive-domain-boundary-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "complete PointMaze v1 Qwen/Falcon null blocks; complete five-seed "
            "Qwen Tour replay estimand; exact Falcon Tour and semantic prefixes"
        ),
        "domain": "point_maze_tour",
        "domain_label": "PointMaze Tour",
        "stratum": "interactive; never pooled with the five static domains",
        "design": {
            "passes": PASSES,
            "target_steps": TARGET_STEPS,
            "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
            "train_rows": TRAIN_ROWS,
            "evaluation_rows": EVAL_ROWS,
            "evaluation_split": "sealed eval",
            "samples_per_prompt": 8,
            "sequential_decisions_per_episode": 4,
            "data_identity_sha256": models["qwen05b"]["data_identity_sha256"],
        },
        "trajectory_metrics": {
            "mean8": "mean validator correctness across eight sampled tours",
            "adjusted_breadth8": "distinct@8 minus pass@8",
        },
        "endpoint_metrics": ["pass8", "adjusted_breadth8"],
        "models": model_records,
        "comparisons": comparisons,
        "point_maze_v1": {
            "domain": "point_maze",
            "domain_label": "PointMaze v1 waypoint interface",
            "relationship_to_tour": (
                "completed predecessor instrument check; separate estimand "
                "from the redesigned PointMaze Tour interface"
            ),
            "design": {
                "passes": PASSES,
                "target_steps": TARGET_STEPS,
                "checkpoint_interval_steps": CHECKPOINT_INTERVAL,
                "train_rows": TRAIN_ROWS,
                "evaluation_rows": EVAL_ROWS,
                "evaluation_split": "sealed eval",
                "samples_per_prompt": 8,
                "maximum_sequential_decisions": 64,
                "data_identity_sha256": v1_models["qwen05b"][
                    "data_identity_sha256"
                ],
            },
            "models": v1_model_records,
            "comparisons": v1_comparisons,
            "aggregation_rule": (
                "both model blocks are complete n=5 same-seed comparisons; "
                "raw effects, means, and paired Student-t intervals are shown"
            ),
        },
        "aggregation_rule": (
            "trajectory means use every available exact seed at each registered "
            "checkpoint; endpoint means and paired 95% Student-t intervals are "
            "emitted only for a complete n=5 terminal pair"
        ),
        "selection_rule": (
            "all registered Tour half-pass checkpoints, every Tour terminal "
            "seed, and both complete v1 model blocks; no checkpoint selection, "
            "imputation, cross-interface, cross-model, or static-domain pooling"
        ),
        "boundary_note": (
            "PointMaze v1 is a separate structurally inert instrument check; "
            "it is displayed beside but never combined with the redesigned Tour"
        ),
        "input_sha256": dict(sorted(input_sha256.items())),
    }


def _plot_trajectory(
    axis: Any,
    model: dict[str, Any],
    metric: str,
    *,
    title: str | None = None,
) -> None:
    style.style_axis(axis, grid="both", title=title)
    axis.set_xlim(0, PASSES)
    axis.set_xticks((0, 2, 4, 6, 8))
    axis.set_xlabel("training pass", fontsize=style.LABEL_FONT)
    for method, record in model["methods"].items():
        visual = METHOD_STYLE[method]
        runs = record["runs"]
        for seed_record in runs.values():
            x = [float(value) for value in seed_record["curve"]]
            y = [seed_record["curve"][f"{value:.1f}"][metric] for value in x]
            axis.plot(
                x,
                y,
                color=visual["color"],
                linestyle=visual["linestyle"],
                linewidth=style.SEED_LW,
                alpha=0.30,
                zorder=2,
            )
        checkpoints = sorted(
            {float(checkpoint) for run in runs.values() for checkpoint in run["curve"]}
        )
        means = []
        for checkpoint in checkpoints:
            key = f"{checkpoint:.1f}"
            values = [
                run["curve"][key][metric]
                for run in runs.values()
                if key in run["curve"]
            ]
            means.append(statistics.fmean(values))
        axis.plot(
            checkpoints,
            means,
            color=visual["color"],
            linestyle=visual["linestyle"],
            linewidth=style.MEAN_LW,
            solid_capstyle="round",
            zorder=4,
        )
    if metric == "mean8":
        axis.set_ylim(0.25, 0.95)
    else:
        axis.set_ylim(1.2, 2.65)


def _comparison_by_key(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["key"]: record for record in payload["comparisons"]}


def _plot_effects(axis: Any, payload: dict[str, Any], metric: str) -> None:
    title = "terminal Δpass@8" if metric == "pass8" else r"terminal Δ($D-P$)"
    style.style_axis(axis, grid="both", title=title)
    axis.axvline(0.0, color=style.MUTED, linewidth=0.75, linestyle=(0, (2, 2)))
    y_positions = (2.0, 1.0, 0.0)
    comparison_records = _comparison_by_key(payload)
    plotted = [0.0]
    for spec, y in zip(COMPARISONS, y_positions):
        record = comparison_records[spec["key"]]
        values = [
            record["per_seed"][str(seed)]["effects"][metric]
            for seed in record["seeds"]
        ]
        plotted.extend(values)
        visual = METHOD_STYLE[record["treatment"]]
        offsets = {
            3: (-0.055, 0.0, 0.055),
            5: (-0.09, -0.045, 0.0, 0.045, 0.09),
        }[len(values)]
        axis.scatter(
            values,
            [y + offset for offset in offsets],
            s=13,
            facecolors=style.WHITE,
            edgecolors=visual["color"],
            linewidths=0.85,
            zorder=5,
        )
        if record["n"] == 5:
            summary = record["summaries"][metric]
            plotted.extend(summary["student_t_95"])
            axis.hlines(
                y,
                summary["student_t_95"][0],
                summary["student_t_95"][1],
                color=visual["color"],
                linewidth=1.3,
                zorder=4,
            )
            axis.scatter(
                [summary["mean"]],
                [y],
                marker="D",
                s=24,
                color=visual["color"],
                edgecolors=style.WHITE,
                linewidths=0.45,
                zorder=6,
            )
    margin = max(0.018, (max(plotted) - min(plotted)) * 0.16)
    axis.set_xlim(min(plotted) - margin, max(plotted) + margin)
    if _same(min(plotted), max(plotted)):
        axis.set_xlim(-0.03, 0.03)
    axis.set_ylim(-0.55, 2.55)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        [
            f"{spec['label']}\n$n={_comparison_by_key(payload)[spec['key']]['n']}$"
            for spec in COMPARISONS
        ],
        fontsize=style.SMALL_FONT,
    )
    axis.set_xlabel("treatment − matched reference", fontsize=style.SMALL_FONT)


def _plot_v1_effects(axis: Any, payload: dict[str, Any], metric: str) -> None:
    title = "v1 Δpass@8" if metric == "pass8" else r"v1 Δ($D-P$)"
    style.style_axis(axis, grid="both", title=title)
    axis.axvline(0.0, color=style.MUTED, linewidth=0.75, linestyle=(0, (2, 2)))
    records = {
        record["key"]: record
        for record in payload["point_maze_v1"]["comparisons"]
    }
    plotted = [0.0]
    y_positions = (1.0, 0.0)
    for spec, y in zip(V1_COMPARISONS, y_positions):
        record = records[spec["key"]]
        values = [
            record["per_seed"][str(seed)]["effects"][metric]
            for seed in record["seeds"]
        ]
        plotted.extend(values)
        summary = record["summaries"][metric]
        plotted.extend(summary["student_t_95"])
        offsets = (-0.09, -0.045, 0.0, 0.045, 0.09)
        axis.scatter(
            values,
            [y + offset for offset in offsets],
            s=12,
            facecolors=style.WHITE,
            edgecolors=style.METHOD,
            linewidths=0.8,
            zorder=5,
        )
        axis.hlines(
            y,
            summary["student_t_95"][0],
            summary["student_t_95"][1],
            color=style.METHOD,
            linewidth=1.25,
            zorder=4,
        )
        axis.scatter(
            [summary["mean"]],
            [y],
            marker="D",
            s=22,
            color=style.METHOD,
            edgecolors=style.WHITE,
            linewidths=0.45,
            zorder=6,
        )
    margin = max(0.018, (max(plotted) - min(plotted)) * 0.16)
    axis.set_xlim(min(plotted) - margin, max(plotted) + margin)
    axis.set_ylim(-0.48, 1.48)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        ["Qwen v1\n$n=5$", "Falcon v1\n$n=5$"],
        fontsize=style.SMALL_FONT,
    )
    axis.set_xlabel("Replay − matched Dr", fontsize=style.SMALL_FONT)


def render(payload: dict[str, Any], output: Path) -> None:
    style.apply_rcparams()
    figure = plt.figure(figsize=(style.WIDTH, 4.25))
    grid = figure.add_gridspec(
        2,
        4,
        left=0.064,
        right=0.995,
        top=0.84,
        bottom=0.18,
        hspace=0.34,
        wspace=0.48,
        width_ratios=(1.12, 1.12, 0.90, 0.78),
    )
    model_by_key = {record["key"]: record for record in payload["models"]}
    top_axes: dict[str, Any] = {}
    for column, key in enumerate(("qwen05b", "falcon1b")):
        model = model_by_key[key]
        for row, metric in enumerate(TRAJECTORY_METRICS):
            axis = figure.add_subplot(grid[row, column])
            if row == 0:
                top_axes[key] = axis
            _plot_trajectory(
                axis,
                model,
                metric,
                title=model["model"] if row == 0 else None,
            )
            if column == 0:
                axis.set_ylabel(
                    TRAJECTORY_LABEL[metric], fontsize=style.LABEL_FONT
                )
            else:
                axis.tick_params(labelleft=False)
        short_label = {
            "drgrpo": "Dr",
            "replay_grpo": "Replay",
            "replay_semantic_maxent": "+Sem",
        }
        terminal_counts = [
            f"{short_label[method]} {len(record['terminal_seeds'])}"
            for method, record in model["methods"].items()
        ]
        top_axis = top_axes[key]
        top_axis.text(
            0.02,
            0.04,
            "terminal $n$: " + " · ".join(terminal_counts),
            transform=top_axis.transAxes,
            fontsize=style.SMALL_FONT,
            color=style.MUTED,
            ha="left",
            va="bottom",
        )

    _plot_effects(figure.add_subplot(grid[0, 2]), payload, "pass8")
    _plot_effects(
        figure.add_subplot(grid[1, 2]), payload, "adjusted_breadth8"
    )
    _plot_v1_effects(figure.add_subplot(grid[0, 3]), payload, "pass8")
    _plot_v1_effects(
        figure.add_subplot(grid[1, 3]), payload, "adjusted_breadth8"
    )

    handles = []
    labels = []
    for method in METHOD_ORDER:
        visual = METHOD_STYLE[method]
        handles.append(
            Line2D(
                [0],
                [0],
                color=visual["color"],
                linestyle=visual["linestyle"],
                linewidth=1.45,
            )
        )
        labels.append(visual["label"])
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=style.WHITE,
                markeredgecolor=style.MUTED,
                markersize=3.6,
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="-",
                color=style.MUTED,
                markerfacecolor=style.MUTED,
                markersize=3.8,
                linewidth=1.1,
            ),
        ]
    )
    labels.extend(("exact seed", "$n=5$ mean + 95% interval"))
    style.bottom_legend(figure, handles, labels, y=0.008, ncol=5)
    figure.suptitle(
        "Interactive-domain boundary: PointMaze v1 and Tour",
        fontsize=style.TITLE_FONT,
        color=style.INK,
        y=0.985,
    )
    figure.text(
        0.5,
        0.945,
        (
            "Separate interfaces and model families are never pooled; Tour "
            "prefixes remain exact-only."
        ),
        ha="center",
        va="top",
        fontsize=style.SMALL_FONT,
        color=style.MUTED,
    )
    style.save(figure, output, png=True, dpi=240)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    payload = build()
    render(payload, output)
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    for suffix in ("pdf", "png", "json"):
        print(f"wrote {output.with_suffix('.' + suffix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
