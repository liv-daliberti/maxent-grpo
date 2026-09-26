#!/usr/bin/env python3
"""Build the complete frozen E112-R1 paired result, or write nothing."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_e105_group_centered_semantic_repair_results as e105  # noqa: E402


# Reuse one endpoint implementation. E112 adds ledger validation and decisions.
DOMAINS = e105.DOMAINS
SCALE_SEEDS = e105.SCALE_SEEDS
EXPECTED_DRAWS = e105.EXPECTED_DRAWS
TERMINAL_METRICS = e105.TERMINAL_METRICS
AUC_METRICS = e105.AUC_METRICS
run_curve = e105.run_curve
run_curve_with_draws = e105.run_curve_with_draws
sampled_prompt_surface = e105.sampled_prompt_surface
registered_steps = e105.registered_steps
normalized_auc = e105.normalized_auc
paired_summary = e105.paired_summary
sha256 = e105.sha256
SAMPLED_METRICS = (
    "sampled_pass8",
    "sampled_mean8",
    "sampled_distinct8",
    "sampled_excess8",
)

LEDGER = ROOT / (
    "var/artifacts/" "e112r1_verified_support_discovery_full_three_scale_jobs.json"
)
OUTPUT = ROOT / ("paper/results/e112r1_verified_support_discovery_three_scale.json")
ANALYSIS_PROTOCOL = ROOT / (
    "paper/preregistration/e112_paired_analysis_specification_20260818.md"
)
IMPLEMENTATION_FREEZE = ROOT / (
    "paper/preregistration/" "e112r1_final_analysis_implementation_freeze_20260825.md"
)
ANALYSIS_PLOTTER = ROOT / (
    "ops/exp_scaling/" "plot_e112r1_verified_support_discovery_effects.py"
)
SHARED_ENDPOINT_BUILDER = ROOT / (
    "ops/exp_scaling/build_e105_group_centered_semantic_repair_results.py"
)
ROW_CONTRACT_AMENDMENT = ROOT / (
    "paper/preregistration/"
    "e112r1_final_analysis_a1_sampled_row_contract_closure_20260825.md"
)
GREEDY_CONTRACT_AMENDMENT = ROOT / (
    "paper/preregistration/"
    "e112r1_final_analysis_a2_greedy_trace_contract_closure_20260825.md"
)
PROMPT_SURFACE_AMENDMENT = ROOT / (
    "paper/preregistration/"
    "e112r1_final_analysis_a3_paired_prompt_surface_closure_20260825.md"
)
DISTINCT_DRAWS_AMENDMENT = ROOT / (
    "paper/preregistration/"
    "e112r1_final_analysis_a4_distinct_request_draws_20260825.md"
)
PAIRED_DRAW_DIAGNOSTIC_AMENDMENT = ROOT / (
    "paper/preregistration/"
    "e112r1_final_analysis_a5_paired_draw_diagnostic_20260825.md"
)
PROMPT_ESTIMAND_SCOPE_AMENDMENT = ROOT / (
    "paper/preregistration/"
    "e112r1_final_analysis_a6_prompt_estimand_scope_20260825.md"
)
LAUNCHER = ROOT / (
    "ops/exp_scaling/" "launch_e112r1_verified_support_discovery_full_three_scale.py"
)
E112_R1_PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e112r1_sampler_contract_repair_and_e112_retirement_20260819.md"
)
E112_UNIT_EVIDENCE = ROOT / (
    "var/artifacts/e112r1_verified_support_discovery_unit_tests.json"
)
E111_LEDGER = ROOT / (
    "var/artifacts/e111_verified_support_discovery_mechanism_gate_jobs.json"
)
E111_AUDIT = ROOT / (
    "var/artifacts/e111_verified_support_discovery_mechanism_gate_audit.json"
)
QWEN3_PLACEMENT = ROOT / (
    "var/artifacts/e105_qwen3_paired_a6000_placement_amendment.json"
)
E112_SNAPSHOT = ROOT / (
    "var/artifacts/source_snapshots/e76_tuned_scale_d546e1d6a3428303"
)
E112_REGISTERED_LAUNCHER_SHA256 = (
    "a8b2d06369be9d20780520707153eb0dc54c8284355336a94c095c4aa986961d"
)
E112_RETIREMENT = ROOT / ("var/artifacts/e112_sampler_contract_failure_retirement.json")
COMPARATOR_LEDGERS = {
    "qwen05b": ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json",
    "falcon1b": ROOT / "var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json",
    "qwen3b": ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json",
}
REPAIRED_PYTHON_LEDGER = ROOT / (
    "var/artifacts/e109_repaired_python_replay_comparators_jobs.json"
)
REPAIRED_PYTHON_PROTOCOL = ROOT / (
    "paper/preregistration/e109_repaired_python_replay_comparators_20260817.md"
)
E109_REGISTERED_LAUNCHER_SHA256 = (
    "58a4ea73fe2aab428bb4688572241f857ea55264e5e64974713fc7d53b3de6e0"
)
E109_CONTINUATION = ROOT / ("var/artifacts/e109r1_qwen3_python_continuation_jobs.json")
PYTHON_SURFACE = "python-factor-response-v2-latex-lambda"
EVAL_SEED_BASES = {
    "graph_coloring": 610100,
    "countdown": 610200,
    "python_factors": 610300,
    "mathir": 610400,
    "pantry_plan": 76299,
}

Cell = tuple[str, str, int]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def _expected_cells() -> set[Cell]:
    return {
        (scale, domain, seed)
        for scale, seeds in SCALE_SEEDS.items()
        for domain in DOMAINS
        for seed in seeds
    }


def _index_runs(
    rows: Any, *, where: str, scale_override: str | None = None
) -> dict[Cell, dict[str, Any]]:
    if not isinstance(rows, list):
        raise RuntimeError(f"{where}: runs must be a list")
    index: dict[Cell, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"{where}: run {position} is not an object")
        try:
            scale = scale_override or str(row["scale"])
            key = (scale, str(row["domain"]), int(row["seed"]))
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{where}: malformed run {position}") from error
        if key in index:
            raise RuntimeError(f"{where}: duplicate cell {key}")
        index[key] = row
    return index


def _validate_bound_file(
    payload: Mapping[str, Any], *, field: str, expected: Path, where: str
) -> None:
    recorded = Path(str(payload.get(field, ""))).resolve()
    if recorded != expected.resolve():
        raise RuntimeError(f"{where}: {field} path drifted")
    if not expected.is_file() or payload.get(f"{field}_sha256") != sha256(expected):
        raise RuntimeError(f"{where}: {field} digest drifted")


def _validate_treatment_provenance(treatment: Mapping[str, Any]) -> None:
    if treatment.get("launcher_sha256") != E112_REGISTERED_LAUNCHER_SHA256:
        raise RuntimeError("E112-R1 registered launcher digest drifted")
    bound_files = {
        "protocol": E112_R1_PROTOCOL,
        "unit_evidence": E112_UNIT_EVIDENCE,
        "e111_ledger": E111_LEDGER,
        "e111_audit": E111_AUDIT,
        "qwen3_paired_placement_artifact": QWEN3_PLACEMENT,
    }
    for field, path in bound_files.items():
        _validate_bound_file(
            treatment,
            field=field,
            expected=path,
            where="E112-R1",
        )
    if treatment.get("original_e112_retirement_sha256") != sha256(E112_RETIREMENT):
        raise RuntimeError("E112-R1 retirement provenance drifted")
    if (
        Path(str(treatment.get("snapshot_root", ""))).resolve()
        != E112_SNAPSHOT.resolve()
    ):
        raise RuntimeError("E112-R1 source snapshot path drifted")
    identity = E112_SNAPSHOT / "SNAPSHOT_IDENTITY.json"
    if not identity.is_file() or treatment.get("snapshot_identity_sha256") != sha256(
        identity
    ):
        raise RuntimeError("E112-R1 source snapshot identity drifted")


def _validate_continuation(
    continuation: Mapping[str, Any], repaired_index: Mapping[Cell, Mapping[str, Any]]
) -> None:
    required = {
        "schema": "e109r1_qwen3_python_continuation_jobs_v1",
        "released": True,
        "pointmaze": "excluded",
        "same_a6000_hardware_class": True,
        "same_run_directories": True,
        "same_scientific_cells": True,
        "scientific_environment_changed": False,
        "outcomes_inspected": False,
        "exact_seeds": [73, 74],
        "exact_original_job_ids": [30659554, 30659555],
    }
    drifted = [key for key, value in required.items() if continuation.get(key) != value]
    if drifted:
        raise RuntimeError(f"E109 continuation provenance drifted: {drifted}")
    if Path(
        str(continuation.get("original_ledger", ""))
    ).resolve() != REPAIRED_PYTHON_LEDGER.resolve() or continuation.get(
        "original_ledger_sha256"
    ) != sha256(
        REPAIRED_PYTHON_LEDGER
    ):
        raise RuntimeError("E109 continuation does not bind the registered ledger")
    for field in ("protocol", "launcher", "routing_amendment"):
        path = Path(str(continuation.get(field, "")))
        _validate_bound_file(
            continuation, field=field, expected=path, where="E109 continuation"
        )
    records = continuation.get("records")
    if not isinstance(records, list) or len(records) != 2:
        raise RuntimeError("E109 continuation must contain exactly two records")
    seen: set[int] = set()
    continuation_jobs: set[int] = set()
    for row in records:
        if not isinstance(row, dict):
            raise RuntimeError("E109 continuation record is malformed")
        seed = int(row.get("seed", -1))
        key = ("qwen3b", "python_factors", seed)
        registered = repaired_index.get(key)
        if registered is None:
            raise RuntimeError(f"E109 continuation names an unregistered seed {seed}")
        expected = {
            "run_stamp": registered["run_stamp"],
            "run_dir": registered["run_dir"],
            "original_job_id": int(registered["job_id"]),
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise RuntimeError(f"E109 continuation binding drifted for seed {seed}")
        continuation_job = row.get("continuation_job_id")
        if not isinstance(continuation_job, int) or isinstance(continuation_job, bool):
            raise RuntimeError("E109 continuation job ID is malformed")
        seen.add(seed)
        continuation_jobs.add(continuation_job)
    if seen != {73, 74} or len(continuation_jobs) != 2:
        raise RuntimeError("E109 continuation cells are not the exact two repairs")


def validate_ledgers(
    treatment: dict[str, Any],
    comparators: Mapping[str, dict[str, Any]],
    repaired_python: dict[str, Any],
    continuation: dict[str, Any],
) -> tuple[dict[Cell, dict[str, Any]], dict[Cell, dict[str, Any]]]:
    """Validate exact grids and bindings without reading outcome files."""
    required = {
        "schema": "e112r1_verified_support_discovery_full_three_scale_jobs_v1",
        "released": True,
        "pointmaze": "excluded",
        "models": list(SCALE_SEEDS),
        "domains": list(DOMAINS),
        "arms": ["verified_support_discovery"],
        "seeds": {scale: list(seeds) for scale, seeds in SCALE_SEEDS.items()},
        "train_rows": 384,
        "passes": 8,
        "target_steps": 3_072,
        "checkpoint_interval_steps": 192,
        "semantic_coefficient": 0.1,
        "replay_weight": 0.1,
        "outcomes_inspected_for_release": False,
    }
    drifted = [key for key, value in required.items() if treatment.get(key) != value]
    if drifted:
        raise RuntimeError(f"E112-R1 design drifted: {drifted}")
    if int(treatment["train_rows"]) * int(treatment["passes"]) != int(
        treatment["target_steps"]
    ):
        raise RuntimeError("E112-R1 horizon is not train rows times passes")
    _validate_treatment_provenance(treatment)

    expected_ledger_provenance = {
        scale: {"path": str(path), "sha256": sha256(path)}
        for scale, path in COMPARATOR_LEDGERS.items()
    }
    if treatment.get("comparator_ledgers") != expected_ledger_provenance:
        raise RuntimeError("E112-R1 historical comparator provenance drifted")
    if Path(
        str(treatment.get("repaired_python_comparator_ledger", ""))
    ).resolve() != REPAIRED_PYTHON_LEDGER.resolve() or treatment.get(
        "repaired_python_comparator_ledger_sha256"
    ) != sha256(
        REPAIRED_PYTHON_LEDGER
    ):
        raise RuntimeError("E112-R1 repaired Python provenance drifted")

    treatment_index = _index_runs(treatment.get("runs"), where="E112-R1")
    expected_cells = _expected_cells()
    if set(treatment_index) != expected_cells or len(treatment_index) != 75:
        raise RuntimeError("E112-R1 does not contain the exact 75-cell grid")
    for key, row in treatment_index.items():
        if row.get("arm") != "verified_support_discovery":
            raise RuntimeError(f"E112-R1 arm drifted: {key}")
        record = str(row.get("held_scheduler_record", ""))
        required_record = (
            "OAT_ZERO_EVAL_PROMPT_INTERVAL=192",
            "OAT_ZERO_EVAL_MODE_COVERAGE_K=8",
            "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4",
            "OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1.0",
            f"OAT_ZERO_EVAL_MODE_COVERAGE_SEED={EVAL_SEED_BASES[key[1]]}",
            "OAT_ZERO_SEMANTIC_SHANNON_COEF=0.1",
        )
        if any(needle not in record for needle in required_record):
            raise RuntimeError(f"E112-R1 evaluation/objective record drifted: {key}")
        if key[1] == "python_factors" and (
            "python_factor_modebench_v1/train" not in record
            or "python_factor_modebench_v1/eval" not in record
        ):
            raise RuntimeError(f"E112-R1 Python data surface drifted: {key}")

    replay_index: dict[Cell, dict[str, Any]] = {}
    for scale, seeds in SCALE_SEEDS.items():
        payload = comparators.get(scale)
        if not isinstance(payload, dict):
            raise RuntimeError(f"missing {scale} historical comparator ledger")
        metadata = {
            "released": True,
            "domains": list(DOMAINS),
            "seeds": list(seeds),
            "train_rows": 384,
            "passes": 8,
            "target_steps": 3_072,
            "checkpoint_interval_steps": 192,
            "replay_weight": 0.1,
        }
        changed = [key for key, value in metadata.items() if payload.get(key) != value]
        if changed:
            raise RuntimeError(f"{scale} comparator metadata drifted: {changed}")
        all_runs = _index_runs(
            [row for row in payload.get("runs", []) if row.get("arm") == "replay"],
            where=f"{scale} replay",
            scale_override=scale,
        )
        expected_scale = {(scale, domain, seed) for domain in DOMAINS for seed in seeds}
        if set(all_runs) != expected_scale or len(all_runs) != 25:
            raise RuntimeError(f"{scale} lacks the exact 25 historical replay cells")
        replay_index.update(
            {key: row for key, row in all_runs.items() if key[1] != "python_factors"}
        )
    if len(replay_index) != 60:
        raise RuntimeError("historical comparator selection is not exactly 60 cells")

    repaired_required = {
        "schema": "e109_repaired_python_replay_comparators_jobs_v1",
        "released": True,
        "pointmaze": "excluded",
        "domain": "python_factors",
        "semantic_coefficient": 0.0,
        "replay_weight": 0.1,
        "train_rows": 384,
        "passes": 8,
        "target_steps": 3_072,
        "checkpoint_interval_steps": 192,
        "parser_surface_version": PYTHON_SURFACE,
        "post_e104_or_e106_update_outcomes_inspected": False,
        "qwen3_a6000_seeds": [73, 74],
        "scales": list(SCALE_SEEDS),
        "seeds": {scale: list(seeds) for scale, seeds in SCALE_SEEDS.items()},
    }
    changed = [
        key
        for key, value in repaired_required.items()
        if repaired_python.get(key) != value
    ]
    if changed:
        raise RuntimeError(f"E109 repaired Python design drifted: {changed}")
    _validate_bound_file(
        repaired_python,
        field="protocol",
        expected=REPAIRED_PYTHON_PROTOCOL,
        where="E109",
    )
    # The mutable root launcher has since acquired operational continuation
    # support. E112 binds the immutable E109 ledger byte-for-byte, so validate
    # the launcher digest frozen inside that ledger rather than today's file.
    if repaired_python.get("launcher_sha256") != E109_REGISTERED_LAUNCHER_SHA256:
        raise RuntimeError("E109 registered launcher digest drifted")
    if Path(
        str(repaired_python.get("qwen3_paired_placement_artifact", ""))
    ).resolve() != Path(
        str(treatment.get("qwen3_paired_placement_artifact", ""))
    ).resolve() or repaired_python.get(
        "qwen3_paired_placement_artifact_sha256"
    ) != treatment.get(
        "qwen3_paired_placement_artifact_sha256"
    ):
        raise RuntimeError("E109/E112 Qwen-3B placement provenance drifted")
    repaired_index = _index_runs(repaired_python.get("runs"), where="E109")
    expected_python = {
        (scale, "python_factors", seed)
        for scale, seeds in SCALE_SEEDS.items()
        for seed in seeds
    }
    if (
        set(repaired_index) != expected_python
        or len(repaired_index) != 15
        or any(row.get("arm") != "replay" for row in repaired_index.values())
    ):
        raise RuntimeError("E109 lacks the exact 15 repaired Python replay cells")
    _validate_continuation(continuation, repaired_index)
    replay_index.update(repaired_index)

    for key, treatment_row in treatment_index.items():
        replay = replay_index[key]
        expected_pair = {
            "run_stamp": str(replay["run_stamp"]),
            "run_dir": str(replay["run_dir"]),
            "job_id": int(replay["job_id"]),
        }
        if treatment_row.get("paired_replay") != expected_pair:
            raise RuntimeError(f"E112-R1 paired comparator binding drifted: {key}")
    return treatment_index, replay_index


def load_and_validate_ledgers() -> (
    tuple[
        dict[str, Any],
        dict[Cell, dict[str, Any]],
        dict[Cell, dict[str, Any]],
        dict[str, Any],
    ]
):
    treatment = _load_json(LEDGER)
    comparators = {
        scale: _load_json(path) for scale, path in COMPARATOR_LEDGERS.items()
    }
    repaired = _load_json(REPAIRED_PYTHON_LEDGER)
    continuation = _load_json(E109_CONTINUATION)
    treatment_index, replay_index = validate_ledgers(
        treatment, comparators, repaired, continuation
    )
    return treatment, treatment_index, replay_index, continuation


def decision_rules(
    terminal_effects: Mapping[Cell, Mapping[str, float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(terminal_effects) != _expected_cells() or len(terminal_effects) != 75:
        raise RuntimeError("decision rules require exactly 75 paired terminal effects")
    improved: list[str] = []
    all_pass: list[float] = []
    scale_results: dict[str, Any] = {}
    for scale, seeds in SCALE_SEEDS.items():
        scale_excess: list[float] = []
        scale_pass: list[float] = []
        for domain in DOMAINS:
            rows = [terminal_effects[(scale, domain, seed)] for seed in seeds]
            excess_mean = statistics.fmean(
                float(row["sampled_excess8"]) for row in rows
            )
            if excess_mean > 0.0:
                improved.append(f"{scale}/{domain}")
            scale_excess.extend(float(row["sampled_excess8"]) for row in rows)
            scale_pass.extend(float(row["sampled_pass8"]) for row in rows)
        excess_mean = statistics.fmean(scale_excess)
        pass_mean = statistics.fmean(scale_pass)
        scale_results[scale] = {
            "terminal_sampled_excess8_effect_mean": excess_mean,
            "terminal_sampled_pass8_effect_mean": pass_mean,
            "adjusted_breadth_positive": excess_mean > 0.0,
            "correctness_nonnegative": pass_mean >= 0.0,
            "scale_success": excess_mean > 0.0 and pass_mean >= 0.0,
            "cells": 25,
        }
        all_pass.extend(scale_pass)
    grand_pass = statistics.fmean(all_pass)
    general = {
        "breadth_metric": "five-seed mean terminal_sampled_excess8 effect > 0",
        "breadth_improved_families": len(improved),
        "breadth_improved_family_ids": improved,
        "breadth_required_families": 8,
        "correctness_metric": (
            "unweighted mean of all 75 terminal_sampled_pass8 paired effects"
        ),
        "grand_terminal_sampled_pass8_effect": grand_pass,
        "systematic_correctness_loss": grand_pass < 0.0,
        "successful_general_extension": len(improved) >= 8 and grand_pass >= 0.0,
    }
    all_scales = {
        "per_scale": scale_results,
        "successful_at_all_three_scales": all(
            row["scale_success"] for row in scale_results.values()
        ),
    }
    return general, all_scales


def paired_draw_summary(
    values: Mapping[int, Mapping[int, float]], *, seeds: Sequence[int]
) -> dict[str, Any]:
    """Keep evaluation-MC and training-seed variation visibly separate."""

    ordered_seeds = tuple(int(seed) for seed in seeds)
    expected_draws = tuple(range(EXPECTED_DRAWS))
    if set(values) != set(ordered_seeds) or len(ordered_seeds) != 5:
        raise RuntimeError("paired-draw summary requires the exact five seeds")
    per_seed: dict[str, Any] = {}
    seed_means: list[float] = []
    for seed in ordered_seeds:
        draw_values = values[seed]
        if set(draw_values) != set(expected_draws):
            raise RuntimeError(
                f"paired-draw summary seed {seed} lacks exact draws 0--3"
            )
        ordered_values = [
            e105.finite(draw_values[draw], where=f"seed {seed} draw {draw}")
            for draw in expected_draws
        ]
        estimate = statistics.fmean(ordered_values)
        seed_means.append(estimate)
        per_seed[str(seed)] = {
            "estimate": estimate,
            "evaluation_mc_se": statistics.stdev(ordered_values)
            / math.sqrt(EXPECTED_DRAWS),
            "draw_effects": {
                str(draw): ordered_values[draw] for draw in expected_draws
            },
        }
    draw_means = {
        str(draw): statistics.fmean(values[seed][draw] for seed in ordered_seeds)
        for draw in expected_draws
    }
    draw_mean_values = [draw_means[str(draw)] for draw in expected_draws]
    estimate = statistics.fmean(seed_means)
    if not math.isclose(estimate, statistics.fmean(draw_mean_values), abs_tol=1e-12):
        raise RuntimeError("paired-draw aggregation axes do not reconcile")
    return {
        "estimate": estimate,
        "n_training_seeds": len(ordered_seeds),
        "n_evaluation_draws": EXPECTED_DRAWS,
        "training_seed_se": statistics.stdev(seed_means)
        / math.sqrt(len(ordered_seeds)),
        "evaluation_mc_se": statistics.stdev(draw_mean_values)
        / math.sqrt(EXPECTED_DRAWS),
        "per_training_seed": per_seed,
        "draw_mean_effects": draw_means,
        "combined_interval": None,
        "p_value": None,
    }


def _curve_payload(
    row: Mapping[str, Any], *, domain: str, steps: Sequence[int], target: int
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, float]],
    dict[int, dict[int, dict[str, float]]],
]:
    sampled_contract = {
        "benchmark": "multi_answer",
        "sample_count": 8,
        "schema_version": 1,
        "seed_base": EVAL_SEED_BASES[domain],
        "temperature": 1.0,
    }
    greedy_contract = {
        "benchmark": "multi_answer",
        "draw_index": None,
        "sample_count": 1,
        "schema_version": 1,
        "seed": 0,
        "temperature": 0.0,
    }
    curve, draw_curves, sources = run_curve_with_draws(
        Path(str(row["run_dir"])),
        steps=steps,
        sampled_contract=sampled_contract,
        greedy_contract=greedy_contract,
    )
    prompt_surface = sampled_prompt_surface(
        Path(str(row["run_dir"])),
        steps=steps,
        draws=range(EXPECTED_DRAWS),
        sampled_contract=sampled_contract,
    )
    for step in steps:
        metrics = curve[int(step)]
        expected = metrics["sampled_distinct8"] - metrics["sampled_pass8"]
        if not math.isclose(metrics["sampled_excess8"], expected, abs_tol=1e-12):
            raise RuntimeError(f"{row['run_dir']}: adjusted breadth algebra drifted")
    auc = {
        metric: normalized_auc(curve, metric=metric, target=target)
        for metric in AUC_METRICS
    }
    return (
        {
            "run_stamp": row["run_stamp"],
            "run_dir": row["run_dir"],
            "registered_job_id": int(row["job_id"]),
            "terminal": curve[target],
            "normalized_auc": auc,
            "curve": {str(step): curve[int(step)] for step in steps},
            "sampled_draw_curves": {
                str(draw): {str(step): draw_curves[draw][int(step)] for step in steps}
                for draw in range(EXPECTED_DRAWS)
            },
            "evaluation_identity": prompt_surface,
            "source_sha256": sources,
        },
        curve,
        draw_curves,
    )


def materialize_results(
    treatment_index: Mapping[Cell, Mapping[str, Any]],
    replay_index: Mapping[Cell, Mapping[str, Any]],
    *,
    steps: Sequence[int],
    target: int,
    provenance: Mapping[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    if (
        set(treatment_index) != _expected_cells()
        or set(replay_index) != _expected_cells()
    ):
        raise RuntimeError("result materialization requires exact paired 75-cell grids")
    if tuple(int(step) for step in steps)[-1] != target:
        raise RuntimeError("registered steps do not end at the target")
    output: dict[str, Any] = {
        "schema": "e112r1_verified_support_discovery_results_v1",
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "pointmaze": "excluded",
        "analysis_provenance": dict(provenance),
        "estimand_scope": {
            "kind": "bundled historical-comparator contrast",
            "contrast": (
                "E112-R1 verified-support-discovery request path and source "
                "snapshot minus registered historical Re:Dr"
            ),
            "isolated_semantic_v7_effect": False,
            "confirmatory_blind": False,
            "prompt_target": "registered_finite_evaluation_bank_within_domain",
            "prompt_population_inference": False,
            "prompt_population_se": None,
            "reason": (
                "request/source provenance differs and private interim outcomes "
                "were inspected before complete-cohort materialization"
            ),
        },
        "metric_contract": {
            "primitive_endpoint_vector": ["sampled_pass8", "sampled_distinct8"],
            "derived_correctness_adjusted_breadth": (
                "sampled_excess8 = sampled_distinct8 - sampled_pass8"
            ),
            "registered_primary_decision_metric": "terminal_sampled_excess8",
            "secondary_diagnostics_change_registered_decisions": False,
            "sampled_evaluation_rows": {
                "benchmark": "multi_answer",
                "sample_count": 8,
                "schema_version": 1,
                "temperature": 1.0,
                "seed_base_by_domain": dict(EVAL_SEED_BASES),
                "draw_seed_formula": "seed_base_by_domain + draw_index",
            },
            "greedy_evaluation_rows": {
                "benchmark": "multi_answer",
                "draw_index": None,
                "evaluation_kind": "deterministic_greedy_trace_neutral",
                "sample_count": 1,
                "schema_version": 1,
                "seed": 0,
                "temperature": 0.0,
            },
            "paired_evaluation_identity": {
                "registered_row_seed_in_request_digest": True,
                "prompt_fields": [
                    "answer_mode_count",
                    "option_ids",
                    "prompt",
                    "prompt_index",
                    "reference",
                ],
                "request_fields": [
                    "option_ids",
                    "prompt_index",
                    "request_seeds_by_option",
                ],
                "response_fields_excluded": [
                    "answer_keys",
                    "metrics",
                    "responses",
                    "rewards",
                ],
                "require_exact_treatment_comparator_match": True,
                "require_constant_prompt_surface_across_grid": True,
                "require_constant_request_surface_within_draw": True,
                "require_distinct_request_surfaces_across_draws": True,
            },
        },
        "design": {
            "scales": list(SCALE_SEEDS),
            "domains": list(DOMAINS),
            "paired_seeds": {
                scale: list(seeds) for scale, seeds in SCALE_SEEDS.items()
            },
            "cells": 75,
            "families": 15,
            "evaluation_draws": EXPECTED_DRAWS,
            "registered_steps": [int(step) for step in steps],
            "target_steps": target,
        },
        "uncertainty": (
            "two-sided 95% Student-t interval over five paired training-seed "
            "effects, df=4; four evaluation draws are averaged, not treated as "
            "independent training seeds"
        ),
        "secondary_paired_draw_diagnostic": {
            "status": "exploratory disclosure after private interim inspection",
            "endpoint_basis": ["sampled_pass8", "sampled_distinct8"],
            "evaluation_mc_se": (
                "SE over four seed-averaged common-draw paired effects"
            ),
            "training_seed_se": (
                "SE over five draw-averaged paired training-seed effects"
            ),
            "combined_interval": None,
            "p_value": None,
            "prompt_population_se": None,
            "generation_draws_change_prompt_identity": False,
            "baseline_centered_formula": (
                "(E112(step)-E112(0)) - (Re:Dr(step)-Re:Dr(0))"
            ),
            "registered_decision_or_gate": False,
            "isolated_component_effect": False,
        },
        "families": {},
    }
    terminal_by_cell: dict[Cell, dict[str, float]] = {}
    for scale, seeds in SCALE_SEEDS.items():
        output["families"][scale] = {}
        for domain in DOMAINS:
            per_seed: dict[str, Any] = {}
            effects: dict[str, dict[int, float]] = defaultdict(dict)
            raw_draw_effects: dict[str, dict[int, dict[int, float]]] = defaultdict(
                lambda: defaultdict(dict)
            )
            centered_draw_effects: dict[str, dict[int, dict[int, float]]] = defaultdict(
                lambda: defaultdict(dict)
            )
            step_zero_draw_effects: dict[
                str, dict[int, dict[int, float]]
            ] = defaultdict(lambda: defaultdict(dict))
            for seed in seeds:
                key = (scale, domain, seed)
                (
                    treatment_payload,
                    treatment_curve,
                    treatment_draw_curves,
                ) = _curve_payload(
                    treatment_index[key], domain=domain, steps=steps, target=target
                )
                replay_payload, replay_curve, replay_draw_curves = _curve_payload(
                    replay_index[key], domain=domain, steps=steps, target=target
                )
                if (
                    treatment_payload["evaluation_identity"]
                    != replay_payload["evaluation_identity"]
                ):
                    raise RuntimeError(
                        "paired evaluation prompt/request surface drifted: " f"{key}"
                    )
                terminal = {
                    metric: treatment_curve[target][metric]
                    - replay_curve[target][metric]
                    for metric in TERMINAL_METRICS
                }
                treatment_auc = treatment_payload["normalized_auc"]
                replay_auc = replay_payload["normalized_auc"]
                auc = {
                    metric: treatment_auc[metric] - replay_auc[metric]
                    for metric in AUC_METRICS
                }
                raw_terminal_by_draw: dict[str, dict[str, float]] = {}
                raw_auc_by_draw: dict[str, dict[str, float]] = {}
                step_zero_by_draw: dict[str, dict[str, float]] = {}
                centered_terminal_by_draw: dict[str, dict[str, float]] = {}
                centered_auc_by_draw: dict[str, dict[str, float]] = {}
                for draw in range(EXPECTED_DRAWS):
                    raw_terminal_by_draw[str(draw)] = {}
                    raw_auc_by_draw[str(draw)] = {}
                    step_zero_by_draw[str(draw)] = {}
                    centered_terminal_by_draw[str(draw)] = {}
                    centered_auc_by_draw[str(draw)] = {}
                    for metric in SAMPLED_METRICS:
                        raw_terminal = (
                            treatment_draw_curves[draw][target][metric]
                            - replay_draw_curves[draw][target][metric]
                        )
                        raw_auc = normalized_auc(
                            treatment_draw_curves[draw], metric=metric, target=target
                        ) - normalized_auc(
                            replay_draw_curves[draw], metric=metric, target=target
                        )
                        step_zero = (
                            treatment_draw_curves[draw][0][metric]
                            - replay_draw_curves[draw][0][metric]
                        )
                        centered_terminal = raw_terminal - step_zero
                        centered_auc = raw_auc - step_zero
                        raw_terminal_by_draw[str(draw)][metric] = raw_terminal
                        raw_auc_by_draw[str(draw)][metric] = raw_auc
                        step_zero_by_draw[str(draw)][metric] = step_zero
                        centered_terminal_by_draw[str(draw)][metric] = centered_terminal
                        centered_auc_by_draw[str(draw)][metric] = centered_auc
                        raw_draw_effects[f"terminal_{metric}"][seed][
                            draw
                        ] = raw_terminal
                        raw_draw_effects[f"auc_{metric}"][seed][draw] = raw_auc
                        centered_draw_effects[f"terminal_{metric}"][seed][
                            draw
                        ] = centered_terminal
                        centered_draw_effects[f"auc_{metric}"][seed][
                            draw
                        ] = centered_auc
                        step_zero_draw_effects[f"step_zero_{metric}"][seed][
                            draw
                        ] = step_zero
                for metric in SAMPLED_METRICS:
                    if not math.isclose(
                        statistics.fmean(
                            raw_terminal_by_draw[str(draw)][metric]
                            for draw in range(EXPECTED_DRAWS)
                        ),
                        terminal[metric],
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError(
                            f"{key}: terminal paired-draw effects do not reconcile"
                        )
                    if not math.isclose(
                        statistics.fmean(
                            raw_auc_by_draw[str(draw)][metric]
                            for draw in range(EXPECTED_DRAWS)
                        ),
                        auc[metric],
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError(
                            f"{key}: AUC paired-draw effects do not reconcile"
                        )
                terminal_by_cell[key] = terminal
                for metric, value in terminal.items():
                    effects[f"terminal_{metric}"][seed] = value
                for metric, value in auc.items():
                    effects[f"auc_{metric}"][seed] = value
                per_seed[str(seed)] = {
                    "treatment": treatment_payload,
                    "replay": replay_payload,
                    "paired_effects": {"terminal": terminal, "normalized_auc": auc},
                    "paired_draw_diagnostics": {
                        "historical_contrast": {
                            "terminal": raw_terminal_by_draw,
                            "normalized_auc": raw_auc_by_draw,
                        },
                        "step_zero_historical_offset": step_zero_by_draw,
                        "baseline_centered_sensitivity": {
                            "terminal": centered_terminal_by_draw,
                            "normalized_auc": centered_auc_by_draw,
                        },
                    },
                }
            output["families"][scale][domain] = {
                "paired_seeds": list(seeds),
                "per_seed": per_seed,
                "paired_effects": {
                    metric: paired_summary(values, seeds=seeds)
                    for metric, values in effects.items()
                },
                "paired_draw_diagnostics": {
                    "historical_contrast": {
                        metric: paired_draw_summary(values, seeds=seeds)
                        for metric, values in raw_draw_effects.items()
                    },
                    "step_zero_historical_offset": {
                        metric: paired_draw_summary(values, seeds=seeds)
                        for metric, values in step_zero_draw_effects.items()
                    },
                    "baseline_centered_sensitivity": {
                        metric: paired_draw_summary(values, seeds=seeds)
                        for metric, values in centered_draw_effects.items()
                    },
                },
            }
    general, all_scales = decision_rules(terminal_by_cell)
    output["registered_general_extension_criterion"] = general
    output["registered_all_three_scales_criterion"] = all_scales
    return output


def atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Create the immutable official output atomically after full validation."""
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing official result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_and_write(
    treatment_index: Mapping[Cell, Mapping[str, Any]],
    replay_index: Mapping[Cell, Mapping[str, Any]],
    *,
    steps: Sequence[int],
    target: int,
    provenance: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    result = materialize_results(
        treatment_index,
        replay_index,
        steps=steps,
        target=target,
        provenance=provenance,
    )
    atomic_write_new(output_path, result)
    return result


def main() -> None:
    treatment, treatment_index, replay_index, continuation = load_and_validate_ledgers()
    target = int(treatment["target_steps"])
    steps = registered_steps(
        interval=int(treatment["checkpoint_interval_steps"]), target=target
    )
    provenance = {
        "treatment_ledger": str(LEDGER),
        "treatment_ledger_sha256": sha256(LEDGER),
        "analysis_protocol": str(ANALYSIS_PROTOCOL),
        "analysis_protocol_sha256": sha256(ANALYSIS_PROTOCOL),
        "implementation_freeze": str(IMPLEMENTATION_FREEZE),
        "implementation_freeze_sha256": sha256(IMPLEMENTATION_FREEZE),
        "analysis_builder": str(Path(__file__).resolve()),
        "analysis_builder_sha256": sha256(Path(__file__)),
        "shared_endpoint_builder": str(SHARED_ENDPOINT_BUILDER),
        "shared_endpoint_builder_sha256": sha256(SHARED_ENDPOINT_BUILDER),
        "sampled_row_contract_amendment": str(ROW_CONTRACT_AMENDMENT),
        "sampled_row_contract_amendment_sha256": sha256(ROW_CONTRACT_AMENDMENT),
        "greedy_trace_contract_amendment": str(GREEDY_CONTRACT_AMENDMENT),
        "greedy_trace_contract_amendment_sha256": sha256(GREEDY_CONTRACT_AMENDMENT),
        "paired_prompt_surface_amendment": str(PROMPT_SURFACE_AMENDMENT),
        "paired_prompt_surface_amendment_sha256": sha256(PROMPT_SURFACE_AMENDMENT),
        "distinct_request_draws_amendment": str(DISTINCT_DRAWS_AMENDMENT),
        "distinct_request_draws_amendment_sha256": sha256(DISTINCT_DRAWS_AMENDMENT),
        "paired_draw_diagnostic_amendment": str(PAIRED_DRAW_DIAGNOSTIC_AMENDMENT),
        "paired_draw_diagnostic_amendment_sha256": sha256(
            PAIRED_DRAW_DIAGNOSTIC_AMENDMENT
        ),
        "prompt_estimand_scope_amendment": str(PROMPT_ESTIMAND_SCOPE_AMENDMENT),
        "prompt_estimand_scope_amendment_sha256": sha256(
            PROMPT_ESTIMAND_SCOPE_AMENDMENT
        ),
        "analysis_plotter": str(ANALYSIS_PLOTTER),
        "analysis_plotter_sha256": sha256(ANALYSIS_PLOTTER),
        "comparator_ledgers": {
            scale: {"path": str(path), "sha256": sha256(path)}
            for scale, path in COMPARATOR_LEDGERS.items()
        },
        "repaired_python_ledger": str(REPAIRED_PYTHON_LEDGER),
        "repaired_python_ledger_sha256": sha256(REPAIRED_PYTHON_LEDGER),
        "e109_continuation": str(E109_CONTINUATION),
        "e109_continuation_sha256": sha256(E109_CONTINUATION),
        "e109_continuation_jobs": [
            int(row["continuation_job_id"]) for row in continuation["records"]
        ],
    }
    build_and_write(
        treatment_index,
        replay_index,
        steps=steps,
        target=target,
        provenance=provenance,
        output_path=OUTPUT,
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
