#!/usr/bin/env python3
"""Build the complete preregistered E105 paired result without selection."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / (
    "var/artifacts/" "e105_group_centered_semantic_repair_full_three_scale_jobs.json"
)
OUTPUT = ROOT / ("paper/results/e105_group_centered_semantic_repair_three_scale.json")
ANALYSIS_PROTOCOL = ROOT / (
    "paper/preregistration/e105_paired_analysis_specification_20260817.md"
)
ANALYSIS_PLOTTER = ROOT / (
    "ops/exp_scaling/plot_e105_group_centered_semantic_endpoint_effects.py"
)
QWEN3_PLACEMENT_PROTOCOL = ROOT / (
    "paper/preregistration/" "e105_qwen3_paired_a6000_placement_amendment_20260817.md"
)
QWEN3_PLACEMENT_SCRIPT = ROOT / (
    "ops/exp_scaling/apply_e105_qwen3_paired_a6000_placement_amendment.py"
)
QWEN3_PLACEMENT_ARTIFACT = ROOT / (
    "var/artifacts/e105_qwen3_paired_a6000_placement_amendment.json"
)
REPAIRED_PYTHON_COMPARATOR_PROTOCOL = ROOT / (
    "paper/preregistration/" "e109_repaired_python_replay_comparators_20260817.md"
)
REPAIRED_PYTHON_COMPARATOR_LAUNCHER = ROOT / (
    "ops/exp_scaling/launch_e109_repaired_python_replay_comparators.py"
)
REPAIRED_PYTHON_COMPARATOR_LEDGER = ROOT / (
    "var/artifacts/e109_repaired_python_replay_comparators_jobs.json"
)
COMPARATOR_LEDGERS = {
    "qwen05b": ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json",
    "falcon1b": ROOT / ("var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json"),
    "qwen3b": ROOT / ("var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json"),
}
SCALE_SEEDS = {
    "qwen05b": (43, 44, 45, 46, 47),
    "falcon1b": (55, 56, 57, 58, 59),
    "qwen3b": (70, 71, 72, 73, 74),
}
DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
EXPECTED_DRAWS = 4
T_CRITICAL_DF4 = 2.7764451051977987
SAMPLED_FIELDS = {
    "sampled_pass8": "any_correct_at_k",
    "sampled_mean8": "mean_at_k",
    "sampled_distinct8": "distinct_correct_modes_at_k",
}
TERMINAL_METRICS = (
    "greedy_pass1",
    "sampled_pass8",
    "sampled_mean8",
    "sampled_distinct8",
    "sampled_excess8",
)
AUC_METRICS = TERMINAL_METRICS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any, *, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"{where}: expected a finite number, got {value!r}")
    return float(value)


def registered_steps(*, interval: int, target: int) -> tuple[int, ...]:
    if interval <= 0 or target <= 0 or target % interval:
        raise RuntimeError("registered interval must exactly divide target")
    return tuple(range(0, target + 1, interval))


def _contract_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and float(actual) == expected
        )
    return actual == expected


def sampled_curve_with_draws(
    run_dir: Path,
    *,
    steps: Sequence[int],
    sampled_contract: Mapping[str, Any] | None = None,
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, dict[int, dict[str, float]]],
    dict[str, str],
]:
    required_contract = {
        "benchmark",
        "sample_count",
        "schema_version",
        "seed_base",
        "temperature",
    }
    if sampled_contract is not None and set(sampled_contract) != required_contract:
        raise RuntimeError(
            "sampled contract must contain exactly " f"{sorted(required_contract)}"
        )
    expected_steps = set(int(step) for step in steps)
    observations: dict[tuple[int, int], dict[str, float]] = {}
    sources: dict[str, str] = {}
    for path in sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        used = False
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, raw in enumerate(handle, start=1):
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                step = row.get("step")
                draw = row.get("draw_index")
                metrics = row.get("metrics")
                if (
                    row.get("evaluation_kind") != "fixed_seed_sampled_k_neutral"
                    or not isinstance(step, int)
                    or isinstance(step, bool)
                    or not isinstance(draw, int)
                    or isinstance(draw, bool)
                    or step not in expected_steps
                    or draw not in range(EXPECTED_DRAWS)
                    or not isinstance(metrics, dict)
                ):
                    continue
                if sampled_contract is not None:
                    expected_row = {
                        "benchmark": sampled_contract["benchmark"],
                        "sample_count": sampled_contract["sample_count"],
                        "schema_version": sampled_contract["schema_version"],
                        "seed": int(sampled_contract["seed_base"]) + draw,
                        "temperature": sampled_contract["temperature"],
                    }
                    drifted = [
                        field
                        for field, expected in expected_row.items()
                        if not _contract_value_matches(row.get(field), expected)
                    ]
                    if drifted:
                        raise RuntimeError(
                            f"{path}:{line_number}: sampled evaluation contract "
                            f"drifted: {drifted}"
                        )
                values = {
                    name: finite(
                        metrics.get(source),
                        where=f"{path}:{line_number}:{source}",
                    )
                    for name, source in SAMPLED_FIELDS.items()
                }
                key = (int(step), int(draw))
                if key in observations and observations[key] != values:
                    raise RuntimeError(
                        f"{run_dir}: conflicting duplicate sampled row {key}"
                    )
                observations[key] = values
                used = True
        if used:
            sources[str(path.resolve())] = sha256(path)

    curve: dict[int, dict[str, float]] = {}
    draw_curves: dict[int, dict[int, dict[str, float]]] = {
        draw: {} for draw in range(EXPECTED_DRAWS)
    }
    for step in steps:
        draws = [observations.get((int(step), draw)) for draw in range(4)]
        if any(row is None for row in draws):
            present = [draw for draw, row in enumerate(draws) if row is not None]
            raise RuntimeError(
                f"{run_dir}: sampled checkpoint {step} has draws {present}, "
                "expected 0--3"
            )
        complete = [row for row in draws if row is not None]
        values = {
            metric: statistics.fmean(row[metric] for row in complete)
            for metric in SAMPLED_FIELDS
        }
        values["sampled_excess8"] = (
            values["sampled_distinct8"] - values["sampled_pass8"]
        )
        curve[int(step)] = values
        for draw, row in enumerate(complete):
            draw_values = dict(row)
            draw_values["sampled_excess8"] = (
                draw_values["sampled_distinct8"] - draw_values["sampled_pass8"]
            )
            draw_curves[draw][int(step)] = draw_values
    return curve, draw_curves, sources


def sampled_curve(
    run_dir: Path,
    *,
    steps: Sequence[int],
    sampled_contract: Mapping[str, Any] | None = None,
) -> tuple[dict[int, dict[str, float]], dict[str, str]]:
    """Return the historical draw-averaged view used by existing analyses."""

    curve, _draw_curves, sources = sampled_curve_with_draws(
        run_dir,
        steps=steps,
        sampled_contract=sampled_contract,
    )
    return curve, sources


def evaluation_trace_contract(
    run_dir: Path,
    *,
    steps: Sequence[int],
    evaluation_kind: str,
    row_contract: Mapping[str, Any],
) -> dict[str, str]:
    """Require exact non-metric metadata for every registered trace step."""

    required_contract = {
        "benchmark",
        "draw_index",
        "sample_count",
        "schema_version",
        "seed",
        "temperature",
    }
    if set(row_contract) != required_contract:
        raise RuntimeError(
            "trace contract must contain exactly " f"{sorted(required_contract)}"
        )
    expected_steps = set(int(step) for step in steps)
    observed_steps: set[int] = set()
    sources: dict[str, str] = {}
    for path in sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        used = False
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, raw in enumerate(handle, start=1):
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if row.get("evaluation_kind") != evaluation_kind:
                    continue
                step = row.get("step")
                if (
                    not isinstance(step, int)
                    or isinstance(step, bool)
                    or step not in expected_steps
                ):
                    continue
                drifted = [
                    field
                    for field, expected in row_contract.items()
                    if not _contract_value_matches(row.get(field), expected)
                ]
                if drifted:
                    raise RuntimeError(
                        f"{path}:{line_number}: {evaluation_kind} contract "
                        f"drifted: {drifted}"
                    )
                observed_steps.add(step)
                used = True
        if used:
            sources[str(path.resolve())] = sha256(path)
    missing = expected_steps.difference(observed_steps)
    if missing:
        raise RuntimeError(
            f"{run_dir}: {evaluation_kind} missing registered steps "
            f"{sorted(missing)}"
        )
    return sources


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sampled_prompt_surface(
    run_dir: Path,
    *,
    steps: Sequence[int],
    draws: Sequence[int],
    sampled_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash response-free prompt and request identities on the exact grid."""

    expected_steps = set(int(step) for step in steps)
    expected_draws = set(int(draw) for draw in draws)
    prompt_fields = (
        "answer_mode_count",
        "option_ids",
        "prompt",
        "prompt_index",
        "reference",
    )
    request_fields = (
        "option_ids",
        "prompt_index",
        "request_seeds_by_option",
    )
    observations: dict[tuple[int, int], tuple[int, str, str]] = {}
    for path in sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line_number, raw in enumerate(handle, start=1):
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if row.get("evaluation_kind") != "fixed_seed_sampled_k_neutral":
                    continue
                step = row.get("step")
                draw = row.get("draw_index")
                if (
                    not isinstance(step, int)
                    or isinstance(step, bool)
                    or step not in expected_steps
                    or not isinstance(draw, int)
                    or isinstance(draw, bool)
                    or draw not in expected_draws
                ):
                    continue
                expected_row = {
                    "benchmark": sampled_contract["benchmark"],
                    "sample_count": sampled_contract["sample_count"],
                    "schema_version": sampled_contract["schema_version"],
                    "seed": int(sampled_contract["seed_base"]) + draw,
                    "temperature": sampled_contract["temperature"],
                }
                if any(
                    not _contract_value_matches(row.get(field), expected)
                    for field, expected in expected_row.items()
                ):
                    raise RuntimeError(
                        f"{path}:{line_number}: cannot hash a drifted sampled row"
                    )
                prompts = row.get("prompts")
                if not isinstance(prompts, list) or not prompts:
                    raise RuntimeError(f"{path}:{line_number}: missing prompt surface")
                if any(not isinstance(prompt, dict) for prompt in prompts):
                    raise RuntimeError(
                        f"{path}:{line_number}: malformed prompt surface"
                    )
                try:
                    prompt_projection = [
                        {field: prompt[field] for field in prompt_fields}
                        for prompt in prompts
                    ]
                    request_projection = [
                        {field: prompt[field] for field in request_fields}
                        for prompt in prompts
                    ]
                except KeyError as error:
                    raise RuntimeError(
                        f"{path}:{line_number}: incomplete prompt identity"
                    ) from error
                value = (
                    len(prompts),
                    _canonical_json_sha256(prompt_projection),
                    _canonical_json_sha256(
                        {
                            "row_seed": int(row["seed"]),
                            "prompts": request_projection,
                        }
                    ),
                )
                key = (step, draw)
                if key in observations and observations[key] != value:
                    raise RuntimeError(
                        f"{run_dir}: conflicting prompt surface at {key}"
                    )
                observations[key] = value

    expected = {(step, draw) for step in expected_steps for draw in expected_draws}
    missing = expected.difference(observations)
    if missing:
        raise RuntimeError(
            f"{run_dir}: prompt surface grid is missing {len(missing)} rows"
        )
    prompt_counts = {value[0] for value in observations.values()}
    prompt_hashes = {value[1] for value in observations.values()}
    if len(prompt_counts) != 1 or len(prompt_hashes) != 1:
        raise RuntimeError(f"{run_dir}: prompt surface changed across the grid")
    request_by_draw = {}
    for draw in sorted(expected_draws):
        hashes = {observations[(step, draw)][2] for step in expected_steps}
        if len(hashes) != 1:
            raise RuntimeError(
                f"{run_dir}: request-seed surface changed for draw {draw}"
            )
        request_by_draw[str(draw)] = next(iter(hashes))
    if len(set(request_by_draw.values())) != len(expected_draws):
        raise RuntimeError(f"{run_dir}: evaluation draws reuse a request-seed surface")
    return {
        "prompt_count": next(iter(prompt_counts)),
        "prompt_sha256": next(iter(prompt_hashes)),
        "request_sha256_by_draw": request_by_draw,
    }


def _greedy_value(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"{path}: greedy result must be a non-empty list")
    correct: list[float] = []
    for row_index, row in enumerate(payload):
        scores = row.get("scores") if isinstance(row, dict) else None
        if not isinstance(scores, list) or not scores:
            raise RuntimeError(f"{path}:{row_index}: missing greedy score")
        score = finite(scores[0], where=f"{path}:{row_index}:scores[0]")
        correct.append(float(score > 0.0))
    return statistics.fmean(correct)


def greedy_curve(
    run_dir: Path, *, steps: Sequence[int]
) -> tuple[dict[int, float], dict[str, str]]:
    curve: dict[int, float] = {}
    sources: dict[str, str] = {}
    for step in steps:
        paths = sorted(
            run_dir.glob(f"debug_job*/eval_results/{int(step)}_multi_answer.json")
        )
        if not paths:
            raise RuntimeError(f"{run_dir}: missing greedy checkpoint {step}")
        values = [_greedy_value(path) for path in paths]
        if any(not math.isclose(value, values[0], abs_tol=1e-12) for value in values):
            raise RuntimeError(
                f"{run_dir}: conflicting duplicate greedy checkpoint {step}"
            )
        curve[int(step)] = values[0]
        for path in paths:
            sources[str(path.resolve())] = sha256(path)
    return curve, sources


def run_curve(
    run_dir: Path,
    *,
    steps: Sequence[int],
    sampled_contract: Mapping[str, Any] | None = None,
    greedy_contract: Mapping[str, Any] | None = None,
) -> tuple[dict[int, dict[str, float]], dict[str, str]]:
    sampled, sampled_sources = sampled_curve(
        run_dir,
        steps=steps,
        sampled_contract=sampled_contract,
    )
    greedy_trace_sources = (
        evaluation_trace_contract(
            run_dir,
            steps=steps,
            evaluation_kind="deterministic_greedy_trace_neutral",
            row_contract=greedy_contract,
        )
        if greedy_contract is not None
        else {}
    )
    greedy, greedy_sources = greedy_curve(run_dir, steps=steps)
    curve = {
        int(step): dict(sampled[int(step)], greedy_pass1=greedy[int(step)])
        for step in steps
    }
    return curve, sampled_sources | greedy_trace_sources | greedy_sources


def run_curve_with_draws(
    run_dir: Path,
    *,
    steps: Sequence[int],
    sampled_contract: Mapping[str, Any] | None = None,
    greedy_contract: Mapping[str, Any] | None = None,
) -> tuple[
    dict[int, dict[str, float]],
    dict[int, dict[int, dict[str, float]]],
    dict[str, str],
]:
    """Return the same validated curve plus its sampled common-draw traces."""

    sampled, draw_curves, sampled_sources = sampled_curve_with_draws(
        run_dir,
        steps=steps,
        sampled_contract=sampled_contract,
    )
    greedy_trace_sources = (
        evaluation_trace_contract(
            run_dir,
            steps=steps,
            evaluation_kind="deterministic_greedy_trace_neutral",
            row_contract=greedy_contract,
        )
        if greedy_contract is not None
        else {}
    )
    greedy, greedy_sources = greedy_curve(run_dir, steps=steps)
    curve = {
        int(step): dict(sampled[int(step)], greedy_pass1=greedy[int(step)])
        for step in steps
    }
    return (
        curve,
        draw_curves,
        sampled_sources | greedy_trace_sources | greedy_sources,
    )


def normalized_auc(
    curve: dict[int, dict[str, float]], *, metric: str, target: int
) -> float:
    steps = sorted(curve)
    if not steps or steps[0] != 0 or steps[-1] != target:
        raise RuntimeError(f"AUC curve does not span 0--{target}: {steps}")
    area = sum(
        (right - left) * (curve[left][metric] + curve[right][metric]) / 2.0
        for left, right in zip(steps, steps[1:])
    )
    return area / float(target)


def paired_summary(values: dict[int, float], *, seeds: Sequence[int]) -> dict[str, Any]:
    ordered = tuple(int(seed) for seed in seeds)
    if tuple(sorted(values)) != tuple(sorted(ordered)) or len(ordered) != 5:
        raise RuntimeError(
            f"expected five paired seeds {ordered}, got {sorted(values)}"
        )
    samples = [finite(values[seed], where=f"paired seed {seed}") for seed in ordered]
    mean = statistics.fmean(samples)
    half = T_CRITICAL_DF4 * statistics.stdev(samples) / math.sqrt(5.0)
    return {
        "mean": mean,
        "n": 5,
        "student_t_95": [mean - half, mean + half],
        "range": [min(samples), max(samples)],
        "per_seed": {str(seed): values[seed] for seed in ordered},
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: ledger must be an object")
    return payload


def _expected_cells() -> set[tuple[str, str, int]]:
    return {
        (scale, domain, seed)
        for scale, seeds in SCALE_SEEDS.items()
        for domain in DOMAINS
        for seed in seeds
    }


def _validate_ledgers(
    treatment: dict[str, Any],
    comparators: dict[str, dict[str, Any]],
    repaired_python: dict[str, Any],
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
]:
    if treatment.get("schema") != (
        "e105_group_centered_semantic_repair_full_three_scale_jobs_v1"
    ):
        raise RuntimeError("unexpected E105 ledger schema")
    if treatment.get("released") is not True:
        raise RuntimeError("E105 ledger is not durably released")
    if treatment.get("pointmaze") != "excluded":
        raise RuntimeError("E105 result build requires PointMaze exclusion")
    if tuple(treatment.get("models", [])) != tuple(SCALE_SEEDS):
        raise RuntimeError("E105 model scales drifted")
    if tuple(treatment.get("domains", [])) != DOMAINS:
        raise RuntimeError("E105 static domains drifted")
    if treatment.get("arms") != ["semantic_group_centered"]:
        raise RuntimeError("E105 treatment arm drifted")
    if treatment.get("seeds") != {
        scale: list(seeds) for scale, seeds in SCALE_SEEDS.items()
    }:
        raise RuntimeError("E105 paired seeds drifted")
    expected_historical_ledgers = {
        scale: {"path": str(path), "sha256": sha256(path)}
        for scale, path in COMPARATOR_LEDGERS.items()
    }
    if treatment.get("historical_comparator_ledgers") != (expected_historical_ledgers):
        raise RuntimeError("E105 historical comparator ledger provenance drifted")

    treatment_index = {
        (str(run["scale"]), str(run["domain"]), int(run["seed"])): run
        for run in treatment.get("runs", [])
    }
    expected = _expected_cells()
    if set(treatment_index) != expected or len(treatment.get("runs", [])) != 75:
        raise RuntimeError("E105 ledger does not contain the exact 75-cell grid")
    if any(
        run.get("arm") != "semantic_group_centered" for run in treatment_index.values()
    ):
        raise RuntimeError("E105 run record names another treatment arm")

    replay_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for scale, seeds in SCALE_SEEDS.items():
        payload = comparators[scale]
        if payload.get("released") is not True:
            raise RuntimeError(f"{scale} comparator is not durably released")
        selected = [
            run for run in payload.get("runs", []) if run.get("arm") == "replay"
        ]
        for run in selected:
            replay_index[(scale, str(run["domain"]), int(run["seed"]))] = run
        expected_scale = {(scale, domain, seed) for domain in DOMAINS for seed in seeds}
        if {key for key in replay_index if key[0] == scale} != expected_scale or len(
            selected
        ) != 25:
            raise RuntimeError(f"{scale} comparator does not contain 25 replay cells")

    if treatment.get("python_comparator_repaired") is not True:
        raise RuntimeError("E105 ledger does not require repaired Python comparators")
    for prefix, path in (
        ("python_comparator_amendment", REPAIRED_PYTHON_COMPARATOR_PROTOCOL),
        ("repaired_python_comparator_ledger", REPAIRED_PYTHON_COMPARATOR_LEDGER),
    ):
        if Path(str(treatment.get(prefix, ""))).resolve() != path.resolve():
            raise RuntimeError(f"E105 ledger names another {prefix}")
        if not path.is_file() or treatment.get(f"{prefix}_sha256") != sha256(path):
            raise RuntimeError(f"E105 {prefix} digest mismatch")

    if repaired_python.get("schema") != (
        "e109_repaired_python_replay_comparators_jobs_v1"
    ):
        raise RuntimeError("unexpected E109 repaired comparator schema")
    if repaired_python.get("released") is not True:
        raise RuntimeError("E109 repaired Python comparator is not durably released")
    if repaired_python.get("pointmaze") != "excluded":
        raise RuntimeError("E109 repaired Python comparator includes PointMaze")
    if repaired_python.get("domain") != "python_factors":
        raise RuntimeError("E109 repaired comparator names another domain")
    if repaired_python.get("semantic_coefficient") != 0.0:
        raise RuntimeError("E109 repaired comparator activates semantic MaxEnt")
    if repaired_python.get("replay_weight") != treatment.get("replay_weight"):
        raise RuntimeError("E109 repaired comparator replay weight drifted")
    if repaired_python.get("snapshot_sha256") != treatment.get("snapshot_sha256"):
        raise RuntimeError("E109 repaired comparator snapshot drifted")
    if repaired_python.get("parser_surface_version") != treatment.get(
        "python_response_surface_version"
    ):
        raise RuntimeError("E109 repaired comparator parser surface drifted")
    if repaired_python.get("target_steps") != treatment.get("target_steps"):
        raise RuntimeError("E109 repaired comparator horizon drifted")
    if repaired_python.get("checkpoint_interval_steps") != treatment.get(
        "checkpoint_interval_steps"
    ):
        raise RuntimeError("E109 repaired comparator checkpoint cadence drifted")
    if repaired_python.get("qwen3_a6000_seeds") != [73, 74]:
        raise RuntimeError("E109 repaired comparator Qwen-3B placement drifted")
    if Path(
        str(repaired_python.get("qwen3_paired_placement_artifact", ""))
    ).resolve() != Path(
        str(treatment.get("qwen3_paired_placement_artifact", ""))
    ).resolve() or repaired_python.get(
        "qwen3_paired_placement_artifact_sha256"
    ) != treatment.get(
        "qwen3_paired_placement_artifact_sha256"
    ):
        raise RuntimeError("E109/E105 Qwen-3B placement artifact drifted")
    if repaired_python.get("scales") != list(SCALE_SEEDS):
        raise RuntimeError("E109 repaired comparator scales drifted")
    if repaired_python.get("seeds") != {
        scale: list(seeds) for scale, seeds in SCALE_SEEDS.items()
    }:
        raise RuntimeError("E109 repaired comparator seeds drifted")
    if Path(str(repaired_python.get("protocol", ""))).resolve() != (
        REPAIRED_PYTHON_COMPARATOR_PROTOCOL.resolve()
    ):
        raise RuntimeError("E109 repaired comparator names another protocol")
    if repaired_python.get("protocol_sha256") != sha256(
        REPAIRED_PYTHON_COMPARATOR_PROTOCOL
    ):
        raise RuntimeError("E109 repaired comparator protocol digest mismatch")
    if repaired_python.get("launcher_sha256") != sha256(
        REPAIRED_PYTHON_COMPARATOR_LAUNCHER
    ):
        raise RuntimeError("E109 repaired comparator launcher digest mismatch")

    repaired_runs = list(repaired_python.get("runs", []))
    expected_repaired = {
        (scale, "python_factors", seed)
        for scale, seeds in SCALE_SEEDS.items()
        for seed in seeds
    }
    repaired_index = {
        (str(run.get("scale")), str(run.get("domain")), int(run.get("seed"))): run
        for run in repaired_runs
        if run.get("arm") == "replay"
    }
    if (
        len(repaired_runs) != 15
        or set(repaired_index) != expected_repaired
        or any(run.get("arm") != "replay" for run in repaired_runs)
    ):
        raise RuntimeError("E109 does not contain the exact 15 Python replay cells")
    replay_index.update(repaired_index)

    for key, run in treatment_index.items():
        paired = run.get("paired_replay", {})
        replay = replay_index[key]
        expected_pair = {
            "run_stamp": str(replay["run_stamp"]),
            "run_dir": str(replay["run_dir"]),
            "job_id": int(replay["job_id"]),
        }
        if paired != expected_pair:
            raise RuntimeError(f"E105 paired comparator binding drifted: {key}")
    return treatment_index, replay_index


def _validate_qwen3_placement_provenance(
    treatment: dict[str, Any],
) -> dict[str, Any]:
    for prefix, path in (
        ("qwen3_paired_placement_protocol", QWEN3_PLACEMENT_PROTOCOL),
        ("qwen3_paired_placement_script", QWEN3_PLACEMENT_SCRIPT),
        ("qwen3_paired_placement_artifact", QWEN3_PLACEMENT_ARTIFACT),
    ):
        recorded = Path(str(treatment.get(prefix, ""))).resolve()
        if recorded != path.resolve():
            raise RuntimeError(f"E105 ledger names another {prefix}")
        if not path.is_file() or treatment.get(f"{prefix}_sha256") != sha256(path):
            raise RuntimeError(f"E105 {prefix} digest mismatch")
    payload = _load_json(QWEN3_PLACEMENT_ARTIFACT)
    if payload.get("schema") != ("e105_qwen3_paired_a6000_placement_amendment_v1"):
        raise RuntimeError("E105 Qwen-3B placement schema mismatch")
    if payload.get("pointmaze") != "excluded":
        raise RuntimeError("E105 Qwen-3B placement includes PointMaze")
    if payload.get("outcome_metrics_inspected") is not False:
        raise RuntimeError("E105 Qwen-3B placement violated outcome blinding")
    paired = {
        (str(row["domain"]), int(row["seed"]))
        for row in payload.get("paired_a6000_cells", [])
    }
    ledger_cells = {
        (str(row["domain"]), int(row["seed"]))
        for row in treatment.get("qwen3_a6000_cells", [])
    }
    if paired != ledger_cells or len(payload.get("paired_a6000_cells", [])) != len(
        paired
    ):
        raise RuntimeError("E105 Qwen-3B moved-cell binding drifted")
    return payload


def main() -> None:
    treatment = _load_json(LEDGER)
    comparators = {
        scale: _load_json(path) for scale, path in COMPARATOR_LEDGERS.items()
    }
    repaired_python = _load_json(REPAIRED_PYTHON_COMPARATOR_LEDGER)
    if treatment.get("analysis_protocol") != str(ANALYSIS_PROTOCOL):
        raise RuntimeError("E105 ledger names another analysis specification")
    if treatment.get("analysis_protocol_sha256") != sha256(ANALYSIS_PROTOCOL):
        raise RuntimeError("E105 analysis specification digest mismatch")
    if treatment.get("analysis_builder_sha256") != sha256(Path(__file__)):
        raise RuntimeError("E105 analysis builder digest mismatch")
    if treatment.get("analysis_plotter") != str(ANALYSIS_PLOTTER):
        raise RuntimeError("E105 ledger names another endpoint plotter")
    if treatment.get("analysis_plotter_sha256") != sha256(ANALYSIS_PLOTTER):
        raise RuntimeError("E105 endpoint plotter digest mismatch")
    qwen3_placement = _validate_qwen3_placement_provenance(treatment)
    treatment_index, replay_index = _validate_ledgers(
        treatment, comparators, repaired_python
    )

    interval = int(treatment["checkpoint_interval_steps"])
    target = int(treatment["target_steps"])
    if target != int(treatment["train_rows"]) * int(treatment["passes"]):
        raise RuntimeError("E105 target does not equal train rows times passes")
    steps = registered_steps(interval=interval, target=target)
    output: dict[str, Any] = {
        "schema": "e105_group_centered_semantic_repair_results_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pointmaze": "excluded",
        "treatment_ledger": str(LEDGER),
        "treatment_ledger_sha256": sha256(LEDGER),
        "analysis_protocol": str(ANALYSIS_PROTOCOL),
        "analysis_protocol_sha256": sha256(ANALYSIS_PROTOCOL),
        "analysis_builder": str(Path(__file__).resolve()),
        "analysis_builder_sha256": sha256(Path(__file__)),
        "analysis_plotter": str(ANALYSIS_PLOTTER),
        "analysis_plotter_sha256": sha256(ANALYSIS_PLOTTER),
        "qwen3_paired_placement": {
            "protocol": str(QWEN3_PLACEMENT_PROTOCOL),
            "protocol_sha256": sha256(QWEN3_PLACEMENT_PROTOCOL),
            "script": str(QWEN3_PLACEMENT_SCRIPT),
            "script_sha256": sha256(QWEN3_PLACEMENT_SCRIPT),
            "artifact": str(QWEN3_PLACEMENT_ARTIFACT),
            "artifact_sha256": sha256(QWEN3_PLACEMENT_ARTIFACT),
            "paired_a6000_cells": qwen3_placement["paired_a6000_cells"],
            "historical_moved_pairs": qwen3_placement["moved_pairs"],
            "prospective_python_cells": qwen3_placement["prospective_python_cells"],
        },
        "comparator_ledgers": {
            scale: {"path": str(path), "sha256": sha256(path)}
            for scale, path in COMPARATOR_LEDGERS.items()
        },
        "repaired_python_comparator": {
            "protocol": str(REPAIRED_PYTHON_COMPARATOR_PROTOCOL),
            "protocol_sha256": sha256(REPAIRED_PYTHON_COMPARATOR_PROTOCOL),
            "launcher": str(REPAIRED_PYTHON_COMPARATOR_LAUNCHER),
            "launcher_sha256": sha256(REPAIRED_PYTHON_COMPARATOR_LAUNCHER),
            "ledger": str(REPAIRED_PYTHON_COMPARATOR_LEDGER),
            "ledger_sha256": sha256(REPAIRED_PYTHON_COMPARATOR_LEDGER),
            "cells": 15,
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
            "registered_steps": list(steps),
            "target_steps": target,
        },
        "uncertainty": (
            "two-sided 95% Student-t interval over five paired seed effects, df=4"
        ),
        "families": {},
    }

    all_terminal_pass_effects: list[float] = []
    breadth_improved_families = 0
    for scale, seeds in SCALE_SEEDS.items():
        output["families"][scale] = {}
        for domain in DOMAINS:
            per_seed: dict[str, Any] = {}
            effects: dict[str, dict[int, float]] = defaultdict(dict)
            for seed in seeds:
                key = (scale, domain, seed)
                treatment_run = treatment_index[key]
                replay_run = replay_index[key]
                treatment_curve, treatment_sources = run_curve(
                    Path(treatment_run["run_dir"]), steps=steps
                )
                replay_curve, replay_sources = run_curve(
                    Path(replay_run["run_dir"]), steps=steps
                )
                terminal_treatment = treatment_curve[target]
                terminal_replay = replay_curve[target]
                terminal_effects = {
                    metric: terminal_treatment[metric] - terminal_replay[metric]
                    for metric in TERMINAL_METRICS
                }
                auc_treatment = {
                    metric: normalized_auc(
                        treatment_curve, metric=metric, target=target
                    )
                    for metric in AUC_METRICS
                }
                auc_replay = {
                    metric: normalized_auc(replay_curve, metric=metric, target=target)
                    for metric in AUC_METRICS
                }
                auc_effects = {
                    metric: auc_treatment[metric] - auc_replay[metric]
                    for metric in AUC_METRICS
                }
                for metric, value in terminal_effects.items():
                    effects[f"terminal_{metric}"][seed] = value
                for metric, value in auc_effects.items():
                    effects[f"auc_{metric}"][seed] = value
                all_terminal_pass_effects.append(terminal_effects["sampled_pass8"])
                per_seed[str(seed)] = {
                    "treatment": {
                        "run_stamp": treatment_run["run_stamp"],
                        "run_dir": treatment_run["run_dir"],
                        "terminal": terminal_treatment,
                        "normalized_auc": auc_treatment,
                        "curve": {str(step): treatment_curve[step] for step in steps},
                        "source_sha256": treatment_sources,
                    },
                    "replay": {
                        "run_stamp": replay_run["run_stamp"],
                        "run_dir": replay_run["run_dir"],
                        "terminal": terminal_replay,
                        "normalized_auc": auc_replay,
                        "curve": {str(step): replay_curve[step] for step in steps},
                        "source_sha256": replay_sources,
                    },
                    "paired_effects": {
                        "terminal": terminal_effects,
                        "normalized_auc": auc_effects,
                    },
                }
            summaries = {
                metric: paired_summary(values, seeds=seeds)
                for metric, values in effects.items()
            }
            if summaries["terminal_sampled_distinct8"]["mean"] > 0.0:
                breadth_improved_families += 1
            output["families"][scale][domain] = {
                "paired_seeds": list(seeds),
                "per_seed": per_seed,
                "paired_effects": summaries,
            }

    if len(all_terminal_pass_effects) != 75:
        raise RuntimeError("E105 result did not materialize 75 paired effects")
    grand_pass_effect = statistics.fmean(all_terminal_pass_effects)
    success = breadth_improved_families >= 8 and grand_pass_effect >= 0.0
    output["registered_general_extension_criterion"] = {
        "breadth_metric": "terminal_sampled_distinct8 paired family mean > 0",
        "breadth_improved_families": breadth_improved_families,
        "breadth_required_families": 8,
        "correctness_metric": (
            "unweighted mean of all 75 terminal_sampled_pass8 paired effects"
        ),
        "grand_terminal_sampled_pass8_effect": grand_pass_effect,
        "systematic_correctness_loss": grand_pass_effect < 0.0,
        "successful_general_extension": success,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
