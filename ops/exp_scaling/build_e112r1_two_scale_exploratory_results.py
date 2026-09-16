#!/usr/bin/env python3
"""Build the 49 integrity-valid pairs from the frozen E112-R1 two-scale set."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_e112r1_verified_support_discovery_results as e112  # noqa: E402


FREEZE = ROOT / (
    "var/artifacts/e112r1_private_interim_unblinding_freeze_20260826_50.json"
)
PUBLIC_PROTOCOL = ROOT / (
    "paper/preregistration/"
    "e112r1_two_scale_exploratory_public_disclosure_20260828.md"
)
IDENTITY_ERRATUM = ROOT / (
    "paper/preregistration/"
    "e112r1_final_analysis_a7_response_free_identity_erratum_20260828.md"
)
INTEGRITY_AMENDMENT = ROOT / (
    "paper/preregistration/"
    "e112r1_two_scale_49pair_terminal_integrity_amendment_20260828.md"
)
PLOTTER = ROOT / (
    "ops/exp_scaling/plot_e112r1_two_scale_exploratory_effects.py"
)
OUTPUT = ROOT / "paper/results/e112r1_two_scale_exploratory_results.json"
SCALES = ("qwen05b", "falcon1b")
INTEGRITY_EXCLUSIONS: dict[e112.Cell, dict[str, Any]] = {
    ("falcon1b", "countdown", 59): {
        "comparator_job_id": 30_269_051,
        "reason": "conflicting duplicate sampled rows in the registered job",
        "source_log": (
            "var/data/xdr_falcon3_1b_instruct_verified_first_replay_"
            "rehearsal_only_e79_falcon_aligned_countdown_replay_s59/"
            "debug_job30269051/eval_mode_coverage_draws.jsonl"
        ),
        "source_log_sha256": "2f6e2ea3419c617dcedafd0ce2f7da7734ead2ad19c73056c0ae579252d64465",
        "outcome_value_selected": False,
    }
}
T_CRITICAL_95 = {3: 3.182446305284263, 4: 2.7764451051977987}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return payload


def _expected_membership() -> set[e112.Cell]:
    return {
        (scale, domain, seed)
        for scale in SCALES
        for domain in e112.DOMAINS
        for seed in e112.SCALE_SEEDS[scale]
    }


def _validate_freeze(
    freeze: Mapping[str, Any],
    treatment_index: Mapping[e112.Cell, Mapping[str, Any]],
    replay_index: Mapping[e112.Cell, Mapping[str, Any]],
) -> set[e112.Cell]:
    required = {
        "schema": "e112r1_private_interim_unblinding_freeze_v1",
        "terminal_cells": 50,
        "target_steps": 3_072,
        "pointmaze": "excluded",
        "user_requested_early_unblinding": True,
        "confirmatory_outcome_blindness_broken": True,
        "efficacy_fields_read_before_freeze": False,
        "campaign_mutation_allowed_from_interim": False,
        "paper_efficacy_output_allowed": False,
    }
    drifted = [key for key, value in required.items() if freeze.get(key) != value]
    if drifted:
        raise RuntimeError(f"E112-R1 50-cell freeze drifted: {drifted}")
    frozen_ledger = Path(str(freeze.get("ledger", "")))
    if not frozen_ledger.is_absolute():
        frozen_ledger = ROOT / frozen_ledger
    if frozen_ledger.resolve() != e112.LEDGER.resolve():
        raise RuntimeError("E112-R1 freeze names another treatment ledger")
    if freeze.get("ledger_sha256") != e112.sha256(e112.LEDGER):
        raise RuntimeError("E112-R1 ledger differs from its 50-cell freeze")
    private_protocol = ROOT / str(freeze.get("protocol", ""))
    if not private_protocol.is_file() or freeze.get("protocol_sha256") != e112.sha256(
        private_protocol
    ):
        raise RuntimeError("E112-R1 private-look protocol digest drifted")
    if (
        not PUBLIC_PROTOCOL.is_file()
        or not IDENTITY_ERRATUM.is_file()
        or not INTEGRITY_AMENDMENT.is_file()
        or not PLOTTER.is_file()
    ):
        raise RuntimeError("public protocols, erratum, or plotter are absent")

    cells = freeze.get("cells")
    if not isinstance(cells, list) or len(cells) != 50:
        raise RuntimeError("E112-R1 freeze does not contain exactly 50 cells")
    observed: set[e112.Cell] = set()
    target = int(freeze["target_steps"])
    for row in cells:
        if not isinstance(row, dict):
            raise RuntimeError("E112-R1 frozen cell is malformed")
        key = (str(row["scale"]), str(row["domain"]), int(row["seed"]))
        if key in observed:
            raise RuntimeError(f"duplicate E112-R1 frozen cell: {key}")
        observed.add(key)
        treatment = treatment_index.get(key)
        replay = replay_index.get(key)
        if treatment is None or replay is None:
            raise RuntimeError(f"frozen E112-R1 pair is absent: {key}")
        if int(row["job_id"]) != int(treatment["job_id"]):
            raise RuntimeError(f"frozen treatment job drifted: {key}")
        if Path(str(row["run_dir"])).resolve() != Path(
            str(treatment["run_dir"])
        ).resolve():
            raise RuntimeError(f"frozen treatment directory drifted: {key}")
        marker = ROOT / str(row["completion_marker"])
        if not marker.is_file() or row.get("completion_marker_sha256") != e112.sha256(
            marker
        ):
            raise RuntimeError(f"frozen completion marker drifted: {key}")
        marker_payload = _load(marker)
        terminal_step = marker_payload.get("terminal_step")
        if not isinstance(terminal_step, int) or terminal_step < target:
            raise RuntimeError(f"frozen treatment is not terminal: {key}")
        paired = row.get("paired_replay")
        expected_pair = {
            "job_id": int(replay["job_id"]),
            "run_dir": str(replay["run_dir"]),
            "run_stamp": str(replay["run_stamp"]),
        }
        if paired != expected_pair:
            raise RuntimeError(f"frozen comparator binding drifted: {key}")
    expected = _expected_membership()
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"50-cell freeze is not the exact two-scale grid; missing={missing}, "
            f"extra={extra}"
        )
    return observed


def _paired_summary(
    values: Mapping[int, float], *, seeds: tuple[int, ...]
) -> dict[str, Any]:
    ordered = tuple(int(seed) for seed in seeds)
    n = len(ordered)
    if n not in (4, 5) or set(values) != set(ordered):
        raise RuntimeError(f"expected four or five paired seeds {ordered}")
    samples = [
        e112.e105.finite(values[seed], where=f"paired seed {seed}")
        for seed in ordered
    ]
    mean = statistics.fmean(samples)
    degrees_freedom = n - 1
    half = (
        T_CRITICAL_95[degrees_freedom]
        * statistics.stdev(samples)
        / math.sqrt(float(n))
    )
    return {
        "mean": mean,
        "n": n,
        "degrees_freedom": degrees_freedom,
        "student_t_95": [mean - half, mean + half],
        "range": [min(samples), max(samples)],
        "per_seed": {str(seed): values[seed] for seed in ordered},
    }


def _paired_draw_summary(
    values: Mapping[int, Mapping[int, float]], *, seeds: tuple[int, ...]
) -> dict[str, Any]:
    ordered = tuple(int(seed) for seed in seeds)
    expected_draws = tuple(range(e112.EXPECTED_DRAWS))
    if len(ordered) not in (4, 5) or set(values) != set(ordered):
        raise RuntimeError("paired-draw summary requires four or five exact seeds")
    per_seed: dict[str, Any] = {}
    seed_means: list[float] = []
    for seed in ordered:
        if set(values[seed]) != set(expected_draws):
            raise RuntimeError(f"paired-draw seed {seed} lacks exact draws 0--3")
        samples = [
            e112.e105.finite(values[seed][draw], where=f"seed {seed} draw {draw}")
            for draw in expected_draws
        ]
        estimate = statistics.fmean(samples)
        seed_means.append(estimate)
        per_seed[str(seed)] = {
            "estimate": estimate,
            "evaluation_mc_se": statistics.stdev(samples)
            / math.sqrt(float(e112.EXPECTED_DRAWS)),
            "draw_effects": {
                str(draw): samples[draw] for draw in expected_draws
            },
        }
    draw_means = {
        str(draw): statistics.fmean(values[seed][draw] for seed in ordered)
        for draw in expected_draws
    }
    draw_mean_values = [draw_means[str(draw)] for draw in expected_draws]
    estimate = statistics.fmean(seed_means)
    if not math.isclose(estimate, statistics.fmean(draw_mean_values), abs_tol=1e-12):
        raise RuntimeError("paired-draw aggregation axes do not reconcile")
    return {
        "estimate": estimate,
        "n_training_seeds": len(ordered),
        "n_evaluation_draws": e112.EXPECTED_DRAWS,
        "training_seed_se": statistics.stdev(seed_means)
        / math.sqrt(float(len(ordered))),
        "evaluation_mc_se": statistics.stdev(draw_mean_values)
        / math.sqrt(float(e112.EXPECTED_DRAWS)),
        "per_training_seed": per_seed,
        "draw_mean_effects": draw_means,
        "combined_interval": None,
        "p_value": None,
    }


def _descriptive_scale_results(result: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scale in SCALES:
        pass_effects: list[float] = []
        breadth_effects: list[float] = []
        positive_families: list[str] = []
        for domain in e112.DOMAINS:
            family = result["families"][scale][domain]
            pass_summary = family["paired_effects"]["terminal_sampled_pass8"]
            breadth_summary = family["paired_effects"]["terminal_sampled_excess8"]
            if float(breadth_summary["mean"]) > 0.0:
                positive_families.append(domain)
            for seed in family["paired_seeds"]:
                pass_effects.append(float(pass_summary["per_seed"][str(seed)]))
                breadth_effects.append(
                    float(breadth_summary["per_seed"][str(seed)])
                )
        cells = len(pass_effects)
        pass_mean = statistics.fmean(pass_effects)
        breadth_mean = statistics.fmean(breadth_effects)
        subset_rule = breadth_mean > 0.0 and pass_mean >= 0.0
        output[scale] = {
            "cells": cells,
            "full_registered_25_cell_scale": cells == 25,
            "terminal_sampled_pass8_effect_mean": pass_mean,
            "terminal_adjusted_breadth_effect_mean": breadth_mean,
            "positive_breadth_family_count": len(positive_families),
            "positive_breadth_family_ids": positive_families,
            "descriptive_completed_scale_rule": subset_rule if cells == 25 else None,
            "descriptive_integrity_valid_subset_rule": subset_rule,
            "interval": None,
            "test": None,
        }
    return output


def _terminal_run_payload(
    row: Mapping[str, Any], *, domain: str, target: int
) -> tuple[dict[str, Any], dict[str, float], dict[int, dict[str, float]]]:
    sampled_contract = {
        "benchmark": "multi_answer",
        "sample_count": 8,
        "schema_version": 1,
        "seed_base": e112.EVAL_SEED_BASES[domain],
        "temperature": 1.0,
    }
    run_dir = Path(str(row["run_dir"]))
    curve, draw_curves, sources = e112.e105.sampled_curve_with_draws(
        run_dir,
        steps=[target],
        sampled_contract=sampled_contract,
    )
    identity = e112.sampled_prompt_surface(
        run_dir,
        steps=[target],
        draws=range(e112.EXPECTED_DRAWS),
        sampled_contract=sampled_contract,
    )
    return (
        {
            "run_stamp": str(row["run_stamp"]),
            "run_dir": str(row["run_dir"]),
            "registered_job_id": int(row["job_id"]),
            "terminal": curve[target],
            "evaluation_identity": identity,
            "source_sha256": sources,
        },
        curve[target],
        {
            draw: draw_curves[draw][target]
            for draw in range(e112.EXPECTED_DRAWS)
        },
    )


def _materialize_terminal_results(
    treatment_index: Mapping[e112.Cell, Mapping[str, Any]],
    replay_index: Mapping[e112.Cell, Mapping[str, Any]],
    *,
    target: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _expected_membership()
    if set(treatment_index) != expected or set(replay_index) != expected:
        raise RuntimeError("terminal materialization requires the exact 50 frozen pairs")
    for key, details in INTEGRITY_EXCLUSIONS.items():
        replay = replay_index[key]
        if int(replay["job_id"]) != int(details["comparator_job_id"]):
            raise RuntimeError(f"integrity exclusion job drifted: {key}")
        source_log = ROOT / str(details["source_log"])
        if (
            not source_log.is_file()
            or e112.sha256(source_log) != details["source_log_sha256"]
        ):
            raise RuntimeError(f"integrity exclusion source drifted: {key}")
        try:
            _terminal_run_payload(replay, domain=key[1], target=target)
        except RuntimeError as error:
            if "conflicting duplicate sampled row" not in str(error):
                raise RuntimeError(
                    f"integrity exclusion failed for another reason: {key}"
                ) from error
        else:
            raise RuntimeError(f"integrity exclusion no longer fails closed: {key}")

    output: dict[str, Any] = {
        "schema": "e112r1_two_scale_exploratory_results_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
            "prompt_target": (
                "registered_finite_terminal_evaluation_bank_within_domain"
            ),
            "prompt_population_inference": False,
            "prompt_population_se": None,
            "trajectory_auc_available": False,
            "trajectory_auc_failure": (
                "the exact frozen 50-pair trajectory result fails the "
                "conflicting-retry contract for one historical comparator"
            ),
        },
        "metric_contract": {
            "primitive_endpoint_vector": ["sampled_pass8", "sampled_distinct8"],
            "derived_correctness_adjusted_breadth": (
                "sampled_excess8 = sampled_distinct8 - sampled_pass8"
            ),
            "terminal_step": target,
            "sampled_evaluation_rows": {
                "benchmark": "multi_answer",
                "sample_count": 8,
                "schema_version": 1,
                "temperature": 1.0,
                "seed_base_by_domain": dict(e112.EVAL_SEED_BASES),
                "draw_seed_formula": "seed_base_by_domain + draw_index",
            },
            "paired_terminal_evaluation_identity": {
                "require_exact_treatment_comparator_match": True,
                "require_constant_prompt_surface_across_terminal_draws": True,
                "require_distinct_request_surfaces_across_terminal_draws": True,
                "terminal_identity_requirement_satisfied_for_included_pairs": True,
                "generated_answer_keys_excluded_from_identity": True,
            },
        },
        "design": {
            "scales": list(SCALES),
            "domains": list(e112.DOMAINS),
            "paired_seeds": {
                scale: list(e112.SCALE_SEEDS[scale]) for scale in SCALES
            },
            "frozen_cells": 50,
            "cells": 49,
            "families": 10,
            "evaluation_draws": e112.EXPECTED_DRAWS,
            "target_steps": target,
            "qwen3b_cells_included": 0,
            "integrity_exclusions": [
                {
                    "scale": scale,
                    "domain": domain,
                    "seed": seed,
                    **details,
                }
                for (scale, domain, seed), details in sorted(
                    INTEGRITY_EXCLUSIONS.items()
                )
            ],
        },
        "uncertainty": (
            "two-sided 95% Student-t interval over paired training-seed "
            "effects: df=4 for n=5 families and df=3 for Falcon Countdown "
            "n=4; four terminal evaluation draws are averaged and are not "
            "treated as independent training seeds"
        ),
        "families": {},
    }
    for scale in SCALES:
        output["families"][scale] = {}
        registered_seeds = e112.SCALE_SEEDS[scale]
        for domain in e112.DOMAINS:
            seeds = tuple(
                seed for seed in registered_seeds
                if (scale, domain, seed) not in INTEGRITY_EXCLUSIONS
            )
            per_seed: dict[str, Any] = {}
            effects = {
                f"terminal_{metric}": {} for metric in e112.SAMPLED_METRICS
            }
            draw_effects = {
                f"terminal_{metric}": {} for metric in e112.SAMPLED_METRICS
            }
            for seed in seeds:
                key = (scale, domain, seed)
                treatment, treatment_terminal, treatment_draws = (
                    _terminal_run_payload(
                        treatment_index[key], domain=domain, target=target
                    )
                )
                replay, replay_terminal, replay_draws = _terminal_run_payload(
                    replay_index[key], domain=domain, target=target
                )
                if treatment["evaluation_identity"] != replay["evaluation_identity"]:
                    raise RuntimeError(
                        f"terminal paired prompt/request surface drifted: {key}"
                    )
                paired = {
                    metric: float(treatment_terminal[metric])
                    - float(replay_terminal[metric])
                    for metric in e112.SAMPLED_METRICS
                }
                paired_by_draw: dict[str, dict[str, float]] = {}
                for draw in range(e112.EXPECTED_DRAWS):
                    paired_by_draw[str(draw)] = {
                        metric: float(treatment_draws[draw][metric])
                        - float(replay_draws[draw][metric])
                        for metric in e112.SAMPLED_METRICS
                    }
                for metric in e112.SAMPLED_METRICS:
                    summary_key = f"terminal_{metric}"
                    effects[summary_key][seed] = paired[metric]
                    draw_effects[summary_key][seed] = {
                        draw: paired_by_draw[str(draw)][metric]
                        for draw in range(e112.EXPECTED_DRAWS)
                    }
                    if not math.isclose(
                        statistics.fmean(
                            draw_effects[summary_key][seed].values()
                        ),
                        paired[metric],
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError(
                            f"terminal paired draws do not reconcile: {key}"
                        )
                per_seed[str(seed)] = {
                    "treatment": treatment,
                    "replay": replay,
                    "paired_effects": paired,
                    "paired_draw_effects": paired_by_draw,
                }
            output["families"][scale][domain] = {
                "paired_seeds": list(seeds),
                "per_seed": per_seed,
                "paired_effects": {
                    metric: _paired_summary(values, seeds=seeds)
                    for metric, values in effects.items()
                },
                "paired_draw_diagnostics": {
                    metric: _paired_draw_summary(values, seeds=seeds)
                    for metric, values in draw_effects.items()
                },
            }
    return output


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build() -> dict[str, Any]:
    freeze = _load(FREEZE)
    treatment, treatment_index, replay_index, continuation = (
        e112.load_and_validate_ledgers()
    )
    membership = _validate_freeze(freeze, treatment_index, replay_index)
    treatment_subset = {key: treatment_index[key] for key in membership}
    replay_subset = {key: replay_index[key] for key in membership}
    target = int(treatment["target_steps"])
    provenance = {
        "treatment_ledger": str(e112.LEDGER),
        "treatment_ledger_sha256": e112.sha256(e112.LEDGER),
        "frozen_membership": str(FREEZE),
        "frozen_membership_sha256": e112.sha256(FREEZE),
        "frozen_private_protocol": str(ROOT / str(freeze["protocol"])),
        "frozen_private_protocol_sha256": str(freeze["protocol_sha256"]),
        "public_disclosure_protocol": str(PUBLIC_PROTOCOL),
        "public_disclosure_protocol_sha256": e112.sha256(PUBLIC_PROTOCOL),
        "response_free_identity_erratum": str(IDENTITY_ERRATUM),
        "response_free_identity_erratum_sha256": e112.sha256(IDENTITY_ERRATUM),
        "integrity_amendment": str(INTEGRITY_AMENDMENT),
        "integrity_amendment_sha256": e112.sha256(INTEGRITY_AMENDMENT),
        "registered_analysis_protocol": str(e112.ANALYSIS_PROTOCOL),
        "registered_analysis_protocol_sha256": e112.sha256(e112.ANALYSIS_PROTOCOL),
        "analysis_builder": str(Path(__file__).resolve()),
        "analysis_builder_sha256": e112.sha256(Path(__file__)),
        "registered_full_builder": str(Path(e112.__file__).resolve()),
        "registered_full_builder_sha256": e112.sha256(Path(e112.__file__)),
        "shared_endpoint_builder": str(e112.SHARED_ENDPOINT_BUILDER),
        "shared_endpoint_builder_sha256": e112.sha256(e112.SHARED_ENDPOINT_BUILDER),
        "analysis_plotter": str(PLOTTER.resolve()),
        "analysis_plotter_sha256": e112.sha256(PLOTTER),
        "comparator_ledgers": {
            scale: {
                "path": str(e112.COMPARATOR_LEDGERS[scale]),
                "sha256": e112.sha256(e112.COMPARATOR_LEDGERS[scale]),
            }
            for scale in SCALES
        },
        "repaired_python_ledger": str(e112.REPAIRED_PYTHON_LEDGER),
        "repaired_python_ledger_sha256": e112.sha256(e112.REPAIRED_PYTHON_LEDGER),
        "e109_continuation": str(e112.E109_CONTINUATION),
        "e109_continuation_sha256": e112.sha256(e112.E109_CONTINUATION),
        "e109_continuation_jobs": [
            int(row["continuation_job_id"]) for row in continuation["records"]
        ],
    }

    result = _materialize_terminal_results(
        treatment_subset,
        replay_subset,
        target=target,
        provenance=provenance,
    )
    result["disclosure"] = {
        "status": "author-requested exploratory public disclosure",
        "confirmatory": False,
        "continuous_outcome_blindness": False,
        "prior_private_looks_terminal_cells": [14, 33, 50],
        "membership_frozen_before_new_50_cell_endpoints_read": True,
        "paper_exploratory_output_allowed_by_author": True,
        "campaign_mutation_allowed_from_outcomes": False,
        "qwen3b": "incomplete registered scale; excluded from all estimates",
        "trajectory_auc": (
            "unavailable for the exact frozen set because one registered "
            "historical comparator has conflicting duplicate evaluation rows"
        ),
        "integrity_valid_pairs": 49,
        "excluded_frozen_pairs": 1,
        "model_or_domain_pooling": False,
        "multiplicity_adjusted_confirmatory_tests": False,
    }
    result["integrity_valid_scale_descriptive_results"] = (
        _descriptive_scale_results(result)
    )
    result["registered_criteria_not_evaluated"] = {
        "general_8_of_15_family_criterion": "not evaluated; only 10 families included",
        "all_three_scales_criterion": "not evaluated; Qwen2.5-3B incomplete and excluded",
    }
    result.pop("registered_general_extension_criterion", None)
    result.pop("registered_all_three_scales_criterion", None)
    if result["estimand_scope"].get("isolated_semantic_v7_effect") is not False:
        raise RuntimeError("two-scale disclosure lost bundled-estimand boundary")
    return result


def main() -> None:
    result = build()
    _atomic_write(OUTPUT, result)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
