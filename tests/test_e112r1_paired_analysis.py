from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / ("ops/exp_scaling/build_e112r1_verified_support_discovery_results.py")
PROTOCOL = ROOT / (
    "paper/preregistration/e112_paired_analysis_specification_20260818.md"
)
IMPLEMENTATION_FREEZE = ROOT / (
    "paper/preregistration/" "e112r1_final_analysis_implementation_freeze_20260825.md"
)
PLOTTER = ROOT / ("ops/exp_scaling/plot_e112r1_verified_support_discovery_effects.py")


def _load():
    spec = importlib.util.spec_from_file_location("e112r1_results", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_plotter():
    spec = importlib.util.spec_from_file_location("e112r1_plot", PLOTTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_indexes(module):
    treatment = {}
    replay = {}
    for scale, seeds in module.SCALE_SEEDS.items():
        for domain in module.DOMAINS:
            for seed in seeds:
                key = (scale, domain, seed)
                treatment[key] = {
                    "run_stamp": f"t-{scale}-{domain}-{seed}",
                    "run_dir": f"/synthetic/t-{scale}-{domain}-{seed}",
                    "job_id": 10_000 + seed,
                }
                replay[key] = {
                    "run_stamp": f"r-{scale}-{domain}-{seed}",
                    "run_dir": f"/synthetic/r-{scale}-{domain}-{seed}",
                    "job_id": 20_000 + seed,
                }
    return treatment, replay


def _synthetic_curve(
    run_dir: Path, *, steps, sampled_contract=None, greedy_contract=None
):
    assert sampled_contract is not None
    assert sampled_contract["benchmark"] == "multi_answer"
    assert sampled_contract["sample_count"] == 8
    assert sampled_contract["schema_version"] == 1
    assert sampled_contract["temperature"] == 1.0
    assert sampled_contract["seed_base"] in {610100, 610200, 610300, 610400, 76299}
    assert greedy_contract == {
        "benchmark": "multi_answer",
        "draw_index": None,
        "sample_count": 1,
        "schema_version": 1,
        "seed": 0,
        "temperature": 0.0,
    }
    treatment = run_dir.name.startswith("t-")
    delta_pass = 0.02 if treatment else 0.0
    delta_distinct = 0.08 if treatment else 0.0
    curve = {}
    for step in steps:
        progress = float(step) / float(steps[-1]) if steps[-1] else 0.0
        pass8 = 0.40 + 0.10 * progress + delta_pass
        distinct8 = 0.70 + 0.10 * progress + delta_distinct
        curve[int(step)] = {
            "greedy_pass1": 0.30 + delta_pass,
            "sampled_pass8": pass8,
            "sampled_mean8": 0.20 + delta_pass,
            "sampled_distinct8": distinct8,
            "sampled_excess8": distinct8 - pass8,
        }
    return curve, {str(run_dir / "synthetic.jsonl"): "digest"}


def _synthetic_curve_with_draws(
    run_dir: Path, *, steps, sampled_contract=None, greedy_contract=None
):
    curve, sources = _synthetic_curve(
        run_dir,
        steps=steps,
        sampled_contract=sampled_contract,
        greedy_contract=greedy_contract,
    )
    treatment = run_dir.name.startswith("t-")
    draw_curves = {}
    for draw in range(4):
        draw_shift = (draw - 1.5) * 0.002 * (2.0 if treatment else 1.0)
        draw_curves[draw] = {}
        for step in steps:
            sampled_pass = curve[int(step)]["sampled_pass8"] + draw_shift
            sampled_mean = curve[int(step)]["sampled_mean8"] + draw_shift
            sampled_distinct = curve[int(step)]["sampled_distinct8"] + draw_shift
            draw_curves[draw][int(step)] = {
                "sampled_pass8": sampled_pass,
                "sampled_mean8": sampled_mean,
                "sampled_distinct8": sampled_distinct,
                "sampled_excess8": sampled_distinct - sampled_pass,
            }
    return curve, draw_curves, sources


def _synthetic_prompt_surface(run_dir: Path, *, steps, draws, sampled_contract):
    return {
        "prompt_count": 2,
        "prompt_sha256": f"prompt-{sampled_contract['seed_base']}",
        "request_sha256_by_draw": {
            str(draw): f"request-{sampled_contract['seed_base']}-{draw}"
            for draw in draws
        },
    }


def test_production_ledgers_validate_without_reading_outcomes():
    module = _load()
    (
        treatment,
        treatment_index,
        replay_index,
        continuation,
    ) = module.load_and_validate_ledgers()

    assert treatment["released"] is True
    assert len(treatment_index) == 75
    assert len(replay_index) == 75
    assert continuation["released"] is True
    assert replay_index[("qwen3b", "python_factors", 74)]["run_stamp"].startswith(
        "e109_"
    )
    assert replay_index[("qwen3b", "graph_coloring", 74)]["run_stamp"].startswith(
        "e80r1_"
    )


def test_validator_rejects_historical_python_rebinding():
    module = _load()
    treatment = module._load_json(module.LEDGER)
    comparators = {
        scale: module._load_json(path)
        for scale, path in module.COMPARATOR_LEDGERS.items()
    }
    repaired = module._load_json(module.REPAIRED_PYTHON_LEDGER)
    continuation = module._load_json(module.E109_CONTINUATION)
    corrupted = copy.deepcopy(treatment)
    historical = next(
        row
        for row in comparators["qwen3b"]["runs"]
        if row["arm"] == "replay"
        and row["domain"] == "python_factors"
        and row["seed"] == 74
    )
    target = next(
        row
        for row in corrupted["runs"]
        if row["scale"] == "qwen3b"
        and row["domain"] == "python_factors"
        and row["seed"] == 74
    )
    target["paired_replay"] = {
        field: historical[field] for field in ("run_stamp", "run_dir", "job_id")
    }

    with pytest.raises(RuntimeError, match="paired comparator binding drifted"):
        module.validate_ledgers(corrupted, comparators, repaired, continuation)


def test_builder_reuses_parser_and_keeps_primitive_endpoint_vector(monkeypatch):
    module = _load()
    assert module.run_curve is module.e105.run_curve
    assert module.run_curve_with_draws is module.e105.run_curve_with_draws
    treatment, replay = _synthetic_indexes(module)
    monkeypatch.setattr(module, "run_curve_with_draws", _synthetic_curve_with_draws)
    monkeypatch.setattr(module, "sampled_prompt_surface", _synthetic_prompt_surface)

    result = module.materialize_results(
        treatment,
        replay,
        steps=(0, 2, 4),
        target=4,
        provenance={"synthetic": True},
        generated_at="frozen",
    )

    assert result["metric_contract"]["primitive_endpoint_vector"] == [
        "sampled_pass8",
        "sampled_distinct8",
    ]
    assert result["metric_contract"]["sampled_evaluation_rows"] == {
        "benchmark": "multi_answer",
        "sample_count": 8,
        "schema_version": 1,
        "temperature": 1.0,
        "seed_base_by_domain": module.EVAL_SEED_BASES,
        "draw_seed_formula": "seed_base_by_domain + draw_index",
    }
    assert result["metric_contract"]["greedy_evaluation_rows"] == {
        "benchmark": "multi_answer",
        "draw_index": None,
        "evaluation_kind": "deterministic_greedy_trace_neutral",
        "sample_count": 1,
        "schema_version": 1,
        "seed": 0,
        "temperature": 0.0,
    }
    assert (
        result["metric_contract"]["paired_evaluation_identity"][
            "require_exact_treatment_comparator_match"
        ]
        is True
    )
    assert (
        result["metric_contract"]["paired_evaluation_identity"][
            "require_distinct_request_surfaces_across_draws"
        ]
        is True
    )
    family = result["families"]["qwen05b"]["countdown"]
    seed = family["per_seed"]["43"]
    terminal = seed["paired_effects"]["terminal"]
    assert terminal["sampled_excess8"] == pytest.approx(
        terminal["sampled_distinct8"] - terminal["sampled_pass8"]
    )
    assert family["paired_effects"]["terminal_sampled_distinct8"]["n"] == 5
    assert family["paired_effects"]["terminal_sampled_excess8"]["n"] == 5
    assert (
        result["registered_general_extension_criterion"]["successful_general_extension"]
        is True
    )
    assert (
        result["registered_all_three_scales_criterion"][
            "successful_at_all_three_scales"
        ]
        is True
    )
    assert result["estimand_scope"]["isolated_semantic_v7_effect"] is False
    assert result["estimand_scope"]["prompt_target"] == (
        "registered_finite_evaluation_bank_within_domain"
    )
    assert result["estimand_scope"]["prompt_population_inference"] is False
    assert result["estimand_scope"]["prompt_population_se"] is None
    diagnostic = family["paired_draw_diagnostics"]
    raw = diagnostic["historical_contrast"]["terminal_sampled_distinct8"]
    centered = diagnostic["baseline_centered_sensitivity"]["terminal_sampled_distinct8"]
    assert raw["estimate"] == pytest.approx(0.08)
    assert raw["evaluation_mc_se"] > 0.0
    assert raw["combined_interval"] is None
    assert centered["estimate"] == pytest.approx(0.0)
    assert (
        result["secondary_paired_draw_diagnostic"]["registered_decision_or_gate"]
        is False
    )
    assert (
        result["secondary_paired_draw_diagnostic"][
            "generation_draws_change_prompt_identity"
        ]
        is False
    )


def test_scale_rule_is_separate_from_general_rule():
    module = _load()
    effects = {}
    for scale, seeds in module.SCALE_SEEDS.items():
        for domain in module.DOMAINS:
            for seed in seeds:
                effects[(scale, domain, seed)] = {
                    "sampled_excess8": -0.1 if scale == "qwen3b" else 0.1,
                    "sampled_pass8": 0.0,
                }
    general, all_scales = module.decision_rules(effects)

    assert general["breadth_improved_families"] == 10
    assert general["successful_general_extension"] is True
    assert all_scales["per_scale"]["qwen3b"]["scale_success"] is False
    assert all_scales["successful_at_all_three_scales"] is False


def test_incomplete_materialization_writes_no_result(tmp_path, monkeypatch):
    module = _load()
    treatment, replay = _synthetic_indexes(module)

    def incomplete(
        run_dir: Path, *, steps, sampled_contract=None, greedy_contract=None
    ):
        assert sampled_contract is not None
        assert greedy_contract is not None
        raise RuntimeError(f"{run_dir}: sampled checkpoint 4 has draws [0, 1]")

    monkeypatch.setattr(module, "run_curve_with_draws", incomplete)
    output = tmp_path / "official.json"
    with pytest.raises(RuntimeError, match="sampled checkpoint"):
        module.build_and_write(
            treatment,
            replay,
            steps=(0, 2, 4),
            target=4,
            provenance={},
            output_path=output,
        )
    assert not output.exists()


def test_paired_prompt_surface_drift_fails_closed(monkeypatch):
    module = _load()
    treatment, replay = _synthetic_indexes(module)
    monkeypatch.setattr(module, "run_curve_with_draws", _synthetic_curve_with_draws)

    def drifted(run_dir: Path, *, steps, draws, sampled_contract):
        surface = _synthetic_prompt_surface(
            run_dir,
            steps=steps,
            draws=draws,
            sampled_contract=sampled_contract,
        )
        if run_dir.name.startswith("t-"):
            surface["prompt_sha256"] = "drift"
        return surface

    monkeypatch.setattr(module, "sampled_prompt_surface", drifted)
    with pytest.raises(RuntimeError, match="paired evaluation prompt/request"):
        module.materialize_results(
            treatment,
            replay,
            steps=(0, 2, 4),
            target=4,
            provenance={},
        )


def test_forest_adapter_materializes_exact_terminal_and_auc_grids(
    tmp_path, monkeypatch
):
    module = _load()
    plotter = _load_plotter()
    treatment, replay = _synthetic_indexes(module)
    monkeypatch.setattr(module, "run_curve_with_draws", _synthetic_curve_with_draws)
    monkeypatch.setattr(module, "sampled_prompt_surface", _synthetic_prompt_surface)
    provenance = {
        "analysis_plotter": str(PLOTTER.resolve()),
        "analysis_plotter_sha256": plotter.sha256(PLOTTER),
    }
    result = module.materialize_results(
        treatment,
        replay,
        steps=(0, 2, 4),
        target=4,
        provenance=provenance,
        generated_at="frozen",
    )
    input_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(result), encoding="utf-8")

    terminal = plotter.plot_payload(
        result, input_path=input_path, effect_kind="terminal"
    )
    auc = plotter.plot_payload(result, input_path=input_path, effect_kind="auc")
    terminal_primitives = plotter.plot_payload(
        result, input_path=input_path, effect_kind="terminal_primitives"
    )
    auc_primitives = plotter.plot_payload(
        result, input_path=input_path, effect_kind="auc_primitives"
    )

    assert len(terminal["cells"]) == 15
    assert len(auc["cells"]) == 15
    assert all(cell["n"] == 5 for cell in terminal["cells"] + auc["cells"])
    assert terminal["metric_order"] == [
        "terminal_sampled_pass8",
        "terminal_sampled_excess8",
    ]
    assert auc["metric_order"] == ["auc_sampled_pass8", "auc_sampled_excess8"]
    assert terminal_primitives["metric_order"] == [
        "terminal_sampled_pass8",
        "terminal_sampled_distinct8",
    ]
    assert auc_primitives["metric_order"] == [
        "auc_sampled_pass8",
        "auc_sampled_distinct8",
    ]
    assert terminal_primitives["supplementary_primitive_vector_view"] is True
    assert terminal["estimand_scope"]["isolated_semantic_v7_effect"] is False
    assert "bundle vs historical" in terminal["title"]
    output = tmp_path / "terminal"
    plotter.render(terminal, output)
    assert output.with_suffix(".pdf").stat().st_size > 0
    assert output.with_suffix(".png").stat().st_size > 0
    assert output.with_suffix(".json").stat().st_size > 0
    primitive_output = tmp_path / "terminal_primitives"
    plotter.render(terminal_primitives, primitive_output)
    assert primitive_output.with_suffix(".pdf").stat().st_size > 0
    assert primitive_output.with_suffix(".png").stat().st_size > 0
    assert primitive_output.with_suffix(".json").stat().st_size > 0


def test_analysis_freeze_is_explicit_about_limits_and_decisions():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    freeze = IMPLEMENTATION_FREEZE.read_text(encoding="utf-8")
    builder = BUILDER.read_text(encoding="utf-8")

    assert "all 75 E112 treatment cells" in protocol
    assert "all four registered" in protocol
    assert "8 of 15 families" in protocol
    assert "all three scales" in protocol
    assert "private interim" in freeze and "cannot be undone" in freeze
    assert "raw distinct@8 - pass@8" in freeze
    assert "must not be described" in freeze and "isolated causal effect" in freeze
    assert "historical comparator selection is not exactly 60 cells" in builder
    assert "E109 lacks the exact 15 repaired Python replay cells" in builder
    assert "refusing to overwrite existing official result" in builder
    assert "isolated_semantic_v7_effect" in builder
    assert "confirmatory_blind" in builder
    assert "thin E112 adapter" in freeze
    assert "plot_e112r1_verified_support_discovery_effects.py" in builder
    assert "e112r1_final_analysis_a4_distinct_request_draws_20260825.md" in builder
    assert "e112r1_final_analysis_a5_paired_draw_diagnostic_20260825.md" in builder
    assert "e112r1_final_analysis_a6_prompt_estimand_scope_20260825.md" in builder
