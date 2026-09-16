from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / (
    "ops/exp_scaling/build_e105_group_centered_semantic_repair_results.py"
)
LAUNCHER = ROOT / (
    "ops/exp_scaling/launch_e105_group_centered_semantic_repair_full_three_scale.py"
)
PROTOCOL = ROOT / (
    "paper/preregistration/e105_paired_analysis_specification_20260817.md"
)
PLOTTER = ROOT / (
    "ops/exp_scaling/plot_e105_group_centered_semantic_endpoint_effects.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("e105_results", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_complete_run(run_dir: Path, *, conflict: bool = False) -> None:
    attempt = run_dir / "debug_job1"
    results = attempt / "eval_results"
    results.mkdir(parents=True)
    sampled = attempt / "eval_mode_coverage_draws.jsonl"
    rows = []
    for step in (0, 2, 4):
        rows.append(
            {
                "benchmark": "multi_answer",
                "draw_index": None,
                "evaluation_kind": "deterministic_greedy_trace_neutral",
                "sample_count": 1,
                "schema_version": 1,
                "seed": 0,
                "step": step,
                "temperature": 0.0,
                "metrics": {},
            }
        )
        for draw in range(4):
            prompts = [
                {
                    "answer_keys": ["a"],
                    "answer_mode_count": 1,
                    "option_ids": [0],
                    "prompt": f"prompt-{index}",
                    "prompt_index": index,
                    "reference": "a",
                    "request_seeds_by_option": [700 + draw],
                }
                for index in range(2)
            ]
            rows.append(
                {
                    "benchmark": "multi_answer",
                    "evaluation_kind": "fixed_seed_sampled_k_neutral",
                    "step": step,
                    "draw_index": draw,
                    "sample_count": 8,
                    "schema_version": 1,
                    "seed": 700 + draw,
                    "temperature": 1.0,
                    "prompts": prompts,
                    "metrics": {
                        "any_correct_at_k": 0.2 + step / 10 + draw / 100,
                        "mean_at_k": 0.1 + step / 20 + draw / 100,
                        "distinct_correct_modes_at_k": 0.4 + step / 5 + draw / 100,
                    },
                }
            )
        greedy = [
            {"scores": [1.0]},
            {"scores": [float(step >= 2)]},
        ]
        (results / f"{step}_multi_answer.json").write_text(
            json.dumps(greedy), encoding="utf-8"
        )
    sampled.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    if conflict:
        retry = run_dir / "debug_job2"
        retry.mkdir()
        bad = dict(rows[-1])
        bad["metrics"] = dict(bad["metrics"], any_correct_at_k=0.99)
        (retry / "eval_mode_coverage_draws.jsonl").write_text(
            json.dumps(bad) + "\n", encoding="utf-8"
        )


def test_complete_curve_contains_every_registered_metric_and_checkpoint(tmp_path):
    module = _load()
    _write_complete_run(tmp_path)

    curve, sources = module.run_curve(tmp_path, steps=(0, 2, 4))

    assert tuple(curve) == (0, 2, 4)
    assert set(curve[4]) == set(module.TERMINAL_METRICS)
    assert curve[0]["greedy_pass1"] == 0.5
    assert curve[4]["greedy_pass1"] == 1.0
    assert curve[4]["sampled_excess8"] == pytest.approx(
        curve[4]["sampled_distinct8"] - curve[4]["sampled_pass8"]
    )
    assert module.normalized_auc(curve, metric="greedy_pass1", target=4) == 0.875
    assert len(sources) == 4


def test_sampled_reader_rejects_conflicting_resume_rows(tmp_path):
    module = _load()
    _write_complete_run(tmp_path, conflict=True)

    with pytest.raises(RuntimeError, match="conflicting duplicate sampled row"):
        module.sampled_curve(tmp_path, steps=(0, 2, 4))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("benchmark", "wrong"),
        ("sample_count", 7),
        ("schema_version", 2),
        ("seed", 999),
        ("temperature", 0.5),
    ),
)
def test_sampled_reader_rejects_row_contract_drift(
    tmp_path: Path, field: str, bad_value: object
):
    module = _load()
    _write_complete_run(tmp_path)
    contract = {
        "benchmark": "multi_answer",
        "sample_count": 8,
        "schema_version": 1,
        "seed_base": 700,
        "temperature": 1.0,
    }
    module.sampled_curve(
        tmp_path,
        steps=(0, 2, 4),
        sampled_contract=contract,
    )

    sampled = tmp_path / "debug_job1/eval_mode_coverage_draws.jsonl"
    rows = [json.loads(line) for line in sampled.read_text().splitlines()]
    target = next(
        row
        for row in rows
        if row["evaluation_kind"] == "fixed_seed_sampled_k_neutral"
    )
    target[field] = bad_value
    sampled.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="sampled evaluation contract drifted"):
        module.sampled_curve(
            tmp_path,
            steps=(0, 2, 4),
            sampled_contract=contract,
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("benchmark", "wrong"),
        ("draw_index", 0),
        ("sample_count", 8),
        ("schema_version", 2),
        ("seed", 1),
        ("temperature", 1.0),
    ),
)
def test_reader_rejects_greedy_trace_contract_drift(
    tmp_path: Path, field: str, bad_value: object
):
    module = _load()
    _write_complete_run(tmp_path)
    contract = {
        "benchmark": "multi_answer",
        "draw_index": None,
        "sample_count": 1,
        "schema_version": 1,
        "seed": 0,
        "temperature": 0.0,
    }
    module.run_curve(
        tmp_path,
        steps=(0, 2, 4),
        greedy_contract=contract,
    )

    sampled = tmp_path / "debug_job1/eval_mode_coverage_draws.jsonl"
    rows = [json.loads(line) for line in sampled.read_text().splitlines()]
    target = next(
        row
        for row in rows
        if row["evaluation_kind"] == "deterministic_greedy_trace_neutral"
    )
    target[field] = bad_value
    sampled.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError, match="deterministic_greedy_trace_neutral contract drifted"
    ):
        module.run_curve(
            tmp_path,
            steps=(0, 2, 4),
            greedy_contract=contract,
        )


def test_prompt_surface_hashes_only_response_free_paired_identity(tmp_path: Path):
    module = _load()
    _write_complete_run(tmp_path)
    contract = {
        "benchmark": "multi_answer",
        "sample_count": 8,
        "schema_version": 1,
        "seed_base": 700,
        "temperature": 1.0,
    }
    surface = module.sampled_prompt_surface(
        tmp_path,
        steps=(0, 2, 4),
        draws=range(4),
        sampled_contract=contract,
    )
    assert surface["prompt_count"] == 2
    assert len(surface["prompt_sha256"]) == 64
    assert set(surface["request_sha256_by_draw"]) == {"0", "1", "2", "3"}

def test_prompt_surface_excludes_generated_answer_keys(tmp_path: Path):
    module = _load()
    _write_complete_run(tmp_path)
    sampled = tmp_path / "debug_job1/eval_mode_coverage_draws.jsonl"
    rows = [json.loads(line) for line in sampled.read_text().splitlines()]
    for row in rows:
        if row["evaluation_kind"] != "fixed_seed_sampled_k_neutral":
            continue
        for prompt_index, prompt in enumerate(row["prompts"]):
            prompt["answer_keys"] = [
                f"generated-{row['step']}-{row['draw_index']}-{prompt_index}"
            ]
    sampled.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    surface = module.sampled_prompt_surface(
        tmp_path,
        steps=(0, 2, 4),
        draws=range(4),
        sampled_contract={
            "benchmark": "multi_answer",
            "sample_count": 8,
            "schema_version": 1,
            "seed_base": 700,
            "temperature": 1.0,
        },
    )

    assert surface["prompt_count"] == 2
    assert len(surface["prompt_sha256"]) == 64


@pytest.mark.parametrize("field", ("prompt", "request_seeds_by_option"))
def test_prompt_surface_rejects_grid_drift(tmp_path: Path, field: str):
    module = _load()
    _write_complete_run(tmp_path)
    sampled = tmp_path / "debug_job1/eval_mode_coverage_draws.jsonl"
    rows = [json.loads(line) for line in sampled.read_text().splitlines()]
    target = next(
        row
        for row in rows
        if row["evaluation_kind"] == "fixed_seed_sampled_k_neutral"
        and row["step"] == 2
        and row["draw_index"] == 1
    )
    target["prompts"][0][field] = "drift"
    sampled.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="surface changed"):
        module.sampled_prompt_surface(
            tmp_path,
            steps=(0, 2, 4),
            draws=range(4),
            sampled_contract={
                "benchmark": "multi_answer",
                "sample_count": 8,
                "schema_version": 1,
                "seed_base": 700,
                "temperature": 1.0,
            },
        )


def test_prompt_surface_uses_registered_row_seed_when_nested_seeds_absent(
    tmp_path: Path,
):
    module = _load()
    _write_complete_run(tmp_path)
    sampled = tmp_path / "debug_job1/eval_mode_coverage_draws.jsonl"
    rows = [json.loads(line) for line in sampled.read_text().splitlines()]
    for row in rows:
        if row["evaluation_kind"] != "fixed_seed_sampled_k_neutral":
            continue
        for prompt in row["prompts"]:
            prompt["request_seeds_by_option"] = []
    sampled.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    surface = module.sampled_prompt_surface(
        tmp_path,
        steps=(0, 2, 4),
        draws=range(4),
        sampled_contract={
            "benchmark": "multi_answer",
            "sample_count": 8,
            "schema_version": 1,
            "seed_base": 700,
            "temperature": 1.0,
        },
    )

    assert len(set(surface["request_sha256_by_draw"].values())) == 4


def test_paired_summary_requires_exactly_five_registered_seeds():
    module = _load()
    summary = module.paired_summary(
        {43: 0.1, 44: 0.2, 45: 0.3, 46: 0.4, 47: 0.5},
        seeds=(43, 44, 45, 46, 47),
    )
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(0.3)
    assert "student_t_95" in summary
    with pytest.raises(RuntimeError, match="expected five paired seeds"):
        module.paired_summary({43: 0.1}, seeds=(43,))


def test_builder_uses_e109_only_for_python_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load()
    historical = {}
    historical_index = {}
    for scale, seeds in module.SCALE_SEEDS.items():
        runs = []
        for domain in module.DOMAINS:
            for seed in seeds:
                run = {
                    "scale": scale,
                    "domain": domain,
                    "seed": seed,
                    "arm": "replay",
                    "job_id": seed * 100 + len(runs),
                    "run_stamp": f"historical_{scale}_{domain}_s{seed}",
                    "run_dir": str(tmp_path / f"historical-{scale}-{domain}-{seed}"),
                }
                runs.append(run)
                historical_index[(scale, domain, seed)] = run
        historical[scale] = {"released": True, "runs": runs}
    historical_paths = {}
    for scale, payload in historical.items():
        path = tmp_path / f"{scale}-historical.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        historical_paths[scale] = path
    monkeypatch.setattr(module, "COMPARATOR_LEDGERS", historical_paths)

    repaired_runs = []
    repaired_index = {}
    for scale, seeds in module.SCALE_SEEDS.items():
        for seed in seeds:
            run = {
                "scale": scale,
                "domain": "python_factors",
                "seed": seed,
                "arm": "replay",
                "job_id": 9_000_000 + seed,
                "run_stamp": f"e109_{scale}_python_s{seed}",
                "run_dir": str(tmp_path / f"e109-{scale}-{seed}"),
            }
            repaired_runs.append(run)
            repaired_index[(scale, "python_factors", seed)] = run
    repaired = {
        "schema": "e109_repaired_python_replay_comparators_jobs_v1",
        "released": True,
        "pointmaze": "excluded",
        "domain": "python_factors",
        "semantic_coefficient": 0.0,
        "replay_weight": 0.1,
        "snapshot_sha256": "snapshot",
        "parser_surface_version": "surface",
        "target_steps": 3_072,
        "checkpoint_interval_steps": 192,
        "qwen3_a6000_seeds": [73, 74],
        "qwen3_paired_placement_artifact": str(tmp_path / "placement.json"),
        "qwen3_paired_placement_artifact_sha256": "placement-hash",
        "scales": list(module.SCALE_SEEDS),
        "seeds": {
            scale: list(seeds) for scale, seeds in module.SCALE_SEEDS.items()
        },
        "protocol": str(module.REPAIRED_PYTHON_COMPARATOR_PROTOCOL),
        "protocol_sha256": module.sha256(
            module.REPAIRED_PYTHON_COMPARATOR_PROTOCOL
        ),
        "launcher_sha256": module.sha256(
            module.REPAIRED_PYTHON_COMPARATOR_LAUNCHER
        ),
        "runs": repaired_runs,
    }
    repaired_ledger = tmp_path / "e109.json"
    repaired_ledger.write_text(json.dumps(repaired), encoding="utf-8")
    monkeypatch.setattr(
        module, "REPAIRED_PYTHON_COMPARATOR_LEDGER", repaired_ledger
    )

    treatment_runs = []
    for scale, seeds in module.SCALE_SEEDS.items():
        for domain in module.DOMAINS:
            for seed in seeds:
                key = (scale, domain, seed)
                comparator = (
                    repaired_index[key]
                    if domain == "python_factors"
                    else historical_index[key]
                )
                treatment_runs.append(
                    {
                        "scale": scale,
                        "domain": domain,
                        "seed": seed,
                        "arm": "semantic_group_centered",
                        "paired_replay": {
                            field: comparator[field]
                            for field in ("run_stamp", "run_dir", "job_id")
                        },
                    }
                )
    treatment = {
        "schema": "e105_group_centered_semantic_repair_full_three_scale_jobs_v1",
        "released": True,
        "pointmaze": "excluded",
        "models": list(module.SCALE_SEEDS),
        "domains": list(module.DOMAINS),
        "arms": ["semantic_group_centered"],
        "seeds": {
            scale: list(seeds) for scale, seeds in module.SCALE_SEEDS.items()
        },
        "historical_comparator_ledgers": {
            scale: {"path": str(path), "sha256": module.sha256(path)}
            for scale, path in historical_paths.items()
        },
        "python_comparator_repaired": True,
        "python_comparator_amendment": str(
            module.REPAIRED_PYTHON_COMPARATOR_PROTOCOL
        ),
        "python_comparator_amendment_sha256": module.sha256(
            module.REPAIRED_PYTHON_COMPARATOR_PROTOCOL
        ),
        "repaired_python_comparator_ledger": str(repaired_ledger),
        "repaired_python_comparator_ledger_sha256": module.sha256(
            repaired_ledger
        ),
        "replay_weight": 0.1,
        "snapshot_sha256": "snapshot",
        "python_response_surface_version": "surface",
        "target_steps": 3_072,
        "checkpoint_interval_steps": 192,
        "qwen3_paired_placement_artifact": str(tmp_path / "placement.json"),
        "qwen3_paired_placement_artifact_sha256": "placement-hash",
        "runs": treatment_runs,
    }

    treatment_index, replay_index = module._validate_ledgers(
        treatment, historical, repaired
    )
    assert len(treatment_index) == 75
    assert len(replay_index) == 75
    assert replay_index[("qwen3b", "python_factors", 74)][
        "run_stamp"
    ] == "e109_qwen3b_python_s74"
    assert replay_index[("qwen3b", "graph_coloring", 74)][
        "run_stamp"
    ] == "historical_qwen3b_graph_coloring_s74"

    historical_python = historical_index[("qwen3b", "python_factors", 74)]
    target = next(
        run
        for run in treatment["runs"]
        if run["scale"] == "qwen3b"
        and run["domain"] == "python_factors"
        and run["seed"] == 74
    )
    target["paired_replay"] = {
        field: historical_python[field]
        for field in ("run_stamp", "run_dir", "job_id")
    }
    with pytest.raises(RuntimeError, match="paired comparator binding drifted"):
        module._validate_ledgers(treatment, historical, repaired)

    target["paired_replay"] = {
        field: repaired_index[("qwen3b", "python_factors", 74)][field]
        for field in ("run_stamp", "run_dir", "job_id")
    }
    with pytest.raises(RuntimeError, match="activates semantic MaxEnt"):
        module._validate_ledgers(
            treatment, historical, dict(repaired, semantic_coefficient=0.1)
        )

    with pytest.raises(RuntimeError, match="placement artifact drifted"):
        module._validate_ledgers(
            treatment,
            historical,
            dict(
                repaired,
                qwen3_paired_placement_artifact_sha256="other-placement",
            ),
        )


def test_analysis_is_frozen_complete_paired_and_static_only():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    builder = BUILDER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "before E105 submission" in protocol
    assert "all 75 E105 treatment cells" in protocol
    assert "all four" in protocol
    assert "Missing cells, checkpoints, draws" in protocol
    assert "at least 8 of the 15 families" in protocol
    assert "PointMaze" in protocol and "excluded" in protocol
    assert "best-checkpoint" in protocol
    assert "E109 repaired-comparator protocol" in protocol
    assert "REPAIRED_PYTHON_COMPARATOR_LEDGER" in builder
    assert "replay_index.update(repaired_index)" in builder
    assert "analysis_protocol_sha256" in launcher
    assert "analysis_builder_sha256" in launcher
    assert "analysis_plotter_sha256" in launcher
    assert "qwen3_paired_placement_artifact_sha256" in launcher
    assert "_validate_qwen3_placement_provenance" in builder
    assert "qwen3_paired_placement" in builder
    assert "paired_a6000_cells" in builder
    assert "existing 3-by-5 baseline" in protocol
    assert "len(all_terminal_pass_effects) != 75" in builder
    assert '"pointmaze": "excluded"' in builder


def test_endpoint_forest_materializes_all_fifteen_exact_n_cells(tmp_path):
    module = _load()
    spec = importlib.util.spec_from_file_location("e105_plot", PLOTTER)
    assert spec is not None and spec.loader is not None
    plotter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plotter)
    families = {}
    for scale, seeds in module.SCALE_SEEDS.items():
        families[scale] = {}
        for domain in module.DOMAINS:
            effects = {}
            for metric in (
                "terminal_sampled_pass8",
                "terminal_sampled_excess8",
            ):
                values = {seed: (seed - seeds[0]) / 100.0 for seed in seeds}
                effects[metric] = module.paired_summary(values, seeds=seeds)
            families[scale][domain] = {
                "paired_seeds": list(seeds),
                "paired_effects": effects,
            }
    result = {
        "schema": "e105_group_centered_semantic_repair_results_v1",
        "pointmaze": "excluded",
        "analysis_plotter": str(PLOTTER.resolve()),
        "analysis_plotter_sha256": plotter.sha256(PLOTTER),
        "design": {
            "scales": list(module.SCALE_SEEDS),
            "domains": list(module.DOMAINS),
        },
        "families": families,
        "registered_general_extension_criterion": {
            "successful_general_extension": True
        },
    }
    input_path = tmp_path / "results.json"
    input_path.write_text(json.dumps(result), encoding="utf-8")

    payload = plotter.plot_payload(result, input_path=input_path)

    assert payload["pointmaze"] == "excluded"
    assert len(payload["cells"]) == 15
    assert all(cell["n"] == 5 for cell in payload["cells"])
    assert payload["baseline"] == "matched ReplayDr.GRPO"
    output = tmp_path / "e105_endpoint_forest"
    plotter.render(payload, output)
    assert output.with_suffix(".pdf").stat().st_size > 0
    assert output.with_suffix(".png").stat().st_size > 0
    assert output.with_suffix(".json").stat().st_size > 0
