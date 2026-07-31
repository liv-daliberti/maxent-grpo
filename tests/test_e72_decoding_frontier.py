"""Tests for the E72 decoding-frontier measurement surface.

The frontier re-measures frozen checkpoints under new decoding settings. Two
properties keep those measurements interpretable and are checked here:

1. a cell inherits every non-decoding argument from its source run, and only
   temperature, top_p, K, and draw count vary; and
2. the reproduction gate fails when a cell drifts from the published terminal
   value it is supposed to reproduce.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launcher = _load("e72_launcher", "ops/exp_scaling/launch_e72_decoding_frontier.py")
aggregator = _load("e72_aggregator", "ops/exp_scaling/aggregate_e72_frontier.py")


SOURCE_CONFIG = {
    "prompt_template": "qwen_boxed",
    "test_split": "multi_answer",
    "prompt_data": "/data/graph/train",
    "eval_data": "/data/graph/eval",
    "eval_input_key": "problem",
    "eval_output_key": "answer",
    "prompt_max_length": 256,
    "generate_max_length": 192,
    "eval_generate_max_length": 192,
    "max_model_len": 512,
    "eval_batch_size": 64,
    "eval_temperature": 0.0,
    "eval_mode_coverage_k": 8,
    "eval_mode_coverage_temperature": 1.0,
    "eval_mode_coverage_draws": 4,
    "eval_mode_coverage_seed": 610100,
    "canonical_action_task": "none",
    "canonical_graph_actions": False,
    "canonical_graph_action_count": 3,
    "canonical_graph_learner_sampling": False,
    "canonical_graph_fixed_shape_sampling": False,
    "verifier_version": "fast",
    "num_samples": 16,
    "rollout_batch_size": 1,
    "train_batch_size_per_device": 1,
    "max_train": 384,
    "zero_stage": 2,
    "vllm_gpu_ratio": 0.25,
    "collocate": True,
    "pretrain": "/models/base",
}


def _manifest(**overrides):
    config = {**SOURCE_CONFIG, **overrides}
    return {
        "runs": [
            {
                "domain": "graph_coloring",
                "arm": "drgrpo",
                "seed": 43,
                "export": {"path": "/ckpt/graph_drgrpo_s43/step_04609"},
                "inherited_eval_config": config,
                "published_terminal": {
                    "greedy": 0.3125,
                    "mean8": 0.3125,
                    "pass8": 0.3125,
                    "distinct8": 0.3125,
                    "mean8_draw_se": 0.0,
                    "pass8_draw_se": 0.0,
                    "distinct8_draw_se": 0.0,
                },
            }
        ]
    }


def _job(temperature=1.0, top_p=1.0, k=8, draws=4, **overrides):
    manifest = _manifest(**overrides)
    return {
        "stage": "a",
        "domain": "graph_coloring",
        "arm": "drgrpo",
        "seed": 43,
        "temperature": temperature,
        "top_p": top_p,
        "k": k,
        "draws": draws,
        "source": manifest["runs"][0],
    }


def test_stage_grids_cover_the_registered_sweep():
    stage_a = launcher.stage_cells("a")
    assert [cell[0] for cell in stage_a] == [0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
    assert {cell[1] for cell in stage_a} == {1.0}
    assert {(cell[2], cell[3]) for cell in stage_a} == {(8, 4)}
    # Temperature one at the published decoding surface must be present: it is
    # the cell the reproduction gate is computed from.
    assert (1.0, 1.0, 8, 4) in stage_a
    assert {cell[1] for cell in launcher.stage_cells("c")} == {0.95}
    with pytest.raises(ValueError):
        launcher.stage_cells("z")


def test_cell_identity_is_filesystem_safe_and_distinct():
    tags = {
        launcher.cell_tag(_job(temperature=t, top_p=p))
        for t in (0.5, 1.0, 1.3)
        for p in (1.0, 0.95)
    }
    assert len(tags) == 6
    assert all("." not in tag and "/" not in tag for tag in tags)
    assert launcher.cell_tag(_job(temperature=1.3, top_p=0.95)) == "T1p3_p0p95_K8_d4"


def test_decoding_settings_are_swept_and_everything_else_inherited(tmp_path):
    job = _job(temperature=1.6, top_p=0.95, k=32, draws=2)
    env = launcher.build_export_vars(ROOT, job, tmp_path)

    # The swept quantities take the cell's values, not the source run's.
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE"] == "1.6"
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P"] == "0.95"
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_K"] == "32"
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS"] == "2"

    # Everything that defines the measurement is inherited verbatim.
    assert env["OAT_ZERO_EVAL_MODE_COVERAGE_SEED"] == "610100"
    assert env["OAT_ZERO_PROMPT_TEMPLATE"] == "qwen_boxed"
    assert env["OAT_ZERO_TEST_SPLIT"] == "multi_answer"
    assert env["OAT_ZERO_EVAL_DATA"] == "/data/graph/eval"
    assert env["OAT_ZERO_EVAL_GENERATE_MAX_LENGTH"] == "192"
    assert env["OAT_ZERO_MAX_MODEL_LEN"] == "512"

    # The checkpoint under test replaces the source run's base model.
    assert env["OAT_ZERO_PRETRAIN"] == "/ckpt/graph_drgrpo_s43/step_04609"
    assert env["OAT_ZERO_EVAL_ONLY"] == "1"


def test_nothing_is_trained_stored_or_recovered(tmp_path):
    env = launcher.build_export_vars(ROOT, _job(), tmp_path)
    assert env["OAT_ZERO_SAVE_CKPT"] == "0"
    assert env["OAT_ZERO_EXPORT_STEPS"] == "-1"
    assert env["OAT_ZERO_AUTO_RESUME"] == "0"
    assert env["OAT_ZERO_WATCHDOG_REQUEUE"] == "0"
    # A frontier cell must not silently inherit a training objective.
    assert env["OAT_ZERO_VARIANT"] == "grpo"


def test_canonical_action_domains_pin_the_audited_vllm_engine(tmp_path):
    plain = launcher.build_export_vars(ROOT, _job(), tmp_path)
    assert "VLLM_USE_V1" not in plain

    pantry = launcher.build_export_vars(
        ROOT,
        _job(canonical_action_task="pantry_support_mask", canonical_graph_learner_sampling=True),
        tmp_path,
    )
    assert pantry["VLLM_USE_V1"] == "0"
    assert pantry["OAT_ZERO_CANONICAL_ACTION_TASK"] == "pantry_support_mask"
    assert pantry["OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING"] == "1"


def test_cells_are_pinned_to_the_node_that_trained_the_checkpoint():
    job = _job()
    job["source"]["source_node"] = "node302"
    assert launcher.resolve_nodelist(job) == "node302"
    # An explicit pool overrides the pin only when asked for.
    assert launcher.resolve_nodelist(job, "node105") == "node105"


def test_missing_source_node_is_an_error_not_a_free_placement():
    job = _job()
    job["source"].pop("source_node", None)
    with pytest.raises(SystemExit):
        launcher.resolve_nodelist(job)


def _cell(**overrides):
    cell = {
        "domain": "graph_coloring",
        "arm": "drgrpo",
        "seed": 43,
        "k": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "greedy": 0.3125,
        "mean": 0.3125,
        "pass": 0.3125,
        "distinct": 0.3125,
        "cell_dir": "/cells/x",
    }
    cell.update(overrides)
    return cell


def test_reproduction_gate_passes_on_an_exact_reproduction():
    gate = aggregator.reproduction_gate([_cell()], _manifest())
    assert gate["passed"] is True
    assert gate["checked"] == 4 and gate["failed"] == 0


def test_reproduction_gate_catches_drift_beyond_tolerance():
    # Well inside the 0.02 rate floor: still a pass.
    assert aggregator.reproduction_gate([_cell(mean=0.3225)], _manifest())["passed"]
    # Beyond it: the cell is not reproducing the published number.
    drifted = aggregator.reproduction_gate([_cell(mean=0.40)], _manifest())
    assert drifted["passed"] is False
    assert any(check["metric"] == "mean" for check in drifted["failures"])
    # Mode counts get the wider floor, so 0.04 passes but 0.10 does not.
    assert aggregator.reproduction_gate([_cell(distinct=0.3525)], _manifest())["passed"]
    assert not aggregator.reproduction_gate([_cell(distinct=0.45)], _manifest())["passed"]


def test_gate_only_reads_the_published_decoding_surface():
    # Cells away from (T=1, top_p=1, K=8) are frontier points, not gate points.
    off_surface = [
        _cell(temperature=1.6, mean=0.9),
        _cell(top_p=0.95, mean=0.9),
        _cell(k=32, mean=0.9),
    ]
    gate = aggregator.reproduction_gate(off_surface, _manifest())
    assert gate["checked"] == 0
    # An empty gate is not a passing gate.
    assert gate["passed"] is False


def test_gate_fails_when_a_cell_has_no_published_reference():
    orphan = _cell(domain="countdown")
    gate = aggregator.reproduction_gate([orphan], _manifest())
    assert gate["passed"] is False
    assert gate["failures"][0]["status"] == "no_published_reference"


def test_temperature_repair_index_is_relative_to_xgrpo_at_temperature_one():
    cells = [
        _cell(arm="drgrpo", temperature=1.0, distinct=0.32, mean=0.31),
        _cell(arm="drgrpo", temperature=1.6, distinct=0.80, mean=0.12),
        _cell(arm="xgrpo", temperature=1.0, distinct=2.40, mean=0.43),
    ]
    summary = aggregator.frontier_summary(cells)
    repair = {row["arm"]: row for row in summary["temperature_repair"]}

    baseline = repair["drgrpo"]
    assert baseline["best_temperature"] == 1.6
    assert baseline["best_distinct_at_8"] == pytest.approx(0.80)
    # Reported with the accuracy it was bought at, so a repair that costs the
    # policy its correctness is visible rather than flattering.
    assert baseline["accuracy_at_best"] == pytest.approx(0.12)
    assert baseline["repair_index_rho"] == pytest.approx(0.80 / 2.40)
    assert repair["xgrpo"]["repair_index_rho"] == pytest.approx(1.0)


def test_frontier_points_average_over_seeds_and_keep_seed_rows():
    cells = [
        _cell(seed=43, temperature=1.0, distinct=2.0, mean=0.40),
        _cell(seed=44, temperature=1.0, distinct=3.0, mean=0.50),
    ]
    point = aggregator.frontier_summary(cells)["points"][0]
    assert point["n_seeds"] == 2 and point["seeds"] == [43, 44]
    assert point["distinct_at_k"] == pytest.approx(2.5)
    assert point["mean_at_k"] == pytest.approx(0.45)
    assert point["per_seed"][43]["distinct_at_k"] == pytest.approx(2.0)


def test_base_cells_are_gated_against_pass_zero_not_terminal():
    # The pre-RL policy has no terminal row; checking it against one would
    # either fail spuriously or, worse, pass against the wrong reference.
    base_cell = _cell(arm="base_node302", seed=0, mean=0.173)
    base_cell["pass"] = 0.436
    base_cell["distinct"] = 0.523
    base_cell["greedy"] = 0.312
    reference = {
        ("graph_coloring", "base_node302"): {
            "greedy": 0.312,
            "mean8": 0.173,
            "pass8": 0.436,
            "distinct8": 0.523,
            "mean8_draw_se": 0.0,
            "pass8_draw_se": 0.0,
            "distinct8_draw_se": 0.0,
        }
    }
    gate = aggregator.reproduction_gate([base_cell], _manifest(), reference)
    assert gate["passed"] is True and gate["checked"] == 4

    # Same cell, wrong GPU group's reference: the hardware difference at pass 0
    # is large enough that the gate must notice.
    drifted = dict(reference)
    drifted[("graph_coloring", "base_node302")] = {
        **reference[("graph_coloring", "base_node302")],
        "distinct8": 0.713,
    }
    assert not aggregator.reproduction_gate([base_cell], _manifest(), drifted)["passed"]


def test_base_cells_without_a_pass_zero_reference_fail_closed():
    base_cell = _cell(arm="base_node999", seed=0)
    gate = aggregator.reproduction_gate([base_cell], _manifest(), {})
    assert gate["passed"] is False
    assert gate["failures"][0]["status"] == "no_published_reference"


def test_truncated_cells_are_excluded_from_the_temperature_frontier():
    # top_p is a separate axis; mixing it into the temperature sweep would
    # attribute a truncation effect to temperature.
    cells = [
        _cell(temperature=1.0, top_p=1.0, distinct=1.0),
        _cell(temperature=1.0, top_p=0.95, distinct=9.0),
    ]
    points = aggregator.frontier_summary(cells)["points"]
    assert len(points) == 1
    assert points[0]["distinct_at_k"] == pytest.approx(1.0)
