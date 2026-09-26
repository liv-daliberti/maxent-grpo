from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / (
    "ops/exp_scaling/build_e105_group_centered_semantic_repair_results.py"
)
PLOTTER = ROOT / (
    "ops/exp_scaling/plot_e105_group_centered_semantic_endpoint_effects.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(per_seed: dict[str, float]) -> dict[str, object]:
    values = list(per_seed.values())
    mean = sum(values) / len(values)
    return {
        "mean": mean,
        "n": 5,
        "student_t_95": [mean - 0.025, mean + 0.025],
        "range": [min(values), max(values)],
        "per_seed": per_seed,
    }


def _synthetic_result(plotter) -> dict[str, object]:
    families: dict[str, object] = {}
    effect_metrics = (
        "terminal_sampled_pass8",
        "terminal_sampled_excess8",
        "auc_sampled_pass8",
        "auc_sampled_excess8",
    )
    for scale_index, scale in enumerate(plotter.SCALES):
        families[scale] = {}
        seeds = list(range(10 * scale_index + 1, 10 * scale_index + 6))
        for domain_index, domain in enumerate(plotter.DOMAINS):
            summaries = {}
            for metric_index, metric in enumerate(effect_metrics):
                per_seed = {
                    str(seed): (
                        0.01 * (scale_index + 1)
                        + 0.002 * domain_index
                        + 0.001 * metric_index
                        + 0.0001 * seed
                    )
                    for seed in seeds
                }
                summaries[metric] = _summary(per_seed)
            families[scale][domain] = {
                "paired_seeds": seeds,
                "paired_effects": summaries,
            }
    return {
        "schema": "e105_group_centered_semantic_repair_results_v1",
        "pointmaze": "excluded",
        "analysis_plotter": str(PLOTTER.resolve()),
        "analysis_plotter_sha256": plotter.sha256(PLOTTER),
        "design": {
            "scales": list(plotter.SCALES),
            "domains": list(plotter.DOMAINS),
        },
        "families": families,
        "registered_general_extension_criterion": {
            "successful_general_extension": True,
        },
    }


def test_builder_uses_all_seventeen_registered_checkpoints_for_normalized_auc():
    builder = _load(BUILDER, "e105_result_builder_test")
    steps = builder.registered_steps(interval=192, target=3_072)
    assert len(steps) == 17
    assert steps == tuple(range(0, 3_073, 192))
    curve = {
        step: {"sampled_pass8": step / 3_072}
        for step in steps
    }
    assert builder.normalized_auc(
        curve,
        metric="sampled_pass8",
        target=3_072,
    ) == pytest.approx(0.5)
    with pytest.raises(RuntimeError, match="does not span"):
        builder.normalized_auc(
            {step: curve[step] for step in steps[:-1]},
            metric="sampled_pass8",
            target=3_072,
        )


def test_dual_forest_payloads_and_assets_cover_all_fifteen_families(tmp_path):
    plotter = _load(PLOTTER, "e105_dual_plotter_test")
    result = _synthetic_result(plotter)
    input_path = tmp_path / "result.json"
    input_path.write_text(json.dumps(result), encoding="utf-8")

    endpoint = plotter.plot_payload(
        result,
        input_path=input_path,
        effect_kind="terminal",
    )
    auc = plotter.plot_payload(
        result,
        input_path=input_path,
        effect_kind="auc",
    )
    assert endpoint["schema"] == (
        "paper-e105-group-centered-semantic-endpoint-effects-v1"
    )
    assert auc["schema"] == "paper-e105-group-centered-semantic-auc-effects-v1"
    assert endpoint["metric_order"] == [
        "terminal_sampled_pass8",
        "terminal_sampled_excess8",
    ]
    assert auc["metric_order"] == [
        "auc_sampled_pass8",
        "auc_sampled_excess8",
    ]
    for payload in (endpoint, auc):
        assert payload["pointmaze"] == "excluded"
        assert len(payload["cells"]) == 15
        assert all(cell["n"] == 5 for cell in payload["cells"])

    endpoint_output = tmp_path / "endpoint"
    auc_output = tmp_path / "auc"
    plotter.render(endpoint, endpoint_output)
    plotter.render(auc, auc_output)
    for output, schema in (
        (
            endpoint_output,
            "paper-e105-group-centered-semantic-endpoint-effects-v1",
        ),
        (auc_output, "paper-e105-group-centered-semantic-auc-effects-v1"),
    ):
        assert output.with_suffix(".pdf").stat().st_size > 0
        assert output.with_suffix(".png").stat().st_size > 0
        payload = json.loads(output.with_suffix(".json").read_text())
        assert payload["schema"] == schema
