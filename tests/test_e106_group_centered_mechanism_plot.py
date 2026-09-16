from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLOTTER = ROOT / "ops/exp_scaling/plot_e106_group_centered_mechanism_diagnostic.py"


def _load():
    spec = importlib.util.spec_from_file_location("e106_mechanism_plot_test", PLOTTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit(plotter) -> dict[str, object]:
    runs = []
    job_id = 1
    for scale_index, scale in enumerate(plotter.SCALES):
        for domain_index, domain in enumerate(plotter.DOMAINS):
            zero = (scale_index, domain_index) in {(0, 2), (2, 3)}
            runs.append(
                {
                    "source": "synthetic_mechanism_only",
                    "scale": scale,
                    "domain": domain,
                    "seed": 40 + job_id,
                    "job_id": job_id,
                    "scheduler_state": "COMPLETED",
                    "completion_receipt_step": 65,
                    "runtime_complete": True,
                    "target_steps": 64,
                    "report": {
                        "group_centered_active_min": 1.0,
                        "legacy_active_max": 0.0,
                        "controller_active_max": 0.0,
                        "semantic_eligible_fraction_max": 0.5,
                        "support_at_least_two_prompt_fraction_max": 0.25,
                        "semantic_rms_max": 0.0 if zero else 0.01,
                        "semantic_effective_mean_abs_max": 1e-18,
                        "replay_gradient_l2_max": 0.1,
                    },
                }
            )
            job_id += 1
    return {
        "schema": "e106_python_lambda_normalization_combined_gate_v1",
        "complete": True,
        "passed": True,
        "pointmaze": "excluded",
        "mechanism_gate_used_outcome_metrics": False,
        "post_update_outcome_metrics_inspected": False,
        "runs": runs,
    }


def test_mechanism_plot_requires_passed_exact_grid_and_renders_zero_cells(tmp_path):
    plotter = _load()
    audit = _audit(plotter)
    input_path = tmp_path / "gate.json"
    input_path.write_text(json.dumps(audit), encoding="utf-8")

    incomplete = dict(audit, complete=False, passed=False)
    with pytest.raises(RuntimeError, match="passed 15-cell gate"):
        plotter.plot_payload(incomplete, input_path=input_path)

    payload = plotter.plot_payload(audit, input_path=input_path)
    assert payload["schema"] == "paper-e106-group-centered-mechanism-diagnostic-v1"
    assert payload["pointmaze"] == "excluded"
    assert payload["outcome_metrics_used"] is False
    assert len(payload["cells"]) == 15
    assert len(payload["zero_semantic_pressure_cells"]) == 2
    assert set(payload["invariant_counts"].values()) == {15}

    output = tmp_path / "mechanism"
    plotter.render(payload, output)
    assert output.with_suffix(".pdf").stat().st_size > 0
    assert output.with_suffix(".png").stat().st_size > 0
    saved = json.loads(output.with_suffix(".json").read_text())
    assert saved["cells"] == payload["cells"]
