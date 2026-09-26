from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PLOTTER = ROOT / "ops/exp_scaling/plot_e111_verified_support_mechanism_diagnostic.py"


def _load():
    spec = importlib.util.spec_from_file_location("e111_mechanism_plot_test", PLOTTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audit(plotter) -> dict[str, object]:
    runs = []
    job_id = 1
    for scale in plotter.SCALES:
        for domain in plotter.DOMAINS:
            runs.append(
                {
                    "scale": scale,
                    "domain": domain,
                    "seed": 40 + job_id,
                    "job_id": job_id,
                    "effective_job_id": 100 + job_id,
                    "scheduler_state": "COMPLETED",
                    "completed_causal_chain": job_id <= 11,
                    "violations": [],
                    "report": {
                        "v7_active_min": 1.0,
                        "v5_active_max": 0.0,
                        "v6_active_max": 0.0,
                        "controller_active_max": 0.0,
                        "proposal_rows_to_ppo_max_abs": 0.0,
                        "proposal_objective_outcome_delta_max_abs": 0.0,
                        "semantic_eligible_fraction_max": 0.5,
                        "support_at_least_two_eligible_fraction_max": 0.25,
                        "semantic_rms_max": 0.01,
                        "proposal_cumulative_admissions": 2.0,
                        "replay_gradient_l2_max": 0.1,
                    },
                }
            )
            job_id += 1
    return {
        "schema": "e111_verified_support_discovery_mechanism_gate_audit_v1",
        "terminal": True,
        "passed": True,
        "violations": [],
        "pointmaze": "excluded",
        "outcomes_used_for_gate": False,
        "runs": runs,
    }


def test_e111_plot_requires_terminal_outcome_blind_exact_grid(tmp_path: Path) -> None:
    plotter = _load()
    audit = _audit(plotter)
    input_path = tmp_path / "audit.json"
    input_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(RuntimeError, match="terminal, passing"):
        plotter.build(dict(audit, terminal=False), input_path=input_path)
    with pytest.raises(RuntimeError, match="outcome-blind"):
        plotter.build(dict(audit, outcomes_used_for_gate=True), input_path=input_path)

    payload = plotter.build(audit, input_path=input_path)
    assert payload["pointmaze"] == "excluded"
    assert payload["outcome_metrics_used_for_gate"] is False
    assert len(payload["cells"]) == 15
    assert payload["completed_causal_chain_cells"] == 11
    assert set(payload["invariant_counts"].values()) == {15}

    output = tmp_path / "e111"
    plotter.render(payload, output)
    assert output.with_suffix(".pdf").stat().st_size > 0
    assert output.with_suffix(".png").stat().st_size > 0
    assert json.loads(output.with_suffix(".json").read_text())["cells"] == payload["cells"]

