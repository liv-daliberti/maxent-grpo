from __future__ import annotations

import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49w_bottom_up_route_calibration.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("e49w_route_calibration", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proposal():
    return {
        "menu": {
            "schema": "math_strategy_action_menu_v1",
            "actions": [
                {"action_id": "A1", "operation": "Factor the polynomial."},
                {"action_id": "A2", "operation": "Use a sign chart."},
                {"action_id": "A3", "operation": "Complete the square."},
                {"action_id": "A4", "operation": "Use the vertex form."},
            ],
            "strategies": [
                {
                    "strategy_id": "S1",
                    "action_ids": ["A1", "A2"],
                    "plan": "Factor, then determine the allowed intervals.",
                },
                {
                    "strategy_id": "S2",
                    "action_ids": ["A3", "A4"],
                    "plan": "Use vertex form to determine the same domain.",
                },
            ],
        },
        "route_sources": [
            {
                "strategy_id": "S1",
                "source_kind": "observed",
                "exemplar_ids": ["sample-a"],
            },
            {
                "strategy_id": "S2",
                "source_kind": "proposed",
                "exemplar_ids": [],
            },
        ],
    }


def test_e49w_candidate_cohort_is_frozen_level5_and_has_19_rows():
    module = _load()
    candidates = module._load_candidates()
    assert len(candidates) == 19
    assert all(candidate["level"] == 5 for candidate in candidates)
    assert all(
        candidate["validator_positive_count"] >= 2
        for candidate in candidates
    )
    observed = [
        (candidate["validator_positive_count"], candidate["problem_id"])
        for candidate in candidates
    ]
    assert observed == sorted(observed, key=lambda row: (-row[0], row[1]))


def test_e49w_menu_requires_observed_s1_and_exact_exemplar_binding():
    module = _load()
    proposal = _proposal()
    assert module._proposal_passes_local_contract(proposal, {"sample-a"})

    proposal["route_sources"][0]["exemplar_ids"] = ["unknown"]
    assert not module._proposal_passes_local_contract(proposal, {"sample-a"})

    proposal = _proposal()
    proposal["route_sources"][0]["source_kind"] = "proposed"
    proposal["route_sources"][0]["exemplar_ids"] = []
    assert not module._proposal_passes_local_contract(proposal, {"sample-a"})


def test_e49w_audit_requires_sound_distinct_nonleaking_routes():
    module = _load()
    proposal = _proposal()
    audit = {
        "finish_reason": "stop",
        "response_id": "response-1",
        "assessment": {
            "strategy_assessments": [
                {
                    "strategy_id": strategy_id,
                    "status": "sound",
                    "failure_code": "none",
                    "derived_answer": "audited value",
                    "matches_reference_answer": True,
                    "all_actions_sufficient": True,
                    "missing_decisive_step": False,
                    "menu_reveals_final_answer": False,
                    "actions_concrete_for_small_model": True,
                }
                for strategy_id in ("S1", "S2")
            ],
            "source_assessments": [
                {
                    "strategy_id": "S1",
                    "binding_status": "observed_bound",
                },
                {
                    "strategy_id": "S2",
                    "binding_status": "proposed_not_applicable",
                },
            ],
            "pair": {
                "relation": "distinct",
                "shared_core": "same target",
                "s1_decisive_operation": "factorization",
                "s2_decisive_operation": "vertex analysis",
            },
        },
    }
    assert module._audit_passes(proposal, audit)
    audit["assessment"]["strategy_assessments"][0][
        "menu_reveals_final_answer"
    ] = True
    assert not module._audit_passes(proposal, audit)


def test_e49w_slurm_uses_one_a100_and_protocol_forbids_relabeling():
    slurm = (
        ROOT / "ops/slurm/e49w_bottom_up_route_calibration_node302.slurm"
    ).read_text(encoding="utf-8")
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e49w_bottom_up_05b_route_menu_calibration_20260726.md"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:a100:1" in slurm
    assert "Failed routes are never relabeled" in protocol
    assert "at least ten training problems pass" in protocol
