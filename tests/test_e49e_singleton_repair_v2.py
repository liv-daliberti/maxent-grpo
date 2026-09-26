from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPAIR = (
    ROOT
    / "ops/math_strategy_calibration/repair_e49e_singleton_gaps_v2.py"
)
AMENDMENT = (
    ROOT
    / "paper/preregistration/"
    "e49e_answer_blind_symbolic_singleton_v2_amendment_20260724.md"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49e_singleton_repair_v2b_job.sh"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49e_singleton_repair_v2_test",
        REPAIR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(module):
    return {
        "schema": module.base.pipeline.MENU_SCHEMA,
        "actions": [
            {
                "action_id": "A1",
                "operation": "Apply the stated recurrence to the target.",
            },
            {
                "action_id": "A2",
                "operation": "Check the resulting symbolic constraint.",
            },
        ],
        "strategies": [
            {
                "strategy_id": "S1",
                "action_ids": ["A1", "A2"],
                "plan": "Execute the recurrence and constraint check.",
            }
        ],
    }


def test_v2_request_is_problem_only(monkeypatch):
    module = _load()
    seen = {}

    def fake_post(_endpoint, request, *, timeout):
        seen["request"] = request
        assert timeout == 30
        return (
            {
                "id": "answer-blind-response",
                "choices": [{"finish_reason": "stop"}],
            },
            json.dumps(_payload(module)),
        )

    monkeypatch.setattr(module.base.pipeline.base, "_post", fake_post)
    record = module._proposal_request(
        endpoint="http://judge/v1",
        model="qwen2.5-72b",
        problem="Use the recurrence with 17 terms.",
        reference_answer="SECRET-ANSWER",
        gold_solution="SECRET-GOLD",
        role=module.PROPOSAL_ROLES[0],
        seed=module.PROPOSAL_SEEDS[0],
        timeout=30,
    )
    request_text = json.dumps(seen["request"], sort_keys=True)
    assert "SECRET-ANSWER" not in request_text
    assert "SECRET-GOLD" not in request_text
    assert record["proposal_input_scope"] == "problem_only"
    assert record["pass"] is True


def test_v2_slots_and_schema_are_frozen_and_bounded():
    module = _load()
    assert module.PROPOSAL_SEEDS == (492241, 492242, 492243, 492244)
    assert len(module.PROPOSAL_ROLES) == 4
    schema = module._singleton_schema()
    assert schema["properties"]["actions"]["maxItems"] == 5
    strategies = schema["properties"]["strategies"]
    assert strategies["minItems"] == strategies["maxItems"] == 1
    assert "FROZEN BEFORE ANY V2" in AMENDMENT.read_text(encoding="utf-8")


def test_v2_launcher_binds_terminal_v1_and_frozen_identity():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "terminal V1 repair evidence is incomplete" in launcher
    assert "frozen_v1_repair_records.jsonl" in launcher
    assert "prior_v1_repair_records_sha256" in launcher
    assert "v2b_judge_requests_at_freeze" in launcher
    assert "e49e_repair_singletons_v2b_node915.slurm" in launcher
    assert "e49e_curated_singleton_contingency_amendment_20260724.md" in launcher
    assert "e49e_curated_singleton_contracts_toy.json" in launcher
    assert "curated_contracts_sha256" in launcher
    slurm = (
        ROOT / "ops/slurm/e49e_repair_singletons_v2b_node915.slurm"
    ).read_text(encoding="utf-8")
    assert "--repair-workers 4" in slurm


def test_v2_still_rejects_problem_external_numerals(monkeypatch):
    module = _load()
    payload = _payload(module)
    payload["actions"][1]["operation"] = "State the derived value 19."

    def fake_post(_endpoint, _request, *, timeout):
        return (
            {
                "id": "leaking-response",
                "choices": [{"finish_reason": "stop"}],
            },
            json.dumps(payload),
        )

    monkeypatch.setattr(module.base.pipeline.base, "_post", fake_post)
    record = module._proposal_request(
        endpoint="http://judge/v1",
        model="qwen2.5-72b",
        problem="Use the recurrence with 17 terms.",
        reference_answer="19",
        gold_solution="The value is 19.",
        role=module.PROPOSAL_ROLES[0],
        seed=module.PROPOSAL_SEEDS[0],
        timeout=30,
    )
    assert record["pass"] is False
    assert record["completed_invalid"] is True
    assert record["menu"] is None


def test_v2_rejects_spelled_out_external_or_reference_numbers():
    module = _load()
    payload = _payload(module)
    payload["actions"][1]["operation"] = "State the derived value nineteen."
    external = module.base.pipeline._menu_from_payload(payload)
    assert not module._proposal_is_nonleaking_v2(
        external,
        problem="Use the recurrence with 17 terms.",
        reference_answer="19",
    )

    payload["actions"][1]["operation"] = "Show that the residue is seven."
    copied_surface = module.base.pipeline._menu_from_payload(payload)
    assert not module._proposal_is_nonleaking_v2(
        copied_surface,
        problem="Choose a residue between 0 and 7.",
        reference_answer="7",
    )


def test_v2_allows_spelled_number_that_is_in_problem_but_not_answer():
    module = _load()
    payload = _payload(module)
    payload["actions"][1]["operation"] = (
        "Apply the fourth-power representation symbolically."
    )
    menu = module.base.pipeline._menu_from_payload(payload)
    assert module._proposal_is_nonleaking_v2(
        menu,
        problem="Solve z^4 = i.",
        reference_answer="11",
    )
