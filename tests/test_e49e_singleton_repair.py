from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPAIR = (
    ROOT
    / "ops/math_strategy_calibration/repair_e49e_singleton_gaps.py"
)
LAUNCHER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e49e_singleton_repair_job.sh"
)


def _load_repair():
    spec = importlib.util.spec_from_file_location(
        "e49e_singleton_repair_test",
        REPAIR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _menu_payload(module, *, leak=False):
    final_operation = (
        "State that the final answer is 40."
        if leak
        else (
            "Multiply the current offer by the future doubling factor "
            "without evaluating or stating the final value."
        )
    )
    return {
        "schema": module.pipeline.MENU_SCHEMA,
        "actions": [
            {
                "action_id": "A1",
                "operation": (
                    "Compute the current offer by doubling the original "
                    "purchase price."
                ),
            },
            {
                "action_id": "A2",
                "operation": final_operation,
            },
        ],
        "strategies": [
            {
                "strategy_id": "S1",
                "action_ids": ["A1", "A2"],
                "plan": "Execute A1 and then A2.",
            }
        ],
    }


def test_answer_leakage_gate_rejects_explicit_reference_answer():
    module = _load_repair()
    safe = module.pipeline._menu_from_payload(_menu_payload(module))
    leaking = module.pipeline._menu_from_payload(
        _menu_payload(module, leak=True)
    )
    assert module._leaks_answer(safe, "\\$40") is False
    assert module._leaks_answer(leaking, "\\$40") is True
    assert module._proposal_is_nonleaking(
        safe,
        problem="The original purchase price is $1.25.",
        reference_answer="\\$40",
    )


def test_answer_leakage_gate_rejects_unseen_derived_numeric_literal():
    module = _load_repair()
    payload = _menu_payload(module)
    payload["actions"][1]["operation"] = (
        "Use the future doubling factor to derive the intermediate 20."
    )
    leaking = module.pipeline._menu_from_payload(payload)
    assert not module._proposal_is_nonleaking(
        leaking,
        problem="The original purchase price is $1.25.",
        reference_answer="\\$40",
    )


def test_singleton_proposal_is_closed_and_cannot_add_diversity(monkeypatch):
    module = _load_repair()
    content = json.dumps(_menu_payload(module))

    def fake_post(_endpoint, _payload, *, timeout):
        assert timeout == 30
        return (
            {
                "id": "proposal-response",
                "choices": [{"finish_reason": "stop"}],
            },
            content,
        )

    monkeypatch.setattr(module.pipeline.base, "_post", fake_post)
    record = module._proposal_request(
        endpoint="http://judge/v1",
        model="qwen2.5-72b",
        problem="A stamp problem.",
        reference_answer="\\$40",
        gold_solution="Auditor-only derivation.",
        role=module.PROPOSAL_ROLES[0],
        seed=module.PROPOSAL_SEEDS[0],
        timeout=30,
    )
    assert record["pass"] is True
    assert record["completed_invalid"] is False
    assert len(record["menu"]["strategies"]) == 1
    assert record["menu"]["strategies"][0]["action_ids"] == ["A1", "A2"]


def test_completed_leaking_proposal_fails_closed(monkeypatch):
    module = _load_repair()
    content = json.dumps(_menu_payload(module, leak=True))

    def fake_post(_endpoint, _payload, *, timeout):
        return (
            {
                "id": "leaking-response",
                "choices": [{"finish_reason": "stop"}],
            },
            content,
        )

    monkeypatch.setattr(module.pipeline.base, "_post", fake_post)
    record = module._proposal_request(
        endpoint="http://judge/v1",
        model="qwen2.5-72b",
        problem="A stamp problem.",
        reference_answer="\\$40",
        gold_solution="",
        role=module.PROPOSAL_ROLES[1],
        seed=module.PROPOSAL_SEEDS[1],
        timeout=30,
    )
    assert record["pass"] is False
    assert record["completed_invalid"] is True
    assert record["menu"] is None


def test_repair_proposal_slots_are_fixed_and_bounded():
    module = _load_repair()
    assert module.PROPOSAL_SEEDS == (492141, 492142)
    assert len(module.PROPOSAL_ROLES) == 2
    schema = module._singleton_schema()
    strategies = schema["properties"]["strategies"]
    assert strategies["minItems"] == strategies["maxItems"] == 1


def _sound_audit(module, menu, role, seed):
    strategy = menu.strategies[0]
    return {
        "trace_contract_version": module.pipeline.TRACE_CONTRACT_VERSION,
        "kind": "soundness",
        "role": role,
        "seed": seed,
        "candidate_menu_sha256": menu.sha256,
        "strategy_id": "S1",
        "response_id": f"sound-{seed}",
        "finish_reason": "stop",
        "content_sha256": f"{seed:064x}"[-64:],
        "assessment": {
            "strategy_id": "S1",
            "action_executions": [
                {
                    "action_id": action_id,
                    "executed_calculation": f"Execute {action_id}.",
                    "output_fact": f"Fact from {action_id}.",
                    "status": "valid",
                    "brief_check": "Valid.",
                }
                for action_id in strategy.action_ids
            ],
            "uses_only_declared_actions": True,
            "self_contained_without_other_strategy": True,
            "derived_answer": "$40",
            "matches_reference_answer": True,
            "status": "sound",
            "decisive_check": "The declared route derives $40.",
        },
        "answer_validator_pass": True,
        "completed_invalid": False,
        "pass": True,
    }


def test_cached_singleton_repair_is_recomputed_not_trusted():
    module = _load_repair()
    problem = "The original purchase price is $1.25."
    reference = r"\$40"
    menu = module.pipeline._menu_from_payload(_menu_payload(module))
    proposal = {
        "repair_version": module.REPAIR_VERSION,
        "kind": "singleton_proposal",
        "role": module.PROPOSAL_ROLES[0],
        "seed": module.PROPOSAL_SEEDS[0],
        "response_id": "proposal",
        "finish_reason": "stop",
        "content_sha256": "a" * 64,
        "menu": json.loads(menu.canonical_json),
        "menu_sha256": menu.sha256,
        "completed_invalid": False,
        "pass": True,
    }
    audits = [
        _sound_audit(module, menu, role, seed)
        for seed, role in zip(
            module.pipeline.SOUNDNESS_SEEDS,
            module.pipeline.SOUNDNESS_ROLES,
            strict=True,
        )
    ]
    record = {
        "schema": "e49e_singleton_repair_record_v1",
        "repair_version": module.REPAIR_VERSION,
        "row_id": "row",
        "problem_sha256": module._sha256_bytes(problem.encode()),
        "reference_answer_sha256": module._sha256_bytes(reference.encode()),
        "attempts": [{"proposal": proposal, "sound_audits": audits}],
        "menu": json.loads(menu.canonical_json),
        "menu_sha256": menu.sha256,
        "pass": True,
    }
    assert module._repair_record_passes(
        record,
        row_id="row",
        problem=problem,
        reference_answer=reference,
    )
    audits[1]["assessment"]["self_contained_without_other_strategy"] = False
    assert not module._repair_record_passes(
        record,
        row_id="row",
        problem=problem,
        reference_answer=reference,
    )


def test_singleton_launcher_binds_numeric_hardening():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert (
        "e49e_singleton_numeric_hardening_amendment_20260724.md"
        in launcher
    )
    assert (
        "e49e_singleton_cache_replay_amendment_20260724.md"
        in launcher
    )
    assert '"singleton_numeric_hardening_amendment_sha256"' in launcher
    assert '"singleton_cache_replay_amendment_sha256"' in launcher
    assert (
        "e49e_repair_origin_replay_amendment_20260724.md"
        in launcher
    )
    assert '"repair_origin_replay_amendment_sha256"' in launcher


def test_augmentation_replay_has_origin_specific_raw_branch():
    source = REPAIR.read_text(encoding="utf-8")
    assert 'record.get("origin") == "original_trace_bank"' in source
    assert "original trace-bank augmentation replay failed" in source
    assert "_canonical_sha256(selected[1])" in source
    assert "kernel augmentation leaked a numeric answer hint" in source
