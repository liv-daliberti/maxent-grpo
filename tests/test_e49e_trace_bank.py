from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PIPELINE = (
    ROOT
    / "ops/math_strategy_calibration/materialize_e49e_trace_bank_data.py"
)


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("e49e_trace_bank", PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proposal(module, *, cross_reference=False):
    payload = {
        "schema": module.MENU_SCHEMA,
        "actions": [
            {"action_id": "A1", "operation": "Apply theorem one."},
            {"action_id": "A2", "operation": "Apply theorem two."},
            {"action_id": "A3", "operation": "Finish the calculation."},
        ],
        "strategies": [
            {
                "strategy_id": "S1",
                "action_ids": ["A1", "A3"],
                "plan": "Apply A1 and then A3.",
            },
            {
                "strategy_id": "S2",
                "action_ids": ["A2", "A3"],
                "plan": (
                    "Apply A1, A2, and A3."
                    if cross_reference
                    else "Apply A2 and then A3."
                ),
            },
        ],
    }
    return module._menu_from_payload(payload)


def _legacy_assessment(*, relation="distinct"):
    return {
        "strategy_assessments": [
            {
                "strategy_id": strategy_id,
                "status": "sound",
                "failure_code": "none",
                "derived_answer": "17",
                "matches_reference_answer": True,
                "brief_derivation": "The exact actions derive 17.",
            }
            for strategy_id in ("S1", "S2")
        ],
        "pair_assessments": [
            {
                "pair_id": "S1__S2",
                "relation": relation,
                "shared_core": "Both solve the target.",
                "left_decisive_operation": "theorem one",
                "right_decisive_operation": "theorem two",
                "brief_check": "The decisive theorems differ.",
            }
        ],
    }


def _legacy_audits(module, *, relation="distinct"):
    assessment = _legacy_assessment(relation=relation)
    return [
        {
            "audit_contract_version": module.base.AUDIT_CONTRACT_VERSION,
            "audit_transport_version": module.base.AUDIT_TRANSPORT_VERSION,
            "audit_role": role,
            "seed": seed,
            "pass": relation == "distinct",
            "response_id": f"legacy-{seed}",
            "finish_reason": "stop",
            "assessment": assessment,
        }
        for seed, role in zip(
            module.base.AUDIT_SEEDS,
            module.base.AUDIT_ROLES,
            strict=True,
        )
    ]


def _input_record(module, *, cross_reference=False):
    proposal = _proposal(module, cross_reference=cross_reference)
    audits = _legacy_audits(module)
    selected = module.base._maximal_certified_subset(proposal, audits)
    assert selected is not None
    return {
        "attempts": [],
        "certification": selected[1],
        "menu": json.loads(selected[0].canonical_json),
        "menu_sha256": selected[0].sha256,
    }


def _sound_record(module, menu, strategy_id, role, seed):
    strategy = menu.strategy(strategy_id)
    assert strategy is not None
    assessment = {
        "strategy_id": strategy_id,
        "action_executions": [
            {
                "action_id": action_id,
                "executed_calculation": f"Execute {action_id} exactly.",
                "output_fact": f"Fact from {action_id}.",
                "status": "valid",
                "brief_check": "The operation is valid.",
            }
            for action_id in strategy.action_ids
        ],
        "uses_only_declared_actions": True,
        "self_contained_without_other_strategy": True,
        "derived_answer": "17",
        "matches_reference_answer": True,
        "status": "sound",
        "decisive_check": "Every action connects and derives 17.",
    }
    return {
        "trace_contract_version": module.TRACE_CONTRACT_VERSION,
        "kind": "soundness",
        "role": role,
        "seed": seed,
        "candidate_menu_sha256": menu.sha256,
        "strategy_id": strategy_id,
        "response_id": f"sound-{strategy_id}-{seed}",
        "finish_reason": "stop",
        "content_sha256": f"{seed:064x}"[-64:],
        "assessment": assessment,
        "answer_validator_pass": True,
        "completed_invalid": False,
        "pass": True,
    }


def _pair_records(module, menu, eligible_ids, *, relation="distinct"):
    pair_ids = [
        f"{left}__{right}"
        for left, right in __import__("itertools").combinations(
            eligible_ids, 2
        )
    ]
    assessment = {
        "pair_assessments": [
            {
                "pair_id": pair_id,
                "relation": relation,
                "shared_core": "Both solve the target.",
                "left_decisive_operation": "theorem one",
                "right_decisive_operation": "theorem two",
                "reducible_by_routine_algebra": relation != "distinct",
                "both_routes_self_contained": True,
                "brief_check": "The decisive operations genuinely differ.",
            }
            for pair_id in pair_ids
        ]
    }
    return [
        {
            "trace_contract_version": module.TRACE_CONTRACT_VERSION,
            "kind": "pair_equivalence",
            "role": role,
            "seed": seed,
            "candidate_menu_sha256": menu.sha256,
            "eligible_strategy_ids": eligible_ids,
            "response_id": f"pair-{seed}",
            "finish_reason": "stop",
            "assessment": assessment,
            "completed_invalid": False,
            "pass": True,
        }
        for seed, role in zip(
            module.PAIR_SEEDS, module.PAIR_ROLES, strict=True
        )
    ]


def test_cross_proposal_bank_reads_durable_certification_and_closes_actions():
    module = _load_pipeline()
    bank = module._candidate_bank(_input_record(module))
    assert bank is not None
    menu, sources = bank
    assert len(menu.strategies) == 2
    assert {source["phase"] for source in sources} == {
        "durable_retained_certification"
    }

    cross_referenced = module._candidate_bank(
        _input_record(module, cross_reference=True)
    )
    assert cross_referenced is not None
    filtered_menu, _ = cross_referenced
    assert len(filtered_menu.strategies) == 1
    retained = filtered_menu.strategies[0]
    references = set(
        module.ACTION_REFERENCE_RE.findall(retained.plan)
    )
    assert references <= set(retained.action_ids)


def test_pruning_rewrites_every_surviving_action_reference():
    module = _load_pipeline()
    proposal = _proposal(module)
    pruned = module._prune_menu_closed(proposal, ("S2",))
    assert len(pruned.strategies) == 1
    strategy = pruned.strategies[0]
    assert strategy.action_ids == ("A1", "A2")
    assert set(module.ACTION_REFERENCE_RE.findall(strategy.plan)) <= {
        "A1",
        "A2",
    }
    assert "A3" not in strategy.plan


def test_trace_certification_requires_two_sound_audits_and_two_pair_vetoes():
    module = _load_pipeline()
    bank = module._candidate_bank(_input_record(module))
    assert bank is not None
    menu, _ = bank
    sound = {
        strategy.strategy_id: [
            _sound_record(module, menu, strategy.strategy_id, role, seed)
            for seed, role in zip(
                module.SOUNDNESS_SEEDS,
                module.SOUNDNESS_ROLES,
                strict=True,
            )
        ]
        for strategy in menu.strategies
    }
    eligible_ids = [strategy.strategy_id for strategy in menu.strategies]
    selected = module._maximal_trace_certified_subset(
        menu,
        sound,
        _pair_records(module, menu, eligible_ids),
        reference_answer="17",
    )
    assert selected is not None
    assert len(selected[0].strategies) == 2

    collapsed = module._maximal_trace_certified_subset(
        menu,
        sound,
        _pair_records(
            module, menu, eligible_ids, relation="equivalent"
        ),
        reference_answer="17",
    )
    assert collapsed is not None
    assert len(collapsed[0].strategies) == 1


def test_trace_certification_fails_closed_on_action_order_or_hidden_import():
    module = _load_pipeline()
    bank = module._candidate_bank(_input_record(module))
    assert bank is not None
    menu, _ = bank
    strategy = menu.strategies[0]
    record = _sound_record(
        module,
        menu,
        strategy.strategy_id,
        module.SOUNDNESS_ROLES[0],
        module.SOUNDNESS_SEEDS[0],
    )
    record["assessment"]["action_executions"].reverse()
    assert (
        module._sound_record_passes(
            menu,
            strategy.strategy_id,
            record,
            reference_answer="17",
        )
        is False
    )


def test_sound_record_rechecks_only_conservative_answer_surface_variants():
    module = _load_pipeline()
    bank = module._candidate_bank(_input_record(module))
    assert bank is not None
    menu, _ = bank
    strategy = menu.strategies[0]
    record = _sound_record(
        module,
        menu,
        strategy.strategy_id,
        module.SOUNDNESS_ROLES[0],
        module.SOUNDNESS_SEEDS[0],
    )
    record["assessment"]["derived_answer"] = "4√2 cm"
    record["answer_validator_pass"] = False
    # The frozen E49E writer folded its narrower answer verifier into this
    # cached convenience bit.  Successor re-evaluation must derive acceptance
    # from the immutable assessment, not preserve that obsolete false
    # negative.
    record["pass"] = False
    assert module._sound_record_passes(
        menu,
        strategy.strategy_id,
        record,
        reference_answer=r"4\sqrt{2}",
    )
    assert not module._sound_record_passes(
        menu,
        strategy.strategy_id,
        record,
        reference_answer=r"4\sqrt{3}",
    )
    record["content_sha256"] = ""
    assert not module._sound_record_passes(
        menu,
        strategy.strategy_id,
        record,
        reference_answer=r"4\sqrt{2}",
    )

    record = _sound_record(
        module,
        menu,
        strategy.strategy_id,
        module.SOUNDNESS_ROLES[0],
        module.SOUNDNESS_SEEDS[0],
    )
    record["assessment"]["self_contained_without_other_strategy"] = False
    assert (
        module._sound_record_passes(
            menu,
            strategy.strategy_id,
            record,
            reference_answer="17",
        )
        is False
    )


def test_record_contract_recomputes_the_frozen_input_bank():
    module = _load_pipeline()
    input_record = _input_record(module)
    bank = module._candidate_bank(input_record)
    assert bank is not None
    menu, sources = bank
    sound = {
        strategy.strategy_id: [
            _sound_record(module, menu, strategy.strategy_id, role, seed)
            for seed, role in zip(
                module.SOUNDNESS_SEEDS,
                module.SOUNDNESS_ROLES,
                strict=True,
            )
        ]
        for strategy in menu.strategies
    }
    eligible_ids = [strategy.strategy_id for strategy in menu.strategies]
    pair_records = _pair_records(module, menu, eligible_ids)
    selected = module._maximal_trace_certified_subset(
        menu,
        sound,
        pair_records,
        reference_answer="17",
    )
    assert selected is not None
    retained, certification = selected
    record = {
        "pass": True,
        "trace_contract_version": module.TRACE_CONTRACT_VERSION,
        "input_record_sha256": module._canonical_sha256(input_record),
        "candidate_sources": sources,
        "menu": json.loads(retained.canonical_json),
        "menu_sha256": retained.sha256,
        "certification": certification,
    }
    assert module._record_passes_contract(
        record,
        reference_answer="17",
        input_record=input_record,
    )

    changed_input = _input_record(module, cross_reference=True)
    assert not module._record_passes_contract(
        record,
        reference_answer="17",
        input_record=changed_input,
    )


def test_completed_audit_cache_is_terminal_but_transport_failure_can_resume(
    tmp_path,
):
    module = _load_pipeline()
    cache_path = tmp_path / "audit.json"
    calls = 0

    def completed_invalid(**_kwargs):
        nonlocal calls
        calls += 1
        return {
            "trace_contract_version": module.TRACE_CONTRACT_VERSION,
            "completed_invalid": True,
            "pass": False,
        }

    first = module._cached_request(cache_path, completed_invalid)
    second = module._cached_request(cache_path, completed_invalid)
    assert first == second
    assert calls == 1

    transport_path = tmp_path / "transport.json"

    def transport_then_complete(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("request failed after transport retries")
        return {
            "trace_contract_version": module.TRACE_CONTRACT_VERSION,
            "completed_invalid": False,
            "pass": False,
        }

    try:
        module._cached_request(transport_path, transport_then_complete)
    except RuntimeError:
        pass
    else:
        raise AssertionError("transport failure unexpectedly completed")
    assert not transport_path.exists()
    resumed = module._cached_request(
        transport_path, transport_then_complete
    )
    assert resumed["completed_invalid"] is False
    assert transport_path.exists()


def test_known_invalid_control_is_bound_to_bank_and_requires_soundness_reject(
    tmp_path,
):
    module = _load_pipeline()
    input_record = _input_record(module)
    bank = module._candidate_bank(input_record)
    assert bank is not None
    menu, _ = bank
    strategy = menu.strategies[0]
    manifest = {
        "schema": "e49e_known_invalid_controls_v1",
        "stage": "toy",
        "controls": [
            {
                "control_id": "synthetic_invalid",
                "row_id": "row",
                "candidate_menu_sha256": menu.sha256,
                "strategy_id": strategy.strategy_id,
                "failure_class": "synthetic",
                "reason": "The route is deliberately invalid.",
            }
        ],
    }
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    controls = module._load_known_invalid_controls(
        path,
        {"row": input_record},
    )
    sound = [
        _sound_record(module, menu, strategy.strategy_id, role, seed)
        for seed, role in zip(
            module.SOUNDNESS_SEEDS,
            module.SOUNDNESS_ROLES,
            strict=True,
        )
    ]
    records = {
        "row": {
            "certification": {
                "sound_audits": {strategy.strategy_id: sound}
            }
        }
    }
    accepted = module._known_invalid_control_results(
        records,
        controls,
        {"row": input_record},
        {"row": "17"},
    )
    assert accepted[0]["double_sound_accept"] is True
    assert accepted[0]["rejected_by_soundness"] is False

    sound[1]["pass"] = False
    sound[1]["assessment"]["self_contained_without_other_strategy"] = False
    rejected = module._known_invalid_control_results(
        records,
        controls,
        {"row": input_record},
        {"row": "17"},
    )
    assert rejected[0]["audits_complete"] is True
    assert rejected[0]["double_sound_accept"] is False
    assert rejected[0]["rejected_by_soundness"] is True

    manifest["controls"][0]["candidate_menu_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        module._load_known_invalid_controls(path, {"row": input_record})
    except RuntimeError:
        pass
    else:
        raise AssertionError("changed control binding unexpectedly passed")


def test_known_equivalent_control_requires_sound_routes_and_no_distinct_edge(
    tmp_path,
):
    module = _load_pipeline()
    input_record = _input_record(module)
    bank = module._candidate_bank(input_record)
    assert bank is not None
    menu, _ = bank
    strategy_ids = [strategy.strategy_id for strategy in menu.strategies]
    manifest = {
        "schema": "e49e_known_equivalent_controls_v1",
        "stage": "toy",
        "controls": [
            {
                "control_id": "synthetic_same",
                "row_id": "row",
                "candidate_menu_sha256": menu.sha256,
                "left_strategy_id": strategy_ids[0],
                "right_strategy_id": strategy_ids[1],
                "reason": "The routes are deliberately equivalent.",
            }
        ],
    }
    path = tmp_path / "equivalent.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    controls = module._load_known_equivalent_controls(
        path,
        {"row": input_record},
    )
    sound = {
        strategy_id: [
            _sound_record(module, menu, strategy_id, role, seed)
            for seed, role in zip(
                module.SOUNDNESS_SEEDS,
                module.SOUNDNESS_ROLES,
                strict=True,
            )
        ]
        for strategy_id in strategy_ids
    }
    record = {
        "row": {
            "certification": {
                "sound_audits": sound,
                "pair_audits": _pair_records(
                    module,
                    menu,
                    strategy_ids,
                    relation="distinct",
                ),
            }
        }
    }
    false_new = module._known_equivalent_control_results(
        record,
        controls,
        {"row": input_record},
        {"row": "17"},
    )
    assert false_new[0]["both_routes_sound"] is True
    assert false_new[0]["double_distinct_false_new"] is True
    assert false_new[0]["rejected_as_new"] is False

    record["row"]["certification"]["pair_audits"] = _pair_records(
        module,
        menu,
        strategy_ids,
        relation="equivalent",
    )
    rejected = module._known_equivalent_control_results(
        record,
        controls,
        {"row": input_record},
        {"row": "17"},
    )
    assert rejected[0]["pair_audits_complete"] is True
    assert rejected[0]["double_distinct_false_new"] is False
    assert rejected[0]["rejected_as_new"] is True


def test_materialized_prompt_embeds_problem_and_menu_exactly_once():
    module = _load_pipeline()
    bank = module._candidate_bank(_input_record(module))
    assert bank is not None
    menu, _ = bank
    prompt = module._embed(
        "A uniquely worded test problem.",
        json.loads(menu.canonical_json),
    )
    assert prompt.count("A uniquely worded test problem.") == 1
    assert prompt.count(module.MENU_START) == 1
    assert prompt.count(module.MENU_END) == 1


def test_known_equivalent_control_requires_no_false_distinct_edge(
    tmp_path,
):
    module = _load_pipeline()
    input_record = _input_record(module)
    bank = module._candidate_bank(input_record)
    assert bank is not None
    menu, _ = bank
    ids = [strategy.strategy_id for strategy in menu.strategies]
    manifest = {
        "schema": "e49e_known_equivalent_controls_v1",
        "stage": "toy",
        "controls": [
            {
                "control_id": "synthetic_equivalent",
                "row_id": "row",
                "candidate_menu_sha256": menu.sha256,
                "left_strategy_id": ids[0],
                "right_strategy_id": ids[1],
                "reason": "The routes are deliberately equivalent.",
            }
        ],
    }
    path = tmp_path / "equivalent-controls.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    controls = module._load_known_equivalent_controls(
        path,
        {"row": input_record},
    )
    sound = {
        strategy_id: [
            _sound_record(module, menu, strategy_id, role, seed)
            for seed, role in zip(
                module.SOUNDNESS_SEEDS,
                module.SOUNDNESS_ROLES,
                strict=True,
            )
        ]
        for strategy_id in ids
    }
    record = {
        "certification": {
            "sound_audits": sound,
            "pair_audits": _pair_records(
                module,
                menu,
                ids,
                relation="distinct",
            ),
        }
    }
    false_new = module._known_equivalent_control_results(
        {"row": record},
        controls,
        {"row": input_record},
        {"row": "17"},
    )
    assert false_new[0]["double_distinct_false_new"] is True
    assert false_new[0]["rejected_as_new"] is False

    record["certification"]["pair_audits"] = _pair_records(
        module,
        menu,
        ids,
        relation="equivalent",
    )
    pair_rejected = module._known_equivalent_control_results(
        {"row": record},
        controls,
        {"row": input_record},
        {"row": "17"},
    )
    assert pair_rejected[0]["pair_audits_complete"] is True
    assert pair_rejected[0]["rejected_as_new"] is True

    sound[ids[1]][1]["pass"] = False
    sound[ids[1]][1]["assessment"][
        "self_contained_without_other_strategy"
    ] = False
    record["certification"]["pair_audits"] = []
    soundness_rejected = module._known_equivalent_control_results(
        {"row": record},
        controls,
        {"row": input_record},
        {"row": "17"},
    )
    assert soundness_rejected[0]["sound_audits_complete"] is True
    assert soundness_rejected[0]["both_routes_sound"] is False
    assert soundness_rejected[0]["rejected_as_new"] is True
