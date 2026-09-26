from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PIPELINE = (
    ROOT
    / "ops/math_strategy_calibration/materialize_e49d_strategy_menu_data.py"
)


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("e49c_menu_pipeline", PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _menu(module):
    payload = {
        "schema": "math_strategy_action_menu_v1",
        "actions": [
            {"action_id": "A1", "operation": "Use theorem one"},
            {"action_id": "A2", "operation": "Use theorem two"},
            {"action_id": "A3", "operation": "Finish"},
        ],
        "strategies": [
            {
                "strategy_id": "S1",
                "action_ids": ["A1", "A3"],
                "plan": "Apply theorem one, then finish.",
            },
            {
                "strategy_id": "S2",
                "action_ids": ["A2", "A3"],
                "plan": "Apply theorem two, then finish.",
            },
        ],
    }
    embedded = (
        f"x\n{module.MENU_START}\n{json.dumps(payload)}"
        f"\n{module.MENU_END}"
    )
    parsed = module.parse_strategy_menu(embedded)
    assert parsed is not None
    return parsed


def _assessment(
    *,
    matches: bool = True,
    same_operations: bool = False,
    relation: str = "distinct",
):
    left = "theorem one"
    right = left if same_operations else "theorem two"
    return {
        "strategy_assessments": [
            {
                "strategy_id": strategy_id,
                "status": "sound",
                "failure_code": "none",
                "derived_answer": "17",
                "matches_reference_answer": matches,
                "brief_derivation": "The declared operations derive 17.",
            }
            for strategy_id in ("S1", "S2")
        ],
        "pair_assessments": [
            {
                "pair_id": "S1__S2",
                "relation": relation,
                "shared_core": "Both seek the same target.",
                "left_decisive_operation": left,
                "right_decisive_operation": right,
                "brief_check": "The decisive theorems differ.",
            }
        ],
    }


def test_v4_audit_is_reference_bound_and_role_specific():
    module = _load_pipeline()
    captured = {}

    def fake_post(endpoint, payload, *, timeout):
        captured["payload"] = payload
        return (
            {"id": "response", "choices": [{"finish_reason": "stop"}]},
            json.dumps(_assessment()),
        )

    module._post = fake_post
    result = module._audit_menu(
        endpoint="http://judge",
        model="qwen2.5-72b",
        problem="Find the value.",
        reference_answer="17",
        menu=_menu(module),
        seed=491711,
        audit_role="soundness_execution",
        timeout=10,
    )
    prompt = captured["payload"]["messages"][1]["content"]
    assert result["pass"] is True
    assert result["audit_role"] == "soundness_execution"
    assert "AUDITOR-ONLY REFERENCE ANSWER:\n17" in prompt
    assert "SOUNDNESS EXECUTION" in prompt


def test_v4_audit_fails_closed_on_answer_mismatch_or_cosmetic_pair():
    module = _load_pipeline()
    responses = iter(
        (
            _assessment(matches=False),
            _assessment(same_operations=True),
        )
    )

    def fake_post(endpoint, payload, *, timeout):
        return (
            {"id": "response", "choices": [{"finish_reason": "stop"}]},
            json.dumps(next(responses)),
        )

    module._post = fake_post
    common = {
        "endpoint": "http://judge",
        "model": "qwen2.5-72b",
        "problem": "Find the value.",
        "reference_answer": "17",
        "menu": _menu(module),
        "seed": 491712,
        "audit_role": "equivalence_attack",
        "timeout": 10,
    }
    assert module._audit_menu(**common)["pass"] is False
    assert module._audit_menu(**common)["pass"] is False


def test_v4_audit_string_bounds_are_enforced_locally_not_by_xgrammar():
    module = _load_pipeline()
    menu = _menu(module)
    schema_text = json.dumps(module._audit_schema(menu))
    assert "minLength" not in schema_text
    assert "maxLength" not in schema_text

    assessment = _assessment()
    assert module._audit_assessment_is_well_formed(menu, assessment) is True
    assessment["pair_assessments"][0]["brief_check"] = "x" * 513
    assert module._audit_assessment_is_well_formed(menu, assessment) is False
    assert (
        module._maximal_certified_subset(
            menu,
            _audits(module, assessment),
        )
        is None
    )
    legacy_audits = _audits(module, assessment)
    for audit in legacy_audits:
        audit.pop("audit_transport_version")
    assert module._maximal_certified_subset(menu, legacy_audits) is not None


def test_audit_feedback_never_relays_reference_answer_or_derivation():
    module = _load_pipeline()
    assessment = _assessment(matches=False)
    assessment["strategy_assessments"][0]["status"] = "unsound"
    assessment["strategy_assessments"][0]["failure_code"] = "answer_mismatch"
    assessment["strategy_assessments"][0][
        "brief_derivation"
    ] = "The hidden reference answer is SECRET_RESULT."
    feedback = module._feedback(
        [{"pass": False, "assessment": assessment}]
    )
    assert "answer_mismatch" in feedback
    assert "SECRET_RESULT" not in feedback
    assert "hidden reference answer" not in feedback


def _audits(module, assessment):
    return [
        {
            "audit_contract_version": module.AUDIT_CONTRACT_VERSION,
            "audit_transport_version": module.AUDIT_TRANSPORT_VERSION,
            "audit_role": role,
            "seed": seed,
            "pass": all(
                row["status"] == "sound"
                and row["matches_reference_answer"]
                for row in assessment["strategy_assessments"]
            )
            and all(
                row["relation"] == "distinct"
                for row in assessment["pair_assessments"]
            ),
            "response_id": f"response-{seed}",
            "finish_reason": "stop",
            "assessment": assessment,
        }
        for seed, role in zip(
            module.AUDIT_SEEDS, module.AUDIT_ROLES, strict=True
        )
    ]


def test_v4_maximal_certified_support_prunes_equivalent_route_to_singleton():
    module = _load_pipeline()
    proposed = _menu(module)
    selected = module._maximal_certified_subset(
        proposed,
        _audits(module, _assessment(relation="equivalent")),
    )
    assert selected is not None
    retained, certification = selected
    assert len(retained.strategies) == 1
    assert certification["retained_original_strategy_ids"] == ["S1"]
    assert retained.strategies[0].action_combo == "A1>A2"
    assert [action.operation for action in retained.actions] == [
        "Use theorem one",
        "Finish",
    ]


def test_v4_maximal_certified_support_retains_double_certified_pair():
    module = _load_pipeline()
    proposed = _menu(module)
    selected = module._maximal_certified_subset(
        proposed,
        _audits(module, _assessment()),
    )
    assert selected is not None
    retained, certification = selected
    assert len(retained.strategies) == 2
    assert certification["retained_original_strategy_ids"] == ["S1", "S2"]


def test_v4_durable_record_recomputes_certification_fail_closed():
    module = _load_pipeline()
    selected = module._maximal_certified_subset(
        _menu(module),
        _audits(module, _assessment(relation="equivalent")),
    )
    assert selected is not None
    retained, certification = selected
    record = {
        "pass": True,
        "audit_contract_version": module.AUDIT_CONTRACT_VERSION,
        "menu": json.loads(retained.canonical_json),
        "menu_sha256": retained.sha256,
        "reference_answer_sha256": "answer-hash",
        "certification": certification,
    }
    assert module._record_passes_contract(record) is True
    record["certification"]["audits"][0]["response_id"] = ""
    assert module._record_passes_contract(record) is False


def _certified_singleton_record(module):
    selected = module._maximal_certified_subset(
        _menu(module),
        _audits(module, _assessment(relation="equivalent")),
    )
    assert selected is not None
    retained, certification = selected
    return {
        "schema": "e49d_strategy_menu_record_v1",
        "row_id": "train:0000:test",
        "problem_sha256": module._sha256_bytes(b"Find the value."),
        "reference_answer_sha256": module._sha256_bytes(b"17"),
        "problem": "Find the value.",
        "menu": json.loads(retained.canonical_json),
        "menu_sha256": retained.sha256,
        "attempts": [],
        "certification": certification,
        "audit_contract_version": module.AUDIT_CONTRACT_VERSION,
        "pass": True,
    }


def _install_recall_fakes(module, *, relation="distinct"):
    observed = {"rescue_attempts": [], "audit_roles": []}

    def fake_route_ideas(**kwargs):
        observed["rescue_attempts"].append(kwargs["rescue_attempt"])
        return (
            [
                {
                    "route_id": "R1",
                    "central_idea": "Use theorem one.",
                    "decisive_operation": "theorem one",
                },
                {
                    "route_id": "R2",
                    "central_idea": "Use theorem two.",
                    "decisive_operation": "theorem two",
                },
            ],
            {"request_seed": 491803},
        )

    def fake_generate_menu(**kwargs):
        assert kwargs["attempt"] == 102
        return _menu(module), {"request_seed": 491803}

    audits = {
        role: audit
        for role, audit in zip(
            module.AUDIT_ROLES,
            _audits(module, _assessment(relation=relation)),
            strict=True,
        )
    }

    def fake_audit_menu(**kwargs):
        observed["audit_roles"].append(kwargs["audit_role"])
        return audits[kwargs["audit_role"]]

    module._generate_route_ideas = fake_route_ideas
    module._generate_menu = fake_generate_menu
    module._audit_menu = fake_audit_menu
    return observed


def test_v4_support_recall_can_promote_existing_singleton_to_pair():
    module = _load_pipeline()
    prior = _certified_singleton_record(module)
    observed = _install_recall_fakes(module)

    recalled = module._recall_existing_singleton(
        endpoint="http://judge",
        model="qwen2.5-72b",
        row_id=prior["row_id"],
        problem="Find the value.",
        reference_answer="17",
        timeout=10,
        prior_record=prior,
    )

    assert observed["rescue_attempts"] == [2]
    assert observed["audit_roles"] == list(module.AUDIT_ROLES)
    assert len(recalled["menu"]["strategies"]) == 2
    assert recalled["support_recall_complete"] is True
    assert module._record_has_support_recall(recalled) is True
    assert module._record_is_materialization_ready(recalled) is True


def test_v4_support_recall_preserves_singleton_when_no_pair_certifies():
    module = _load_pipeline()
    prior = _certified_singleton_record(module)
    _install_recall_fakes(module, relation="equivalent")

    recalled = module._recall_existing_singleton(
        endpoint="http://judge",
        model="qwen2.5-72b",
        row_id=prior["row_id"],
        problem="Find the value.",
        reference_answer="17",
        timeout=10,
        prior_record=prior,
    )

    assert recalled["menu_sha256"] == prior["menu_sha256"]
    assert len(recalled["menu"]["strategies"]) == 1
    assert module._record_has_support_recall(recalled) is True
    assert module._record_is_materialization_ready(recalled) is True


def test_v4_support_recall_transport_retry_retains_error_and_completes_once():
    module = _load_pipeline()
    prior = _certified_singleton_record(module)

    def failed_route_ideas(**kwargs):
        raise RuntimeError("temporary transport failure")

    module._generate_route_ideas = failed_route_ideas
    failed = module._recall_existing_singleton(
        endpoint="http://judge",
        model="qwen2.5-72b",
        row_id=prior["row_id"],
        problem="Find the value.",
        reference_answer="17",
        timeout=10,
        prior_record=prior,
    )
    assert failed["support_recall_complete"] is False
    assert module._record_is_materialization_ready(failed) is False

    _install_recall_fakes(module, relation="equivalent")
    recalled = module._recall_existing_singleton(
        endpoint="http://judge",
        model="qwen2.5-72b",
        row_id=prior["row_id"],
        problem="Find the value.",
        reference_answer="17",
        timeout=10,
        prior_record=failed,
    )
    tagged = [
        attempt
        for attempt in recalled["attempts"]
        if attempt.get("support_recall_version")
        == module.SUPPORT_RECALL_VERSION
    ]
    assert len(tagged) == 2
    assert sum(bool(attempt.get("error")) for attempt in tagged) == 1
    assert module._record_has_support_recall(recalled) is True
    assert module._record_is_materialization_ready(recalled) is True


def test_v4_fresh_singleton_runs_and_tags_exactly_two_rescues():
    module = _load_pipeline()
    observed = {"menu_attempts": [], "rescue_attempts": []}

    def fake_generate_menu(**kwargs):
        observed["menu_attempts"].append(kwargs["attempt"])
        return _menu(module), {"request_seed": 491701 + kwargs["attempt"]}

    def fake_route_ideas(**kwargs):
        observed["rescue_attempts"].append(kwargs["rescue_attempt"])
        return (
            [
                {
                    "route_id": "R1",
                    "central_idea": "Use theorem one.",
                    "decisive_operation": "theorem one",
                },
                {
                    "route_id": "R2",
                    "central_idea": "Use theorem two.",
                    "decisive_operation": "theorem two",
                },
            ],
            {
                "request_seed": (
                    491701 + 100 + kwargs["rescue_attempt"]
                )
            },
        )

    audits = {
        role: audit
        for role, audit in zip(
            module.AUDIT_ROLES,
            _audits(module, _assessment(relation="equivalent")),
            strict=True,
        )
    }

    module._generate_menu = fake_generate_menu
    module._generate_route_ideas = fake_route_ideas
    module._audit_menu = lambda **kwargs: audits[kwargs["audit_role"]]

    record = module._build_one(
        endpoint="http://judge",
        model="qwen2.5-72b",
        row_id="train:0000:fresh",
        problem="Find the value.",
        reference_answer="17",
        timeout=10,
    )

    assert observed["menu_attempts"] == [1, 101, 102]
    assert observed["rescue_attempts"] == [1, 2]
    assert len(record["menu"]["strategies"]) == 1
    assert record["support_recall_version"] == module.SUPPORT_RECALL_VERSION
    assert record["support_recall_complete"] is True
    assert module._record_has_support_recall(record) is True
    assert module._record_is_materialization_ready(record) is True


def test_v4_generator_collapses_identical_action_combos_to_singleton():
    module = _load_pipeline()
    payload = {
        "schema": "math_strategy_action_menu_v1",
        "actions": [
            {"action_id": "A1", "operation": "Set up the equation"},
            {"action_id": "A2", "operation": "Solve the equation"},
            {"action_id": "A3", "operation": "Check the domain"},
        ],
        "strategies": [
            {
                "strategy_id": "S1",
                "action_ids": ["A1", "A2", "A3"],
                "plan": "Set up, solve, and check.",
            },
            {
                "strategy_id": "S2",
                "action_ids": ["A1", "A2", "A3"],
                "plan": "The same operations in different words.",
            },
        ],
    }

    def fake_post(endpoint, request, *, timeout):
        return (
            {"id": "proposal", "choices": [{"finish_reason": "stop"}]},
            json.dumps(payload),
        )

    module._post = fake_post
    menu, generation = module._generate_menu(
        endpoint="http://judge",
        model="qwen2.5-72b",
        problem="Solve it.",
        attempt=1,
        feedback="",
        route_ideas=None,
        timeout=10,
    )
    assert len(menu.strategies) == 1
    assert generation["collapsed_duplicate_strategy_count"] == 1


def test_v4_guided_schema_defers_action_id_uniqueness_to_strict_parser():
    module = _load_pipeline()
    action_ids = module._menu_schema()["properties"]["strategies"]["items"][
        "properties"
    ]["action_ids"]
    assert "uniqueItems" not in action_ids

    payload = json.loads(_menu(module).canonical_json)
    payload["strategies"][0]["action_ids"] = ["A1", "A1"]
    embedded = (
        f"x\n{module.MENU_START}\n{json.dumps(payload)}"
        f"\n{module.MENU_END}"
    )
    try:
        module.parse_strategy_menu(embedded)
    except ValueError as exc:
        assert "strategy action combo is invalid" in str(exc)
    else:
        raise AssertionError("duplicate action IDs must fail local validation")
