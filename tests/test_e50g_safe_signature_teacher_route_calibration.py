from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50g_safe_signature_teacher_route_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("e50g_safe_route", SOURCE)
assert SPEC is not None and SPEC.loader is not None
E50G = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = E50G
SPEC.loader.exec_module(E50G)
PREPARE_NEXT = (
    ROOT / "ops/math_strategy_calibration/prepare_e50_next_toy_and_probe.sh"
)
MATERIALIZE_E50D = (
    ROOT
    / "ops/math_strategy_calibration/"
    "materialize_e50d_teacher_route_math_toy.sh"
)
LAUNCH_E50G = (
    ROOT
    / "ops/math_strategy_calibration/"
    "launch_e50g_safe_signature_teacher_route_calibration.sh"
)


def _proposal() -> dict[str, object]:
    return {
        "route_sources": [
            {
                "strategy_id": "S1",
                "source_kind": "observed",
                "exemplar_ids": ["a"],
            },
            {
                "strategy_id": "S2",
                "source_kind": "observed",
                "exemplar_ids": ["b"],
            },
        ]
    }


def _audit(relation: str = "equivalent") -> dict[str, object]:
    return {
        "finish_reason": "stop",
        "response_id": "response",
        "assessment": {
            "strategy_assessments": [
                {
                    "strategy_id": strategy_id,
                    "status": "sound",
                    "failure_code": "none",
                    "derived_answer": "1",
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
                    "strategy_id": strategy_id,
                    "binding_status": "observed_bound",
                }
                for strategy_id in ("S1", "S2")
            ],
            "pair": {
                "relation": relation,
                "s1_decisive_operation": "Euclidean remainders",
                "s2_decisive_operation": "prime factors",
            },
        },
    }


def test_soundness_gate_does_not_delegate_identity_to_relation_vote() -> None:
    assert E50G._sound_and_bound_audit(
        _proposal(), _audit("equivalent")
    )
    assert E50G._sound_and_bound_audit(
        _proposal(), _audit("distinct")
    )
    source = SOURCE.read_text(encoding="utf-8")
    assert "menu_records.partial.jsonl" in source
    assert "menu_checkpoint_identity.json" in source
    assert "menu checkpoint identity drifted" in source


def test_soundness_gate_still_rejects_route_or_binding_failure() -> None:
    unsound = _audit()
    unsound["assessment"]["strategy_assessments"][0]["status"] = "unsound"
    assert not E50G._sound_and_bound_audit(_proposal(), unsound)

    unbound = _audit()
    unbound["assessment"]["source_assessments"][1][
        "binding_status"
    ] = "invalid"
    assert not E50G._sound_and_bound_audit(_proposal(), unbound)


def test_conditioned_record_contract_binds_requests_and_execution_batches():
    problem = {
        "problem_order": 3,
        "source_index": 17,
        "unique_id": "row",
        "problem": "problem only",
    }
    proposal = {
        "methods": [
            {"label": "the Euclidean algorithm", "actions": ["a", "b"]},
            {"label": "prime factorization", "actions": ["c", "d"]},
        ],
        "decisive_difference": "different engines",
    }
    request = E50G.BASE._proposal_payload("judge", "problem only", 3)
    request_hash = hashlib.sha256(
        json.dumps(
            request, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    executions = []
    for route_index in (0, 1):
        for choice_index in range(8):
            text = f"route {route_index} choice {choice_index}"
            executions.append(
                {
                    "route_index": route_index,
                    "choice_index": choice_index,
                    "response_id": f"batch-{route_index}",
                    "finish_reason": "stop",
                    "response": text,
                    "response_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                }
            )
    record = {
        "problem_order": 3,
        "source_index": 17,
        "row_id": "row",
        "proposal": proposal,
        "proposal_response_id": "proposal",
        "proposal_finish_reason": "stop",
        "proposal_request_sha256": request_hash,
        "answer_was_not_in_proposal_payload": True,
        "answer_was_not_in_execution_payloads": True,
        "executions": executions,
    }
    assert (
        E50G._generated_record_contract_failure(
            generated=record,
            problem=problem,
            judge_model="judge",
            attempt_index=0,
        )
        is None
    )
    record["executions"][0]["response"] = "tampered"
    assert E50G._generated_record_contract_failure(
        generated=record,
        problem=problem,
        judge_model="judge",
        attempt_index=0,
    ) == "execution_response_hash_mismatch"


def test_second_record_contract_reconstructs_exclusion_bound_request():
    problem = {
        "problem_order": 3,
        "source_index": 17,
        "unique_id": "row",
        "problem": "problem only",
    }
    first = {
        "proposal": {
            "methods": [
                {"label": "calculus", "actions": ["a", "b"]},
                {"label": "sharp inequality", "actions": ["c", "d"]},
            ]
        }
    }
    excluded = ("calculus", "sharp inequality")
    proposal = {
        "methods": [
            {"label": "the Euclidean algorithm", "actions": ["a", "b"]},
            {"label": "prime factorization", "actions": ["c", "d"]},
        ],
        "decisive_difference": "different engines",
    }
    request = E50G.E50H._proposal_payload(
        "judge", "problem only", 3, excluded
    )
    request_hash = hashlib.sha256(
        json.dumps(
            request, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    executions = []
    for route_index in (0, 1):
        for choice_index in range(8):
            text = f"route {route_index} choice {choice_index}"
            executions.append(
                {
                    "route_index": route_index,
                    "choice_index": choice_index,
                    "response_id": f"batch-{route_index}",
                    "finish_reason": "stop",
                    "response": text,
                    "response_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                }
            )
    record = {
        "problem_order": 3,
        "source_index": 17,
        "row_id": "row",
        "attempt_index": 1,
        "proposal": proposal,
        "proposal_response_id": "proposal",
        "proposal_finish_reason": "stop",
        "proposal_request_sha256": request_hash,
        "excluded_first_pair_labels": list(excluded),
        "repeated_excluded_label_pair": False,
        "answer_was_not_in_proposal_payload": True,
        "answer_was_not_in_execution_payloads": True,
        "executions": executions,
    }
    assert E50G._generated_record_contract_failure(
        generated=record,
        problem=problem,
        judge_model="judge",
        attempt_index=1,
        first_generated=first,
    ) is None
    record["excluded_first_pair_labels"][0] = "tampered"
    assert E50G._generated_record_contract_failure(
        generated=record,
        problem=problem,
        judge_model="judge",
        attempt_index=1,
        first_generated=first,
    ) == "excluded_first_pair_mismatch"


def test_third_record_contract_reconstructs_both_exclusion_bound_requests():
    problem = {
        "problem_order": 3,
        "source_index": 17,
        "unique_id": "row",
        "problem": "problem only",
    }
    first = {
        "proposal": {
            "methods": [
                {"label": "calculus", "actions": ["a", "b"]},
                {"label": "sharp inequality", "actions": ["c", "d"]},
            ]
        }
    }
    second = {
        "proposal": {
            "methods": [
                {"label": "Euclidean algorithm", "actions": ["a", "b"]},
                {"label": "prime factorization", "actions": ["c", "d"]},
            ]
        }
    }
    excluded = (
        ("calculus", "sharp inequality"),
        ("Euclidean algorithm", "prime factorization"),
    )
    proposal = {
        "methods": [
            {"label": "vector geometry", "actions": ["a", "b"]},
            {"label": "synthetic geometry", "actions": ["c", "d"]},
        ],
        "decisive_difference": "different engines",
    }
    request = E50G.E50I._proposal_payload(
        "judge", "problem only", 3, excluded
    )
    request_hash = hashlib.sha256(
        json.dumps(
            request, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    executions = []
    for route_index in (0, 1):
        for choice_index in range(8):
            text = f"route {route_index} choice {choice_index}"
            executions.append(
                {
                    "route_index": route_index,
                    "choice_index": choice_index,
                    "response_id": f"batch-{route_index}",
                    "finish_reason": "stop",
                    "response": text,
                    "response_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                }
            )
    record = {
        "problem_order": 3,
        "source_index": 17,
        "row_id": "row",
        "attempt_index": 2,
        "proposal": proposal,
        "proposal_response_id": "proposal",
        "proposal_finish_reason": "stop",
        "proposal_request_sha256": request_hash,
        "excluded_first_two_pair_labels": [
            list(pair) for pair in excluded
        ],
        "repeated_excluded_label_pair": False,
        "answer_was_not_in_proposal_payload": True,
        "answer_was_not_in_execution_payloads": True,
        "executions": executions,
    }
    assert E50G._generated_record_contract_failure(
        generated=record,
        problem=problem,
        judge_model="judge",
        attempt_index=2,
        first_generated=first,
        second_generated=second,
    ) is None
    record["excluded_first_two_pair_labels"][1][0] = "tampered"
    assert E50G._generated_record_contract_failure(
        generated=record,
        problem=problem,
        judge_model="judge",
        attempt_index=2,
        first_generated=first,
        second_generated=second,
    ) == "excluded_first_two_pairs_mismatch"


def test_fourth_record_contract_reconstructs_all_available_exclusions():
    problem = {
        "problem_order": 3,
        "source_index": 17,
        "unique_id": "row",
        "problem": "problem only",
    }
    priors = [
        {
            "proposal": {
                "methods": [
                    {"label": left, "actions": ["a", "b"]},
                    {"label": right, "actions": ["c", "d"]},
                ]
            }
        }
        for left, right in (
            ("calculus", "sharp inequality"),
            ("Euclidean algorithm", "prime factorization"),
            ("vector geometry", "synthetic geometry"),
        )
    ]
    excluded = tuple(
        (
            prior["proposal"]["methods"][0]["label"],
            prior["proposal"]["methods"][1]["label"],
        )
        for prior in priors
    )
    proposal = {
        "methods": [
            {"label": "dynamic programming", "actions": ["a", "b"]},
            {"label": "closed-form count", "actions": ["c", "d"]},
        ],
        "decisive_difference": "different engines",
    }
    request = E50G.E50J._proposal_payload(
        "judge", "problem only", 3, excluded
    )
    request_hash = hashlib.sha256(
        json.dumps(
            request, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    executions = []
    for route_index in (0, 1):
        for choice_index in range(8):
            text = f"route {route_index} choice {choice_index}"
            executions.append(
                {
                    "route_index": route_index,
                    "choice_index": choice_index,
                    "response_id": f"batch-{route_index}",
                    "finish_reason": "stop",
                    "response": text,
                    "response_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                }
            )
    record = {
        "problem_order": 3,
        "source_index": 17,
        "row_id": "row",
        "attempt_index": 3,
        "proposal": proposal,
        "proposal_response_id": "proposal",
        "proposal_finish_reason": "stop",
        "proposal_request_sha256": request_hash,
        "excluded_prior_pair_labels": [list(pair) for pair in excluded],
        "repeated_excluded_label_pair": False,
        "answer_was_not_in_proposal_payload": True,
        "answer_was_not_in_execution_payloads": True,
        "executions": executions,
    }
    assert E50G._generated_record_contract_failure(
        generated=record,
        problem=problem,
        judge_model="judge",
        attempt_index=3,
        first_generated=priors[0],
        second_generated=priors[1],
        third_generated=priors[2],
    ) is None
    record["excluded_prior_pair_labels"][2][0] = "tampered"
    assert E50G._generated_record_contract_failure(
        generated=record,
        problem=problem,
        judge_model="judge",
        attempt_index=3,
        first_generated=priors[0],
        second_generated=priors[1],
        third_generated=priors[2],
    ) == "excluded_prior_pairs_mismatch"


def test_fifth_record_contract_reconstructs_all_available_exclusions():
    problem = {
        "problem_order": 3,
        "source_index": 17,
        "unique_id": "row",
        "problem": "problem only",
    }
    priors = [
        {
            "proposal": {
                "methods": [
                    {"label": left, "actions": ["a", "b"]},
                    {"label": right, "actions": ["c", "d"]},
                ]
            }
        }
        for left, right in (
            ("calculus", "sharp inequality"),
            ("Euclidean algorithm", "prime factorization"),
            ("vector geometry", "synthetic geometry"),
            ("dynamic programming", "closed-form count"),
        )
    ]
    excluded = tuple(
        (
            prior["proposal"]["methods"][0]["label"],
            prior["proposal"]["methods"][1]["label"],
        )
        for prior in priors
    )
    proposal = {
        "methods": [
            {"label": "complementary count", "actions": ["a", "b"]},
            {"label": "inclusion-exclusion", "actions": ["c", "d"]},
        ],
        "decisive_difference": "different engines",
    }
    request = E50G.E50K._proposal_payload(
        "judge", "problem only", 3, excluded
    )
    request_hash = hashlib.sha256(
        json.dumps(
            request, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    executions = []
    for route_index in (0, 1):
        for choice_index in range(8):
            text = f"route {route_index} choice {choice_index}"
            executions.append(
                {
                    "route_index": route_index,
                    "choice_index": choice_index,
                    "response_id": f"batch-{route_index}",
                    "finish_reason": "stop",
                    "response": text,
                    "response_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                }
            )
    record = {
        "problem_order": 3,
        "source_index": 17,
        "row_id": "row",
        "attempt_index": 4,
        "proposal": proposal,
        "proposal_response_id": "proposal",
        "proposal_finish_reason": "stop",
        "proposal_request_sha256": request_hash,
        "excluded_prior_pair_labels": [list(pair) for pair in excluded],
        "repeated_excluded_label_pair": False,
        "answer_was_not_in_proposal_payload": True,
        "answer_was_not_in_execution_payloads": True,
        "executions": executions,
    }
    assert E50G._generated_record_contract_failure(
        generated=record,
        problem=problem,
        judge_model="judge",
        attempt_index=4,
        first_generated=priors[0],
        second_generated=priors[1],
        third_generated=priors[2],
        fourth_generated=priors[3],
    ) is None
    record["excluded_prior_pair_labels"][3][0] = "tampered"
    assert E50G._generated_record_contract_failure(
        generated=record,
        problem=problem,
        judge_model="judge",
        attempt_index=4,
        first_generated=priors[0],
        second_generated=priors[1],
        third_generated=priors[2],
        fourth_generated=priors[3],
    ) == "excluded_prior_pairs_mismatch"


def test_candidate_attempt_ranking_is_frozen_and_one_per_problem():
    rows = [
        {
            "problem_order": 0,
            "attempt_index": 0,
            "minimum_teacher_execution_count": 2,
            "combined_teacher_execution_count": 8,
        },
        {
            "problem_order": 0,
            "attempt_index": 1,
            "minimum_teacher_execution_count": 3,
            "combined_teacher_execution_count": 6,
        },
        {
            "problem_order": 1,
            "attempt_index": 0,
            "minimum_teacher_execution_count": 4,
            "combined_teacher_execution_count": 9,
        },
        {
            "problem_order": 1,
            "attempt_index": 1,
            "minimum_teacher_execution_count": 4,
            "combined_teacher_execution_count": 9,
        },
    ]
    selected = E50G._select_one_candidate_per_problem(rows)
    assert [row["attempt_index"] for row in selected] == [1, 0]
    assert all(row["eligible_attempt_count"] == 2 for row in selected)


def test_downstream_toy_launcher_is_fail_closed_to_e50g_only():
    source = PREPARE_NEXT.read_text(encoding="utf-8")
    assert "e50g_safe_signature_teacher_route_calibration_v1" in source
    assert "variant=e50d" in source
    for stale in ("e49aa", "e49ab", "e49ac", "variant=e50b"):
        assert stale not in source
    materializer = MATERIALIZE_E50D.read_text(encoding="utf-8")
    assert "e50g_safe_signature_teacher_route_calibration_v1" in materializer
    assert ".cross_rendering_false_new_count == 0" in materializer
    assert ".wrong_route_success_count == 0" in materializer
    for stale in (
        "e50c_72b_teacher_route_calibration_v1",
        "e50f_conditioned_teacher_route_calibration_v1",
    ):
        assert stale not in materializer
    launch = LAUNCH_E50G.read_text(encoding="utf-8")
    assert "e50k_fifth_conditioned_teacher_contingency_job.txt" in launch
    assert '--dependency="afterany:$e50k_job"' in launch
    assert "e50f_job" not in launch
    assert "e50i_job" not in launch
    assert "e50j_job" not in launch


def test_activation_requires_hashed_quarantine_and_corpus_results(
    tmp_path, monkeypatch
) -> None:
    generated = tmp_path / "generated.jsonl"
    generated.write_text("{}\n", encoding="utf-8")
    generated_sha = hashlib.sha256(generated.read_bytes()).hexdigest()

    e50c = tmp_path / "e50c.json"
    e50c.write_text(
        json.dumps(
            {
                "schema": "e50c_72b_teacher_route_calibration_v1",
                "pass": False,
                "failure": (
                    "open_relation_judge_quarantined_before_route_discovery"
                ),
                "teacher_completed_problem_count": 36,
                "teacher_logged_sample_count": 576,
                "teacher_private_response_count": 0,
                "selected_source_indices": [],
                "wrong_route_success_count": 0,
                "identity": {
                    "e50c_script_sha256": hashlib.sha256(
                        E50G.E50C.SCRIPT.read_bytes()
                    ).hexdigest(),
                    "quarantine_script_sha256": hashlib.sha256(
                        E50G.E50C_QUARANTINE_SCRIPT.read_bytes()
                    ).hexdigest(),
                    "quarantine_protocol_sha256": hashlib.sha256(
                        E50G.E50C_QUARANTINE_PROTOCOL.read_bytes()
                    ).hexdigest(),
                    "endpoint_record_sha256": hashlib.sha256(
                        E50G.E50C.ENDPOINT_RECORD.read_bytes()
                    ).hexdigest(),
                    "e47_manifest_sha256": hashlib.sha256(
                        E50G.E50C.E47_MANIFEST.read_bytes()
                    ).hexdigest(),
                    "e47_problems_sha256": hashlib.sha256(
                        E50G.E50C.E47_PROBLEMS.read_bytes()
                    ).hexdigest(),
                    "false_new_failure_sha256": {
                        schema: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path, schema in E50G.OPEN_RELATION_FAILURES
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    e50f = tmp_path / "e50f.json"
    e50f.write_text(
        json.dumps(
            {
                "schema": "e50f_conditioned_teacher_route_calibration_v1",
                "pass": False,
                "failure": (
                    "open_relation_judge_quarantined_after_frozen_"
                    "conditioned_corpus_generation"
                ),
                "generation_problem_count": 50,
                "generation_error_count": 0,
                "conditioned_execution_count": 800,
                "selected_source_indices": [],
                "conditioned_teacher_records_sha256": generated_sha,
                "identity": {
                    "script_sha256": hashlib.sha256(
                        E50G.BASE_PATH.read_bytes()
                    ).hexdigest(),
                    "protocol_sha256": hashlib.sha256(
                        E50G.BASE.PROTOCOL.read_bytes()
                    ).hexdigest(),
                    "e50c_result_sha256": hashlib.sha256(
                        e50c.read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    second_generated = tmp_path / "second_generated.jsonl"
    second_generated.write_text("{}\n", encoding="utf-8")
    second_generated_sha = hashlib.sha256(
        second_generated.read_bytes()
    ).hexdigest()
    e50h = tmp_path / "e50h.json"
    e50h.write_text(
        json.dumps(
            {
                "schema": "e50h_second_conditioned_teacher_corpus_v1",
                "pass": False,
                "failure": "corpus_only_second_attempt_no_training_authority",
                "generation_problem_count": 50,
                "generation_error_count": 0,
                "conditioned_execution_count": 800,
                "selected_source_indices": [],
                "second_conditioned_records_sha256": second_generated_sha,
                "identity": {
                    "script_sha256": hashlib.sha256(
                        E50G.E50H_PATH.read_bytes()
                    ).hexdigest(),
                    "protocol_sha256": hashlib.sha256(
                        E50G.E50H.PROTOCOL.read_bytes()
                    ).hexdigest(),
                    "base_e50f_script_sha256": hashlib.sha256(
                        E50G.BASE_PATH.read_bytes()
                    ).hexdigest(),
                    "e50f_result_sha256": hashlib.sha256(
                        e50f.read_bytes()
                    ).hexdigest(),
                    "e50f_generated_sha256": generated_sha,
                    "endpoint_record_sha256": hashlib.sha256(
                        E50G.E50C.ENDPOINT_RECORD.read_bytes()
                    ).hexdigest(),
                    "e47_manifest_sha256": hashlib.sha256(
                        E50G.E50C.E47_MANIFEST.read_bytes()
                    ).hexdigest(),
                    "e47_problems_sha256": hashlib.sha256(
                        E50G.E50C.E47_PROBLEMS.read_bytes()
                    ).hexdigest(),
                    "proposal_seed": E50G.E50H.PROPOSAL_SEED,
                    "execution_seed": E50G.E50H.EXECUTION_SEED,
                },
            }
        ),
        encoding="utf-8",
    )
    third_generated = tmp_path / "third_generated.jsonl"
    third_generated.write_text("", encoding="utf-8")
    third_generated_sha = hashlib.sha256(
        third_generated.read_bytes()
    ).hexdigest()
    prethird_orders = list(range(E50G.E50I.TRIGGER_THRESHOLD))
    e50i = tmp_path / "e50i.json"
    e50i.write_text(
        json.dumps(
            {
                "schema": "e50i_third_conditioned_teacher_contingency_v1",
                "pass": False,
                "failure": "corpus_only_third_attempt_no_training_authority",
                "triggered": False,
                "trigger_threshold": E50G.E50I.TRIGGER_THRESHOLD,
                "prethird_eligible_problem_count": len(prethird_orders),
                "prethird_eligible_problem_orders": prethird_orders,
                "attempted_problem_orders": [],
                "generation_problem_count": 0,
                "generation_error_count": 0,
                "conditioned_execution_count": 0,
                "selected_source_indices": [],
                "third_conditioned_records_sha256": third_generated_sha,
                "identity": {
                    "script_sha256": hashlib.sha256(
                        E50G.E50I_PATH.read_bytes()
                    ).hexdigest(),
                    "protocol_sha256": hashlib.sha256(
                        E50G.E50I.PROTOCOL.read_bytes()
                    ).hexdigest(),
                    "e50h_script_sha256": hashlib.sha256(
                        E50G.E50H_PATH.read_bytes()
                    ).hexdigest(),
                    "signature_source_sha256": hashlib.sha256(
                        E50G.SIGNATURE_PATH.read_bytes()
                    ).hexdigest(),
                    "e50f_result_sha256": hashlib.sha256(
                        e50f.read_bytes()
                    ).hexdigest(),
                    "e50f_generated_sha256": generated_sha,
                    "e50h_result_sha256": hashlib.sha256(
                        e50h.read_bytes()
                    ).hexdigest(),
                    "e50h_generated_sha256": second_generated_sha,
                    "endpoint_record_sha256": hashlib.sha256(
                        E50G.E50C.ENDPOINT_RECORD.read_bytes()
                    ).hexdigest(),
                    "e47_manifest_sha256": hashlib.sha256(
                        E50G.E50C.E47_MANIFEST.read_bytes()
                    ).hexdigest(),
                    "e47_problems_sha256": hashlib.sha256(
                        E50G.E50C.E47_PROBLEMS.read_bytes()
                    ).hexdigest(),
                    "prethird_eligible_problem_orders": prethird_orders,
                    "trigger_threshold": E50G.E50I.TRIGGER_THRESHOLD,
                    "proposal_seed": E50G.E50I.PROPOSAL_SEED,
                    "execution_seed": E50G.E50I.EXECUTION_SEED,
                },
            }
        ),
        encoding="utf-8",
    )
    fourth_generated = tmp_path / "fourth_generated.jsonl"
    fourth_generated.write_text("", encoding="utf-8")
    fourth_generated_sha = hashlib.sha256(
        fourth_generated.read_bytes()
    ).hexdigest()
    prefourth_orders = list(range(E50G.E50J.TRIGGER_THRESHOLD))
    e50j = tmp_path / "e50j.json"
    e50j.write_text(
        json.dumps(
            {
                "schema": "e50j_fourth_conditioned_teacher_contingency_v1",
                "pass": False,
                "failure": "corpus_only_fourth_attempt_no_training_authority",
                "triggered": False,
                "trigger_threshold": E50G.E50J.TRIGGER_THRESHOLD,
                "prefourth_eligible_problem_count": len(prefourth_orders),
                "prefourth_eligible_problem_orders": prefourth_orders,
                "attempted_problem_orders": [],
                "generation_problem_count": 0,
                "generation_error_count": 0,
                "conditioned_execution_count": 0,
                "selected_source_indices": [],
                "fourth_conditioned_records_sha256": fourth_generated_sha,
                "identity": {
                    "script_sha256": hashlib.sha256(
                        E50G.E50J_PATH.read_bytes()
                    ).hexdigest(),
                    "protocol_sha256": hashlib.sha256(
                        E50G.E50J.PROTOCOL.read_bytes()
                    ).hexdigest(),
                    "e50i_script_sha256": hashlib.sha256(
                        E50G.E50I_PATH.read_bytes()
                    ).hexdigest(),
                    "signature_source_sha256": hashlib.sha256(
                        E50G.SIGNATURE_PATH.read_bytes()
                    ).hexdigest(),
                    "e50f_result_sha256": hashlib.sha256(
                        e50f.read_bytes()
                    ).hexdigest(),
                    "e50f_generated_sha256": generated_sha,
                    "e50h_result_sha256": hashlib.sha256(
                        e50h.read_bytes()
                    ).hexdigest(),
                    "e50h_generated_sha256": second_generated_sha,
                    "e50i_result_sha256": hashlib.sha256(
                        e50i.read_bytes()
                    ).hexdigest(),
                    "e50i_generated_sha256": third_generated_sha,
                    "endpoint_record_sha256": hashlib.sha256(
                        E50G.E50C.ENDPOINT_RECORD.read_bytes()
                    ).hexdigest(),
                    "e47_manifest_sha256": hashlib.sha256(
                        E50G.E50C.E47_MANIFEST.read_bytes()
                    ).hexdigest(),
                    "e47_problems_sha256": hashlib.sha256(
                        E50G.E50C.E47_PROBLEMS.read_bytes()
                    ).hexdigest(),
                    "prefourth_eligible_problem_orders": prefourth_orders,
                    "trigger_threshold": E50G.E50J.TRIGGER_THRESHOLD,
                    "proposal_seed": E50G.E50J.PROPOSAL_SEED,
                    "execution_seed": E50G.E50J.EXECUTION_SEED,
                },
            }
        ),
        encoding="utf-8",
    )
    fifth_generated = tmp_path / "fifth_generated.jsonl"
    fifth_generated.write_text("", encoding="utf-8")
    fifth_generated_sha = hashlib.sha256(
        fifth_generated.read_bytes()
    ).hexdigest()
    prefifth_orders = list(range(E50G.E50K.TRIGGER_THRESHOLD))
    e50k = tmp_path / "e50k.json"
    e50k.write_text(
        json.dumps(
            {
                "schema": "e50k_fifth_conditioned_teacher_contingency_v1",
                "pass": False,
                "failure": "corpus_only_fifth_attempt_no_training_authority",
                "triggered": False,
                "trigger_threshold": E50G.E50K.TRIGGER_THRESHOLD,
                "prefifth_eligible_problem_count": len(prefifth_orders),
                "prefifth_eligible_problem_orders": prefifth_orders,
                "attempted_problem_orders": [],
                "generation_problem_count": 0,
                "generation_error_count": 0,
                "conditioned_execution_count": 0,
                "selected_source_indices": [],
                "fifth_conditioned_records_sha256": fifth_generated_sha,
                "identity": {
                    "script_sha256": hashlib.sha256(
                        E50G.E50K_PATH.read_bytes()
                    ).hexdigest(),
                    "protocol_sha256": hashlib.sha256(
                        E50G.E50K.PROTOCOL.read_bytes()
                    ).hexdigest(),
                    "e50j_script_sha256": hashlib.sha256(
                        E50G.E50J_PATH.read_bytes()
                    ).hexdigest(),
                    "signature_source_sha256": hashlib.sha256(
                        E50G.SIGNATURE_PATH.read_bytes()
                    ).hexdigest(),
                    "e50f_result_sha256": hashlib.sha256(
                        e50f.read_bytes()
                    ).hexdigest(),
                    "e50f_generated_sha256": generated_sha,
                    "e50h_result_sha256": hashlib.sha256(
                        e50h.read_bytes()
                    ).hexdigest(),
                    "e50h_generated_sha256": second_generated_sha,
                    "e50i_result_sha256": hashlib.sha256(
                        e50i.read_bytes()
                    ).hexdigest(),
                    "e50i_generated_sha256": third_generated_sha,
                    "e50j_result_sha256": hashlib.sha256(
                        e50j.read_bytes()
                    ).hexdigest(),
                    "e50j_generated_sha256": fourth_generated_sha,
                    "endpoint_record_sha256": hashlib.sha256(
                        E50G.E50C.ENDPOINT_RECORD.read_bytes()
                    ).hexdigest(),
                    "e47_manifest_sha256": hashlib.sha256(
                        E50G.E50C.E47_MANIFEST.read_bytes()
                    ).hexdigest(),
                    "e47_problems_sha256": hashlib.sha256(
                        E50G.E50C.E47_PROBLEMS.read_bytes()
                    ).hexdigest(),
                    "prefifth_eligible_problem_orders": prefifth_orders,
                    "trigger_threshold": E50G.E50K.TRIGGER_THRESHOLD,
                    "proposal_seed": E50G.E50K.PROPOSAL_SEED,
                    "execution_seed": E50G.E50K.EXECUTION_SEED,
                },
            }
        ),
        encoding="utf-8",
    )
    development = tmp_path / "development.json"
    development.write_text(
        json.dumps(
            {
                "schema": "safe_math_strategy_signature_development_audit_v1",
                "counts": {
                    "false_new_count": 0,
                    "false_merge_count": 1,
                    "sound_pair_count": 18,
                },
            }
        ),
        encoding="utf-8",
    )
    rewrite = tmp_path / "rewrite.json"
    rewrite.write_text(
        json.dumps(
            {
                "schema": "e50g_full_injection_rewrite_audit_v1",
                "pass": True,
                "counts": {
                    "problem_count": 50,
                    "comparison_count": 150,
                    "false_new_count": 0,
                    "signature_change_count": 0,
                },
                "identity": {
                    "signature_source": hashlib.sha256(
                        E50G.SIGNATURE_PATH.read_bytes()
                    ).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    route_confusion = tmp_path / "route_confusion.json"
    route_confusion.write_text(
        json.dumps(
            {
                "schema": "e49t_route_confusion_calibration_result_v1",
                "pass": True,
            }
        ),
        encoding="utf-8",
    )
    declaration = tmp_path / "declaration.json"
    declaration.write_text(
        json.dumps(
            {
                "schema": "e49t_declaration_mismatch_result_v1",
                "pass": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(E50G, "E50C_RESULT", e50c)
    monkeypatch.setattr(E50G, "E50F_RESULT", e50f)
    monkeypatch.setattr(E50G, "E50F_GENERATED", generated)
    monkeypatch.setattr(E50G, "E50H_RESULT", e50h)
    monkeypatch.setattr(E50G, "E50H_GENERATED", second_generated)
    monkeypatch.setattr(E50G, "E50I_RESULT", e50i)
    monkeypatch.setattr(E50G, "E50I_GENERATED", third_generated)
    monkeypatch.setattr(E50G, "E50J_RESULT", e50j)
    monkeypatch.setattr(E50G, "E50J_GENERATED", fourth_generated)
    monkeypatch.setattr(E50G, "E50K_RESULT", e50k)
    monkeypatch.setattr(E50G, "E50K_GENERATED", fifth_generated)
    monkeypatch.setattr(E50G, "DEVELOPMENT_AUDIT", development)
    monkeypatch.setattr(E50G, "FULL_REWRITE_AUDIT", rewrite)
    monkeypatch.setattr(E50G, "ROUTE_CONFUSION_RESULT", route_confusion)
    monkeypatch.setattr(E50G, "DECLARATION_MISMATCH_RESULT", declaration)
    (
        observed_c,
        observed_f,
        observed_h,
        observed_i,
        observed_j,
        observed_k,
    ) = (
        E50G._verify_activation()
    )
    assert observed_c["pass"] is False
    assert observed_f["pass"] is False
    assert observed_h["pass"] is False
    assert observed_i["pass"] is False
    assert observed_j["pass"] is False
    assert observed_k["pass"] is False
