from __future__ import annotations

import json
import random
import re

from oat_drgrpo.math_strategy_canonicalizer import MathStrategyCanonicalizer


def _items(payload):
    prompt = payload["messages"][1]["content"]
    rendered = re.split(
        r"(?:(?:SOLUTIONS|ANSWER-MATCHED RESPONSES) "
        r"\(opaque IDs, randomly permuted\):|ITEMS:)\n",
        prompt,
        maxsplit=1,
    )[1]
    return re.findall(
        r"### ID ([A-Z]+_\d+)\n(.*?)(?=\n\n### ID |\Z)",
        rendered,
        flags=re.DOTALL,
    )


def _response(clusters, ambiguous=()):
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "clusters": [
                                {
                                    "cluster_id": f"c{index}",
                                    "strategy": strategy,
                                    "member_ids": members,
                                }
                                for index, (strategy, members) in enumerate(
                                    clusters, start=1
                                )
                            ],
                            "ambiguous_ids": list(ambiguous),
                        }
                    )
                }
            }
        ]
    }


def _is_integrity(payload):
    schema = payload["response_format"]["json_schema"]["schema"]
    assessments = schema["properties"].get("assessments", {})
    properties = assessments.get("items", {}).get("properties", {})
    return "status" in properties


def _is_relation(payload):
    schema = payload["response_format"]["json_schema"]["schema"]
    assessments = schema["properties"].get("assessments", {})
    properties = assessments.get("items", {}).get("properties", {})
    return "relation" in properties


def _relation_pairs(payload):
    prompt = payload["messages"][1]["content"]
    return re.findall(
        r"(PAIR_\d+): ([A-Z]+_\d+) versus ([A-Z]+_\d+)",
        prompt,
    )


def _relation_response(relations):
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "assessments": [
                                {
                                    "pair_id": pair_id,
                                    "brief_check": "synthetic pair comparison",
                                    "relation": relation,
                                }
                                for pair_id, relation in relations
                            ]
                        }
                    )
                }
            }
        ]
    }


def _partition_response_as_relations(payload, response):
    content = json.loads(response["choices"][0]["message"]["content"])
    assignment = {}
    for cluster in content["clusters"]:
        for member_id in cluster["member_ids"]:
            assignment[member_id] = cluster["cluster_id"]
    ambiguous = set(content["ambiguous_ids"])
    return _relation_response(
        [
            (
                pair_id,
                "ambiguous"
                if (
                    left in ambiguous
                    or right in ambiguous
                    or left not in assignment
                    or right not in assignment
                )
                else (
                    "same"
                    if assignment[left] == assignment[right]
                    else "different"
                ),
            )
            for pair_id, left, right in _relation_pairs(payload)
        ]
    )


def _integrity_response(statuses):
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "assessments": [
                                {
                                    "item_id": item_id,
                                    "brief_check": "synthetic test decision",
                                    "status": status,
                                }
                                for item_id, status in statuses
                            ]
                        }
                    )
                }
            }
        ]
    }


def _with_valid_integrity(partition_transport):
    def transport(payload):
        if _is_integrity(payload):
            return _integrity_response(
                [(item_id, "valid") for item_id, _ in _items(payload)]
            )
        response = partition_transport(payload)
        if _is_relation(payload):
            return _partition_response_as_relations(payload, response)
        return response

    return transport


def _semantic_transport(payload):
    if _is_integrity(payload):
        return _integrity_response(
            [
                (
                    item_id,
                    "invalid" if "INCOHERENT" in text else "valid",
                )
                for item_id, text in _items(payload)
            ]
        )
    clusters = {}
    ambiguous = []
    for item_id, text in _items(payload):
        if "INCOHERENT" in text:
            ambiguous.append(item_id)
        elif "FACTOR" in text:
            clusters.setdefault("factor", []).append(item_id)
        elif "GEOMETRY" in text:
            clusters.setdefault("geometry", []).append(item_id)
        else:
            clusters.setdefault("other", []).append(item_id)
    response = _response(list(clusters.items()), ambiguous)
    if _is_relation(payload):
        return _partition_response_as_relations(payload, response)
    schema = payload["response_format"]["json_schema"]["schema"]
    cluster = schema["properties"]["clusters"]["items"]
    assert "strategy" not in cluster["properties"]
    assert schema["properties"]["clusters"]["maxItems"] > 0
    assert cluster["properties"]["member_ids"]["maxItems"] > 0
    return response


def test_two_pass_admission_is_validator_gated_and_persistent():
    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_semantic_transport,
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[1, 2]] * 5,
        prompt_texts=["problem"] * 5,
        response_texts=[
            "FACTOR route one",
            "FACTOR paraphrase",
            "GEOMETRY route",
            "INCOHERENT",
            "FACTOR but validator rejected",
        ],
        task_reward_positive=[True, True, True, True, False],
        active_mask=[True] * 5,
        num_samples=5,
    )
    assert keys[0] == keys[1]
    assert keys[0] != keys[2]
    assert keys[3:] == [None, None]
    assert diagnostics.validator_positive_rows == 4
    assert diagnostics.accepted_rows == 3
    assert diagnostics.rejected_integrity_rows == 1
    assert diagnostics.rejected_ambiguous_rows == 0
    assert diagnostics.new_strategy_count == 2

    second_keys, second_diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[1, 2]] * 2,
        prompt_texts=["problem"] * 2,
        response_texts=["FACTOR a third way of writing it", "rejected"],
        task_reward_positive=[True, False],
        active_mask=[True, True],
        num_samples=2,
    )
    assert second_keys == [keys[0], None]
    assert second_diagnostics.matched_existing_rows == 1
    assert second_diagnostics.new_strategy_count == 0


def test_permutation_disagreement_coarsens_instead_of_false_new():
    def disagreeing_transport(payload):
        ids = [item_id for item_id, _ in _items(payload)]
        if payload["seed"] == 470721:
            return _response([("same", ids)])
        return _response(
            [("left", [ids[0]]), ("right", ids[1:])]
        )

    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_with_valid_integrity(disagreeing_transport),
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[3]] * 2,
        prompt_texts=["problem"] * 2,
        response_texts=["solution a", "solution b"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert keys[0] == keys[1]
    assert diagnostics.rejected_disagreement_rows == 0
    assert diagnostics.new_strategy_count == 1


def test_two_pass_union_uses_transitive_components():
    def chained_transport(payload):
        ids = sorted(item_id for item_id, _ in _items(payload))
        assert ids == ["CAND_0000", "CAND_0001", "CAND_0002"]
        if payload["seed"] == 470721:
            return _response([("ab", ids[:2]), ("c", ids[2:])])
        return _response([("a", ids[:1]), ("bc", ids[1:])])

    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_with_valid_integrity(chained_transport),
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[4]] * 3,
        prompt_texts=["problem"] * 3,
        response_texts=["route a", "route b", "route c"],
        task_reward_positive=[True] * 3,
        active_mask=[True] * 3,
        num_samples=3,
    )
    assert len(set(keys)) == 1
    assert diagnostics.new_strategy_count == 1
    assert diagnostics.new_strategy_rows == 3


def test_state_round_trip_preserves_strategy_ids():
    first = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_semantic_transport,
    )
    keys, _ = first.canonicalize(
        prompt_token_ids=[[8], [8]],
        prompt_texts=["problem", "problem"],
        response_texts=["FACTOR", "GEOMETRY"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    second = MathStrategyCanonicalizer(
        endpoint="http://new-judge/v1",
        transport=_semantic_transport,
    )
    second.load_state_dict(first.state_dict())
    restored_keys, diagnostics = second.canonicalize(
        prompt_token_ids=[[8], [8]],
        prompt_texts=["problem", "problem"],
        response_texts=["FACTOR wording", "GEOMETRY wording"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert restored_keys == keys
    assert diagnostics.matched_existing_rows == 2


def _malformed_response():
    return {
        "choices": [
            {
                "message": {
                    "content": '{"assessments":[{"item_id":"truncated'
                }
            }
        ]
    }


def test_structurally_invalid_integrity_json_fails_group_closed():
    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=lambda payload: _malformed_response(),
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[9], [9]],
        prompt_texts=["problem", "problem"],
        response_texts=["a", "b"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert keys == [None, None]
    assert diagnostics.judge_calls == 2
    assert diagnostics.rejected_ambiguous_rows == 2
    assert diagnostics.judge_format_failure_rows == 2


def test_structurally_invalid_partition_json_fails_group_closed():
    def transport(payload):
        if _is_integrity(payload):
            return _integrity_response(
                [(item_id, "valid") for item_id, _ in _items(payload)]
            )
        return _malformed_response()

    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=transport,
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[15], [15]],
        prompt_texts=["problem", "problem"],
        response_texts=["a", "b"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert keys == [None, None]
    assert diagnostics.judge_calls == 4
    assert diagnostics.rejected_ambiguous_rows == 2
    assert diagnostics.judge_format_failure_rows == 2


def test_structurally_invalid_relation_json_fails_group_closed():
    def transport(payload):
        if _is_integrity(payload):
            return _integrity_response(
                [(item_id, "valid") for item_id, _ in _items(payload)]
            )
        if _is_relation(payload):
            return _malformed_response()
        ids = sorted(item_id for item_id, _ in _items(payload))
        return _response(
            [("left", [ids[0]]), ("right", [ids[1]])]
        )

    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=transport,
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[16], [16]],
        prompt_texts=["problem", "problem"],
        response_texts=["route a", "route b"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert keys == [None, None]
    assert diagnostics.judge_calls == 6
    assert diagnostics.rejected_ambiguous_rows == 2
    assert diagnostics.judge_format_failure_rows == 2


def test_omitted_id_is_rejected_as_ambiguous_not_rewarded():
    def omitting_transport(payload):
        ids = [item_id for item_id, _ in _items(payload)]
        return _response(
            [("only", [item_id for item_id in ids if item_id != "CAND_0000"])]
        )

    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_with_valid_integrity(omitting_transport),
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[10], [10]],
        prompt_texts=["problem", "problem"],
        response_texts=["omitted", "present"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert keys[0] is None
    assert keys[1] is not None
    assert diagnostics.rejected_ambiguous_rows == 1


def test_omitted_stored_representative_rejects_round_without_graph_error():
    def omitting_rep_transport(payload):
        ids = sorted(item_id for item_id, _ in _items(payload))
        representatives = [
            item_id for item_id in ids if item_id.startswith("REP_")
        ]
        candidates = [
            item_id for item_id in ids if item_id.startswith("CAND_")
        ]
        if not representatives:
            return _response([("seed", candidates)])
        return _response([("candidates", candidates)])

    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_with_valid_integrity(omitting_rep_transport),
    )
    canonicalizer.canonicalize(
        prompt_token_ids=[[12], [12]],
        prompt_texts=["problem", "problem"],
        response_texts=["seed", "seed paraphrase"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[12], [12]],
        prompt_texts=["problem", "problem"],
        response_texts=["candidate one", "candidate two"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert keys == [None, None]
    assert diagnostics.accepted_rows == 0
    assert diagnostics.rejected_disagreement_rows == 2
    assert diagnostics.new_strategy_count == 0


def test_unanimous_existing_representative_match_survives_unrelated_pair():
    call_count = 0

    def transport(payload):
        nonlocal call_count
        call_count += 1
        items = dict(_items(payload))
        ids = list(items)
        rep = next((item_id for item_id in ids if item_id.startswith("REP_")), None)
        if rep is None:
            return _response([("seed", ids)])
        candidates = sorted(
            item_id for item_id in ids if item_id.startswith("CAND_")
        )
        if payload["seed"] == 470721:
            return _response([("existing", [rep, *candidates])])
        if len(candidates) == 1:
            return _response([("existing", [rep, candidates[0]])])
        return _response(
            [
                ("existing", [rep, candidates[0]]),
                ("unstable", [candidates[1]]),
            ]
        )

    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_with_valid_integrity(transport),
    )
    first_keys, _ = canonicalizer.canonicalize(
        prompt_token_ids=[[11], [11]],
        prompt_texts=["problem", "problem"],
        response_texts=["seed", "seed paraphrase"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    second_keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[11], [11]],
        prompt_texts=["problem", "problem"],
        response_texts=["stable existing", "unstable candidate"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert second_keys == [first_keys[0], first_keys[0]]
    assert diagnostics.matched_existing_rows == 2
    assert diagnostics.rejected_disagreement_rows == 0
    assert call_count == 6


def test_conflicting_candidate_component_does_not_reject_unrelated_match():
    def component_transport(payload):
        items = dict(_items(payload))
        representatives = sorted(
            item_id
            for item_id in items
            if item_id.startswith("REP_")
        )
        candidates = sorted(
            item_id
            for item_id in items
            if item_id.startswith("CAND_")
        )
        if not representatives:
            return _response(
                [
                    ("first", [candidates[0]]),
                    ("second", [candidates[1]]),
                ]
            )
        if payload["seed"] == 470721:
            return _response(
                [
                    ("conflict", [*representatives, candidates[0]]),
                    ("unrelated", [candidates[1]]),
                ]
            )
        return _response(
            [
                ("first", [representatives[0], candidates[0]]),
                ("second", [representatives[1], candidates[1]]),
            ]
        )

    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_with_valid_integrity(component_transport),
    )
    first_keys, _ = canonicalizer.canonicalize(
        prompt_token_ids=[[14], [14]],
        prompt_texts=["problem", "problem"],
        response_texts=["first seed", "second seed"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert first_keys[0] != first_keys[1]
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[14], [14]],
        prompt_texts=["problem", "problem"],
        response_texts=["conflicting bridge", "stable second match"],
        task_reward_positive=[True, True],
        active_mask=[True, True],
        num_samples=2,
    )
    assert keys == [None, first_keys[1]]
    assert diagnostics.accepted_rows == 1
    assert diagnostics.matched_existing_rows == 1
    assert diagnostics.rejected_disagreement_rows == 1
    assert diagnostics.new_strategy_count == 0


def test_random_omissions_and_partition_disagreements_always_fail_closed():
    round_index = 0

    def adversarial_transport(payload):
        ids = sorted(item_id for item_id, _ in _items(payload))
        rng = random.Random(
            f"{round_index}:{payload['seed']}:{','.join(ids)}"
        )
        ambiguous = [
            item_id for item_id in ids if rng.random() < 0.15
        ]
        assigned = [item_id for item_id in ids if item_id not in ambiguous]
        clusters = {}
        for item_id in assigned:
            clusters.setdefault(f"route-{rng.randrange(3)}", []).append(
                item_id
            )
        return _response(list(clusters.items()), ambiguous)

    canonicalizer = MathStrategyCanonicalizer(
        endpoint="http://judge/v1",
        transport=_with_valid_integrity(adversarial_transport),
    )
    for round_index in range(40):
        row_count = 1 + round_index % 5
        keys, diagnostics = canonicalizer.canonicalize(
            prompt_token_ids=[[13]] * row_count,
            prompt_texts=["problem"] * row_count,
            response_texts=[
                f"candidate {round_index}-{index}"
                for index in range(row_count)
            ],
            task_reward_positive=[True] * row_count,
            active_mask=[True] * row_count,
            num_samples=row_count,
        )
        assert len(keys) == row_count
        assert diagnostics.accepted_rows + (
            diagnostics.rejected_integrity_rows
            + diagnostics.rejected_ambiguous_rows
            + diagnostics.rejected_disagreement_rows
        ) == row_count
        assert all(
            key is None or key.startswith("math_strategy_")
            for key in keys
        )
