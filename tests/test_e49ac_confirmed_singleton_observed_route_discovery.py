from __future__ import annotations

import importlib.util
import json
import pathlib
import types


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49ac_confirmed_singleton_observed_route_discovery.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49ac_confirmed_singleton", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e49ac_singleton_selection_is_frozen_by_support_then_key():
    module = _load()
    candidate = {
        "source_index": 7,
        "row_id": "row",
        "subject": "algebra",
        "level": 5,
        "exemplars": [
            {
                "sample_id": "a",
                "response_sha256": "03",
                "text": "a",
            },
            {
                "sample_id": "b",
                "response_sha256": "01",
                "text": "b",
            },
            {
                "sample_id": "c",
                "response_sha256": "02",
                "text": "c",
            },
        ],
    }
    cluster = {
        "outcomes": [
            {"sample_id": "a", "strategy_key": "route_b"},
            {"sample_id": "b", "strategy_key": "route_b"},
            {"sample_id": "c", "strategy_key": "route_a"},
        ]
    }
    record = module._singleton_candidate(candidate, cluster)
    assert record["candidate_pair"] is True
    assert [row["strategy_key"] for row in record["clusters"]] == [
        "route_b",
        "route_a",
    ]
    assert record["clusters"][0]["member_ids"] == ["b", "a"]
    assert record["minimum_cluster_size"] == 1
    assert record["combined_cluster_size"] == 3


def test_e49ac_protocol_is_fail_closed_and_uses_stronger_execution_gate():
    module = _load()
    protocol = module.PROTOCOL.read_text(encoding="utf-8")
    slurm = (
        ROOT
        / "ops/slurm/"
        "e49ac_confirmed_singleton_route_discovery_node302.slurm"
    ).read_text(encoding="utf-8")
    assert module.RELATION_SEEDS == (470731, 470732)
    assert module.FORCED_SAMPLE_COUNT == 64
    assert module.MIN_FORCED_SUCCESSES == 2
    assert module.UNFORCED_SAMPLE_COUNT == 64
    assert module.MIN_NATURAL_PER_ROUTE == 2
    assert module.MIN_NATURAL_TOTAL == 8
    assert "no lower-ranked pair may" in protocol
    assert "replace a rejected pair" in protocol
    assert "Both audits must return" in protocol
    assert "Both routes must achieve at least two" in protocol
    assert "also draw 64 independent" in protocol
    assert "at least eight samples are accepted" in protocol
    assert "#SBATCH --gres=gpu:a100:1" in slurm


def test_e49ac_uses_exact_frozen_pairwise_source():
    module = _load()
    assert (
        module._sha256(module.PAIRWISE_SOURCE)
        == "91a1a8fdc3b29fa49b1089154f5955ec2d9e759cc85e94b35bc7d101d81bf988"
    )


def test_e49ac_natural_support_and_rank_are_frozen():
    module = _load()
    assert module._natural_support({"S1": 2, "S2": 6}) is True
    assert module._natural_support({"S1": 1, "S2": 20}) is False
    assert module._natural_support({"S1": 2, "S2": 5}) is False
    rows = [
        {
            "source_index": 8,
            "minimum_cluster_size": 1,
            "unforced_route_success_counts": {"S1": 3, "S2": 9},
            "unforced_accepted_count": 12,
        },
        {
            "source_index": 7,
            "minimum_cluster_size": 3,
            "unforced_route_success_counts": {"S1": 3, "S2": 10},
            "unforced_accepted_count": 13,
        },
        {
            "source_index": 6,
            "minimum_cluster_size": 2,
            "unforced_route_success_counts": {"S1": 4, "S2": 4},
            "unforced_accepted_count": 8,
        },
    ]
    assert [
        row["source_index"]
        for row in sorted(rows, key=module._problem_rank)
    ] == [6, 7, 8]


def test_e49ac_retains_both_direct_audit_checks():
    module = _load()

    class FakeJudge:
        def __init__(self, **_kwargs):
            pass

        def _post(self, payload):
            content = json.dumps(
                {
                    "assessments": [
                        {
                            "pair_id": "PAIR_0000",
                            "brief_check": "different decisive operations",
                            "relation": "different",
                        }
                    ]
                }
            )
            return {
                "id": f"response-{payload['seed']}",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": content},
                    }
                ],
            }

        def _judge_relations(
            self, *, problem, base_items, pairs, seed
        ):
            assert problem and len(base_items) == 2 and len(pairs) == 1
            response = self._post({"seed": seed})
            payload = json.loads(response["choices"][0]["message"]["content"])
            return {
                row["pair_id"]: row["relation"]
                for row in payload["assessments"]
            }

    candidate = {
        "source_index": 1,
        "row_id": "row",
        "subject": "algebra",
        "level": 5,
        "problem": "problem",
        "exemplars": [
            {"sample_id": "a", "text": "route a"},
            {"sample_id": "b", "text": "route b"},
        ],
    }
    singleton = {
        "source_index": 1,
        "row_id": "row",
        "subject": "algebra",
        "level": 5,
        "emitted_key_count": 2,
        "minimum_cluster_size": 1,
        "combined_cluster_size": 2,
        "clusters": [
            {"member_ids": ["a"]},
            {"member_ids": ["b"]},
        ],
    }
    record = module._confirm_candidate(
        pairwise_module=types.SimpleNamespace(
            MathStrategyCanonicalizer=FakeJudge
        ),
        endpoint="http://invalid",
        model="fake",
        candidate=candidate,
        singleton=singleton,
        timeout=1,
    )
    assert record["singleton_confirmation_pass"] is True
    assert [row["seed"] for row in record["relations"]] == [470731, 470732]
    assert all(
        row["assessment"]["assessments"][0]["brief_check"]
        == "different decisive operations"
        for row in record["relations"]
    )
