from __future__ import annotations

import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49y_observed_route_discovery.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("e49y_route_discovery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e49y_partition_requires_exact_once_complete_assignment():
    module = _load()
    sample_ids = ["a", "b", "c", "d"]
    partition = {
        "clusters": [
            {
                "cluster_id": "C1",
                "strategy": "method one",
                "member_ids": ["a", "b"],
            },
            {
                "cluster_id": "C2",
                "strategy": "method two",
                "member_ids": ["c", "d"],
            },
        ]
    }
    assignment, clusters = module._validate_partition(partition, sample_ids)
    assert len(clusters) == 2
    assert assignment == {"a": "C1", "b": "C1", "c": "C2", "d": "C2"}

    partition["clusters"][1]["member_ids"] = ["b", "c", "d"]
    try:
        module._validate_partition(partition, sample_ids)
    except ValueError as error:
        assert "twice" in str(error)
    else:
        raise AssertionError("duplicate route assignment did not fail closed")


def test_e49y_pairwise_relation_is_label_permutation_invariant():
    module = _load()
    sample_ids = ["a", "b", "c", "d"]
    first = {"a": "C1", "b": "C1", "c": "C2", "d": "C2"}
    permuted = {"a": "C2", "b": "C2", "c": "C1", "d": "C1"}
    changed = {"a": "C1", "b": "C2", "c": "C2", "d": "C2"}
    assert module._same_relation(first, sample_ids) == module._same_relation(
        permuted, sample_ids
    )
    assert module._same_relation(first, sample_ids) != module._same_relation(
        changed, sample_ids
    )


def test_exact_assignment_schema_requires_every_response_property():
    module = _load()
    schema = module._exact_assignment_partition_schema(["a", "b", "c"])
    assignments = schema["properties"]["assignments"]
    assert assignments["required"] == ["a", "b", "c"]
    assert assignments["additionalProperties"] is False
    assignment, clusters = module._validate_exact_assignment_partition(
        {
            "assignments": {"a": "C1", "b": "C1", "c": "C2"},
            "cluster_strategies": [
                {"cluster_id": "C1", "strategy": "first"},
                {"cluster_id": "C2", "strategy": "second"},
            ],
        },
        ["a", "b", "c"],
    )
    assert assignment == {"a": "C1", "b": "C1", "c": "C2"}
    assert [row["member_ids"] for row in clusters] == [["a", "b"], ["c"]]


def test_consensus_retains_only_groups_both_partitions_separate():
    module = _load()
    module.PARTITION_MODE = "consensus_intersection"
    sample_ids = ["a", "b", "c", "d", "e", "f"]
    candidate = {
        "exemplars": [{"sample_id": sample_id} for sample_id in sample_ids]
    }
    passes = [
        {
            "assignment": {
                "a": "C1",
                "b": "C1",
                "c": "C2",
                "d": "C2",
                "e": "C3",
                "f": "C3",
            },
            "clusters": [
                {"cluster_id": "C1", "strategy": "one"},
                {"cluster_id": "C2", "strategy": "two"},
                {"cluster_id": "C3", "strategy": "three"},
            ],
        },
        {
            "assignment": {
                "a": "C2",
                "b": "C2",
                "c": "C1",
                "d": "C1",
                "e": "C1",
                "f": "C1",
            },
            "clusters": [
                {"cluster_id": "C1", "strategy": "alpha"},
                {"cluster_id": "C2", "strategy": "beta"},
            ],
        },
    ]
    original = module._partition_once
    module._partition_once = lambda **kwargs: passes[kwargs["pass_index"]]
    try:
        selected, _ = module._stable_clusters(
            endpoint="unused",
            model="unused",
            candidate=candidate,
            timeout=1,
        )
    finally:
        module._partition_once = original
    assert [row["member_ids"] for row in selected] == [
        ["a", "b"],
        ["c", "d"],
    ]


def test_e49y_protocol_forbids_invented_routes_and_uses_exact_hard_pool():
    protocol = (
        ROOT
        / "paper/preregistration/e49y_observed_route_discovery_20260726.md"
    ).read_text(encoding="utf-8")
    slurm = (
        ROOT / "ops/slurm/e49y_observed_route_discovery_node302.slurm"
    ).read_text(encoding="utf-8")
    assert "all 211 level-4 or level-5 rows" in protocol
    assert "64 independent natural" in protocol
    assert "proposed routes are forbidden" in protocol
    assert "at least ten problems" in protocol
    assert "#SBATCH --gres=gpu:a100:1" in slurm
