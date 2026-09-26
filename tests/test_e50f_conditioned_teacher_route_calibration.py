from __future__ import annotations

import importlib.util
import json
import pathlib
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50f_conditioned_teacher_route_calibration.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e50f_conditioned_teacher", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e50f_frozen_sampling_and_relation_contract():
    module = _load()
    assert module.EXECUTION_SAMPLES == 8
    assert module.FORCED_SAMPLES == 16
    assert module.UNFORCED_SAMPLES == 64
    assert module.RELATION_SEEDS == (470741, 470742)
    assert module.QUARANTINED_CORPUS_ONLY is True
    source = SCRIPT.read_text(encoding="utf-8")
    assert "conditioned_teacher_records.partial.jsonl" in source
    assert "conditioned_teacher_checkpoint_identity.json" in source
    assert "partial checkpoint identity drifted" in source
    assert "list(range(50))" in source
    schema = module._proposal_schema()
    assert schema["properties"]["methods"]["minItems"] == 2
    assert schema["properties"]["methods"]["maxItems"] == 2
    actions = schema["properties"]["methods"]["items"]["properties"]["actions"]
    assert actions["minItems"] == 2
    assert actions["maxItems"] == 6
    slurm = (
        ROOT
        / "ops/slurm/"
        "e50f_conditioned_teacher_route_calibration_node302.slurm"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --nodelist=node302" in slurm
    assert "#SBATCH --gres=gpu:a100:1" in slurm


def test_e50f_protocol_never_credits_proposal_text():
    module = _load()
    protocol = module.PROTOCOL.read_text(encoding="utf-8")
    assert "problem but not its reference answer" in protocol
    assert "Proposed prose alone never establishes a route" in protocol
    assert "at least two independently sampled" in protocol
    assert "cross-route comparison genuinely different" in protocol
    assert "least two counted samples from each route" in protocol
    assert "performs no policy update" in protocol


def test_e50f_proposal_payload_is_answer_blind_and_fixed_pair():
    module = _load()
    payload = module._proposal_payload("judge", "problem only", 3)
    rendered = str(payload)
    assert "problem only" in rendered
    assert "reference answer" not in rendered
    assert payload["temperature"] == 0.7
    assert payload["seed"] == module.PROPOSAL_SEED + 3
    assert (
        payload["response_format"]["json_schema"]["schema"]
        == module._proposal_schema()
    )
    rendered_prompt = payload["messages"][1]["content"]
    assert len(module.SAFE_PAIR_OPTIONS) == 12
    for left, right in module.SAFE_PAIR_OPTIONS:
        assert f"{left} VERSUS {right}" in rendered_prompt
    assert "Choose one and only one pair" in rendered_prompt
    assert "problem only" in rendered_prompt


def test_e50f_prompt_pairs_exactly_match_frozen_safe_pairs():
    module = _load()
    source = (
        ROOT
        / "ops/math_strategy_calibration/"
        "safe_math_strategy_signatures.py"
    )
    spec = importlib.util.spec_from_file_location("e50f_safe_pairs", source)
    assert spec is not None and spec.loader is not None
    signatures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signatures)
    observed = set()
    for left, right in module.SAFE_PAIR_OPTIONS:
        distinct, left_key, right_key = signatures.safe_distinct_pair(
            {"label": left}, {"label": right}
        )
        assert distinct is True
        observed.add(frozenset((left_key, right_key)))
    assert observed == signatures.SAFE_DISTINCT_PAIRS


def test_e50f_relation_gate_requires_within_same_and_cross_different():
    module = _load()

    class Judge:
        def __init__(self, **_kwargs):
            pass

        def _judge_relations(self, *, pairs, **_kwargs):
            return {
                pair_id: (
                    "same"
                    if left.split("_")[0] == right.split("_")[0]
                    else "different"
                )
                for pair_id, left, right in pairs
            }

    candidate = {
        "problem_order": 0,
        "source_index": 3,
        "row_id": "row",
        "subject": "Algebra",
        "level": 5,
        "problem": "problem",
        "proposal": {
            "methods": [
                {"label": "first", "actions": ["a", "b"]},
                {"label": "second", "actions": ["c", "d"]},
            ]
        },
    }
    members = [
        [
            {"sample_id": "a1", "text": "one"},
            {"sample_id": "a2", "text": "two"},
        ],
        [
            {"sample_id": "b1", "text": "three"},
            {"sample_id": "b2", "text": "four"},
        ],
    ]
    record = module._relation_record(
        pairwise_module=SimpleNamespace(MathStrategyCanonicalizer=Judge),
        endpoint="endpoint",
        model="model",
        candidate=candidate,
        route_members=members,
        timeout=1,
    )
    assert record["cluster_eligible"] is True
    assert record["minimum_cluster_size"] == 2
    assert record["combined_cluster_size"] == 4
    assert record["route_positive_counts"] == [2, 2]
    assert record["minimum_teacher_execution_count"] == 2
    assert record["combined_teacher_execution_count"] == 4
    assert [row["member_ids"] for row in record["clusters"]] == [
        ["a1", "a2"],
        ["b1", "b2"],
    ]


def test_e50f_keeps_passed_e50c_only_as_quarantined_input(
    tmp_path, monkeypatch
):
    module = _load()
    result = tmp_path / "e50c.json"
    result.write_text(
        json.dumps(
            {
                "schema": "e50c_72b_teacher_route_calibration_v1",
                "pass": True,
                "selected_source_indices": list(range(10)),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "E50C_RESULT", result)
    assert module._verify_activation()["pass"] is True


def test_e50d_uses_passing_safe_signature_source(tmp_path):
    path = (
        ROOT
        / "ops/math_strategy_calibration/"
        "materialize_e50d_teacher_route_math_toy.py"
    )
    spec = importlib.util.spec_from_file_location("e50d_materializer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failed = tmp_path / "failed.json"
    passed = tmp_path / "passed.json"
    failed.write_text(
        json.dumps(
            {
                "schema": "first",
                "pass": False,
                "selected_source_indices": [],
            }
        ),
        encoding="utf-8",
    )
    passed.write_text(
        json.dumps(
            {
                "schema": "second",
                "pass": True,
                "selected_source_indices": list(range(10)),
                "bidirectionally_executable_count": 10,
            }
        ),
        encoding="utf-8",
    )
    module.CALIBRATIONS = (
        (failed, "first", "first_origin"),
        (passed, "second", "second_origin"),
    )
    assert module._select_calibration() == (
        passed,
        "second",
        "second_origin",
    )


def test_e50d_default_source_is_only_e50g():
    path = (
        ROOT
        / "ops/math_strategy_calibration/"
        "materialize_e50d_teacher_route_math_toy.py"
    )
    spec = importlib.util.spec_from_file_location("e50d_default_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len(module.CALIBRATIONS) == 1
    assert module.CALIBRATIONS[0][1:] == (
        "e50g_safe_signature_teacher_route_calibration_v1",
        "e50g_safe_signature_teacher_05b_natural_supported",
    )
