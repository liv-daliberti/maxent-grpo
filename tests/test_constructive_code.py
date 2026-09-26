from __future__ import annotations

import hashlib

import pytest

from oat_drgrpo.constructive_code import (
    CheckedWitness,
    ConstructiveCodeError,
    ReleasedCheckerDecision,
    WitnessSchema,
    canonicalize_accepted_witness,
    canonicalize_checked_behavior,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decision(output: str, *, accepted: bool = True, input_text: str = "input"):
    return ReleasedCheckerDecision(
        checker_sha256=_sha(b"checker"),
        input_sha256=_sha(input_text.encode()),
        output_sha256=_sha(output.encode()),
        accepted=accepted,
        exit_code=0 if accepted else 1,
    )


def _canonical(output: str, schema: WitnessSchema):
    return canonicalize_accepted_witness(output, schema, _decision(output))


def test_ordered_sequence_collapses_formatting_but_preserves_semantic_order():
    schema = WitnessSchema(
        family="ordered_sequence",
        exact_item_count=4,
        count_header="item_count",
        integer_minimum=1,
        integer_maximum=9,
    )

    first = _canonical("4\n1 2 3 4\n", schema)
    formatted = _canonical("4  1\r\n2\t3 4", schema)
    reordered = _canonical("4 2 1 3 4", schema)

    assert first.canonical_key == formatted.canonical_key
    assert first.canonical_key != reordered.canonical_key


def test_unordered_set_collapses_order_and_rejects_duplicates():
    schema = WitnessSchema(
        family="unordered_set",
        expected_universe=(2, 3, 5),
        integer_minimum=1,
        integer_maximum=5,
    )

    assert (
        _canonical("5 2 3", schema).canonical_key
        == _canonical("2\n3 5\n", schema).canonical_key
    )
    with pytest.raises(ConstructiveCodeError, match="duplicate"):
        _canonical("2 3 3", schema)


def test_assignment_collapses_pair_order_but_not_key_value_semantics():
    schema = WitnessSchema(
        family="assignment",
        expected_keys=(1, 2, 3),
        count_header="item_count",
        integer_minimum=0,
        integer_maximum=9,
    )

    first = _canonical("3\n1 7\n2 8\n3 9", schema)
    reordered = _canonical("3 3 9 1 7 2 8", schema)
    changed = _canonical("3 1 8 2 7 3 9", schema)

    assert first.canonical_key == reordered.canonical_key
    assert first.canonical_key != changed.canonical_key
    with pytest.raises(ConstructiveCodeError, match="duplicate key"):
        _canonical("3 1 7 1 8 3 9", schema)


def test_partition_collapses_group_and_member_order_only():
    schema = WitnessSchema(
        family="unordered_partition",
        expected_universe=(1, 2, 3, 4),
        exact_group_count=2,
        count_header="group_count",
        integer_minimum=1,
        integer_maximum=4,
    )

    first = _canonical("2\n1 3\n2 4\n", schema)
    reordered = _canonical("2\r\n4 2\r\n3 1", schema)
    changed = _canonical("2\n1 2\n3 4", schema)

    assert first.canonical_key == reordered.canonical_key
    assert first.canonical_key != changed.canonical_key
    with pytest.raises(ConstructiveCodeError, match="across groups"):
        _canonical("2\n1 2\n2 3 4", schema)


def test_canonicalizer_is_bound_to_positive_released_checker_decision():
    schema = WitnessSchema(family="ordered_sequence", exact_item_count=2)

    with pytest.raises(ConstructiveCodeError, match="SHA-256"):
        canonicalize_accepted_witness("1 2", schema, _decision("2 1"))
    with pytest.raises(ConstructiveCodeError, match="rejected"):
        canonicalize_accepted_witness(
            "1 2",
            schema,
            _decision("1 2", accepted=False),
        )
    with pytest.raises(ConstructiveCodeError, match="exit successfully"):
        ReleasedCheckerDecision(
            checker_sha256=_sha(b"checker"),
            input_sha256=_sha(b"input"),
            output_sha256=_sha(b"output"),
            accepted=True,
            exit_code=1,
        )


def test_behavior_key_uses_ordered_hidden_inputs_and_canonical_witnesses():
    schema = WitnessSchema(family="ordered_sequence", exact_item_count=2)
    first_output = "1 2"
    second_output = "3 4"
    first = CheckedWitness(
        _decision(first_output, input_text="input-a"),
        canonicalize_accepted_witness(
            first_output,
            schema,
            _decision(first_output, input_text="input-a"),
        ),
    )
    second = CheckedWitness(
        _decision(second_output, input_text="input-b"),
        canonicalize_accepted_witness(
            second_output,
            schema,
            _decision(second_output, input_text="input-b"),
        ),
    )
    suite_sha = _sha(b"suite")

    behavior = canonicalize_checked_behavior(
        "Codeforces:1_A", suite_sha, [first, second]
    )
    repeated = canonicalize_checked_behavior(
        "Codeforces:1_A",
        suite_sha,
        [first, second],
    )
    reversed_behavior = canonicalize_checked_behavior(
        "Codeforces:1_A",
        suite_sha,
        [second, first],
    )

    assert behavior.canonical_key == repeated.canonical_key
    assert behavior.canonical_key != reversed_behavior.canonical_key

    with pytest.raises(ConstructiveCodeError, match="does not match"):
        canonicalize_checked_behavior(
            "Codeforces:1_A",
            suite_sha,
            [CheckedWitness(first.decision, second.witness)],
        )


def test_schema_contract_rejects_ambiguous_family_shapes():
    with pytest.raises(ConstructiveCodeError, match="exact item count"):
        WitnessSchema(family="ordered_sequence")
    with pytest.raises(ConstructiveCodeError, match="expected universe"):
        WitnessSchema(family="unordered_partition", exact_group_count=2)
    with pytest.raises(ConstructiveCodeError, match="must be unique"):
        WitnessSchema(family="assignment", expected_keys=(1, 1))
