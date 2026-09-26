from __future__ import annotations

import hashlib
import json

import pytest

from oat_drgrpo.constructive_code import (
    ConstructiveCodeError,
    ReleasedCheckerDecision,
)
from oat_drgrpo.constructive_code_adapters import canonicalize_task_output
from oat_drgrpo.constructive_code_adapters import (
    canonicalize_task_witness,
    registered_task_adapters,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decision(input_bytes: bytes, output_bytes: bytes) -> ReleasedCheckerDecision:
    return ReleasedCheckerDecision(
        checker_sha256=_sha(b"checker"),
        input_sha256=_sha(input_bytes),
        output_sha256=_sha(output_bytes),
        accepted=True,
        exit_code=0,
    )


def _slate_canonical(
    problem_id: str,
    adapter_id: str,
    input_data: str,
    output: str,
):
    input_bytes = input_data.encode()
    output_bytes = output.encode()
    return canonicalize_task_witness(
        problem_id=problem_id,
        adapter_id=adapter_id,
        input_data=input_bytes,
        output=output_bytes,
        decision=_decision(input_bytes, output_bytes),
    )


def test_fixed_sequence_preserves_order():
    input_bytes = b"3 2\n"
    first = b"1 3 2\n"
    second = b"2 3 1\n"

    first_key = canonicalize_task_output(
        "fixed_integer_sequence_v1",
        input_bytes,
        first,
        _decision(input_bytes, first),
    ).canonical_key
    second_key = canonicalize_task_output(
        "fixed_integer_sequence_v1",
        input_bytes,
        second,
        _decision(input_bytes, second),
    ).canonical_key

    assert first_key != second_key


def test_status_set_collapses_index_order_and_handles_no_sentinel():
    input_bytes = b"5 2\n1 2 1 3 4\n"
    first = b"YES\n2 4\n"
    second = b"YES\n4 2\n"
    no_output = b"NO\n"

    first_witness = canonicalize_task_output(
        "status_integer_set_v1",
        input_bytes,
        first,
        _decision(input_bytes, first),
    )
    second_witness = canonicalize_task_output(
        "status_integer_set_v1",
        input_bytes,
        second,
        _decision(input_bytes, second),
    )
    no_witness = canonicalize_task_output(
        "status_integer_set_v1",
        input_bytes,
        no_output,
        _decision(input_bytes, no_output),
    )

    assert first_witness.canonical_key == second_witness.canonical_key
    assert no_witness.canonical_key != first_witness.canonical_key


def test_matrix_assignment_inserts_row_major_keys():
    input_bytes = b"2 2 5\n5 4\n3 5\n1 1\n1 1\n"
    first = b"3 4\n2 5\n"
    second = b"3 4 2 4\n"

    first_witness = canonicalize_task_output(
        "matrix_assignment_v1",
        input_bytes,
        first,
        _decision(input_bytes, first),
    )
    second_witness = canonicalize_task_output(
        "matrix_assignment_v1",
        input_bytes,
        second,
        _decision(input_bytes, second),
    )

    assert '"value":[[1,3],[2,4],[3,2],[4,5]]' in first_witness.canonical_json
    assert first_witness.canonical_key != second_witness.canonical_key


def test_two_group_partition_collapses_group_and_member_order():
    input_bytes = b"4\n1 2 3 4\n"
    first = b"2 1 3\n2 2 4\n"
    second = b"2 4 2\n2 3 1\n"

    first_witness = canonicalize_task_output(
        "two_group_partition_v1",
        input_bytes,
        first,
        _decision(input_bytes, first),
    )
    second_witness = canonicalize_task_output(
        "two_group_partition_v1",
        input_bytes,
        second,
        _decision(input_bytes, second),
    )

    assert first_witness.canonical_key == second_witness.canonical_key


def test_adapter_view_remains_bound_to_exact_raw_output():
    input_bytes = b"3 2\n"
    output_bytes = b"1 3 2\n"
    wrong_decision = _decision(input_bytes, b"2 3 1\n")

    with pytest.raises(ConstructiveCodeError, match="SHA-256"):
        canonicalize_task_output(
            "fixed_integer_sequence_v1",
            input_bytes,
            output_bytes,
            wrong_decision,
        )


@pytest.mark.parametrize(
    ("problem_id", "adapter_id", "input_data", "output"),
    [
        ("327_B", "fixed_integer_sequence_v1", "3\n", "2 9 15\n"),
        ("359_B", "fixed_integer_sequence_v1", "2 1\n", "1 2 3 4\n"),
        ("361_B", "sentinel_integer_sequence_v1", "3 1\n", "-1\nignored\n"),
        ("482_A", "fixed_integer_sequence_v1", "4 2\n", "1 4 2 3\n"),
        ("659_C", "counted_integer_set_v1", "1 20\n2\n", "3 5 1 3\n"),
        (
            "988_A",
            "status_integer_set_v1",
            "4 2\n1 2 1 3\n",
            "YES 1 2\n",
        ),
        (
            "1294_C",
            "multi_case_status_integer_set_v1",
            "2\n30\n7\n",
            "YES 2 3 5\nNO\n",
        ),
        ("1516_C", "counted_integer_set_v1", "3\n1 2 3\n", "1 2\n"),
        (
            "1153_B",
            "matrix_assignment_v1",
            "2 2 3\n2 3\n3 2\n1 1\n1 1\n",
            "1 2\n3 2\n",
        ),
        ("1208_C", "matrix_assignment_v1", "2\n", "0 1\n2 3\n"),
        ("1283_C", "implicit_assignment_v1", "3\n0 2 0\n", "3 2 1\n"),
        (
            "1408_A",
            "multi_case_implicit_assignment_v1",
            "1\n3\n1 2 3\n2 3 1\n3 1 2\n",
            "1 2 3\n",
        ),
        (
            "1051_B",
            "status_pair_partition_v1",
            "1 4\n",
            "YES\n1 2\n3 4\n",
        ),
        (
            "1102_B",
            "status_label_partition_v1",
            "4 2\n1 2 1 2\n",
            "YES\n1 1 2 2\n",
        ),
        (
            "1399_D",
            "multi_case_label_partition_v1",
            "1\n4\n0101\n",
            "1\n1 1 1 1\n",
        ),
        (
            "149_C",
            "two_group_partition_v1",
            "4\n1 2 3 4\n",
            "2 1 4\n2 2 3\n",
        ),
    ],
)
def test_every_registered_adapter_builds_a_hash_bound_witness(
    problem_id,
    adapter_id,
    input_data,
    output,
):
    witness = _slate_canonical(problem_id, adapter_id, input_data, output)

    payload = json.loads(witness.canonical_json)
    assert payload["adapter"] == adapter_id
    assert witness.canonical_key.startswith(
        f"constructive_witness:{witness.family}:v1:"
    )


def test_registry_is_exactly_the_frozen_review_slate():
    expected = {
        ("327_B", "fixed_integer_sequence_v1"),
        ("359_B", "fixed_integer_sequence_v1"),
        ("361_B", "sentinel_integer_sequence_v1"),
        ("482_A", "fixed_integer_sequence_v1"),
        ("659_C", "counted_integer_set_v1"),
        ("988_A", "status_integer_set_v1"),
        ("1294_C", "multi_case_status_integer_set_v1"),
        ("1516_C", "counted_integer_set_v1"),
        ("1153_B", "matrix_assignment_v1"),
        ("1208_C", "matrix_assignment_v1"),
        ("1283_C", "implicit_assignment_v1"),
        ("1408_A", "multi_case_implicit_assignment_v1"),
        ("1051_B", "status_pair_partition_v1"),
        ("1102_B", "status_label_partition_v1"),
        ("1399_D", "multi_case_label_partition_v1"),
        ("149_C", "two_group_partition_v1"),
    }

    assert set(registered_task_adapters()) == expected


def test_slate_set_identity_ignores_order_but_not_membership():
    first = _slate_canonical(
        "659_C",
        "counted_integer_set_v1",
        "1 20\n2\n",
        "3 5 1 3",
    )
    alias = _slate_canonical(
        "659_C",
        "counted_integer_set_v1",
        "1 20\n2\n",
        "3\n3 5 1 trailing",
    )
    changed = _slate_canonical(
        "659_C",
        "counted_integer_set_v1",
        "1 20\n2\n",
        "3 3 5 7",
    )

    assert first.canonical_key == alias.canonical_key
    assert first.canonical_key != changed.canonical_key


def test_slate_partition_quotients_label_names_but_preserves_grouping():
    input_data = "4 2\n1 2 1 2\n"
    first = _slate_canonical(
        "1102_B",
        "status_label_partition_v1",
        input_data,
        "YES 1 1 2 2",
    )
    relabeled = _slate_canonical(
        "1102_B",
        "status_label_partition_v1",
        input_data,
        "YES 2 2 1 1",
    )
    regrouped = _slate_canonical(
        "1102_B",
        "status_label_partition_v1",
        input_data,
        "YES 1 2 1 2",
    )

    assert first.canonical_key == relabeled.canonical_key
    assert first.canonical_key != regrouped.canonical_key


def test_slate_input_hash_binding_is_fail_closed():
    input_bytes = b"1\n"
    output_bytes = b"1\n"
    decision = ReleasedCheckerDecision(
        checker_sha256=_sha(b"checker"),
        input_sha256="2" * 64,
        output_sha256=_sha(output_bytes),
        accepted=True,
        exit_code=0,
    )

    with pytest.raises(ConstructiveCodeError, match="input does not match"):
        canonicalize_task_witness(
            problem_id="327_B",
            adapter_id="fixed_integer_sequence_v1",
            input_data=input_bytes,
            output=output_bytes,
            decision=decision,
        )


def test_magic_grid_adapter_accepts_valid_witness_sizes_above_four_mib():
    n = 800
    input_bytes = f"{n}\n".encode()
    output_bytes = (" ".join(str(value) for value in range(n * n)) + "\n").encode()
    assert len(output_bytes) > 4 * 1024 * 1024

    witness = canonicalize_task_witness(
        problem_id="1208_C",
        adapter_id="matrix_assignment_v1",
        input_data=input_bytes,
        output=output_bytes,
        decision=_decision(input_bytes, output_bytes),
    )

    assert witness.family == "assignment"
