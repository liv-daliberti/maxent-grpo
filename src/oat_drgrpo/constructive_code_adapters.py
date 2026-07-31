"""Trusted input-derived witness adapters for the frozen executable slate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import prod
from typing import Callable

from .constructive_code import (
    CanonicalWitness,
    ConstructiveCodeError,
    ReleasedCheckerDecision,
    WITNESS_SCHEMA_VERSION,
    WitnessFamily,
    WitnessSchema,
    canonicalize_accepted_witness_view,
    sha256_bytes,
)


Adapter = Callable[[bytes, bytes, ReleasedCheckerDecision], CanonicalWitness]


def _integers(value: bytes, *, label: str) -> list[int]:
    try:
        tokens = value.decode("utf-8").split()
        return [int(token) for token in tokens]
    except (UnicodeDecodeError, ValueError) as error:
        raise ConstructiveCodeError(f"{label} is not integer text") from error


def _tokens(value: bytes) -> list[str]:
    try:
        return value.decode("utf-8").split()
    except UnicodeDecodeError as error:
        raise ConstructiveCodeError("output is not UTF-8") from error


def fixed_integer_sequence_v1(
    input_bytes: bytes,
    output_bytes: bytes,
    decision: ReleasedCheckerDecision,
) -> CanonicalWitness:
    values = _integers(input_bytes, label="input")
    if not values:
        raise ConstructiveCodeError("sequence input lacks n")
    n = values[0]
    schema = WitnessSchema(
        family="ordered_sequence",
        exact_item_count=n,
        integer_minimum=1,
        integer_maximum=n,
    )
    return canonicalize_accepted_witness_view(
        output_bytes,
        output_bytes.decode("utf-8"),
        schema,
        decision,
    )


def status_integer_set_v1(
    input_bytes: bytes,
    output_bytes: bytes,
    decision: ReleasedCheckerDecision,
) -> CanonicalWitness:
    values = _integers(input_bytes, label="input")
    if len(values) < 2:
        raise ConstructiveCodeError("set input lacks n and k")
    n, k = values[:2]
    tokens = _tokens(output_bytes)
    if not tokens:
        raise ConstructiveCodeError("set output is empty")
    status = tokens[0].upper()
    if status == "NO":
        schema = WitnessSchema(
            family="unordered_set",
            exact_item_count=1,
            integer_minimum=0,
            integer_maximum=0,
        )
        witness_view = "0"
    elif status == "YES":
        schema = WitnessSchema(
            family="unordered_set",
            exact_item_count=k,
            integer_minimum=1,
            integer_maximum=n,
        )
        witness_view = " ".join(tokens[1:])
    else:
        raise ConstructiveCodeError("set output lacks YES/NO status")
    return canonicalize_accepted_witness_view(
        output_bytes, witness_view, schema, decision
    )


def matrix_assignment_v1(
    input_bytes: bytes,
    output_bytes: bytes,
    decision: ReleasedCheckerDecision,
) -> CanonicalWitness:
    values = _integers(input_bytes, label="input")
    if len(values) < 3:
        raise ConstructiveCodeError("matrix input lacks n, m, and h")
    n, m, h = values[:3]
    output_values = _integers(output_bytes, label="matrix output")
    if len(output_values) != n * m:
        raise ConstructiveCodeError("matrix output has the wrong cell count")
    pairs = []
    for index, value in enumerate(output_values, start=1):
        pairs.extend((index, value))
    schema = WitnessSchema(
        family="assignment",
        exact_item_count=n * m,
        expected_keys=tuple(range(1, n * m + 1)),
        integer_minimum=0,
        integer_maximum=max(h, n * m),
    )
    return canonicalize_accepted_witness_view(
        output_bytes,
        " ".join(str(value) for value in pairs),
        schema,
        decision,
    )


def two_group_partition_v1(
    input_bytes: bytes,
    output_bytes: bytes,
    decision: ReleasedCheckerDecision,
) -> CanonicalWitness:
    values = _integers(input_bytes, label="input")
    if not values:
        raise ConstructiveCodeError("partition input lacks n")
    n = values[0]
    output_values = _integers(output_bytes, label="partition output")
    if not output_values:
        raise ConstructiveCodeError("partition output is empty")
    first_count = output_values[0]
    if first_count < 0 or len(output_values) <= first_count + 1:
        raise ConstructiveCodeError("first partition count is malformed")
    first = output_values[1 : first_count + 1]
    second_count = output_values[first_count + 1]
    second = output_values[first_count + 2 :]
    if second_count < 0 or len(second) != second_count:
        raise ConstructiveCodeError("second partition count is malformed")
    schema = WitnessSchema(
        family="unordered_partition",
        expected_universe=tuple(range(1, n + 1)),
        exact_group_count=2,
        integer_minimum=1,
        integer_maximum=n,
    )
    witness_view = (
        " ".join(str(value) for value in first)
        + "\n"
        + " ".join(str(value) for value in second)
    )
    return canonicalize_accepted_witness_view(
        output_bytes, witness_view, schema, decision
    )


ADAPTERS: dict[str, Adapter] = {
    "fixed_integer_sequence_v1": fixed_integer_sequence_v1,
    "status_integer_set_v1": status_integer_set_v1,
    "matrix_assignment_v1": matrix_assignment_v1,
    "two_group_partition_v1": two_group_partition_v1,
}


def canonicalize_task_output(
    adapter_name: str,
    input_bytes: bytes,
    output_bytes: bytes,
    decision: ReleasedCheckerDecision,
) -> CanonicalWitness:
    try:
        adapter = ADAPTERS[adapter_name]
    except KeyError as error:
        raise ConstructiveCodeError("unregistered task adapter") from error
    return adapter(input_bytes, output_bytes, decision)


# The generic adapters above are the API frozen by the four-task executable
# slate. The full review slate additionally binds every adapter to a problem ID
# so similarly shaped tasks cannot accidentally share input semantics.
_SLATE_MAX_OUTPUT_BYTES = 16 * 1024 * 1024
_SLATE_MAX_TOKENS = 2_000_000


@dataclass
class _SlateCursor:
    tokens: list[str]
    position: int = 0

    @classmethod
    def from_text(cls, text: str) -> _SlateCursor:
        tokens = text.split()
        if len(tokens) > _SLATE_MAX_TOKENS:
            raise ConstructiveCodeError("task output has too many tokens")
        return cls(tokens=tokens)

    def token(self) -> str:
        if self.position >= len(self.tokens):
            raise ConstructiveCodeError("task output ended early")
        value = self.tokens[self.position]
        self.position += 1
        return value

    def integer(self) -> int:
        raw = self.token()
        try:
            return int(raw)
        except ValueError as error:
            raise ConstructiveCodeError(
                "task output contains a non-integer"
            ) from error

    def integers(self, count: int) -> list[int]:
        if count < 0:
            raise ConstructiveCodeError("task output declares a negative count")
        return [self.integer() for _ in range(count)]

    def require_end(self) -> None:
        if self.position != len(self.tokens):
            raise ConstructiveCodeError("task output contains trailing tokens")


SlateAdapter = Callable[[str, str, str], object]


def _trusted_text(
    value: str | bytes,
    label: str,
    maximum: int,
) -> tuple[bytes, str]:
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    elif isinstance(value, bytes):
        encoded = value
    else:
        raise ConstructiveCodeError(f"{label} must be text or bytes")
    if not encoded or len(encoded) > maximum:
        raise ConstructiveCodeError(f"{label} has an invalid byte count")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConstructiveCodeError(f"{label} is not UTF-8") from error
    if "\x00" in text:
        raise ConstructiveCodeError(f"{label} contains a NUL byte")
    return encoded, text


def _input_integers(text: str) -> list[int]:
    cursor = _SlateCursor.from_text(text)
    return cursor.integers(len(cursor.tokens))


def _sequence_adapter(
    problem_id: str,
    input_text: str,
    output_text: str,
) -> object:
    inputs = _input_integers(input_text)
    count = 2 * inputs[0] if problem_id == "359_B" else inputs[0]
    cursor = _SlateCursor.from_text(output_text)
    return cursor.integers(count)


def _sentinel_sequence_adapter(
    _problem_id: str,
    input_text: str,
    output_text: str,
) -> object:
    n = _input_integers(input_text)[0]
    lines = output_text.splitlines()
    first_line = lines[0].strip() if lines else ""
    cursor = _SlateCursor.from_text(first_line)
    first = cursor.integer()
    if first == -1:
        cursor.require_end()
        return {"status": "impossible"}
    values = [first, *cursor.integers(n - 1)]
    cursor.require_end()
    return values


def _counted_set_adapter(
    _problem_id: str,
    _input_text: str,
    output_text: str,
) -> object:
    cursor = _SlateCursor.from_text(output_text)
    count = cursor.integer()
    values = cursor.integers(count)
    if len(values) != len(set(values)):
        raise ConstructiveCodeError("set witness contains duplicates")
    return sorted(values)


def _status_set_adapter(
    _problem_id: str,
    input_text: str,
    output_text: str,
) -> object:
    count = _input_integers(input_text)[1]
    cursor = _SlateCursor.from_text(output_text)
    status = cursor.token()
    if status == "NO":
        cursor.require_end()
        return {"status": "impossible"}
    if status != "YES":
        raise ConstructiveCodeError("set witness has an invalid status")
    values = cursor.integers(count)
    cursor.require_end()
    if len(values) != len(set(values)):
        raise ConstructiveCodeError("set witness contains duplicates")
    return {"status": "ok", "values": sorted(values)}


def _multi_status_set_adapter(
    _problem_id: str,
    input_text: str,
    output_text: str,
) -> object:
    inputs = _input_integers(input_text)
    test_count = inputs[0]
    cases = inputs[1:]
    if len(cases) != test_count:
        raise ConstructiveCodeError("trusted multi-case input is malformed")
    cursor = _SlateCursor.from_text(output_text)
    values: list[object] = []
    for expected_product in cases:
        status = cursor.token()
        if status == "NO":
            values.append({"status": "impossible"})
            continue
        if status != "YES":
            raise ConstructiveCodeError("set witness has an invalid status")
        factors = cursor.integers(3)
        if len(set(factors)) != 3 or prod(factors) != expected_product:
            raise ConstructiveCodeError("accepted factor witness is malformed")
        values.append({"status": "ok", "values": sorted(factors)})
    return values


def _matrix_assignment_adapter(
    problem_id: str,
    input_text: str,
    output_text: str,
) -> object:
    inputs = _input_integers(input_text)
    if problem_id == "1153_B":
        rows, columns = inputs[:2]
    else:
        rows = columns = inputs[0]
    cursor = _SlateCursor.from_text(output_text)
    values = cursor.integers(rows * columns)
    return [[index, value] for index, value in enumerate(values, start=1)]


def _implicit_assignment_adapter(
    _problem_id: str,
    input_text: str,
    output_text: str,
) -> object:
    count = _input_integers(input_text)[0]
    values = _SlateCursor.from_text(output_text).integers(count)
    return [[index, value] for index, value in enumerate(values, start=1)]


def _multi_assignment_adapter(
    _problem_id: str,
    input_text: str,
    output_text: str,
) -> object:
    inputs = _input_integers(input_text)
    test_count = inputs[0]
    input_position = 1
    cursor = _SlateCursor.from_text(output_text)
    cases: list[object] = []
    for _ in range(test_count):
        if input_position >= len(inputs):
            raise ConstructiveCodeError("trusted multi-case input is malformed")
        count = inputs[input_position]
        input_position += 1 + 3 * count
        values = cursor.integers(count)
        cases.append(
            [[index, value] for index, value in enumerate(values, start=1)]
        )
    if input_position != len(inputs):
        raise ConstructiveCodeError(
            "trusted multi-case input has trailing values"
        )
    cursor.require_end()
    return cases


def _canonical_groups(
    groups: list[list[int]],
    universe: set[int],
) -> list[list[int]]:
    normalized = [sorted(group) for group in groups]
    flattened = [value for group in normalized for value in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != universe:
        raise ConstructiveCodeError("partition witness differs from its universe")
    return sorted(normalized)


def _status_pair_partition_adapter(
    _problem_id: str,
    input_text: str,
    output_text: str,
) -> object:
    left, right = _input_integers(input_text)[:2]
    cursor = _SlateCursor.from_text(output_text)
    if cursor.token() != "YES":
        raise ConstructiveCodeError("pair partition lacks YES status")
    pairs = [cursor.integers(2) for _ in range((right - left + 1) // 2)]
    return _canonical_groups(pairs, set(range(left, right + 1)))


def _status_label_partition_adapter(
    _problem_id: str,
    input_text: str,
    output_text: str,
) -> object:
    inputs = _input_integers(input_text)
    count, group_count = inputs[:2]
    cursor = _SlateCursor.from_text(output_text)
    status = cursor.token()
    if status == "NO":
        return {"status": "impossible"}
    if status != "YES":
        raise ConstructiveCodeError("label partition has an invalid status")
    labels = cursor.integers(count)
    groups = [[] for _ in range(group_count)]
    for index, label in enumerate(labels, start=1):
        if not 1 <= label <= group_count:
            raise ConstructiveCodeError("partition label is outside its range")
        groups[label - 1].append(index)
    return {
        "status": "ok",
        "groups": _canonical_groups(groups, set(range(1, count + 1))),
    }


def _multi_label_partition_adapter(
    _problem_id: str,
    input_text: str,
    output_text: str,
) -> object:
    input_cursor = _SlateCursor.from_text(input_text)
    test_count = input_cursor.integer()
    output_cursor = _SlateCursor.from_text(output_text)
    cases: list[object] = []
    for _ in range(test_count):
        count = input_cursor.integer()
        input_cursor.token()
        group_count = output_cursor.integer()
        labels = output_cursor.integers(count)
        groups = [[] for _ in range(group_count)]
        for index, label in enumerate(labels, start=1):
            if not 1 <= label <= group_count:
                raise ConstructiveCodeError(
                    "partition label is outside its range"
                )
            groups[label - 1].append(index)
        cases.append(_canonical_groups(groups, set(range(1, count + 1))))
    input_cursor.require_end()
    return cases


def _two_group_partition_adapter(
    _problem_id: str,
    input_text: str,
    output_text: str,
) -> object:
    count = _input_integers(input_text)[0]
    cursor = _SlateCursor.from_text(output_text)
    first_count = cursor.integer()
    first = cursor.integers(first_count)
    second_count = cursor.integer()
    second = cursor.integers(second_count)
    return _canonical_groups([first, second], set(range(1, count + 1)))


_SLATE_ADAPTERS: dict[
    tuple[str, str],
    tuple[WitnessFamily, SlateAdapter],
] = {
    ("327_B", "fixed_integer_sequence_v1"): (
        "ordered_sequence",
        _sequence_adapter,
    ),
    ("359_B", "fixed_integer_sequence_v1"): (
        "ordered_sequence",
        _sequence_adapter,
    ),
    ("361_B", "sentinel_integer_sequence_v1"): (
        "ordered_sequence",
        _sentinel_sequence_adapter,
    ),
    ("482_A", "fixed_integer_sequence_v1"): (
        "ordered_sequence",
        _sequence_adapter,
    ),
    ("659_C", "counted_integer_set_v1"): (
        "unordered_set",
        _counted_set_adapter,
    ),
    ("988_A", "status_integer_set_v1"): (
        "unordered_set",
        _status_set_adapter,
    ),
    ("1294_C", "multi_case_status_integer_set_v1"): (
        "unordered_set",
        _multi_status_set_adapter,
    ),
    ("1516_C", "counted_integer_set_v1"): (
        "unordered_set",
        _counted_set_adapter,
    ),
    ("1153_B", "matrix_assignment_v1"): (
        "assignment",
        _matrix_assignment_adapter,
    ),
    ("1208_C", "matrix_assignment_v1"): (
        "assignment",
        _matrix_assignment_adapter,
    ),
    ("1283_C", "implicit_assignment_v1"): (
        "assignment",
        _implicit_assignment_adapter,
    ),
    ("1408_A", "multi_case_implicit_assignment_v1"): (
        "assignment",
        _multi_assignment_adapter,
    ),
    ("1051_B", "status_pair_partition_v1"): (
        "unordered_partition",
        _status_pair_partition_adapter,
    ),
    ("1102_B", "status_label_partition_v1"): (
        "unordered_partition",
        _status_label_partition_adapter,
    ),
    ("1399_D", "multi_case_label_partition_v1"): (
        "unordered_partition",
        _multi_label_partition_adapter,
    ),
    ("149_C", "two_group_partition_v1"): (
        "unordered_partition",
        _two_group_partition_adapter,
    ),
}


def registered_task_adapters() -> tuple[tuple[str, str], ...]:
    return tuple(sorted(_SLATE_ADAPTERS))


def canonicalize_task_witness(
    *,
    problem_id: str,
    adapter_id: str,
    input_data: str | bytes,
    output: str | bytes,
    decision: ReleasedCheckerDecision,
) -> CanonicalWitness:
    """Canonicalize checker-consumed semantics for one accepted slate output."""

    try:
        family, adapter = _SLATE_ADAPTERS[(problem_id, adapter_id)]
    except KeyError as error:
        raise ConstructiveCodeError("unregistered task adapter") from error
    input_bytes, input_text = _trusted_text(
        input_data,
        "task input",
        16 * 1024 * 1024,
    )
    output_bytes, output_text = _trusted_text(
        output,
        "task output",
        _SLATE_MAX_OUTPUT_BYTES,
    )
    if sha256_bytes(input_bytes) != decision.input_sha256:
        raise ConstructiveCodeError("input does not match checker-bound SHA-256")
    if sha256_bytes(output_bytes) != decision.output_sha256:
        raise ConstructiveCodeError("output does not match checker-bound SHA-256")
    if not decision.accepted:
        raise ConstructiveCodeError("cannot canonicalize a rejected checker output")
    payload = {
        "adapter": adapter_id,
        "family": family,
        "schema_version": WITNESS_SCHEMA_VERSION,
        "value": adapter(problem_id, input_text, output_text),
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = sha256_bytes(canonical_json.encode("ascii"))
    return CanonicalWitness(
        family=family,
        checker_sha256=decision.checker_sha256,
        input_sha256=decision.input_sha256,
        output_sha256=decision.output_sha256,
        canonical_json=canonical_json,
        canonical_key=f"constructive_witness:{family}:v1:{digest}",
    )
