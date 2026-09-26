"""Fail-closed semantic identity for constructive programming witnesses.

The released problem checker remains the correctness authority.  This module
only canonicalizes the exact output that an accepted checker decision binds by
SHA-256.  A task-specific adapter must derive the trusted ``WitnessSchema``
from the problem input; model-authored text never controls the schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Literal, Sequence


WITNESS_SCHEMA_VERSION = "constructive-code-witness-v1"
BEHAVIOR_KEY_VERSION = "constructive-code-behavior-v1"
WitnessFamily = Literal[
    "ordered_sequence",
    "unordered_set",
    "assignment",
    "unordered_partition",
]
CountHeader = Literal["none", "item_count", "group_count"]

_INTEGER_TOKEN = re.compile(r"[+-]?[0-9]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_OUTPUT_BYTES = 1024 * 1024
_MAX_INTEGER_TOKENS = 131072


class ConstructiveCodeError(ValueError):
    """Raised when checker evidence or witness identity is malformed."""


@dataclass(frozen=True)
class WitnessSchema:
    """Trusted, input-derived parsing and equivalence contract for one test."""

    family: WitnessFamily
    exact_item_count: int | None = None
    expected_keys: tuple[int, ...] = ()
    expected_universe: tuple[int, ...] = ()
    exact_group_count: int | None = None
    count_header: CountHeader = "none"
    integer_minimum: int = -(2**63)
    integer_maximum: int = 2**63 - 1
    schema_version: str = WITNESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != WITNESS_SCHEMA_VERSION:
            raise ConstructiveCodeError("unsupported witness schema version")
        if self.family not in {
            "ordered_sequence",
            "unordered_set",
            "assignment",
            "unordered_partition",
        }:
            raise ConstructiveCodeError("unsupported witness family")
        if self.count_header not in {"none", "item_count", "group_count"}:
            raise ConstructiveCodeError("unsupported count header")
        if self.integer_minimum > self.integer_maximum:
            raise ConstructiveCodeError("invalid integer bounds")
        if self.exact_item_count is not None and self.exact_item_count < 0:
            raise ConstructiveCodeError("exact item count cannot be negative")
        if self.exact_group_count is not None and self.exact_group_count < 1:
            raise ConstructiveCodeError("exact group count must be positive")
        if len(set(self.expected_keys)) != len(self.expected_keys):
            raise ConstructiveCodeError("expected assignment keys must be unique")
        if len(set(self.expected_universe)) != len(self.expected_universe):
            raise ConstructiveCodeError("expected universe must be unique")
        trusted_values = (*self.expected_keys, *self.expected_universe)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in trusted_values
        ):
            raise ConstructiveCodeError("trusted schema values must be integers")
        if any(
            value < self.integer_minimum or value > self.integer_maximum
            for value in trusted_values
        ):
            raise ConstructiveCodeError("trusted schema value is outside bounds")

        if self.family == "ordered_sequence":
            if self.exact_item_count is None or self.exact_item_count < 1:
                raise ConstructiveCodeError(
                    "ordered sequence requires a positive exact item count"
                )
            if self.expected_keys or self.expected_universe:
                raise ConstructiveCodeError(
                    "ordered sequence cannot declare keys or an unordered universe"
                )
            if self.exact_group_count is not None:
                raise ConstructiveCodeError(
                    "ordered sequence cannot declare a group count"
                )
            if self.count_header == "group_count":
                raise ConstructiveCodeError(
                    "ordered sequence cannot use a group-count header"
                )
        elif self.family == "unordered_set":
            if self.exact_item_count is None and not self.expected_universe:
                raise ConstructiveCodeError(
                    "unordered set requires a count or expected universe"
                )
            if self.expected_keys or self.exact_group_count is not None:
                raise ConstructiveCodeError(
                    "unordered set cannot declare assignment or partition fields"
                )
            if self.count_header == "group_count":
                raise ConstructiveCodeError(
                    "unordered set cannot use a group-count header"
                )
        elif self.family == "assignment":
            if self.exact_item_count is None and not self.expected_keys:
                raise ConstructiveCodeError(
                    "assignment requires a count or expected keys"
                )
            if self.expected_universe or self.exact_group_count is not None:
                raise ConstructiveCodeError(
                    "assignment cannot declare set or partition fields"
                )
            if self.count_header == "group_count":
                raise ConstructiveCodeError(
                    "assignment cannot use a group-count header"
                )
        else:
            if not self.expected_universe:
                raise ConstructiveCodeError(
                    "unordered partition requires an expected universe"
                )
            if self.expected_keys or self.exact_item_count is not None:
                raise ConstructiveCodeError(
                    "unordered partition cannot declare item or assignment fields"
                )
            if self.count_header == "item_count":
                raise ConstructiveCodeError(
                    "unordered partition cannot use an item-count header"
                )


@dataclass(frozen=True)
class ReleasedCheckerDecision:
    """Hash-bound result emitted by the trusted released-checker runner."""

    checker_sha256: str
    input_sha256: str
    output_sha256: str
    accepted: bool
    exit_code: int
    timed_out: bool = False

    def __post_init__(self) -> None:
        for name in ("checker_sha256", "input_sha256", "output_sha256"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ConstructiveCodeError(f"{name} is not a lowercase SHA-256")
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise ConstructiveCodeError("checker exit code must be an integer")
        if self.accepted and (self.timed_out or self.exit_code != 0):
            raise ConstructiveCodeError(
                "accepted checker decision must exit successfully without timeout"
            )


@dataclass(frozen=True)
class CanonicalWitness:
    """Canonical semantic payload for one checker-accepted output."""

    family: WitnessFamily
    checker_sha256: str
    input_sha256: str
    output_sha256: str
    canonical_json: str
    canonical_key: str


@dataclass(frozen=True)
class CheckedWitness:
    """One hidden-test checker decision paired with its canonical witness."""

    decision: ReleasedCheckerDecision
    witness: CanonicalWitness


@dataclass(frozen=True)
class CanonicalBehavior:
    """Ordered hidden-suite behavior of one generated program."""

    canonical_json: str
    canonical_key: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _output_bytes(output: str | bytes) -> bytes:
    if isinstance(output, bytes):
        encoded = output
    elif isinstance(output, str):
        encoded = output.encode("utf-8")
    else:
        raise ConstructiveCodeError("witness output must be text or bytes")
    if not encoded or len(encoded) > _MAX_OUTPUT_BYTES:
        raise ConstructiveCodeError("witness output is empty or too large")
    return encoded


def _output_text(output: str | bytes) -> tuple[bytes, str]:
    encoded = _output_bytes(output)
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConstructiveCodeError("witness output is not UTF-8") from error
    if "\x00" in text:
        raise ConstructiveCodeError("witness output contains a NUL byte")
    return encoded, text


def _parse_integer_tokens(text: str) -> list[int]:
    raw_tokens = text.split()
    if not raw_tokens or len(raw_tokens) > _MAX_INTEGER_TOKENS:
        raise ConstructiveCodeError("witness has an invalid token count")
    values: list[int] = []
    for token in raw_tokens:
        if _INTEGER_TOKEN.fullmatch(token) is None:
            raise ConstructiveCodeError("witness contains a non-integer token")
        values.append(int(token))
    return values


def _check_bounds(values: Sequence[int], schema: WitnessSchema) -> None:
    if any(
        value < schema.integer_minimum or value > schema.integer_maximum
        for value in values
    ):
        raise ConstructiveCodeError("witness integer is outside schema bounds")


def _strip_item_count(values: list[int], schema: WitnessSchema) -> list[int]:
    if schema.count_header == "none":
        return values
    if schema.count_header != "item_count" or not values:
        raise ConstructiveCodeError("invalid witness count header")
    declared = values[0]
    payload = values[1:]
    if schema.family == "assignment":
        actual = len(payload) // 2 if len(payload) % 2 == 0 else -1
    else:
        actual = len(payload)
    if declared != actual:
        raise ConstructiveCodeError("declared item count does not match output")
    return payload


def _canonical_payload(text: str, schema: WitnessSchema) -> object:
    if schema.family != "unordered_partition":
        values = _strip_item_count(_parse_integer_tokens(text), schema)
        _check_bounds(values, schema)

    if schema.family == "ordered_sequence":
        if len(values) != schema.exact_item_count:
            raise ConstructiveCodeError("ordered sequence has the wrong length")
        return values

    if schema.family == "unordered_set":
        if len(values) != len(set(values)):
            raise ConstructiveCodeError("unordered set contains a duplicate")
        if (
            schema.exact_item_count is not None
            and len(values) != schema.exact_item_count
        ):
            raise ConstructiveCodeError("unordered set has the wrong size")
        if schema.expected_universe and set(values) != set(schema.expected_universe):
            raise ConstructiveCodeError("unordered set differs from expected universe")
        return sorted(values)

    if schema.family == "assignment":
        if len(values) % 2 != 0:
            raise ConstructiveCodeError("assignment must contain key-value pairs")
        pairs = list(zip(values[::2], values[1::2]))
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ConstructiveCodeError("assignment contains a duplicate key")
        if (
            schema.exact_item_count is not None
            and len(pairs) != schema.exact_item_count
        ):
            raise ConstructiveCodeError("assignment has the wrong size")
        if schema.expected_keys and set(keys) != set(schema.expected_keys):
            raise ConstructiveCodeError("assignment keys differ from expected keys")
        return [list(pair) for pair in sorted(pairs)]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if schema.count_header == "group_count":
        if not lines or _INTEGER_TOKEN.fullmatch(lines[0]) is None:
            raise ConstructiveCodeError("partition lacks its group-count header")
        declared_groups = int(lines.pop(0))
    else:
        declared_groups = None
    if not lines:
        raise ConstructiveCodeError("partition contains no groups")
    groups: list[list[int]] = []
    for line in lines:
        group = _parse_integer_tokens(line)
        _check_bounds(group, schema)
        if len(group) != len(set(group)):
            raise ConstructiveCodeError("partition group contains a duplicate")
        groups.append(sorted(group))
    if declared_groups is not None and declared_groups != len(groups):
        raise ConstructiveCodeError("declared group count does not match output")
    if schema.exact_group_count is not None and len(groups) != schema.exact_group_count:
        raise ConstructiveCodeError("partition has the wrong group count")
    flattened = [value for group in groups for value in group]
    if len(flattened) != len(set(flattened)):
        raise ConstructiveCodeError("partition repeats an element across groups")
    if set(flattened) != set(schema.expected_universe):
        raise ConstructiveCodeError("partition differs from expected universe")
    return sorted(groups)


def canonicalize_accepted_witness(
    output: str | bytes,
    schema: WitnessSchema,
    decision: ReleasedCheckerDecision,
) -> CanonicalWitness:
    """Canonicalize exactly the bytes accepted by the released checker."""

    encoded, text = _output_text(output)
    if sha256_bytes(encoded) != decision.output_sha256:
        raise ConstructiveCodeError("output does not match checker-bound SHA-256")
    if not decision.accepted:
        raise ConstructiveCodeError("cannot canonicalize a rejected checker output")
    payload = {
        "schema_version": schema.schema_version,
        "family": schema.family,
        "value": _canonical_payload(text, schema),
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = sha256_bytes(canonical_json.encode("ascii"))
    return CanonicalWitness(
        family=schema.family,
        checker_sha256=decision.checker_sha256,
        input_sha256=decision.input_sha256,
        output_sha256=decision.output_sha256,
        canonical_json=canonical_json,
        canonical_key=f"constructive_witness:{schema.family}:v1:{digest}",
    )


def canonicalize_accepted_witness_view(
    output: str | bytes,
    witness_view: str,
    schema: WitnessSchema,
    decision: ReleasedCheckerDecision,
) -> CanonicalWitness:
    """Canonicalize a trusted adapter view while binding the exact raw output.

    Some released tasks wrap the witness in a YES/NO token, omit assignment
    keys for position-indexed matrices, or encode partitions with per-group
    counts. The task adapter may transform that syntax into the generic
    schema's integer view, but the checker decision and returned witness remain
    bound to the exact unmodified program output bytes.
    """

    encoded, _ = _output_text(output)
    if sha256_bytes(encoded) != decision.output_sha256:
        raise ConstructiveCodeError("output does not match checker-bound SHA-256")
    if not decision.accepted:
        raise ConstructiveCodeError("cannot canonicalize a rejected checker output")
    payload = {
        "schema_version": schema.schema_version,
        "family": schema.family,
        "value": _canonical_payload(str(witness_view), schema),
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = sha256_bytes(canonical_json.encode("ascii"))
    return CanonicalWitness(
        family=schema.family,
        checker_sha256=decision.checker_sha256,
        input_sha256=decision.input_sha256,
        output_sha256=decision.output_sha256,
        canonical_json=canonical_json,
        canonical_key=f"constructive_witness:{schema.family}:v1:{digest}",
    )


def canonicalize_checked_behavior(
    problem_key: str,
    suite_sha256: str,
    observations: Sequence[CheckedWitness],
) -> CanonicalBehavior:
    """Return the semantic mode for the ordered frozen hidden-test suite."""

    if not str(problem_key).strip():
        raise ConstructiveCodeError("problem key is empty")
    if _SHA256.fullmatch(suite_sha256) is None:
        raise ConstructiveCodeError("suite_sha256 is not a lowercase SHA-256")
    if not observations:
        raise ConstructiveCodeError("behavior requires at least one observation")
    input_hashes = [item.decision.input_sha256 for item in observations]
    if len(input_hashes) != len(set(input_hashes)):
        raise ConstructiveCodeError("behavior suite contains duplicate input hashes")
    for item in observations:
        if not item.decision.accepted:
            raise ConstructiveCodeError("behavior contains a rejected checker decision")
        if (
            item.witness.checker_sha256 != item.decision.checker_sha256
            or item.witness.input_sha256 != item.decision.input_sha256
            or item.witness.output_sha256 != item.decision.output_sha256
        ):
            raise ConstructiveCodeError(
                "behavior witness does not match its checker decision"
            )

    payload = {
        "schema_version": BEHAVIOR_KEY_VERSION,
        "problem_key": str(problem_key),
        "suite_sha256": suite_sha256,
        "observations": [
            {
                "input_sha256": item.decision.input_sha256,
                "witness": json.loads(item.witness.canonical_json),
            }
            for item in observations
        ],
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = sha256_bytes(canonical_json.encode("ascii"))
    return CanonicalBehavior(
        canonical_json=canonical_json,
        canonical_key=f"constructive_behavior:v1:{digest}",
    )
