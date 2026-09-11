#!/usr/bin/env python3
"""Exhaustively verify E16's two frozen finite-action datasets and codec.

This preflight intentionally does more than hash Arrow files.  It normalizes
the logical rows, exhausts every one of Countdown's 108 action codes, and
compares their semantic grader keys with the independently generated complete
answer set.  A small AST check also makes target/solution access in the decoder
an explicit protocol violation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import textwrap
from typing import Any

from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from make_modebench_data import _countdown_expression_map  # noqa: E402
from oat_drgrpo.canonical_actions import (  # noqa: E402
    COUNTDOWN_ACTIONS_BY_POSITION,
    decode_countdown_action_code,
    enumerate_countdown_action_codes,
)
from oat_drgrpo.math_grader import (  # noqa: E402
    _canonical_countdown_expression_key,
    _verify_countdown_expression,
)

try:  # package import in tests
    from .verify_e14_dataset import (
        EXPECTED_COMBINED_CONTENT_HASH as EXPECTED_GRAPH_COMBINED_CONTENT_HASH,
        validate_e14_rows,
    )
except ImportError:  # direct script execution
    from verify_e14_dataset import (  # type: ignore[no-redef]
        EXPECTED_COMBINED_CONTENT_HASH as EXPECTED_GRAPH_COMBINED_CONTENT_HASH,
        validate_e14_rows,
    )


EXPECTED_COUNTDOWN_COMBINED_CONTENT_HASH = (
    "eea53a7e80f1c500a71062a68c07869c77665948249aa29a02402137181821ca"
)
EXPECTED_COUNTDOWN_TRAIN_CONTENT_HASH = (
    "75532b08ab6c59fbea35b8a40af5f00cf2b4c0c5e1ac6b0d73c96dd592b38c8a"
)
EXPECTED_COUNTDOWN_EVAL_CONTENT_HASH = (
    "2262c3386fb7145408c2b65e3a54b2ffd95f19805f2310044615e4c954506903"
)


class DatasetCodecError(ValueError):
    """Raised when E16 data or its label-free codec leaves the frozen contract."""


def _content_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decoder_reference_keys() -> set[str]:
    """Return all reference-dictionary keys read by the Countdown decoder."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(decode_countdown_action_code)))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            key_node = node.slice
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                keys.add(key_node.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
    return keys


def validate_countdown_codec_static() -> dict[str, Any]:
    """Reject target/solution access and any drift from the 6 x 3 x 6 grammar."""

    support_sizes = tuple(len(support) for support in COUNTDOWN_ACTIONS_BY_POSITION)
    if support_sizes != (6, 3, 6):
        raise DatasetCodecError(
            f"Countdown action supports drifted: {support_sizes!r}"
        )
    codes = enumerate_countdown_action_codes()
    if len(codes) != 108 or len(set(codes)) != 108:
        raise DatasetCodecError("Countdown codec must expose exactly 108 unique codes")
    if codes[0] != "111" or codes[-1] != "636":
        raise DatasetCodecError("Countdown code ordering or position support drifted")
    reference_keys = _decoder_reference_keys()
    if reference_keys != {"numbers", "verifier"}:
        raise DatasetCodecError(
            "Countdown decoder may read only public operands and verifier; "
            f"observed keys={sorted(reference_keys)!r}"
        )
    return {
        "code_count": len(codes),
        "decoder_reference_keys": sorted(reference_keys),
        "first_code": codes[0],
        "horizon": 3,
        "last_code": codes[-1],
        "position_support_sizes": list(support_sizes),
        "target_or_solution_access": False,
    }


def _normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": str(row["answer"]),
        "answer_mode_count": int(row["answer_mode_count"]),
        "answer_mode_split": str(row["answer_mode_split"]),
        "modebench_task": str(row["modebench_task"]),
        "problem": str(row["problem"]),
    }


def validate_e16_countdown_rows(
    rows: list[dict[str, Any]], *, split_tag: str
) -> list[dict[str, Any]]:
    """Exhaust every code and return the normalized frozen row records."""

    expected_split = (
        "train_multi_answer" if split_tag == "train" else "eval_multi_answer"
    )
    codes = enumerate_countdown_action_codes()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        label = f"{split_tag}[{index}]"
        try:
            spec = json.loads(str(row["answer"]))
            numbers = [int(value) for value in spec["numbers"]]
            target = int(spec["target"])
            declared_count = int(row["answer_mode_count"])
            problem = str(row["problem"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise DatasetCodecError(f"{label} is not a Countdown row") from error
        if spec.get("verifier") != "countdown" or row.get("modebench_task") != "countdown":
            raise DatasetCodecError(f"{label} has the wrong task or verifier")
        if len(numbers) != 3 or len(set(numbers)) != 3:
            raise DatasetCodecError(f"{label} does not have three distinct operands")
        if str(row.get("answer_mode_split")) != expected_split:
            raise DatasetCodecError(f"{label} has the wrong answer-mode split")
        if int(spec.get("num_completions", -1)) != declared_count:
            raise DatasetCodecError(f"{label} num_completions drifted")
        if int(spec.get("num_expressions", -1)) != declared_count:
            raise DatasetCodecError(f"{label} num_expressions drifted")
        for required in (
            f"Using the numbers {numbers}",
            f"equals {target}",
            "Use each given number exactly once.",
            "inside \\boxed{}.",
        ):
            if required not in problem:
                raise DatasetCodecError(
                    f"{label} is missing prompt contract {required!r}"
                )

        key_to_code: dict[str, str] = {}
        valid_keys: set[str] = set()
        for code in codes:
            expression = decode_countdown_action_code(code, spec)
            key = _canonical_countdown_expression_key(expression, spec)
            if key is None:
                raise DatasetCodecError(f"{label} code {code} has no semantic key")
            if key in key_to_code:
                raise DatasetCodecError(
                    f"{label} codes {key_to_code[key]} and {code} alias {key}"
                )
            key_to_code[key] = code
            if _verify_countdown_expression(expression, spec):
                valid_keys.add(key)
        if len(key_to_code) != 108:
            raise DatasetCodecError(f"{label} does not expose 108 distinct keys")

        declared_expressions = _countdown_expression_map(numbers).get(target, set())
        declared_keys = {
            key
            for expression in declared_expressions
            if (key := _canonical_countdown_expression_key(expression, spec)) is not None
        }
        if valid_keys != declared_keys:
            missing = sorted(declared_keys - valid_keys)
            extra = sorted(valid_keys - declared_keys)
            raise DatasetCodecError(
                f"{label} codec does not bijectively cover declared modes: "
                f"missing={missing!r} extra={extra!r}"
            )
        if len(valid_keys) != declared_count or not 2 <= declared_count <= 8:
            raise DatasetCodecError(
                f"{label} valid-mode count mismatch: declared={declared_count} "
                f"codec={len(valid_keys)}"
            )
        normalized.append(_normalized_row(row))
    return normalized


def verify_e16_canonical_datasets(
    *, graph_data_root: Path, countdown_data_root: Path
) -> dict[str, Any]:
    """Verify both pools and return one campaign data identity."""

    codec = validate_countdown_codec_static()
    graph_train = validate_e14_rows(
        list(load_from_disk(str(graph_data_root / "train"))["train"]),
        split_tag="train",
    )
    graph_eval = validate_e14_rows(
        list(load_from_disk(str(graph_data_root / "eval"))["multi_answer"]),
        split_tag="multi_answer",
    )
    if len(graph_train) != 192 or len(graph_eval) != 96:
        raise DatasetCodecError(
            "E16 requires the E15 graph pools with 192/96 rows"
        )
    graph_hash = _content_hash({"eval": graph_eval, "train": graph_train})
    if graph_hash != EXPECTED_GRAPH_COMBINED_CONTENT_HASH:
        raise DatasetCodecError(
            f"graph data hash drifted: {graph_hash}"
        )

    countdown_train = validate_e16_countdown_rows(
        list(load_from_disk(str(countdown_data_root / "train"))["train"]),
        split_tag="train",
    )
    countdown_eval = validate_e16_countdown_rows(
        list(load_from_disk(str(countdown_data_root / "eval"))["multi_answer"]),
        split_tag="eval",
    )
    if len(countdown_train) != 384 or len(countdown_eval) != 128:
        raise DatasetCodecError(
            "E16 requires the frozen Countdown pools with 384/128 rows"
        )
    countdown_train_hash = _content_hash(countdown_train)
    countdown_eval_hash = _content_hash(countdown_eval)
    countdown_hash = _content_hash(
        {"eval": countdown_eval, "train": countdown_train}
    )
    expected_hashes = (
        EXPECTED_COUNTDOWN_TRAIN_CONTENT_HASH,
        EXPECTED_COUNTDOWN_EVAL_CONTENT_HASH,
        EXPECTED_COUNTDOWN_COMBINED_CONTENT_HASH,
    )
    observed_hashes = (
        countdown_train_hash,
        countdown_eval_hash,
        countdown_hash,
    )
    if observed_hashes != expected_hashes:
        raise DatasetCodecError(
            "Countdown logical content drifted: "
            f"expected={expected_hashes!r} observed={observed_hashes!r}"
        )
    return {
        "codec": codec,
        "countdown": {
            "combined_content_hash": countdown_hash,
            "eval_content_hash": countdown_eval_hash,
            "eval_rows": len(countdown_eval),
            "max_valid_modes": 8,
            "min_valid_modes": 2,
            "train_content_hash": countdown_train_hash,
            "train_rows": len(countdown_train),
        },
        "graph_coloring": {
            "combined_content_hash": graph_hash,
            "eval_rows": len(graph_eval),
            "train_rows": len(graph_train),
        },
        "schema": "e16_canonical_dataset_identity_v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-data-root", type=Path, required=True)
    parser.add_argument("--countdown-data-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        identity = verify_e16_canonical_datasets(
            graph_data_root=args.graph_data_root,
            countdown_data_root=args.countdown_data_root,
        )
    except (DatasetCodecError, FileNotFoundError, KeyError) as error:
        raise SystemExit(f"E16 canonical data/codec rejected: {error}") from error
    print(json.dumps(identity, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
