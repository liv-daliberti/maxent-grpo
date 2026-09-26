#!/usr/bin/env python3
"""Materialize the restricted executable MathIR linear-algebra benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.mathir import (  # noqa: E402
    MATHIR_VERIFIER,
    MATHIR_VERSION,
    certified_mathir_strategy_keys,
)


@dataclass(frozen=True)
class Family:
    name: str
    initial_lhs: str
    initial_rhs: str
    display_equation: str
    certified_programs: tuple[str, ...]


FAMILIES = (
    Family(
        name="ax_plus_b_eq_c",
        initial_lhs="add(mul(a,x),b)",
        initial_rhs="c",
        display_equation="a*x + b = c",
        certified_programs=(
            "sub(b);div(a)",
            "div(a);sub(div(b,a))",
        ),
    ),
    Family(
        name="x_over_a_plus_b_eq_c",
        initial_lhs="add(div(x,a),b)",
        initial_rhs="c",
        display_equation="x/a + b = c",
        certified_programs=(
            "sub(b);mul(a)",
            "mul(a);sub(mul(a,b))",
        ),
    ),
)


def _prompt(family: Family, bindings: dict[str, int]) -> str:
    binding_text = ", ".join(f"{name}={bindings[name]}" for name in sorted(bindings))
    starter = family.certified_programs[0]
    return (
        "MathIR linear-v0. Solve for x with executable equation commands. "
        f"Bindings: {binding_text}. Initial equation: {family.display_equation}. "
        "A command add(E), sub(E), mul(E), or div(E) applies to BOTH sides "
        "and exact-simplifies. Separate 1-4 commands with semicolons. "
        "E may use only a,b,c,x and add/sub/mul/div/neg; never use numbers. "
        "Finish with x isolated. Return a different valid path if you know one. "
        "Otherwise COPY this exact valid response and nothing else: "
        f"\\boxed{{{starter}}}"
    )


def _sample_bindings(family: Family, rng: random.Random) -> dict[str, int]:
    nonzero_small = [value for value in range(-9, 10) if value != 0]
    nonzero_offset = [value for value in range(-12, 13) if value != 0]
    solution = rng.choice(nonzero_small)
    a = rng.choice(nonzero_small)
    b = rng.choice(nonzero_offset)
    if family.name == "ax_plus_b_eq_c":
        return {"a": a, "b": b, "c": a * solution + b}
    if family.name == "x_over_a_plus_b_eq_c":
        # Construct an integer solution while retaining the formal x/a state.
        quotient = solution
        return {"a": a, "b": b, "c": quotient + b}
    raise RuntimeError(f"unknown MathIR family: {family.name}")


def _row(
    *,
    family: Family,
    bindings: dict[str, int],
    split_tag: str,
    seed: int,
    index: int,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "verifier": MATHIR_VERIFIER,
        "mathir_version": MATHIR_VERSION,
        "bindings": bindings,
        "initial_lhs": family.initial_lhs,
        "initial_rhs": family.initial_rhs,
        "max_steps": 4,
        "support_is_open": True,
        "source": "synthetic_mathir_linear_v0",
        "family": family.name,
        "instance_id": f"{split_tag}-{seed}-{index}",
    }
    certified_keys = certified_mathir_strategy_keys(spec, family.certified_programs)
    if len(certified_keys) < 2:
        raise RuntimeError(f"{family.name} did not retain two certified strategies")
    certified_digest = hashlib.sha256(
        "\n".join(sorted(certified_keys)).encode("utf-8")
    ).hexdigest()
    starter_validation_keys = certified_mathir_strategy_keys(
        spec,
        (family.certified_programs[0],),
    )
    spec["num_certified_strategies"] = len(certified_keys)
    spec["certified_strategy_key_sha256"] = certified_digest
    spec["public_seed_program"] = family.certified_programs[0]
    spec["public_seed_key"] = next(iter(starter_validation_keys))
    return {
        "problem": _prompt(family, bindings),
        "answer": json.dumps(spec, sort_keys=True, separators=(",", ":")),
        "modebench_task": MATHIR_VERIFIER,
        # Zero is deliberate: the executable grammar has open growing support,
        # so normalized total-mode coverage has no honest denominator.
        "answer_mode_count": 0,
        "answer_mode_split": split_tag,
        "mathir_family": family.name,
    }


def _build_rows(
    count: int,
    *,
    seed: int,
    split_tag: str,
    excluded_identities: set[tuple[str, tuple[tuple[str, int], ...]]] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    excluded = excluded_identities or set()
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    rows: list[dict[str, Any]] = []
    family_cursor = 0
    attempts = 0
    while len(rows) < int(count):
        attempts += 1
        if attempts > max(10_000, int(count) * 200):
            raise RuntimeError(f"could not build {count} unique MathIR rows")
        family = FAMILIES[family_cursor % len(FAMILIES)]
        family_cursor += 1
        bindings = _sample_bindings(family, rng)
        identity = (family.name, tuple(sorted(bindings.items())))
        if identity in excluded or identity in seen:
            continue
        candidate = _row(
            family=family,
            bindings=bindings,
            split_tag=split_tag,
            seed=seed,
            index=len(rows),
        )
        seen.add(identity)
        rows.append(candidate)
    rng.shuffle(rows)
    return rows


def _identities(rows: list[dict[str, Any]]) -> set[tuple[str, tuple[tuple[str, int], ...]]]:
    identities = set()
    for row in rows:
        spec = json.loads(row["answer"])
        identities.add((str(spec["family"]), tuple(sorted(spec["bindings"].items()))))
    return identities


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=384)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=4500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.train_size <= 0 or args.eval_size <= 0:
        raise SystemExit("train and eval sizes must be positive")
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} already exists; pass --overwrite")
        if output_root.name != "mathir_algebra_v0_probe":
            raise SystemExit(
                "refusing overwrite outside the exact mathir_algebra_v0_probe target"
            )
        shutil.rmtree(output_root)

    train_rows = _build_rows(
        args.train_size,
        seed=args.seed,
        split_tag="train_open_support",
    )
    eval_rows = _build_rows(
        args.eval_size,
        seed=args.seed + 10_000,
        split_tag="eval_open_support",
        excluded_identities=_identities(train_rows),
    )
    if _identities(train_rows) & _identities(eval_rows):
        raise RuntimeError("train/eval MathIR identity overlap")

    DatasetDict({"train": Dataset.from_list(train_rows)}).save_to_disk(
        str(output_root / "train")
    )
    DatasetDict({"multi_answer": Dataset.from_list(eval_rows)}).save_to_disk(
        str(output_root / "eval")
    )
    identity = {
        "schema": "mathir_algebra_linear_v0_dataset_v1",
        "seed": int(args.seed),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_rows_sha256": _rows_sha256(train_rows),
        "eval_rows_sha256": _rows_sha256(eval_rows),
        "families": [family.name for family in FAMILIES],
        "support": "open_growing",
        "answer_mode_count": None,
        "certified_strategy_sets_are_exhaustive": False,
    }
    (output_root / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote MathIR train={len(train_rows)} eval={len(eval_rows)} "
        f"to {output_root}"
    )
    print(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()
