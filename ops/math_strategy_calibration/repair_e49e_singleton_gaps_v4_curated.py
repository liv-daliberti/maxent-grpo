#!/usr/bin/env python3
"""Carry V3 successes and audit two corrected precommitted contracts."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
V3_PATH = HERE / "repair_e49e_singleton_gaps_v3_curated.py"
DEFAULT_V3_CONTRACTS = HERE / "e49e_curated_singleton_contracts_toy.json"
DEFAULT_V4_CONTRACTS = (
    HERE / "e49e_curated_singleton_contracts_toy_v4.json"
)


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, V3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load E49E curated V3 implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


impl = _load_module("e49e_curated_v4_impl")
v3_validator = _load_module("e49e_curated_v3_validator")

REPAIR_VERSION = "e49e_precommitted_curated_singleton_repair_v4"
REPAIR_ORIGIN = "precommitted_curated_singleton_repair_v4"
PRIOR_REPAIR_VERSION = v3_validator.REPAIR_VERSION
_prior_records: dict[str, dict[str, Any]] = {}
_impl_record_passes = impl._repair_record_passes


def _carried_prior(
    *,
    row_id: str,
    problem: str,
    reference_answer: str,
) -> dict[str, Any] | None:
    prior = _prior_records.get(row_id)
    if not isinstance(prior, dict) or prior.get("pass") is not True:
        return None
    if not v3_validator._repair_record_passes(
        prior,
        row_id=row_id,
        problem=problem,
        reference_answer=reference_answer,
    ):
        raise RuntimeError(f"prior V3 singleton repair changed: {row_id}")
    return {
        "schema": "e49e_singleton_repair_record_v1",
        "repair_version": REPAIR_VERSION,
        "row_id": row_id,
        "problem_sha256": impl._sha256_bytes(problem.encode("utf-8")),
        "reference_answer_sha256": impl._sha256_bytes(
            reference_answer.encode("utf-8")
        ),
        "attempts": [],
        "carry_origin": PRIOR_REPAIR_VERSION,
        "prior_repair_record_sha256": impl._canonical_sha256(prior),
        "prior_repair_record": prior,
        "menu": prior["menu"],
        "menu_sha256": prior["menu_sha256"],
        "pass": True,
    }


def _repair_record_passes(
    record: dict[str, Any],
    *,
    row_id: str,
    problem: str,
    reference_answer: str,
) -> bool:
    if record.get("carry_origin") == PRIOR_REPAIR_VERSION:
        prior = record.get("prior_repair_record")
        return bool(
            isinstance(prior, dict)
            and record.get("repair_version") == REPAIR_VERSION
            and record.get("row_id") == row_id
            and record.get("attempts") == []
            and record.get("prior_repair_record_sha256")
            == impl._canonical_sha256(prior)
            and v3_validator._repair_record_passes(
                prior,
                row_id=row_id,
                problem=problem,
                reference_answer=reference_answer,
            )
            and record.get("menu") == prior.get("menu")
            and record.get("menu_sha256") == prior.get("menu_sha256")
            and record.get("pass") is True
        )
    return _impl_record_passes(
        record,
        row_id=row_id,
        problem=problem,
        reference_answer=reference_answer,
    )


def _initialize_validator(path: pathlib.Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    v3_validator._contracts.clear()
    v3_validator._contracts.update(payload)
    v3_validator._contracts_sha256 = v3_validator._sha256_bytes(
        path.read_bytes()
    )
    v3_validator._configure()


def _configure_impl(path: pathlib.Path) -> None:
    impl.REPAIR_VERSION = REPAIR_VERSION
    impl.REPAIR_ORIGIN = REPAIR_ORIGIN
    impl.PRIOR_REPAIR_VERSION = PRIOR_REPAIR_VERSION
    impl._prior_records.clear()
    impl._prior_records.update(_prior_records)
    impl._contracts.clear()
    impl._contracts.update(json.loads(path.read_text(encoding="utf-8")))
    impl._contracts_sha256 = impl._sha256_bytes(path.read_bytes())
    impl._carried_prior = _carried_prior
    impl.base.REPAIR_VERSION = REPAIR_VERSION
    impl.base.REPAIR_ORIGIN = REPAIR_ORIGIN
    impl.base._repair_one = impl._repair_one
    impl.base._repair_record_passes = _repair_record_passes


def main() -> None:
    prior_path = pathlib.Path(
        os.environ.get("E49E_PRIOR_REPAIR_V3_RECORDS", "")
    )
    v3_contracts = pathlib.Path(
        os.environ.get("E49E_V3_CURATED_CONTRACTS", str(DEFAULT_V3_CONTRACTS))
    )
    v4_contracts = pathlib.Path(
        os.environ.get("E49E_V4_CURATED_CONTRACTS", str(DEFAULT_V4_CONTRACTS))
    )
    if not prior_path.is_file():
        raise RuntimeError("terminal V3 repair records are missing")
    if not v3_contracts.is_file() or not v4_contracts.is_file():
        raise RuntimeError("curated contract manifest is missing")
    _prior_records.update(impl.base.pipeline._load_latest(prior_path))
    _initialize_validator(v3_contracts)
    _configure_impl(v4_contracts)
    impl.base.main()


if __name__ == "__main__":
    main()
