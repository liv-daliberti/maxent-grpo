#!/usr/bin/env python3
"""Fresh-seed answer-blind repair after V2's in-flight placement abort."""

from __future__ import annotations

import importlib.util
import os
import pathlib


HERE = pathlib.Path(__file__).resolve().parent
V2_PATH = HERE / "repair_e49e_singleton_gaps_v2.py"
SPEC = importlib.util.spec_from_file_location(
    "e49e_singleton_repair_v2b_base",
    V2_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load E49E singleton-repair V2 implementation")
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)

REPAIR_VERSION = "e49e_answer_blind_symbolic_singleton_repair_v2b"
REPAIR_ORIGIN = "answer_blind_symbolic_singleton_repair_v2b"
PROPOSAL_SEEDS = (492251, 492252, 492253, 492254)
PROPOSAL_ROLES = v2.PROPOSAL_ROLES


def _configure() -> None:
    v2.REPAIR_VERSION = REPAIR_VERSION
    v2.REPAIR_ORIGIN = REPAIR_ORIGIN
    v2.PROPOSAL_SEEDS = PROPOSAL_SEEDS
    v2.PROPOSAL_ROLES = PROPOSAL_ROLES
    v2._configure_base()


def main() -> None:
    prior_path = pathlib.Path(os.environ.get("E49E_PRIOR_REPAIR_RECORDS", ""))
    if prior_path.is_file():
        v2._prior_records.update(v2.base.pipeline._load_latest(prior_path))
    _configure()
    v2.base.main()


if __name__ == "__main__":
    main()
