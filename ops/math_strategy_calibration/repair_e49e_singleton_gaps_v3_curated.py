#!/usr/bin/env python3
"""Certify precommitted toy singleton contracts for any post-V2 gaps."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
BASE_PATH = HERE / "repair_e49e_singleton_gaps.py"
V2B_PATH = HERE / "repair_e49e_singleton_gaps_v2b.py"
DEFAULT_CONTRACTS = HERE / "e49e_curated_singleton_contracts_toy.json"


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_module("e49e_curated_repair_base", BASE_PATH)
v2b_validator = _load_module(
    "e49e_curated_repair_v2b_validator",
    V2B_PATH,
)
v2b_validator._configure()

REPAIR_VERSION = "e49e_precommitted_curated_singleton_repair_v3"
REPAIR_ORIGIN = "precommitted_curated_singleton_repair"
PRIOR_REPAIR_VERSION = v2b_validator.REPAIR_VERSION
_prior_records: dict[str, dict[str, Any]] = {}
_contracts: dict[str, dict[str, Any]] = {}
_contracts_sha256 = ""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _audit_is_well_formed_and_exact(
    menu: Any,
    audit: dict[str, Any],
    *,
    reference_answer: str,
) -> bool:
    strategy = menu.strategy("S1")
    if strategy is None:
        return False
    assessment = audit.get("assessment")
    return bool(
        audit.get("trace_contract_version")
        == base.pipeline.TRACE_CONTRACT_VERSION
        and audit.get("kind") == "soundness"
        and audit.get("role") in base.pipeline.SOUNDNESS_ROLES
        and audit.get("seed") in base.pipeline.SOUNDNESS_SEEDS
        and audit.get("candidate_menu_sha256") == menu.sha256
        and audit.get("strategy_id") == "S1"
        and audit.get("finish_reason") == "stop"
        and str(audit.get("response_id") or "")
        and str(audit.get("content_sha256") or "")
        and base.pipeline._sound_assessment_is_well_formed(
            assessment,
            action_ids=list(strategy.action_ids),
            strategy_id="S1",
        )
        and base.pipeline.audited_answer_matches(
            assessment["derived_answer"],
            reference_answer,
        )
    )


def _audit_affirms_route_integrity(audit: dict[str, Any]) -> bool:
    assessment = audit["assessment"]
    return bool(
        all(
            action["status"] == "valid"
            for action in assessment["action_executions"]
        )
        and assessment["uses_only_declared_actions"] is True
        and assessment["self_contained_without_other_strategy"] is True
    )


def _objective_double_sound(
    menu: Any,
    audits: list[dict[str, Any]],
    *,
    reference_answer: str,
) -> bool:
    return bool(
        len(audits) == 2
        and {
            (audit.get("seed"), audit.get("role"))
            for audit in audits
            if isinstance(audit, dict)
        }
        == set(
            zip(
                base.pipeline.SOUNDNESS_SEEDS,
                base.pipeline.SOUNDNESS_ROLES,
                strict=True,
            )
        )
        and all(
            _audit_is_well_formed_and_exact(
                menu,
                audit,
                reference_answer=reference_answer,
            )
            for audit in audits
        )
        and any(_audit_affirms_route_integrity(audit) for audit in audits)
    )


def _carried_prior(
    *,
    row_id: str,
    problem: str,
    reference_answer: str,
) -> dict[str, Any] | None:
    prior = _prior_records.get(row_id)
    if not isinstance(prior, dict) or prior.get("pass") is not True:
        return None
    if not v2b_validator.v2._repair_record_passes(
        prior,
        row_id=row_id,
        problem=problem,
        reference_answer=reference_answer,
    ):
        raise RuntimeError(f"prior V2 singleton repair changed: {row_id}")
    return {
        "schema": "e49e_singleton_repair_record_v1",
        "repair_version": REPAIR_VERSION,
        "row_id": row_id,
        "problem_sha256": _sha256_bytes(problem.encode("utf-8")),
        "reference_answer_sha256": _sha256_bytes(
            reference_answer.encode("utf-8")
        ),
        "attempts": [],
        "carry_origin": PRIOR_REPAIR_VERSION,
        "prior_repair_record_sha256": _canonical_sha256(prior),
        "prior_repair_record": prior,
        "menu": prior["menu"],
        "menu_sha256": prior["menu_sha256"],
        "pass": True,
    }


def _repair_one(
    *,
    endpoint: str,
    model: str,
    row_id: str,
    problem: str,
    reference_answer: str,
    gold_solution: str,
    cache_root: pathlib.Path,
    timeout: int,
) -> dict[str, Any]:
    del gold_solution
    carried = _carried_prior(
        row_id=row_id,
        problem=problem,
        reference_answer=reference_answer,
    )
    if carried is not None:
        return carried
    payload = _contracts.get(row_id)
    if not isinstance(payload, dict):
        raise RuntimeError(f"no precommitted singleton contract: {row_id}")
    menu = base.pipeline._menu_from_payload(payload)
    closed = base.pipeline._extract_closed_candidate(menu, "S1")
    if (
        closed is None
        or closed.sha256 != menu.sha256
        or len(menu.strategies) != 1
        or list(menu.strategies[0].action_ids)
        != [action.action_id for action in menu.actions]
        or not v2b_validator.v2._proposal_is_nonleaking_v2(
            menu,
            problem=problem,
            reference_answer=reference_answer,
        )
    ):
        raise RuntimeError(
            f"precommitted singleton contract failed local gates: {row_id}"
        )

    row_hash = _sha256_bytes(row_id.encode("utf-8"))[:20]
    audits = []
    for seed, role in zip(
        base.pipeline.SOUNDNESS_SEEDS,
        base.pipeline.SOUNDNESS_ROLES,
        strict=True,
    ):
        audit_path = (
            cache_root
            / row_hash
            / f"curated-soundness-S1-{seed}-{menu.sha256[:16]}.json"
        )
        audits.append(
            base.pipeline._cached_request(
                audit_path,
                base.pipeline._sound_request,
                endpoint=endpoint,
                model=model,
                problem=problem,
                reference_answer=reference_answer,
                menu=menu,
                strategy_id="S1",
                role=role,
                seed=seed,
                timeout=timeout,
            )
        )
    double_sound = _objective_double_sound(
        menu,
        audits,
        reference_answer=reference_answer,
    )
    proposal = {
        "repair_version": REPAIR_VERSION,
        "kind": "precommitted_curated_singleton_contract",
        "proposal_input_scope": "frozen_problem_specific_contract",
        "contracts_manifest_sha256": _contracts_sha256,
        "menu": json.loads(menu.canonical_json),
        "menu_sha256": menu.sha256,
        "pass": True,
    }
    record = {
        "schema": "e49e_singleton_repair_record_v1",
        "repair_version": REPAIR_VERSION,
        "row_id": row_id,
        "problem_sha256": _sha256_bytes(problem.encode("utf-8")),
        "reference_answer_sha256": _sha256_bytes(
            reference_answer.encode("utf-8")
        ),
        "attempts": [
            {
                "proposal": proposal,
                "sound_audits": audits,
            }
        ],
        "pass": double_sound,
    }
    if double_sound:
        record["menu"] = json.loads(menu.canonical_json)
        record["menu_sha256"] = menu.sha256
    else:
        record["error"] = "precommitted contract failed double-sound audit"
    return record


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
            == _canonical_sha256(prior)
            and v2b_validator.v2._repair_record_passes(
                prior,
                row_id=row_id,
                problem=problem,
                reference_answer=reference_answer,
            )
            and record.get("menu") == prior.get("menu")
            and record.get("menu_sha256") == prior.get("menu_sha256")
            and record.get("pass") is True
        )
    if (
        not isinstance(record, dict)
        or record.get("schema") != "e49e_singleton_repair_record_v1"
        or record.get("repair_version") != REPAIR_VERSION
        or record.get("row_id") != row_id
        or record.get("problem_sha256")
        != _sha256_bytes(problem.encode("utf-8"))
        or record.get("reference_answer_sha256")
        != _sha256_bytes(reference_answer.encode("utf-8"))
        or not isinstance(record.get("attempts"), list)
        or len(record["attempts"]) != 1
    ):
        return False
    attempt = record["attempts"][0]
    proposal = attempt.get("proposal")
    audits = attempt.get("sound_audits")
    if (
        not isinstance(proposal, dict)
        or proposal.get("repair_version") != REPAIR_VERSION
        or proposal.get("kind")
        != "precommitted_curated_singleton_contract"
        or proposal.get("proposal_input_scope")
        != "frozen_problem_specific_contract"
        or proposal.get("contracts_manifest_sha256")
        != _contracts_sha256
        or proposal.get("pass") is not True
        or not isinstance(proposal.get("menu"), dict)
        or not isinstance(audits, list)
        or len(audits) != 2
    ):
        return False
    try:
        menu = base.pipeline._menu_from_payload(proposal["menu"])
        expected = base.pipeline._menu_from_payload(_contracts[row_id])
    except (KeyError, TypeError, ValueError):
        return False
    if (
        menu.canonical_json != expected.canonical_json
        or menu.sha256 != proposal.get("menu_sha256")
        or not v2b_validator.v2._proposal_is_nonleaking_v2(
            menu,
            problem=problem,
            reference_answer=reference_answer,
        )
        or {
            (audit.get("seed"), audit.get("role"))
            for audit in audits
            if isinstance(audit, dict)
        }
        != set(
            zip(
                base.pipeline.SOUNDNESS_SEEDS,
                base.pipeline.SOUNDNESS_ROLES,
                strict=True,
            )
        )
    ):
        return False
    double_sound = _objective_double_sound(
        menu,
        audits,
        reference_answer=reference_answer,
    )
    if record.get("pass") is not True:
        return not double_sound and not isinstance(record.get("menu"), dict)
    return bool(
        double_sound
        and record.get("menu_sha256") == menu.sha256
        and record.get("menu") == json.loads(menu.canonical_json)
    )


def _configure() -> None:
    base.REPAIR_VERSION = REPAIR_VERSION
    base.REPAIR_ORIGIN = REPAIR_ORIGIN
    base._repair_one = _repair_one
    base._repair_record_passes = _repair_record_passes


def main() -> None:
    global _contracts_sha256
    prior_path = pathlib.Path(
        os.environ.get("E49E_PRIOR_REPAIR_V2B_RECORDS", "")
    )
    contracts_path = pathlib.Path(
        os.environ.get("E49E_CURATED_CONTRACTS", str(DEFAULT_CONTRACTS))
    )
    if prior_path.is_file():
        _prior_records.update(base.pipeline._load_latest(prior_path))
    if not contracts_path.is_file():
        raise RuntimeError("precommitted singleton contracts are missing")
    _contracts.update(json.loads(contracts_path.read_text(encoding="utf-8")))
    _contracts_sha256 = _sha256_bytes(contracts_path.read_bytes())
    _configure()
    base.main()


if __name__ == "__main__":
    main()
