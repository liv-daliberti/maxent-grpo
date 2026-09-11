#!/usr/bin/env python3
"""Calibrate conditioned teacher routes behind deterministic safe signatures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50g_safe_signature_teacher_route_calibration_20260726.md"
)
RUNTIME_CONTROL_SCHEMA_REPAIR = (
    ROOT
    / "paper/preregistration/"
    "e50g_runtime_control_schema_repair_20260726.md"
)
BASE_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50f_conditioned_teacher_route_calibration.py"
)
E50H_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50h_second_conditioned_teacher_corpus.py"
)
E50I_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50i_third_conditioned_teacher_contingency.py"
)
E50J_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50j_fourth_conditioned_teacher_contingency.py"
)
E50K_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50k_fifth_conditioned_teacher_contingency.py"
)
SIGNATURE_PATH = (
    ROOT
    / "ops/math_strategy_calibration/safe_math_strategy_signatures.py"
)
DEVELOPMENT_AUDIT = (
    ROOT
    / "var/artifacts/"
    "safe_math_strategy_signature_development_audit_v1.json"
)
FULL_REWRITE_AUDIT = (
    ROOT / "var/artifacts/e50g_full_injection_rewrite_audit_v1.json"
)
E50C_RESULT = (
    ROOT
    / "var/artifacts/e50c_72b_teacher_route_calibration_v1/result.json"
)
E50C_QUARANTINE_SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "quarantine_e50c_open_relation_result.py"
)
E50C_QUARANTINE_PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50c_open_relation_quarantine_20260726.md"
)
E50F_ROOT = (
    ROOT
    / "var/artifacts/e50f_conditioned_teacher_route_calibration_v1"
)
E50F_RESULT = E50F_ROOT / "result.json"
E50F_GENERATED = E50F_ROOT / "private/conditioned_teacher_records.jsonl"
E50H_ROOT = (
    ROOT
    / "var/artifacts/e50h_second_conditioned_teacher_corpus_v1"
)
E50H_RESULT = E50H_ROOT / "result.json"
E50H_GENERATED = E50H_ROOT / "private/second_conditioned_records.jsonl"
E50I_ROOT = (
    ROOT
    / "var/artifacts/e50i_third_conditioned_teacher_contingency_v1"
)
E50I_RESULT = E50I_ROOT / "result.json"
E50I_GENERATED = E50I_ROOT / "private/third_conditioned_records.jsonl"
E50J_ROOT = (
    ROOT
    / "var/artifacts/e50j_fourth_conditioned_teacher_contingency_v1"
)
E50J_RESULT = E50J_ROOT / "result.json"
E50J_GENERATED = E50J_ROOT / "private/fourth_conditioned_records.jsonl"
E50K_ROOT = (
    ROOT
    / "var/artifacts/e50k_fifth_conditioned_teacher_contingency_v1"
)
E50K_RESULT = E50K_ROOT / "result.json"
E50K_GENERATED = E50K_ROOT / "private/fifth_conditioned_records.jsonl"
ROUTE_CONFUSION_RESULT = (
    ROOT
    / "var/artifacts/e49t_route_confusion_calibration_v1/result.json"
)
DECLARATION_MISMATCH_RESULT = (
    ROOT
    / "var/artifacts/e49t_declaration_mismatch_calibration_v1/result.json"
)
OPEN_RELATION_FAILURES = (
    (
        ROOT
        / "var/artifacts/"
        "e50f2_frozen_pairwise_relation_calibration_v1/result.json",
        "e50f2_frozen_pairwise_relation_calibration_v1",
    ),
    (
        ROOT
        / "var/artifacts/"
        "e50f4_pairwise_finite_family_calibration_v1/result.json",
        "e50f4_pairwise_finite_family_calibration_v1",
    ),
)
SCHEMA = "e50g_safe_signature_teacher_route_calibration_v1"
FORCED_SAMPLES = 16
UNFORCED_SAMPLES = 64
SEED = 500811


def _load_module(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module("e50g_conditioned_base", BASE_PATH)
E50H = _load_module("e50g_second_conditioned", E50H_PATH)
E50I = _load_module("e50g_third_conditioned", E50I_PATH)
E50J = _load_module("e50g_fourth_conditioned", E50J_PATH)
E50K = _load_module("e50g_fifth_conditioned", E50K_PATH)
SIGNATURES = _load_module("e50g_safe_signatures", SIGNATURE_PATH)
E50C = BASE.BASE


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_activation() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if (
        not E50C_RESULT.is_file()
        or not E50F_RESULT.is_file()
        or not E50H_RESULT.is_file()
        or not E50I_RESULT.is_file()
        or not E50J_RESULT.is_file()
        or not E50K_RESULT.is_file()
    ):
        raise RuntimeError(
            "E50G requires terminal E50C/F/H/I/J/K results"
        )
    e50c = json.loads(E50C_RESULT.read_text(encoding="utf-8"))
    e50f = json.loads(E50F_RESULT.read_text(encoding="utf-8"))
    e50h = json.loads(E50H_RESULT.read_text(encoding="utf-8"))
    e50i = json.loads(E50I_RESULT.read_text(encoding="utf-8"))
    e50j = json.loads(E50J_RESULT.read_text(encoding="utf-8"))
    e50k = json.loads(E50K_RESULT.read_text(encoding="utf-8"))
    if e50c.get("schema") != "e50c_72b_teacher_route_calibration_v1":
        raise RuntimeError("E50G E50C schema mismatch")
    e50c_identity = e50c.get("identity") or {}
    expected_relation_failures = {
        schema: _sha256(path)
        for path, schema in OPEN_RELATION_FAILURES
    }
    if (
        e50c.get("pass") is not False
        or e50c.get("failure")
        != "open_relation_judge_quarantined_before_route_discovery"
        or e50c.get("teacher_completed_problem_count") != 36
        or e50c.get("teacher_logged_sample_count") != 576
        or e50c.get("teacher_private_response_count") != 0
        or e50c.get("selected_source_indices") != []
        or e50c.get("wrong_route_success_count") != 0
        or e50c_identity.get("e50c_script_sha256")
        != _sha256(E50C.SCRIPT)
        or e50c_identity.get("quarantine_script_sha256")
        != _sha256(E50C_QUARANTINE_SCRIPT)
        or e50c_identity.get("quarantine_protocol_sha256")
        != _sha256(E50C_QUARANTINE_PROTOCOL)
        or e50c_identity.get("endpoint_record_sha256")
        != _sha256(E50C.ENDPOINT_RECORD)
        or e50c_identity.get("e47_manifest_sha256")
        != _sha256(E50C.E47_MANIFEST)
        or e50c_identity.get("e47_problems_sha256")
        != _sha256(E50C.E47_PROBLEMS)
        or e50c_identity.get("false_new_failure_sha256")
        != expected_relation_failures
    ):
        raise RuntimeError("E50G E50C quarantine contract mismatch")
    if e50f.get("schema") != "e50f_conditioned_teacher_route_calibration_v1":
        raise RuntimeError("E50G E50F schema mismatch")
    e50f_identity = e50f.get("identity") or {}
    if (
        e50f.get("pass") is not False
        or e50f.get("failure")
        != (
            "open_relation_judge_quarantined_after_frozen_"
            "conditioned_corpus_generation"
        )
        or e50f.get("generation_problem_count") != 50
        or e50f.get("generation_error_count") != 0
        or e50f.get("conditioned_execution_count") != 800
        or e50f.get("selected_source_indices") != []
        or e50f_identity.get("script_sha256") != _sha256(BASE_PATH)
        or e50f_identity.get("protocol_sha256")
        != _sha256(BASE.PROTOCOL)
        or e50f_identity.get("e50c_result_sha256")
        != _sha256(E50C_RESULT)
    ):
        raise RuntimeError("E50G E50F quarantined corpus contract mismatch")
    if not E50F_GENERATED.is_file():
        raise RuntimeError("E50G first conditioned-teacher corpus is missing")
    expected_generated = e50f.get("conditioned_teacher_records_sha256")
    if expected_generated != _sha256(E50F_GENERATED):
        raise RuntimeError(
            "E50G first conditioned-teacher corpus hash mismatch"
        )
    if e50h.get("schema") != "e50h_second_conditioned_teacher_corpus_v1":
        raise RuntimeError("E50G E50H schema mismatch")
    e50h_identity = e50h.get("identity") or {}
    if (
        e50h.get("pass") is not False
        or e50h.get("failure")
        != "corpus_only_second_attempt_no_training_authority"
        or e50h.get("generation_problem_count") != 50
        or e50h.get("generation_error_count") != 0
        or e50h.get("conditioned_execution_count") != 800
        or e50h.get("selected_source_indices") != []
        or e50h_identity.get("script_sha256") != _sha256(E50H_PATH)
        or e50h_identity.get("protocol_sha256")
        != _sha256(E50H.PROTOCOL)
        or e50h_identity.get("base_e50f_script_sha256")
        != _sha256(BASE_PATH)
        or e50h_identity.get("e50f_result_sha256")
        != _sha256(E50F_RESULT)
        or e50h_identity.get("e50f_generated_sha256")
        != _sha256(E50F_GENERATED)
        or e50h_identity.get("endpoint_record_sha256")
        != _sha256(E50C.ENDPOINT_RECORD)
        or e50h_identity.get("e47_manifest_sha256")
        != _sha256(E50C.E47_MANIFEST)
        or e50h_identity.get("e47_problems_sha256")
        != _sha256(E50C.E47_PROBLEMS)
        or e50h_identity.get("proposal_seed") != E50H.PROPOSAL_SEED
        or e50h_identity.get("execution_seed") != E50H.EXECUTION_SEED
    ):
        raise RuntimeError("E50G E50H corpus contract mismatch")
    if not E50H_GENERATED.is_file():
        raise RuntimeError("E50G second conditioned-teacher corpus is missing")
    if (
        e50h.get("second_conditioned_records_sha256")
        != _sha256(E50H_GENERATED)
    ):
        raise RuntimeError(
            "E50G second conditioned-teacher corpus hash mismatch"
        )
    if (
        e50i.get("schema")
        != "e50i_third_conditioned_teacher_contingency_v1"
    ):
        raise RuntimeError("E50G E50I schema mismatch")
    e50i_identity = e50i.get("identity") or {}
    attempted_orders = e50i.get("attempted_problem_orders")
    prethird_orders = e50i.get("prethird_eligible_problem_orders")
    if (
        e50i.get("pass") is not False
        or e50i.get("failure")
        != "corpus_only_third_attempt_no_training_authority"
        or e50i.get("trigger_threshold") != E50I.TRIGGER_THRESHOLD
        or not isinstance(attempted_orders, list)
        or not isinstance(prethird_orders, list)
        or attempted_orders
        != sorted(set(int(order) for order in attempted_orders))
        or prethird_orders
        != sorted(set(int(order) for order in prethird_orders))
        or not set(attempted_orders).isdisjoint(prethird_orders)
        or (
            len(prethird_orders) < E50I.TRIGGER_THRESHOLD
            and set(attempted_orders) | set(prethird_orders) != set(range(50))
        )
        or (
            len(prethird_orders) >= E50I.TRIGGER_THRESHOLD
            and attempted_orders != []
        )
        or e50i.get("prethird_eligible_problem_count")
        != len(prethird_orders)
        or e50i.get("triggered")
        is not (len(prethird_orders) < E50I.TRIGGER_THRESHOLD)
        or e50i.get("generation_problem_count") != len(attempted_orders)
        or e50i.get("generation_error_count") != 0
        or e50i.get("conditioned_execution_count")
        != len(attempted_orders) * 16
        or e50i.get("selected_source_indices") != []
        or e50i_identity.get("script_sha256") != _sha256(E50I_PATH)
        or e50i_identity.get("protocol_sha256")
        != _sha256(E50I.PROTOCOL)
        or e50i_identity.get("e50h_script_sha256")
        != _sha256(E50H_PATH)
        or e50i_identity.get("signature_source_sha256")
        != _sha256(SIGNATURE_PATH)
        or e50i_identity.get("e50f_result_sha256")
        != _sha256(E50F_RESULT)
        or e50i_identity.get("e50f_generated_sha256")
        != _sha256(E50F_GENERATED)
        or e50i_identity.get("e50h_result_sha256")
        != _sha256(E50H_RESULT)
        or e50i_identity.get("e50h_generated_sha256")
        != _sha256(E50H_GENERATED)
        or e50i_identity.get("endpoint_record_sha256")
        != _sha256(E50C.ENDPOINT_RECORD)
        or e50i_identity.get("e47_manifest_sha256")
        != _sha256(E50C.E47_MANIFEST)
        or e50i_identity.get("e47_problems_sha256")
        != _sha256(E50C.E47_PROBLEMS)
        or e50i_identity.get("prethird_eligible_problem_orders")
        != prethird_orders
        or e50i_identity.get("trigger_threshold")
        != E50I.TRIGGER_THRESHOLD
        or e50i_identity.get("proposal_seed") != E50I.PROPOSAL_SEED
        or e50i_identity.get("execution_seed") != E50I.EXECUTION_SEED
    ):
        raise RuntimeError("E50G E50I contingency contract mismatch")
    if not E50I_GENERATED.is_file():
        raise RuntimeError("E50G third conditioned-teacher corpus is missing")
    if (
        e50i.get("third_conditioned_records_sha256")
        != _sha256(E50I_GENERATED)
    ):
        raise RuntimeError(
            "E50G third conditioned-teacher corpus hash mismatch"
        )
    if (
        e50j.get("schema")
        != "e50j_fourth_conditioned_teacher_contingency_v1"
    ):
        raise RuntimeError("E50G E50J schema mismatch")
    e50j_identity = e50j.get("identity") or {}
    attempted_j = e50j.get("attempted_problem_orders")
    prefourth_orders = e50j.get("prefourth_eligible_problem_orders")
    if (
        e50j.get("pass") is not False
        or e50j.get("failure")
        != "corpus_only_fourth_attempt_no_training_authority"
        or e50j.get("trigger_threshold") != E50J.TRIGGER_THRESHOLD
        or not isinstance(attempted_j, list)
        or not isinstance(prefourth_orders, list)
        or attempted_j != sorted(set(int(order) for order in attempted_j))
        or prefourth_orders
        != sorted(set(int(order) for order in prefourth_orders))
        or not set(attempted_j).isdisjoint(prefourth_orders)
        or (
            len(prefourth_orders) < E50J.TRIGGER_THRESHOLD
            and set(attempted_j) | set(prefourth_orders) != set(range(50))
        )
        or (
            len(prefourth_orders) >= E50J.TRIGGER_THRESHOLD
            and attempted_j != []
        )
        or e50j.get("prefourth_eligible_problem_count")
        != len(prefourth_orders)
        or e50j.get("triggered")
        is not (len(prefourth_orders) < E50J.TRIGGER_THRESHOLD)
        or e50j.get("generation_problem_count") != len(attempted_j)
        or e50j.get("generation_error_count") != 0
        or e50j.get("conditioned_execution_count")
        != len(attempted_j) * 16
        or e50j.get("selected_source_indices") != []
        or e50j_identity.get("script_sha256") != _sha256(E50J_PATH)
        or e50j_identity.get("protocol_sha256")
        != _sha256(E50J.PROTOCOL)
        or e50j_identity.get("e50i_script_sha256")
        != _sha256(E50I_PATH)
        or e50j_identity.get("signature_source_sha256")
        != _sha256(SIGNATURE_PATH)
        or e50j_identity.get("e50f_result_sha256")
        != _sha256(E50F_RESULT)
        or e50j_identity.get("e50f_generated_sha256")
        != _sha256(E50F_GENERATED)
        or e50j_identity.get("e50h_result_sha256")
        != _sha256(E50H_RESULT)
        or e50j_identity.get("e50h_generated_sha256")
        != _sha256(E50H_GENERATED)
        or e50j_identity.get("e50i_result_sha256")
        != _sha256(E50I_RESULT)
        or e50j_identity.get("e50i_generated_sha256")
        != _sha256(E50I_GENERATED)
        or e50j_identity.get("endpoint_record_sha256")
        != _sha256(E50C.ENDPOINT_RECORD)
        or e50j_identity.get("e47_manifest_sha256")
        != _sha256(E50C.E47_MANIFEST)
        or e50j_identity.get("e47_problems_sha256")
        != _sha256(E50C.E47_PROBLEMS)
        or e50j_identity.get("prefourth_eligible_problem_orders")
        != prefourth_orders
        or e50j_identity.get("trigger_threshold")
        != E50J.TRIGGER_THRESHOLD
        or e50j_identity.get("proposal_seed") != E50J.PROPOSAL_SEED
        or e50j_identity.get("execution_seed") != E50J.EXECUTION_SEED
    ):
        raise RuntimeError("E50G E50J contingency contract mismatch")
    if not E50J_GENERATED.is_file():
        raise RuntimeError("E50G fourth conditioned-teacher corpus is missing")
    if (
        e50j.get("fourth_conditioned_records_sha256")
        != _sha256(E50J_GENERATED)
    ):
        raise RuntimeError(
            "E50G fourth conditioned-teacher corpus hash mismatch"
        )
    if (
        e50k.get("schema")
        != "e50k_fifth_conditioned_teacher_contingency_v1"
    ):
        raise RuntimeError("E50G E50K schema mismatch")
    e50k_identity = e50k.get("identity") or {}
    attempted_k = e50k.get("attempted_problem_orders")
    prefifth_orders = e50k.get("prefifth_eligible_problem_orders")
    if (
        e50k.get("pass") is not False
        or e50k.get("failure")
        != "corpus_only_fifth_attempt_no_training_authority"
        or e50k.get("trigger_threshold") != E50K.TRIGGER_THRESHOLD
        or not isinstance(attempted_k, list)
        or not isinstance(prefifth_orders, list)
        or attempted_k != sorted(set(int(order) for order in attempted_k))
        or prefifth_orders
        != sorted(set(int(order) for order in prefifth_orders))
        or not set(attempted_k).isdisjoint(prefifth_orders)
        or (
            len(prefifth_orders) < E50K.TRIGGER_THRESHOLD
            and set(attempted_k) | set(prefifth_orders) != set(range(50))
        )
        or (
            len(prefifth_orders) >= E50K.TRIGGER_THRESHOLD
            and attempted_k != []
        )
        or e50k.get("prefifth_eligible_problem_count")
        != len(prefifth_orders)
        or e50k.get("triggered")
        is not (len(prefifth_orders) < E50K.TRIGGER_THRESHOLD)
        or e50k.get("generation_problem_count") != len(attempted_k)
        or e50k.get("generation_error_count") != 0
        or e50k.get("conditioned_execution_count")
        != len(attempted_k) * 16
        or e50k.get("selected_source_indices") != []
        or e50k_identity.get("script_sha256") != _sha256(E50K_PATH)
        or e50k_identity.get("protocol_sha256")
        != _sha256(E50K.PROTOCOL)
        or e50k_identity.get("e50j_script_sha256")
        != _sha256(E50J_PATH)
        or e50k_identity.get("signature_source_sha256")
        != _sha256(SIGNATURE_PATH)
        or e50k_identity.get("e50f_result_sha256")
        != _sha256(E50F_RESULT)
        or e50k_identity.get("e50f_generated_sha256")
        != _sha256(E50F_GENERATED)
        or e50k_identity.get("e50h_result_sha256")
        != _sha256(E50H_RESULT)
        or e50k_identity.get("e50h_generated_sha256")
        != _sha256(E50H_GENERATED)
        or e50k_identity.get("e50i_result_sha256")
        != _sha256(E50I_RESULT)
        or e50k_identity.get("e50i_generated_sha256")
        != _sha256(E50I_GENERATED)
        or e50k_identity.get("e50j_result_sha256")
        != _sha256(E50J_RESULT)
        or e50k_identity.get("e50j_generated_sha256")
        != _sha256(E50J_GENERATED)
        or e50k_identity.get("endpoint_record_sha256")
        != _sha256(E50C.ENDPOINT_RECORD)
        or e50k_identity.get("e47_manifest_sha256")
        != _sha256(E50C.E47_MANIFEST)
        or e50k_identity.get("e47_problems_sha256")
        != _sha256(E50C.E47_PROBLEMS)
        or e50k_identity.get("prefifth_eligible_problem_orders")
        != prefifth_orders
        or e50k_identity.get("trigger_threshold")
        != E50K.TRIGGER_THRESHOLD
        or e50k_identity.get("proposal_seed") != E50K.PROPOSAL_SEED
        or e50k_identity.get("execution_seed") != E50K.EXECUTION_SEED
    ):
        raise RuntimeError("E50G E50K contingency contract mismatch")
    if not E50K_GENERATED.is_file():
        raise RuntimeError("E50G fifth conditioned-teacher corpus is missing")
    if (
        e50k.get("fifth_conditioned_records_sha256")
        != _sha256(E50K_GENERATED)
    ):
        raise RuntimeError(
            "E50G fifth conditioned-teacher corpus hash mismatch"
        )

    development = json.loads(DEVELOPMENT_AUDIT.read_text(encoding="utf-8"))
    if (
        development.get("schema")
        != "safe_math_strategy_signature_development_audit_v1"
        or development.get("counts", {}).get("false_new_count") != 0
        or development.get("counts", {}).get("false_merge_count") != 1
        or development.get("counts", {}).get("sound_pair_count") != 18
    ):
        raise RuntimeError("E50G signature development evidence drifted")
    rewrite = json.loads(FULL_REWRITE_AUDIT.read_text(encoding="utf-8"))
    if (
        rewrite.get("schema") != "e50g_full_injection_rewrite_audit_v1"
        or rewrite.get("pass") is not True
        or rewrite.get("counts", {}).get("problem_count") != 50
        or rewrite.get("counts", {}).get("comparison_count") != 150
        or rewrite.get("counts", {}).get("false_new_count") != 0
        or rewrite.get("counts", {}).get("signature_change_count") != 0
        or rewrite.get("identity", {}).get("signature_source")
        != _sha256(SIGNATURE_PATH)
    ):
        raise RuntimeError("E50G full rewrite false-new evidence drifted")
    for path, schema in OPEN_RELATION_FAILURES:
        relation_result = json.loads(path.read_text(encoding="utf-8"))
        checks = relation_result.get("checks") or {}
        if (
            relation_result.get("schema") != schema
            or relation_result.get("pass") is not False
            or (
                checks.get("zero_false_new")
                if "zero_false_new" in checks
                else checks.get("zero_false_new_each_pass")
            )
            is not False
        ):
            raise RuntimeError(
                f"E50G requires frozen false-new quarantine evidence: {path}"
            )
    for path, schema in (
        (
            ROUTE_CONFUSION_RESULT,
            "e49t_route_confusion_calibration_result_v1",
        ),
        (
            DECLARATION_MISMATCH_RESULT,
            "e49t_declaration_mismatch_result_v1",
        ),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != schema or payload.get("pass") is not True:
            raise RuntimeError(f"E50G requires passing runtime control: {path}")
    return e50c, e50f, e50h, e50i, e50j, e50k


def _method_route(method: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": str(method.get("label") or ""),
        "actions": [
            str(action) for action in (method.get("actions") or ())
        ],
    }


def _generated_record_contract_failure(
    *,
    generated: dict[str, Any],
    problem: dict[str, Any],
    judge_model: str,
    attempt_index: int,
    first_generated: dict[str, Any] | None = None,
    second_generated: dict[str, Any] | None = None,
    third_generated: dict[str, Any] | None = None,
    fourth_generated: dict[str, Any] | None = None,
) -> str | None:
    """Verify that one conditioned record came from the frozen requests."""

    problem_order = int(problem["problem_order"])
    if (
        int(generated.get("problem_order", -1)) != problem_order
        or int(generated.get("source_index", -1))
        != int(problem["source_index"])
        or str(generated.get("row_id") or "")
        != str(problem["unique_id"])
    ):
        return "problem_identity_mismatch"
    observed_attempt = generated.get("attempt_index")
    if (
        (attempt_index == 0 and observed_attempt not in (None, 0))
        or (attempt_index == 1 and observed_attempt != 1)
        or (attempt_index == 2 and observed_attempt != 2)
        or (attempt_index == 3 and observed_attempt != 3)
        or (attempt_index == 4 and observed_attempt != 4)
    ):
        return "attempt_identity_mismatch"
    if generated.get("error"):
        return "generation_error"
    if (
        generated.get("proposal_finish_reason") != "stop"
        or not str(generated.get("proposal_response_id") or "")
        or generated.get("answer_was_not_in_proposal_payload") is not True
        or generated.get("answer_was_not_in_execution_payloads") is not True
    ):
        return "proposal_provenance_failure"
    if attempt_index == 0:
        expected_payload = BASE._proposal_payload(
            judge_model, str(problem["problem"]), problem_order
        )
    elif attempt_index == 1:
        first_proposal = (
            first_generated.get("proposal")
            if isinstance(first_generated, dict)
            else None
        )
        first_methods = (
            first_proposal.get("methods")
            if isinstance(first_proposal, dict)
            else None
        )
        if not isinstance(first_methods, list) or len(first_methods) != 2:
            return "missing_first_attempt_for_exclusion"
        excluded_labels = (
            str(first_methods[0].get("label") or ""),
            str(first_methods[1].get("label") or ""),
        )
        if (
            not all(excluded_labels)
            or generated.get("excluded_first_pair_labels")
            != list(excluded_labels)
        ):
            return "excluded_first_pair_mismatch"
        expected_payload = E50H._proposal_payload(
            judge_model,
            str(problem["problem"]),
            problem_order,
            excluded_labels,
        )
        excluded_pairs = (excluded_labels,)
    elif attempt_index == 2:
        first_proposal = (
            first_generated.get("proposal")
            if isinstance(first_generated, dict)
            else None
        )
        second_proposal = (
            second_generated.get("proposal")
            if isinstance(second_generated, dict)
            else None
        )
        first_methods = (
            first_proposal.get("methods")
            if isinstance(first_proposal, dict)
            else None
        )
        second_methods = (
            second_proposal.get("methods")
            if isinstance(second_proposal, dict)
            else None
        )
        if (
            not isinstance(first_methods, list)
            or len(first_methods) != 2
            or not isinstance(second_methods, list)
            or len(second_methods) != 2
        ):
            return "missing_prior_attempts_for_exclusion"
        excluded_pairs = (
            (
                str(first_methods[0].get("label") or ""),
                str(first_methods[1].get("label") or ""),
            ),
            (
                str(second_methods[0].get("label") or ""),
                str(second_methods[1].get("label") or ""),
            ),
        )
        if (
            not all(label for pair in excluded_pairs for label in pair)
            or generated.get("excluded_first_two_pair_labels")
            != [list(pair) for pair in excluded_pairs]
        ):
            return "excluded_first_two_pairs_mismatch"
        expected_payload = E50I._proposal_payload(
            judge_model,
            str(problem["problem"]),
            problem_order,
            excluded_pairs,
        )
    elif attempt_index == 3:
        prior_generated = [
            record
            for record in (
                first_generated,
                second_generated,
                third_generated,
            )
            if isinstance(record, dict)
        ]
        prior_methods = [
            (record.get("proposal") or {}).get("methods")
            for record in prior_generated
        ]
        if (
            len(prior_methods) not in (2, 3)
            or any(
                not isinstance(methods, list) or len(methods) != 2
                for methods in prior_methods
            )
        ):
            return "missing_prior_attempts_for_fourth_exclusion"
        excluded_pairs = tuple(
            (
                str(methods[0].get("label") or ""),
                str(methods[1].get("label") or ""),
            )
            for methods in prior_methods
        )
        if (
            not all(label for pair in excluded_pairs for label in pair)
            or generated.get("excluded_prior_pair_labels")
            != [list(pair) for pair in excluded_pairs]
        ):
            return "excluded_prior_pairs_mismatch"
        expected_payload = E50J._proposal_payload(
            judge_model,
            str(problem["problem"]),
            problem_order,
            excluded_pairs,
        )
    elif attempt_index == 4:
        prior_generated = [
            record
            for record in (
                first_generated,
                second_generated,
                third_generated,
                fourth_generated,
            )
            if isinstance(record, dict)
        ]
        prior_methods = [
            (record.get("proposal") or {}).get("methods")
            for record in prior_generated
        ]
        if (
            len(prior_methods) not in (3, 4)
            or any(
                not isinstance(methods, list) or len(methods) != 2
                for methods in prior_methods
            )
        ):
            return "missing_prior_attempts_for_fifth_exclusion"
        excluded_pairs = tuple(
            (
                str(methods[0].get("label") or ""),
                str(methods[1].get("label") or ""),
            )
            for methods in prior_methods
        )
        if (
            not all(label for pair in excluded_pairs for label in pair)
            or generated.get("excluded_prior_pair_labels")
            != [list(pair) for pair in excluded_pairs]
        ):
            return "excluded_prior_pairs_mismatch"
        expected_payload = E50K._proposal_payload(
            judge_model,
            str(problem["problem"]),
            problem_order,
            excluded_pairs,
        )
    else:
        return "unsupported_attempt_index"
    expected_request_sha256 = _sha256_text(
        json.dumps(
            expected_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if generated.get("proposal_request_sha256") != expected_request_sha256:
        return "proposal_request_hash_mismatch"

    proposal = generated.get("proposal")
    methods = proposal.get("methods") if isinstance(proposal, dict) else None
    if (
        not isinstance(methods, list)
        or len(methods) != 2
        or any(
            not isinstance(method, dict)
            or not str(method.get("label") or "").strip()
            or not isinstance(method.get("actions"), list)
            or not 2 <= len(method["actions"]) <= 6
            or any(
                not str(action).strip() for action in method["actions"]
            )
            for method in methods
        )
    ):
        return "proposal_schema_failure"
    if attempt_index in (1, 2, 3, 4):
        normalized_observed_labels = {
            " ".join(str(method["label"]).lower().split())
            for method in methods
        }
        if attempt_index == 1:
            excluded_pairs = (
                tuple(generated["excluded_first_pair_labels"]),
            )
        elif attempt_index == 2:
            excluded_pairs = tuple(
                tuple(pair)
                for pair in generated["excluded_first_two_pair_labels"]
            )
        else:
            excluded_pairs = tuple(
                tuple(pair)
                for pair in generated["excluded_prior_pair_labels"]
            )
        repeats_excluded = any(
            normalized_observed_labels
            == {
                " ".join(str(label).lower().split())
                for label in pair
            }
            for pair in excluded_pairs
        )
        if (
            generated.get("repeated_excluded_label_pair")
            is not repeats_excluded
        ):
            return "excluded_pair_flag_mismatch"
    executions = generated.get("executions")
    if not isinstance(executions, list) or len(executions) != 16:
        return "execution_count_mismatch"
    for route_index in (0, 1):
        route_rows = [
            row
            for row in executions
            if int(row.get("route_index", -1)) == route_index
        ]
        if (
            len(route_rows) != 8
            or {int(row.get("choice_index", -1)) for row in route_rows}
            != set(range(8))
            or len(
                {
                    str(row.get("response_id") or "")
                    for row in route_rows
                }
            )
            != 1
            or not str(route_rows[0].get("response_id") or "")
        ):
            return f"route_{route_index + 1}_batch_provenance_failure"
    route_response_ids = {
        str(row["response_id"])
        for row in executions
        if int(row["choice_index"]) == 0
    }
    if len(route_response_ids) != 2:
        return "execution_batches_not_independent"
    if any(
        _sha256_text(str(row.get("response") or ""))
        != str(row.get("response_sha256") or "")
        for row in executions
    ):
        return "execution_response_hash_mismatch"
    return None


def _select_one_candidate_per_problem(
    candidate_pool: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the frozen E50H attempt-ranking rule without 0.5B outcomes."""

    by_problem: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidate_pool:
        by_problem.setdefault(int(candidate["problem_order"]), []).append(
            candidate
        )
    selected = []
    for problem_order in sorted(by_problem):
        ranked = sorted(
            by_problem[problem_order],
            key=lambda row: (
                -int(row["minimum_teacher_execution_count"]),
                -int(row["combined_teacher_execution_count"]),
                int(row["attempt_index"]),
            ),
        )
        winner = {
            **ranked[0],
            "eligible_attempt_count": len(ranked),
            "eligible_attempt_indices": sorted(
                int(row["attempt_index"]) for row in ranked
            ),
        }
        selected.append(winner)
    return selected


def _sound_and_bound_audit(
    proposal: dict[str, Any],
    audit: dict[str, Any],
) -> bool:
    """Apply the frozen menu audit except its uncalibrated relation label."""

    if (
        audit.get("finish_reason") != "stop"
        or not str(audit.get("response_id") or "")
    ):
        return False
    assessment = audit.get("assessment")
    if not isinstance(assessment, dict):
        return False
    strategies = assessment.get("strategy_assessments")
    if not isinstance(strategies, list) or {
        row.get("strategy_id") for row in strategies if isinstance(row, dict)
    } != {"S1", "S2"}:
        return False
    if any(
        row.get("status") != "sound"
        or row.get("failure_code") != "none"
        or row.get("matches_reference_answer") is not True
        or row.get("all_actions_sufficient") is not True
        or row.get("missing_decisive_step") is not False
        or row.get("menu_reveals_final_answer") is not False
        or row.get("actions_concrete_for_small_model") is not True
        or not str(row.get("derived_answer") or "").strip()
        for row in strategies
    ):
        return False
    sources = assessment.get("source_assessments")
    if not isinstance(sources, list):
        return False
    actual = {
        str(row.get("strategy_id")): row.get("binding_status")
        for row in sources
        if isinstance(row, dict)
    }
    expected = {
        str(row["strategy_id"]): (
            "observed_bound"
            if row["source_kind"] == "observed"
            else "proposed_not_applicable"
        )
        for row in proposal["route_sources"]
    }
    if actual != expected:
        return False
    pair = assessment.get("pair")
    return bool(
        isinstance(pair, dict)
        and str(pair.get("s1_decisive_operation") or "").strip()
        and str(pair.get("s2_decisive_operation") or "").strip()
    )


def _identity(
    e50c: dict[str, Any],
    e50f: dict[str, Any],
    e50h: dict[str, Any],
    e50i: dict[str, Any],
    e50j: dict[str, Any],
    e50k: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_sha256": _sha256(PROTOCOL),
        "runtime_control_schema_repair_sha256": _sha256(
            RUNTIME_CONTROL_SCHEMA_REPAIR
        ),
        "script_sha256": _sha256(SCRIPT),
        "base_e50f_script_sha256": _sha256(BASE_PATH),
        "signature_source_sha256": _sha256(SIGNATURE_PATH),
        "signature_development_audit_sha256": _sha256(DEVELOPMENT_AUDIT),
        "full_rewrite_audit_sha256": _sha256(FULL_REWRITE_AUDIT),
        "e50c_result_sha256": _sha256(E50C_RESULT),
        "e50c_result_schema": e50c["schema"],
        "e50c_quarantine_script_sha256": _sha256(
            E50C_QUARANTINE_SCRIPT
        ),
        "e50c_quarantine_protocol_sha256": _sha256(
            E50C_QUARANTINE_PROTOCOL
        ),
        "e50f_result_sha256": _sha256(E50F_RESULT),
        "e50f_result_schema": e50f["schema"],
        "conditioned_teacher_records_sha256": _sha256(E50F_GENERATED),
        "e50h_script_sha256": _sha256(E50H_PATH),
        "e50h_protocol_sha256": _sha256(E50H.PROTOCOL),
        "e50h_result_sha256": _sha256(E50H_RESULT),
        "e50h_result_schema": e50h["schema"],
        "second_conditioned_records_sha256": _sha256(E50H_GENERATED),
        "e50i_script_sha256": _sha256(E50I_PATH),
        "e50i_protocol_sha256": _sha256(E50I.PROTOCOL),
        "e50i_result_sha256": _sha256(E50I_RESULT),
        "e50i_result_schema": e50i["schema"],
        "third_conditioned_records_sha256": _sha256(E50I_GENERATED),
        "e50j_script_sha256": _sha256(E50J_PATH),
        "e50j_protocol_sha256": _sha256(E50J.PROTOCOL),
        "e50j_result_sha256": _sha256(E50J_RESULT),
        "e50j_result_schema": e50j["schema"],
        "fourth_conditioned_records_sha256": _sha256(E50J_GENERATED),
        "e50k_script_sha256": _sha256(E50K_PATH),
        "e50k_protocol_sha256": _sha256(E50K.PROTOCOL),
        "e50k_result_sha256": _sha256(E50K_RESULT),
        "e50k_result_schema": e50k["schema"],
        "fifth_conditioned_records_sha256": _sha256(E50K_GENERATED),
        "e47_manifest_sha256": _sha256(E50C.E47_MANIFEST),
        "e47_problems_sha256": _sha256(E50C.E47_PROBLEMS),
        "endpoint_record_sha256": _sha256(E50C.ENDPOINT_RECORD),
        "e49t_identity_sha256": _sha256(E50C.E49T_IDENTITY),
        "route_confusion_result_sha256": _sha256(ROUTE_CONFUSION_RESULT),
        "declaration_mismatch_result_sha256": _sha256(
            DECLARATION_MISMATCH_RESULT
        ),
        "open_relation_failure_sha256": {
            schema: _sha256(path)
            for path, schema in OPEN_RELATION_FAILURES
        },
        "forced_samples": FORCED_SAMPLES,
        "unforced_samples": UNFORCED_SAMPLES,
        "seed": SEED,
    }


def _failure(
    *,
    result_path: pathlib.Path,
    reason: str,
    e50c: dict[str, Any],
    e50f: dict[str, Any],
    e50h: dict[str, Any],
    e50i: dict[str, Any],
    e50j: dict[str, Any],
    e50k: dict[str, Any],
    generated_contract_path: pathlib.Path,
    candidate_path: pathlib.Path,
    menu_path: pathlib.Path,
    candidates: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
) -> None:
    _write_json(
        result_path,
        {
            "schema": SCHEMA,
            "pass": False,
            "failure": reason,
            "signature_candidate_count": len(candidates),
            "sound_bound_signature_menu_count": len(accepted),
            "bidirectionally_executable_count": 0,
            "selected_source_indices": [],
            "wrong_route_success_count": 0,
            "cross_rendering_false_new_count": sum(
                int(row.get("cross_rendering_false_new_count", 0))
                for row in accepted
            ),
            "identity": _identity(
                e50c, e50f, e50h, e50i, e50j, e50k
            ),
            "conditioned_record_contract_sha256": _sha256(
                generated_contract_path
            ),
            "candidate_records_sha256": _sha256(candidate_path),
            "menu_records_sha256": _sha256(menu_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    output = args.output_root.resolve()
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError(f"fresh E50G result required: {result_path}")
    e50c, e50f, e50h, e50i, e50j, e50k = _verify_activation()

    manifest = json.loads(E50C.E47_MANIFEST.read_text(encoding="utf-8"))
    problems = E50C._load_jsonl(E50C.E47_PROBLEMS)
    expected_indices = manifest["selection"]["source_indices"]
    if (
        len(problems) != 50
        or [int(row["source_index"]) for row in problems] != expected_indices
        or any(int(row["level"]) != 5 for row in problems)
    ):
        raise RuntimeError("E50G frozen E47 problem cohort drifted")
    for problem_order, problem in enumerate(problems):
        problem["problem_order"] = problem_order

    from oat_drgrpo.math_grader import boxed_reward_fn
    from oat_drgrpo.templates import apply_qwen_math_template

    endpoint, judge_model = E50C._endpoint(E50C.ENDPOINT_RECORD)
    first_generated_rows = _read_jsonl(E50F_GENERATED)
    second_generated_rows = _read_jsonl(E50H_GENERATED)
    third_generated_rows = _read_jsonl(E50I_GENERATED)
    fourth_generated_rows = _read_jsonl(E50J_GENERATED)
    fifth_generated_rows = _read_jsonl(E50K_GENERATED)
    for label, rows in (
        ("first", first_generated_rows),
        ("second", second_generated_rows),
    ):
        if (
            len(rows) != 50
            or sorted(int(row["problem_order"]) for row in rows)
            != list(range(50))
        ):
            raise RuntimeError(
                f"E50G {label} conditioned-teacher corpus is incomplete"
            )
    first_by_order = {
        int(row["problem_order"]): row for row in first_generated_rows
    }
    second_by_order = {
        int(row["problem_order"]): row for row in second_generated_rows
    }
    third_by_order = {
        int(row["problem_order"]): row for row in third_generated_rows
    }
    fourth_by_order = {
        int(row["problem_order"]): row for row in fourth_generated_rows
    }
    attempted_third_orders = [
        int(order) for order in e50i["attempted_problem_orders"]
    ]
    if (
        len(third_generated_rows) != len(attempted_third_orders)
        or sorted(
            int(row["problem_order"]) for row in third_generated_rows
        )
        != attempted_third_orders
    ):
        raise RuntimeError(
            "E50G third conditioned-teacher corpus is incomplete"
        )
    attempted_fourth_orders = [
        int(order) for order in e50j["attempted_problem_orders"]
    ]
    if (
        len(fourth_generated_rows) != len(attempted_fourth_orders)
        or sorted(
            int(row["problem_order"]) for row in fourth_generated_rows
        )
        != attempted_fourth_orders
    ):
        raise RuntimeError(
            "E50G fourth conditioned-teacher corpus is incomplete"
        )
    attempted_fifth_orders = [
        int(order) for order in e50k["attempted_problem_orders"]
    ]
    if (
        len(fifth_generated_rows) != len(attempted_fifth_orders)
        or sorted(
            int(row["problem_order"]) for row in fifth_generated_rows
        )
        != attempted_fifth_orders
    ):
        raise RuntimeError(
            "E50G fifth conditioned-teacher corpus is incomplete"
        )
    generated_attempts = [
        (0, row) for row in first_generated_rows
    ] + [
        (1, row) for row in second_generated_rows
    ] + [
        (2, row) for row in third_generated_rows
    ] + [
        (3, row) for row in fourth_generated_rows
    ] + [
        (4, row) for row in fifth_generated_rows
    ]
    problem_by_order = {
        int(problem["problem_order"]): problem for problem in problems
    }
    candidate_pool = []
    generated_contract = []
    for attempt_index, generated in generated_attempts:
        problem_order = int(generated["problem_order"])
        problem = problem_by_order[problem_order]
        contract_failure = _generated_record_contract_failure(
            generated=generated,
            problem=problem,
            judge_model=judge_model,
            attempt_index=attempt_index,
            first_generated=(
                first_by_order[problem_order]
                if attempt_index in (1, 2, 3, 4)
                else None
            ),
            second_generated=(
                second_by_order[problem_order]
                if attempt_index in (2, 3, 4)
                else None
            ),
            third_generated=(
                third_by_order.get(problem_order)
                if attempt_index in (3, 4)
                else None
            ),
            fourth_generated=(
                fourth_by_order.get(problem_order)
                if attempt_index == 4
                else None
            ),
        )
        source_corpus = {
            0: "E50F",
            1: "E50H",
            2: "E50I",
            3: "E50J",
            4: "E50K",
        }[
            attempt_index
        ]
        contract_record = {
            "problem_order": problem_order,
            "source_index": int(problem["source_index"]),
            "attempt_index": attempt_index,
            "source_corpus": source_corpus,
            "source_record_sha256": _sha256_text(
                json.dumps(
                    generated,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "pass": contract_failure is None,
            "failure": contract_failure,
            "candidate_eligible": False,
            "candidate_eligibility_failure": None,
        }
        generated_contract.append(contract_record)
        if contract_failure is not None:
            continue
        proposal = generated.get("proposal")
        methods = proposal.get("methods") if isinstance(proposal, dict) else None
        if not isinstance(methods, list) or len(methods) != 2:
            contract_record["candidate_eligibility_failure"] = (
                "proposal_method_count"
            )
            continue
        proposal_routes = [_method_route(method) for method in methods]
        distinct, left_signature, right_signature = (
            SIGNATURES.safe_distinct_pair(*proposal_routes)
        )
        if not distinct:
            contract_record["candidate_eligibility_failure"] = (
                "unsafe_or_merged_signature_pair"
            )
            continue
        if attempt_index in (1, 2, 3, 4):
            if generated.get("repeated_excluded_label_pair") is True:
                contract_record["candidate_eligibility_failure"] = (
                    "repeated_excluded_label_pair"
                )
                continue
            prior_records = [first_by_order[problem_order]]
            if attempt_index in (2, 3, 4):
                prior_records.append(second_by_order[problem_order])
            if attempt_index in (3, 4) and problem_order in third_by_order:
                prior_records.append(third_by_order[problem_order])
            if attempt_index == 4 and problem_order in fourth_by_order:
                prior_records.append(fourth_by_order[problem_order])
            repeated_prior_signature = False
            for prior_record in prior_records:
                prior_proposal = prior_record.get("proposal")
                prior_methods = (
                    prior_proposal.get("methods")
                    if isinstance(prior_proposal, dict)
                    else None
                )
                if not (
                    isinstance(prior_methods, list)
                    and len(prior_methods) == 2
                ):
                    continue
                prior_pair = SIGNATURES.safe_distinct_pair(
                    *[
                        _method_route(method)
                        for method in prior_methods
                    ]
                )
                if (
                    prior_pair[0]
                    and {left_signature, right_signature}
                    == {prior_pair[1], prior_pair[2]}
                ):
                    repeated_prior_signature = True
                    break
            if repeated_prior_signature:
                contract_record["candidate_eligibility_failure"] = (
                    "repeated_prior_signature_pair"
                )
                continue

        route_members: list[list[dict[str, Any]]] = [[], []]
        for execution in generated.get("executions") or ():
            text = str(execution.get("response") or "")
            response_sha256 = str(execution.get("response_sha256") or "")
            if (
                _sha256_text(text) != response_sha256
                or len(text) > 4000
                or int(execution.get("route_index", -1)) not in (0, 1)
                or execution.get("finish_reason") != "stop"
            ):
                continue
            _, reward = boxed_reward_fn(
                text, str(problem["answer"]), fast=False
            )
            if float(reward) <= 0:
                continue
            route_index = int(execution["route_index"])
            route_members[route_index].append(
                {
                    "sample_id": (
                        f"a{attempt_index}r{route_index + 1}s"
                        f"{int(execution['choice_index']):02d}_"
                        f"{response_sha256[:12]}"
                    ),
                    "sample_index": int(execution["choice_index"]),
                    "text": text,
                    "response_sha256": response_sha256,
                }
            )
        for members in route_members:
            members.sort(key=lambda row: row["response_sha256"])
        if any(len(members) < 2 for members in route_members):
            contract_record["candidate_eligibility_failure"] = (
                "insufficient_exact_teacher_executions"
            )
            continue

        contract_record["candidate_eligible"] = True
        source_index = int(problem["source_index"])
        selected_members = [members[:3] for members in route_members]
        clusters = [
            {
                "cluster_id": f"C{route_index + 1}",
                "strategy": (
                    str(methods[route_index]["label"])
                    + ": "
                    + " -> ".join(
                        str(action)
                        for action in methods[route_index]["actions"]
                    )
                ),
                "strategy_key": (
                    left_signature if route_index == 0 else right_signature
                ),
                "member_ids": [
                    str(member["sample_id"])
                    for member in selected_members[route_index]
                ],
            }
            for route_index in range(2)
        ]
        exemplars = [
            member for members in route_members for member in members
        ]
        candidate_pool.append(
            {
                "problem_order": problem_order,
                "source_index": source_index,
                "attempt_index": attempt_index,
                "source_corpus": source_corpus,
                "source_record_sha256": contract_record[
                    "source_record_sha256"
                ],
                "row_id": str(problem["unique_id"]),
                "problem": str(problem["problem"]),
                "answer": str(problem["answer"]),
                "subject": str(problem["subject"]),
                "level": int(problem["level"]),
                "proposal": proposal,
                "proposal_signatures": [left_signature, right_signature],
                "route_positive_counts": [
                    len(route_members[0]),
                    len(route_members[1]),
                ],
                "minimum_teacher_execution_count": min(
                    len(route_members[0]), len(route_members[1])
                ),
                "combined_teacher_execution_count": sum(
                    len(members) for members in route_members
                ),
                "exemplars": exemplars,
                "cluster_record": {
                    "source_index": source_index,
                    "row_id": str(problem["unique_id"]),
                    "subject": str(problem["subject"]),
                    "level": int(problem["level"]),
                    "minimum_cluster_size": min(
                        len(members) for members in selected_members
                    ),
                    "combined_cluster_size": sum(
                        len(members) for members in selected_members
                    ),
                    "clusters": clusters,
                    "cluster_eligible": True,
                },
            }
        )
    observed_prethird_eligible = sorted(
        {
            int(candidate["problem_order"])
            for candidate in candidate_pool
            if int(candidate["attempt_index"]) in (0, 1)
        }
    )
    if (
        observed_prethird_eligible
        != e50i["prethird_eligible_problem_orders"]
    ):
        raise RuntimeError(
            "E50G recomputed E50F/E50H eligible union disagrees "
            "with the preregistered E50I trigger"
        )
    observed_prefourth_eligible = sorted(
        {
            int(candidate["problem_order"])
            for candidate in candidate_pool
            if int(candidate["attempt_index"]) in (0, 1, 2)
        }
    )
    if (
        observed_prefourth_eligible
        != e50j["prefourth_eligible_problem_orders"]
    ):
        raise RuntimeError(
            "E50G recomputed E50F/E50H/E50I eligible union disagrees "
            "with the preregistered E50J trigger"
        )
    observed_prefifth_eligible = sorted(
        {
            int(candidate["problem_order"])
            for candidate in candidate_pool
            if int(candidate["attempt_index"]) in (0, 1, 2, 3)
        }
    )
    if (
        observed_prefifth_eligible
        != e50k["prefifth_eligible_problem_orders"]
    ):
        raise RuntimeError(
            "E50G recomputed E50F/E50H/E50I/E50J eligible union "
            "disagrees with the preregistered E50K trigger"
        )
    generated_contract.sort(
        key=lambda row: (
            int(row["problem_order"]),
            int(row["attempt_index"]),
        )
    )
    candidates = _select_one_candidate_per_problem(candidate_pool)
    generated_contract_path = (
        output / "conditioned_record_contract.jsonl"
    )
    _write_jsonl(generated_contract_path, generated_contract)
    candidate_path = output / "candidate_records.jsonl"
    _write_jsonl(candidate_path, candidates)
    print(
        json.dumps(
            {
                "phase": "safe_signature_candidates",
                "count": len(candidates),
                "eligible_attempt_count": len(candidate_pool),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    menu_partial_path = output / "private/menu_records.partial.jsonl"
    menu_checkpoint_identity_path = (
        output / "private/menu_checkpoint_identity.json"
    )
    menu_checkpoint_identity = {
        **_identity(e50c, e50f, e50h, e50i, e50j, e50k),
        "candidate_records_sha256": _sha256(candidate_path),
        "conditioned_record_contract_sha256": _sha256(
            generated_contract_path
        ),
    }
    if menu_checkpoint_identity_path.is_file():
        observed_menu_identity = json.loads(
            menu_checkpoint_identity_path.read_text(encoding="utf-8")
        )
        if observed_menu_identity != menu_checkpoint_identity:
            raise RuntimeError("E50G menu checkpoint identity drifted")
    else:
        _write_json(
            menu_checkpoint_identity_path, menu_checkpoint_identity
        )
    menu_records = (
        _read_jsonl(menu_partial_path)
        if menu_partial_path.is_file()
        else []
    )
    completed_sources = [
        int(row["source_index"]) for row in menu_records
    ]
    candidate_sources = {
        int(row["source_index"]) for row in candidates
    }
    if (
        len(completed_sources) != len(set(completed_sources))
        or not set(completed_sources) <= candidate_sources
    ):
        raise RuntimeError("E50G partial menu checkpoint is invalid")
    remaining_candidates = [
        candidate
        for candidate in candidates
        if int(candidate["source_index"]) not in set(completed_sources)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                E50C._menu_candidate,
                endpoint=endpoint,
                model=judge_model,
                candidate=candidate,
                cluster_record=candidate["cluster_record"],
                timeout=args.timeout,
            )
            for candidate in remaining_candidates
        ]
        for future in as_completed(futures):
            menu_records.append(future.result())
            menu_records.sort(key=lambda row: int(row["source_index"]))
            _write_jsonl(menu_partial_path, menu_records)
    if {
        int(row["source_index"]) for row in menu_records
    } != candidate_sources:
        raise RuntimeError("E50G menu audit corpus is incomplete")
    candidate_by_source = {
        int(row["source_index"]): row for row in candidates
    }
    evaluated_menus = []
    for record in menu_records:
        source_index = int(record["source_index"])
        candidate = candidate_by_source[source_index]
        if record.get("error") or not isinstance(record.get("menu"), dict):
            evaluated_menus.append(record)
            continue
        proposal_view = {
            "menu": record["menu"],
            "route_sources": record["route_sources"],
        }
        audits = record.get("audits") or ()
        sound_bound = (
            len(audits) == 2
            and all(
                _sound_and_bound_audit(proposal_view, audit)
                for audit in audits
            )
        )
        final_distinct, final_left, final_right = (
            SIGNATURES.safe_menu_pair(record["menu"])
        )
        expected_left, expected_right = candidate["proposal_signatures"]
        final_routes = [
            SIGNATURES.strategy_route(record["menu"], "S1"),
            SIGNATURES.strategy_route(record["menu"], "S2"),
        ]
        proposal_routes = [
            _method_route(method)
            for method in candidate["proposal"]["methods"]
        ]
        cross_false_new = sum(
            int(SIGNATURES.safe_distinct_pair(left, right)[0])
            for left, right in zip(
                proposal_routes, final_routes, strict=True
            )
        )
        reverse_decision = SIGNATURES.safe_distinct_pair(
            final_routes[1], final_routes[0]
        )
        signature_pass = bool(
            final_distinct
            and [final_left, final_right]
            == [expected_left, expected_right]
            and cross_false_new == 0
            and reverse_decision[0] is True
            and [reverse_decision[1], reverse_decision[2]]
            == [final_right, final_left]
        )
        evaluated_menus.append(
            {
                **record,
                "old_relation_double_audit_pass": bool(
                    record.get("double_audit_pass")
                ),
                "sound_bound_double_audit_pass": sound_bound,
                "safe_signature_pass": signature_pass,
                "proposal_signatures": [expected_left, expected_right],
                "menu_signatures": [final_left, final_right],
                "attempt_index": candidate["attempt_index"],
                "source_corpus": candidate["source_corpus"],
                "source_record_sha256": candidate[
                    "source_record_sha256"
                ],
                "cross_rendering_false_new_count": cross_false_new,
                "route_positive_counts": candidate[
                    "route_positive_counts"
                ],
                "minimum_teacher_execution_count": candidate[
                    "minimum_teacher_execution_count"
                ],
                "combined_teacher_execution_count": candidate[
                    "combined_teacher_execution_count"
                ],
            }
        )
        print(
            json.dumps(
                {
                    "phase": "safe_signature_menu",
                    "source_index": source_index,
                    "attempt_index": candidate["attempt_index"],
                    "sound_bound": sound_bound,
                    "signature_pass": signature_pass,
                    "old_relation_pass": bool(
                        record.get("double_audit_pass")
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    evaluated_menus.sort(key=lambda row: int(row["source_index"]))
    menu_path = output / "menu_records.jsonl"
    _write_jsonl(menu_path, evaluated_menus)
    accepted = [
        row
        for row in evaluated_menus
        if row.get("sound_bound_double_audit_pass") is True
        and row.get("safe_signature_pass") is True
    ]
    accepted.sort(
        key=lambda row: (
            -int(row["minimum_teacher_execution_count"]),
            -int(row["combined_teacher_execution_count"]),
            int(row["source_index"]),
        )
    )
    if not accepted:
        _failure(
            result_path=result_path,
            reason="no_sound_bound_safe_signature_menu",
            e50c=e50c,
            e50f=e50f,
            e50h=e50h,
            e50i=e50i,
            e50j=e50j,
            e50k=e50k,
            generated_contract_path=generated_contract_path,
            candidate_path=candidate_path,
            menu_path=menu_path,
            candidates=candidates,
            accepted=accepted,
        )
        print(result_path)
        return

    import vllm

    forced_cases = []
    unforced_cases = []
    for record in accepted:
        candidate = candidate_by_source[int(record["source_index"])]
        menu = E50C._parse_menu(record["menu"])
        for strategy in menu.strategies:
            forced_cases.append(
                {
                    "source_index": int(record["source_index"]),
                    "row_id": str(record["row_id"]),
                    "problem": E50C._forced_problem(
                        candidate["problem"], menu, strategy
                    ),
                    "answer": candidate["answer"],
                    "menu": menu,
                    "strategy": strategy,
                }
            )
        unforced_cases.append(
            {
                "source_index": int(record["source_index"]),
                "row_id": str(record["row_id"]),
                "problem": E50C._neutral_problem(
                    candidate["problem"], menu
                ),
                "answer": candidate["answer"],
                "menu": menu,
            }
        )
    llm = vllm.LLM(
        model=str(E50C.MODEL),
        dtype="bfloat16",
        max_model_len=3072,
        gpu_memory_utilization=0.80,
        swap_space=8.0,
        enable_prefix_caching=True,
        seed=SEED,
    )
    forced_outputs = llm.generate(
        [apply_qwen_math_template(case["problem"]) for case in forced_cases],
        vllm.SamplingParams(
            n=FORCED_SAMPLES,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            seed=SEED + 1,
        ),
    )
    unforced_outputs = llm.generate(
        [apply_qwen_math_template(case["problem"]) for case in unforced_cases],
        vllm.SamplingParams(
            n=UNFORCED_SAMPLES,
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            seed=SEED + 2,
        ),
    )

    private_05b = []
    forced_tokens: list[list[int]] = []
    forced_prompts: list[str] = []
    forced_responses: list[str] = []
    forced_positive: list[bool] = []
    for case_index, (case, request_output) in enumerate(
        zip(forced_cases, forced_outputs, strict=True)
    ):
        if len(request_output.outputs) != FORCED_SAMPLES:
            raise RuntimeError("E50G forced sample count mismatch")
        for sample_index, sample in enumerate(request_output.outputs):
            text = str(sample.text)
            _, reward = boxed_reward_fn(text, case["answer"], fast=False)
            positive = float(reward) > 0.0
            forced_tokens.append(list(request_output.prompt_token_ids))
            forced_prompts.append(case["problem"])
            forced_responses.append(text)
            forced_positive.append(positive)
            private_05b.append(
                {
                    "kind": "forced",
                    "case_index": case_index,
                    "source_index": case["source_index"],
                    "row_id": case["row_id"],
                    "strategy_id": case["strategy"].strategy_id,
                    "sample_index": sample_index,
                    "answer_positive": positive,
                    "response_sha256": _sha256_text(text),
                    "response": text,
                }
            )
    unforced_tokens: list[list[int]] = []
    unforced_prompts: list[str] = []
    unforced_responses: list[str] = []
    unforced_positive: list[bool] = []
    for case_index, (case, request_output) in enumerate(
        zip(unforced_cases, unforced_outputs, strict=True)
    ):
        if len(request_output.outputs) != UNFORCED_SAMPLES:
            raise RuntimeError("E50G unforced sample count mismatch")
        for sample_index, sample in enumerate(request_output.outputs):
            text = str(sample.text)
            _, reward = boxed_reward_fn(text, case["answer"], fast=False)
            positive = float(reward) > 0.0
            unforced_tokens.append(list(request_output.prompt_token_ids))
            unforced_prompts.append(case["problem"])
            unforced_responses.append(text)
            unforced_positive.append(positive)
            private_05b.append(
                {
                    "kind": "unforced",
                    "case_index": case_index,
                    "source_index": case["source_index"],
                    "row_id": case["row_id"],
                    "sample_index": sample_index,
                    "answer_positive": positive,
                    "response_sha256": _sha256_text(text),
                    "response": text,
                }
            )
    private_05b_path = output / "private/base_05b_responses.jsonl"
    _write_jsonl(private_05b_path, private_05b)

    canonicalizer = E50C._load_frozen_canonicalizer()
    forced_keys, forced_diagnostics = E50C._canonical_keys(
        canonicalizer=canonicalizer,
        endpoint=endpoint,
        model=judge_model,
        prompt_tokens=forced_tokens,
        prompt_texts=forced_prompts,
        responses=forced_responses,
        positives=forced_positive,
        samples_per_prompt=FORCED_SAMPLES,
        timeout=args.timeout,
        workers=args.workers,
    )
    unforced_keys, unforced_diagnostics = E50C._canonical_keys(
        canonicalizer=canonicalizer,
        endpoint=endpoint,
        model=judge_model,
        prompt_tokens=unforced_tokens,
        prompt_texts=unforced_prompts,
        responses=unforced_responses,
        positives=unforced_positive,
        samples_per_prompt=UNFORCED_SAMPLES,
        timeout=args.timeout,
        workers=args.workers,
    )

    forced_results = []
    for case_index, case in enumerate(forced_cases):
        start = case_index * FORCED_SAMPLES
        stop = start + FORCED_SAMPLES
        probe = canonicalizer(endpoint=endpoint, model=judge_model)
        expected_key = probe._menu_strategy_key(
            case["menu"], case["strategy"].strategy_id
        )
        forced_results.append(
            {
                "source_index": case["source_index"],
                "strategy_id": case["strategy"].strategy_id,
                "answer_success_count": sum(forced_positive[start:stop]),
                "forced_route_success_count": sum(
                    key == expected_key for key in forced_keys[start:stop]
                ),
                "wrong_route_success_count": sum(
                    key is not None and key != expected_key
                    for key in forced_keys[start:stop]
                ),
            }
        )
    menu_by_source = {
        int(record["source_index"]): record for record in accepted
    }
    problems_out = []
    for case_index, case in enumerate(unforced_cases):
        start = case_index * UNFORCED_SAMPLES
        stop = start + UNFORCED_SAMPLES
        probe = canonicalizer(endpoint=endpoint, model=judge_model)
        route_counts = {
            strategy.strategy_id: sum(
                key
                == probe._menu_strategy_key(
                    case["menu"], strategy.strategy_id
                )
                for key in unforced_keys[start:stop]
            )
            for strategy in case["menu"].strategies
        }
        forced_for_problem = [
            row
            for row in forced_results
            if row["source_index"] == case["source_index"]
        ]
        forced_ok = (
            len(forced_for_problem) == 2
            and all(
                row["forced_route_success_count"] >= 1
                and row["wrong_route_success_count"] == 0
                for row in forced_for_problem
            )
        )
        unforced_ok = (
            min(route_counts.values()) >= 2
            and sum(route_counts.values()) >= 8
        )
        record = menu_by_source[case["source_index"]]
        candidate = candidate_by_source[case["source_index"]]
        problems_out.append(
            {
                "problem_order": candidate["problem_order"],
                "source_index": case["source_index"],
                "attempt_index": candidate["attempt_index"],
                "source_corpus": candidate["source_corpus"],
                "source_record_sha256": candidate[
                    "source_record_sha256"
                ],
                "row_id": case["row_id"],
                "menu": record["menu"],
                "menu_sha256": record["menu_sha256"],
                "route_sources": record["route_sources"],
                "proposal_signatures": record["proposal_signatures"],
                "menu_signatures": record["menu_signatures"],
                "minimum_cluster_size": record["minimum_cluster_size"],
                "combined_cluster_size": record["combined_cluster_size"],
                "route_positive_counts": record["route_positive_counts"],
                "minimum_teacher_execution_count": record[
                    "minimum_teacher_execution_count"
                ],
                "combined_teacher_execution_count": record[
                    "combined_teacher_execution_count"
                ],
                "forced_route_success_counts": {
                    row["strategy_id"]: row["forced_route_success_count"]
                    for row in forced_for_problem
                },
                "forced_wrong_route_counts": {
                    row["strategy_id"]: row["wrong_route_success_count"]
                    for row in forced_for_problem
                },
                "unforced_route_success_counts": route_counts,
                "unforced_accepted_count": sum(route_counts.values()),
                "bidirectionally_executable": forced_ok and unforced_ok,
            }
        )
    problems_out.sort(
        key=lambda row: (
            -min(row["unforced_route_success_counts"].values()),
            -int(row["unforced_accepted_count"]),
            -int(row["minimum_teacher_execution_count"]),
            int(row["problem_order"]),
        )
    )
    selected = [
        row["source_index"]
        for row in problems_out
        if row["bidirectionally_executable"]
    ][:10]
    wrong_route_count = sum(
        row["wrong_route_success_count"] for row in forced_results
    )
    cross_false_new = sum(
        int(row["cross_rendering_false_new_count"]) for row in accepted
    )
    checks = {
        "ten_naturally_supported_problems": len(selected) == 10,
        "zero_forced_wrong_route_accepts": wrong_route_count == 0,
        "zero_cross_rendering_false_new": cross_false_new == 0,
        "all_accepted_signatures_stable": all(
            row["proposal_signatures"] == row["menu_signatures"]
            and row["safe_signature_pass"] is True
            for row in accepted
        ),
    }
    payload = {
        "schema": SCHEMA,
        "pass": all(checks.values()),
        "checks": checks,
        "signature_candidate_count": len(candidates),
        "eligible_attempt_count": len(candidate_pool),
        "sound_bound_signature_menu_count": len(accepted),
        "bidirectionally_executable_count": sum(
            row["bidirectionally_executable"] for row in problems_out
        ),
        "selected_source_indices": selected,
        "wrong_route_success_count": wrong_route_count,
        "cross_rendering_false_new_count": cross_false_new,
        "forced_canonicalizer_diagnostics": forced_diagnostics,
        "unforced_canonicalizer_diagnostics": unforced_diagnostics,
        "identity": _identity(
            e50c, e50f, e50h, e50i, e50j, e50k
        ),
        "conditioned_record_contract_sha256": _sha256(
            generated_contract_path
        ),
        "candidate_records_sha256": _sha256(candidate_path),
        "menu_records_sha256": _sha256(menu_path),
        "base_05b_responses_sha256": _sha256(private_05b_path),
        "forced_cases": forced_results,
        "problems": problems_out,
    }
    _write_json(result_path, payload)
    print(result_path)


if __name__ == "__main__":
    main()
