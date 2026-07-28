from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "audit_e49r_combined_toy_bank.py"
)
SOURCE = ROOT / "var/data/e49b_math_strategy_toy"
BASE = ROOT / "var/artifacts/e49e_trace_bank_math_toy_repair_v5"
H = ROOT / "var/artifacts/e49h_curated_distinct_routes_toy_v1"
STRICT = [
    H,
    ROOT / "var/artifacts/e49k_curated_eval_buffer_v1",
    ROOT / "var/artifacts/e49l_curated_train_buffer_v1",
    ROOT / "var/artifacts/e49p_curated_eval_reserve_v1",
]
RECOVERY = ROOT / "var/artifacts/e49q_recovered_routes_v1"
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49r_combined_manual_audit_20260724.md"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49r_combined_audit_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_survivors_reach_buffered_support_before_manual_review():
    module = _load()
    strict = module._strict_survivors(STRICT)
    recovery = module._recovery_survivors(RECOVERY)
    claims = strict + recovery
    assert len(strict) == 16
    assert len(recovery) == 4
    assert len({row["record"]["row_id"] for row in claims}) == 20
    assert sum(row["record"]["split"] == "train" for row in claims) == 11
    assert sum(row["record"]["split"] == "eval" for row in claims) == 9


def test_zero_support_row_has_two_exact_singleton_traces():
    module = _load()
    rows = module._source_rows(SOURCE)
    _, key = module._singleton_payload(
        e49h_evidence=H,
        source_rows=rows,
    )
    assert key["row_id"] == "eval:0010:834bdfcea94c2ee5ca2f"
    assert key["strategy_id"] == "S2"


def test_prepare_writes_26_pair_blinded_packet(tmp_path):
    module = _load()
    evidence = tmp_path / "evidence"
    args = argparse.Namespace(
        source=SOURCE,
        base_evidence=BASE,
        strict_evidence=STRICT,
        recovery_evidence=RECOVERY,
        e49h_evidence=H,
        evidence=evidence,
    )
    module.prepare(args)
    manifest = json.loads(
        (evidence / "manual_audit_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["pair_count"] == 26
    assert manifest["retained_claim_count"] == 20
    assert manifest["singleton_repair_control_count"] == 1
    assert manifest["blinded_equivalent_control_count"] == 5
    assert len(
        (evidence / "manual_audit_packet.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 26


def test_protocol_requires_zero_false_new_and_no_zero_support():
    text = PROTOCOL.read_text(encoding="utf-8").casefold()
    assert "frozen before e49r packet generation or labeling" in text
    assert "manual false-new is exactly zero" in text
    assert "all 100 rows retain at least one validated route" in text
