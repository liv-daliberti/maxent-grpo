from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
AUDIT = (
    ROOT
    / "ops/math_strategy_calibration/audit_e49e_repaired_calibration.py"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "e49e_repaired_calibration_test",
        AUDIT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manual_packet_does_not_reveal_pair_kind():
    module = _load()
    packet = module._packet_row(
        pair_id="PAIR_123",
        problem="A problem.",
        reference_answer="17",
        routes=[{"plan": "left"}, {"plan": "right"}],
    )
    assert packet["schema"] == "e49e_repaired_blinded_manual_pair_v1"
    assert "kind" not in packet
    assert "expected_distinct" not in packet
    assert set(packet["questions"]) == {
        "route_a_sound_and_self_contained",
        "route_b_sound_and_self_contained",
        "genuinely_distinct_decisive_strategy",
        "rationale",
    }


def test_manual_pruning_uses_deterministic_maximum_distinct_clique():
    module = _load()
    retained = ["S1", "S2", "S3", "S4"]
    edges = {
        frozenset(("S1", "S2")),
        frozenset(("S1", "S3")),
        frozenset(("S2", "S3")),
        frozenset(("S2", "S4")),
    }
    assert module._maximum_manual_clique(
        retained,
        set(retained),
        edges,
    ) == ("S1", "S2", "S3")
    assert module._maximum_manual_clique(
        retained,
        {"S2", "S3", "S4"},
        edges,
    ) == ("S2", "S3")
    assert module._maximum_manual_clique(
        retained,
        set(),
        edges,
    ) == ()


def test_final_audit_replays_evidence_and_gates_false_new_support():
    source = AUDIT.read_text(encoding="utf-8")
    for required in (
        "_repair_record_passes",
        "_augmentation_selected",
        "_prune_menu_closed",
        "e49e_manually_audited_materialization_v1",
        "no_zero_support_rows_after_manual_pruning",
        "manual_rejected_claim_count",
        "final_retained_pairs_all_manually_distinct",
        "manual_equivalent_controls_all_recognized",
        "overall_multi_at_least_20",
        "eval_multi_at_least_10",
        "all_prompts_at_most_2048_tokens",
        "len(invalid_controls) == 5",
        "len(equivalent_controls) == 3",
        "blinded_equivalent_control",
        "audit_script_sha256",
        "base_calibration_amendment_sha256",
        "manual_pruning_amendment_sha256",
        "e49e_singleton_repair_v5_frozen_identity_v1",
        "repair_e49e_singleton_gaps_v5_curated.py",
        "_configure_v4_validator",
        "module.impl.base._augmentation_selected",
        "v5_calibration_replay_amendment_sha256",
        '"__pycache__" not in candidate.parts',
    ):
        assert required in source
