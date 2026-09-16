"""Reporting gates for the reproducible dated terminal census."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops/exp_scaling"))
from build_paper_latest_results import campaign_summary, summarize_effect


def endpoint(seed, *, arm, domain="graph_coloring", value=0.0):
    return {
        "model_key": "qwen05b", "domain": domain, "seed": seed, "arm": arm,
        "run_dir": f"/test/{domain}/{arm}/{seed}",
        "endpoint_status": "admitted", "training_completion_receipt_present": True,
        "endpoint": {"pass8": value, "distinct8": 2 * value, "breadth8": value, "mean8": value / 2},
    }


def test_partial_intersection_never_receives_an_interval():
    left = {s: endpoint(s, arm="replay_maxrl", value=s / 100) for s in range(43, 48)}
    right = {s: endpoint(s, arm="maxrl", value=0.1) for s in range(43, 47)}
    result = summarize_effect(left, right)
    assert result["paired_seeds"] == [43, 44, 45, 46]
    assert not result["complete_five_seed_block"]
    assert all("student_t_95" not in s for s in result["summaries"].values())


def test_four_arm_completion_is_separate_from_complete_replay_pair():
    rows = [endpoint(s, arm=arm, value=s / 100)
            for arm in ("maxrl", "replay_maxrl", "drgrpo", "replay_drgrpo")
            for s in range(43, 48) if (arm, s) != ("replay_drgrpo", 47)]
    result = campaign_summary("e119", {"rows": rows, "ledger": "test", "ledger_sha256": "test"})
    assert result["complete_five_seed_blocks"] == 0
    block = result["blocks"][0]
    assert block["contrasts"]["replay_maxrl_minus_maxrl"]["complete_five_seed_block"]
    assert block["contrasts"]["four_arm_replay_maxrl_minus_maxrl"]["n"] == 4


def test_e120_effect_sign_bootstrap_and_mechanism_gate_are_explicit():
    rows = [endpoint(s, arm="frequency", value=.2) for s in range(43, 48)]
    comparators = [endpoint(s, arm="uniform", value=.4) for s in range(43, 48)]
    telemetry = {row["run_dir"]: {"status": "pass" if row["seed"] != 47 else "provisional"} for row in rows}
    result = campaign_summary("e120", {"rows": rows, "comparator_rows": comparators, "frequency_telemetry": telemetry, "ledger": "test", "ledger_sha256": "test"})
    assert result["complete_five_seed_blocks"] == 1
    assert result["mechanism_validated_complete_blocks"] == 0
    contrast = result["blocks"][0]["contrasts"]["uniform_minus_frequency"]
    assert contrast["summaries"]["pass8"]["mean"] == .2
    assert contrast["summaries"]["pass8"]["paired_bootstrap_percentile_95"] == [.2, .2]
    assert contrast["summaries"]["pass8"]["bootstrap_ordered_resamples"] == 3125
