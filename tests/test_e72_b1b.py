from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "ops" / "exp_scaling" / "aggregate_e72_b1b.py"
SPEC = importlib.util.spec_from_file_location("aggregate_e72_b1b", PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _domain(x_verdict: str, dr_verdict: str, x_mean: float = -0.1):
    return {
        "reportable": True,
        "comparisons": {
            "xgrpo": {"verdict": x_verdict, "mean": x_mean},
            "drgrpo": {"verdict": dr_verdict, "mean": 1.0},
        },
    }


def test_registered_verdict_boundaries():
    assert MODULE.verdict([-0.149, 0.149]) == "E"
    assert MODULE.verdict([-0.30, -0.01]) == "W"
    assert MODULE.verdict([0.151, 0.30]) == "B"
    assert MODULE.verdict([-0.20, 0.10]) == "U"
    assert MODULE.verdict([-0.10, 0.20]) == "U"


def test_p1_requires_three_xgrpo_equivalences():
    domains = [
        _domain("E", "B"),
        _domain("E", "B"),
        _domain("E", "B"),
        _domain("W", "B"),
        _domain("U", "B"),
    ]
    assert "P1" in MODULE.registered_interpretations(domains)
    domains[2] = _domain("U", "B")
    assert "P1" not in MODULE.registered_interpretations(domains)


def test_p2_and_p3_follow_registered_majority_rule():
    p2 = [
        _domain("W", "B"),
        _domain("W", "B"),
        _domain("W", "B"),
        _domain("E", "B"),
        _domain("U", "B"),
    ]
    assert "P2" in MODULE.registered_interpretations(p2)
    p3 = [
        _domain("W", "E"),
        _domain("W", "E"),
        _domain("W", "E"),
        _domain("W", "B"),
        _domain("W", "B"),
    ]
    assert "P3" in MODULE.registered_interpretations(p3)


def test_p4_reports_any_positive_point_estimate():
    domains = [_domain("E", "B") for _ in range(5)]
    assert "P4" not in MODULE.registered_interpretations(domains)
    domains[4] = _domain("U", "B", x_mean=0.01)
    assert "P4" in MODULE.registered_interpretations(domains)


def test_global_interpretation_is_withheld_until_every_domain_reports():
    domains = [_domain("E", "B") for _ in range(5)]
    domains[0] = {"reportable": False}
    assert MODULE.registered_interpretations(domains) == ["WITHHELD"]


def test_frozen_b1b_surface():
    assert MODULE.SEEDS == (43, 44, 45, 46, 47)
    assert MODULE.TERMINAL_STEP == 4608
    assert MODULE.BOOTSTRAP_RESAMPLES == 10_000
    assert MODULE.MARGIN == 0.15
    assert MODULE.EXPECTED_OVERRIDES[
        "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE"
    ] == "verified_likelihood_per_rollout"


def test_summary_retains_every_preregistered_comparator():
    cells = {}
    references = {}
    for domain, _ in MODULE.DOMAINS:
        for seed in MODULE.SEEDS:
            cells[(domain, seed)] = {
                "terminal": True,
                "metrics": {
                    "greedy": 0.5,
                    "mean8": 0.5,
                    "pass8": 0.75,
                    "distinct8": 1.0,
                },
                "integrity": {"violations": []},
            }
            references[(domain, "drgrpo", seed)] = {
                metric: 0.25 for metric in MODULE.METRIC_KEYS
            }
            references[(domain, "b3a", seed)] = {
                metric: 0.30 for metric in MODULE.METRIC_KEYS
            }
            references[(domain, "b1a", seed)] = {"distinct8": 1.0}
            references[(domain, "xgrpo", seed)] = {
                metric: 1.0 for metric in MODULE.METRIC_KEYS
            }

    summary = MODULE.summarize(cells, references)
    assert summary["all_25_cells_terminal_and_valid"]
    assert summary["registered_interpretations"] == ["P1"]
    for row in summary["domains"]:
        assert set(row["means"]) == {"drgrpo", "b3a", "b1b", "b1a", "xgrpo"}
        assert set(row["comparisons"]) == {"drgrpo", "b3a", "b1a", "xgrpo"}
        assert row["comparisons"]["xgrpo"]["verdict"] == "E"
