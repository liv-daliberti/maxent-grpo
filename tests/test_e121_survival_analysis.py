"""Focused checks for the registered E121 descriptive analysis."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def analysis():
    spec = importlib.util.spec_from_file_location(
        "e121_analysis_test", ROOT / "ops/exp_scaling/build_paper_e121_survival.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def identity(prompt, value, *, sequence=None):
    return dict(
        prompt_fingerprint=prompt,
        delta_mean_logprob=value,
        delta_sequence_logprob=value if sequence is None else sequence,
        worst_mean_intermediate_delta=min(0, value),
        worst_sequence_intermediate_delta=min(0, value if sequence is None else sequence),
    )


def test_quantiles_and_decline_threshold_are_signed_and_strict(analysis):
    rows = [identity(i, value, sequence=10 * value)
            for i, value in enumerate([-1, -.5, 0, 1])]
    actual = analysis.statistics(analysis.array_from_identities(rows))
    np.testing.assert_allclose(actual[:, 0], [-.25, -.85, -1, .25, -1])
    np.testing.assert_allclose(actual[:, 1], [-2.5, -8.5, -10, .5, -10])


def test_pooled_statistics_weight_identities_not_prompt_or_seed_medians(analysis):
    # One retained bank has nine exemplars; the other has one. Equal weighting
    # of prompt summaries would give a different median and decline fraction.
    rows = [identity(10, -1) for _ in range(9)] + [identity(20, 9)]
    stats = analysis.named_statistics(analysis.array_from_identities(rows))
    assert stats["mean_logprob"]["median"] == -1
    assert stats["mean_logprob"]["fraction_below_minus_0_5"] == .9


def test_clustered_sample_preserves_all_exemplars_and_multiplicity(analysis):
    class FixedRng:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 3, 3)
            return np.array([2, 0, 2])

    groups = [np.array([[10.0] * 4, [11.0] * 4]),
              np.array([[20.0] * 4]),
              np.array([[30.0] * 4, [31.0] * 4, [32.0] * 4])]
    sample = analysis.clustered_sample(groups, FixedRng())
    np.testing.assert_array_equal(sample[:, 0], [30, 31, 32, 10, 11, 30, 31, 32])


def test_bootstrap_resamples_prompts_then_seeds_and_reuses_prompt_draw(analysis, monkeypatch):
    # Prompt fingerprints overlap across seeds on purpose: seeds are separate
    # clusters. A repeated selected seed reuses its already drawn prompt bank.
    seeds = [[identity(10, 1), identity(10, 2), identity(20, 30)],
             [identity(10, 100), identity(20, 200), identity(20, 201)]]
    calls = []
    answers = iter([np.array([0, 0]), np.array([1, 0]), np.array([1, 1])])

    class ScriptedRng:
        def integers(self, low, high, size):
            calls.append((low, high, size))
            return next(answers)

    monkeypatch.setattr(analysis.np.random, "default_rng", lambda seed: ScriptedRng())
    result = analysis.hierarchical_bootstrap(seeds, draws=1, rng_seed=121)
    expected_sample = np.array([[value, value, 0, 0]
                                for value in [200, 201, 100, 200, 201, 100]])
    expected = analysis.named_statistics(expected_sample)
    assert calls == [(0, 2, 2)] * 3
    for metric, stats in expected.items():
        for name, value in stats.items():
            assert result[metric][name] == [value, value]


@pytest.fixture
def history_file(tmp_path, analysis, monkeypatch):
    monkeypatch.setattr(analysis, "FREEZE", 2)
    monkeypatch.setattr(analysis, "HORIZON", 4)
    prefix = "train/canonical_replay_"
    records = []
    for step, mean in [(2, -1.0), (3, -2.0), (4, -.8)]:
        records.append({
            "trainer/step": step,
            "trainer/policy_sgd_step": step,
            prefix + "bank_membership_frozen": 1,
            prefix + "prompt_fingerprint_row_00": 10,
            prefix + "outcome_fingerprint_row_00": 20,
            prefix + "exemplar_mean_logprob_row_00": mean,
            prefix + "exemplar_sequence_logprob_row_00": mean * 4,
        })
    terminal = dict(records[-1], **{"trainer/step": 5})
    records.append(terminal)
    path = tmp_path / "train_metrics.jsonl"
    run = dict(seed=43, metrics=str(path), frozen_identities=1,
               frozen_prompts=1, identity_observations=3)
    return records, path, run


def write_records(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


def test_history_excludes_terminal_duplicate_and_retains_worst_visit(analysis, history_file):
    records, path, run = history_file
    write_records(path, records)
    rows = analysis.read_histories(run)
    assert len(rows) == 1
    row = rows[0]
    assert row["visit_count"] == 3
    assert row["delta_mean_logprob"] == pytest.approx(.2)
    assert row["delta_sequence_logprob"] == pytest.approx(.8)
    assert row["worst_mean_intermediate_delta"] == -1
    assert row["worst_sequence_intermediate_delta"] == -4


@pytest.mark.parametrize("mutation", ["missing", "repeated", "nonfinite", "population"])
def test_history_fails_on_missing_repeated_or_invalid_observations(analysis, history_file, mutation):
    records, path, run = history_file
    if mutation == "missing":
        records.pop(1)
    elif mutation == "repeated":
        records.insert(1, records[1].copy())
    elif mutation == "nonfinite":
        records[1]["train/canonical_replay_exemplar_mean_logprob_row_00"] = float("nan")
    elif mutation == "population":
        run["frozen_identities"] = 2
    write_records(path, records)
    with pytest.raises(ValueError):
        analysis.read_histories(run)


@pytest.mark.parametrize("changed", ["protocol", "ledger", "metrics", "receipt"])
def test_audit_gate_rejects_changed_source_bytes(analysis, tmp_path, monkeypatch, changed):
    protocol, ledger, audit_path = [tmp_path / name for name in ("protocol.md", "ledger.json", "audit.json")]
    protocol.write_text("registered protocol")
    ledger.write_text("{}")
    runs = []
    for seed in analysis.SEEDS:
        directory = tmp_path / str(seed)
        directory.mkdir()
        metrics = directory / "train_metrics.jsonl"
        receipt = directory / "TRAINING_COMPLETE.json"
        metrics.write_text("{}\n")
        receipt.write_text("{}")
        runs.append(dict(
            seed=seed, passed=True, missing_identity_count=0, insufficient_visit_count=0,
            complete_finite_score_fraction=1, complete_round_robin_schedule=True,
            metrics=str(metrics), metrics_sha256=analysis.sha256(metrics), run_dir=str(directory),
            completion_receipt_sha256=analysis.sha256(receipt),
        ))
    audit_path.write_text(json.dumps(dict(
        schema="e121_independent_integrity_audit_v1", passed=True,
        ledger_sha256=analysis.sha256(ledger), preregistration_sha256=analysis.sha256(protocol), runs=runs,
    )))
    monkeypatch.setattr(analysis, "AUDIT", audit_path)
    monkeypatch.setattr(analysis, "PROTOCOL", protocol)
    monkeypatch.setattr(analysis, "LEDGER", ledger)
    assert analysis.checked_audit()["passed"] is True
    path = {"protocol": protocol, "ledger": ledger, "metrics": Path(runs[0]["metrics"]),
            "receipt": Path(runs[0]["run_dir"]) / "TRAINING_COMPLETE.json"}[changed]
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="differs|differ"):
        analysis.checked_audit()
