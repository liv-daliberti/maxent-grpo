"""Scientific invariants of the frozen E120 primary-estimand analysis."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e120_primary", ROOT / "ops/exp_scaling/build_paper_e120_primary_breadth.py"
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


@pytest.fixture
def source():
    return json.loads(BUILDER.SOURCE.read_text())


def replace_endpoints(source, domain, uniform_distinct):
    cell = source["cells"]["qwen05b"][domain]
    for arm in ("uniform_key_replay", "fresh_frequency"):
        for seed in BUILDER.SEEDS:
            cell[arm][str(seed)] = {"pass8": 0.5, "distinct8": 1.0}
    for seed, distinct in zip(BUILDER.SEEDS, uniform_distinct):
        cell["uniform_key_replay"][str(seed)]["distinct8"] = distinct
    for metric in ("pass8", "distinct8"):
        values = {
            str(seed): cell["fresh_frequency"][str(seed)][metric] - cell["uniform_key_replay"][str(seed)][metric]
            for seed in BUILDER.SEEDS
        }
        cell["fresh_frequency_minus_uniform"][metric] = {
            "per_seed": values, "mean": sum(values.values()) / 5
        }


def test_primary_contrast_reverses_source_sign_and_removes_success(source):
    rows = BUILDER.analyze(source)
    graph = rows["graph_coloring"]["uniform_minus_frequency"]
    assert graph["distinct8"]["mean"] == pytest.approx(0.6625)
    assert graph["pass8"]["mean"] == pytest.approx(0.066015625)
    assert graph["breadth8"]["mean"] == pytest.approx(0.596484375)
    # Python seed 43 gains only success, not additional correct alternatives.
    python = rows["python_factors"]["uniform_minus_frequency"]
    assert python["pass8"]["per_seed"]["43"] == 0.828125
    assert python["breadth8"]["per_seed"]["43"] == 0.0


def test_pooled_interval_preserves_seed_pairing_across_domains(source):
    for domain in BUILDER.DOMAINS:
        replace_endpoints(source, domain, [1.0] * 5)
    # Opposite effects cancel within every shared seed. Independent domain
    # resampling or treating the 25 cells as replicates would invent variance.
    replace_endpoints(source, "graph_coloring", [0.5, 0.75, 1.0, 1.25, 1.5])
    replace_endpoints(source, "countdown", [1.5, 1.25, 1.0, 0.75, 0.5])
    rows = BUILDER.analyze(source)
    pooled = rows["five_domain_mean"]["uniform_minus_frequency"]["breadth8"]
    assert pooled["per_seed"] == {str(seed): 0.0 for seed in BUILDER.SEEDS}
    assert pooled["paired_bootstrap_percentile_95"] == [0.0, 0.0]
    assert rows["graph_coloring"]["uniform_minus_frequency"]["breadth8"]["paired_bootstrap_percentile_95"] != [0.0, 0.0]


@pytest.mark.parametrize("mutation", ["incomplete", "missing_seed", "duplicate_seed", "nonfinite", "wrong_delta", "wrong_mean", "wrong_schema", "wrong_sign", "wrong_freeze"])
def test_invalid_source_fails_before_reporting(source, mutation):
    cell = source["cells"]["qwen05b"]["graph_coloring"]
    if mutation == "incomplete":
        cell["complete_block"] = False
    elif mutation == "missing_seed":
        del cell["uniform_key_replay"]["47"]
    elif mutation == "duplicate_seed":
        cell["terminal_seeds"][-1] = 46
    elif mutation == "nonfinite":
        cell["fresh_frequency"]["43"]["pass8"] = float("nan")
    elif mutation == "wrong_delta":
        cell["fresh_frequency_minus_uniform"]["pass8"]["per_seed"]["43"] *= -1
    elif mutation == "wrong_mean":
        cell["fresh_frequency_minus_uniform"]["pass8"]["mean"] += 0.1
    elif mutation == "wrong_schema":
        source["schema"] = "different"
    elif mutation == "wrong_sign":
        source["contrast"] = "uniform minus frequency"
    elif mutation == "wrong_freeze":
        source["generated_at"] = "2026-09-06T00:00:00+00:00"
    with pytest.raises(ValueError):
        BUILDER.analyze(source)


def test_bootstrap_counts_ordered_resamples_and_interpolates_percentiles():
    # Four zero effects and one unit effect produce Binomial(5, .2)/5.
    result = BUILDER.paired_bootstrap(dict(zip(map(str, BUILDER.SEEDS), [0, 0, 0, 0, 1])))
    assert result["mean"] == 0.2
    assert result["paired_bootstrap_percentile_95"] == [0.0, 0.6]


def test_source_is_unchanged_by_analysis(source):
    original = copy.deepcopy(source)
    BUILDER.analyze(source)
    assert source == original


def test_build_rejects_replaced_snapshot_even_if_date_is_retained(source, tmp_path, monkeypatch):
    # A refreshed artifact must not silently enter this fixed-snapshot analysis.
    replacement = tmp_path / "source.json"
    replacement.write_text(json.dumps(source) + "\n")
    monkeypatch.setattr(BUILDER, "SOURCE", replacement)
    with pytest.raises(ValueError, match="source bytes differ"):
        BUILDER.build()
