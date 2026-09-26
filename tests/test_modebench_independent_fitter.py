"""The revised fit must authenticate new development evidence before ranking."""
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "ops/exp_scaling/fit_modebench_level3_independent.py"
SPEC = importlib.util.spec_from_file_location("independent_fitter", PATH)
fitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fitter)


@pytest.fixture
def receipts(tmp_path, monkeypatch):
    paths = []
    for index in range(5):
        path = tmp_path / f"receipt{index}.json"
        path.write_text(json.dumps({"domain": "graph_coloring", "split": "dev",
                                   "identity": {"seeds": list(fitter.DEVELOPMENT_DRAW_LABELS)}}))
        paths.append(path)
    monkeypatch.setattr(fitter.mixture, "receipt_rows", lambda receipt: [{"problem": "graph"}])
    return paths


def test_all_seed_receipts_authenticated_before_unchanged_fit(receipts, monkeypatch):
    calls = []
    monkeypatch.setattr(fitter, "validate_seed_receipt", lambda receipt, rows: calls.append("validate"))

    def fit(*args, **kwargs):
        assert calls == ["validate"] * 5
        assert kwargs == {"seed": fitter.mixture.SELECTION_SEED}
        return {"provenance": {}, "information_boundary": {}, "weights": [0, 0, 0, 1]}

    monkeypatch.setattr(fitter.mixture, "fit_recipe", fit)
    result = fitter.fit_recipe(receipts[0], receipts[1:], "graph_coloring")
    assert result["weights"] == [0, 0, 0, 1]
    assert result["sampling_validation"]["legacy_receipts_used"] is False


@pytest.mark.parametrize("field,value", [("split", "eval"), ("domain", "pantry")])
def test_other_split_or_domain_cannot_enter_fit(receipts, monkeypatch, field, value):
    receipt = json.loads(receipts[0].read_text())
    receipt[field] = value
    receipts[0].write_text(json.dumps(receipt))
    monkeypatch.setattr(fitter.mixture, "fit_recipe", lambda *a, **k: pytest.fail("must not rank"))
    with pytest.raises(ValueError, match="registered v2 development"):
        fitter.fit_recipe(receipts[0], receipts[1:], "graph_coloring")


def test_legacy_seed_validation_failure_prevents_ranking(receipts, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("legacy seed schema")
    monkeypatch.setattr(fitter, "validate_seed_receipt", reject)
    monkeypatch.setattr(fitter.mixture, "fit_recipe", lambda *a, **k: pytest.fail("must not rank"))
    with pytest.raises(ValueError, match="legacy seed schema"):
        fitter.fit_recipe(receipts[0], receipts[1:], "graph_coloring")


def test_input_mutation_prevents_recipe_publication(receipts, monkeypatch, tmp_path):
    monkeypatch.setattr(fitter, "validate_seed_receipt", lambda *a, **k: None)
    def mutate(*args, **kwargs):
        receipts[1].write_text("{}")
        return {"provenance": {}, "information_boundary": {}}
    monkeypatch.setattr(fitter.mixture, "fit_recipe", mutate)
    output = tmp_path / "recipe.json"
    with pytest.raises(ValueError, match="changed during"):
        fitter.fit_recipe(receipts[0], receipts[1:], "graph_coloring", output)
    assert not output.exists()
