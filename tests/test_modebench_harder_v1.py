from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "ops/exp_scaling/materialize_modebench_harder_v1.py"
SPEC = importlib.util.spec_from_file_location("harder_v1", PATH)
assert SPEC and SPEC.loader
harder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(harder)


def test_harder_contract_knobs_are_strictly_above_easy():
    assert harder.ROWS_PER_DOMAIN == 128
    assert harder.DOMAIN_ORDER[-1] == "pantry"
    assert harder.HARD_MATHIR_FAMILIES == {
        "ax_plus_b_eq_dx_plus_c", "ax_plus_b_eq_c_minus_dx"
    }


def test_materialized_harder_mode(tmp_path):
    output = tmp_path / "modebench_harder_v1"
    manifest = harder.materialize(output)
    assert manifest["schema"] == "modebench_harder_v1"
    assert manifest["verifier_contract"] == "unchanged"
    assert json.loads((output / "identity.json").read_text()) == manifest
    for domain in harder.DOMAIN_ORDER:
        assert (output / domain / "eval" / "dataset_dict.json").is_file()
