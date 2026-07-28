from __future__ import annotations

import importlib.util
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "materialize_e50b_observed_route_math_toy.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("e50b_materializer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e50b_source_preference_is_fixed(tmp_path):
    module = _load()
    preferred = tmp_path / "preferred.json"
    fallback = tmp_path / "fallback.json"
    for path, schema in (
        (preferred, "preferred_v1"),
        (fallback, "fallback_v1"),
    ):
        path.write_text(
            json.dumps(
                {
                    "schema": schema,
                    "pass": True,
                    "bidirectionally_executable_count": 10,
                    "selected_source_indices": list(range(10)),
                }
            ),
            encoding="utf-8",
        )
    module.SOURCES = (
        (preferred, "preferred_v1", "preferred"),
        (fallback, "fallback_v1", "fallback"),
    )
    assert module._select_source() == (
        preferred,
        "preferred_v1",
        "preferred",
    )


def test_e50b_protocol_requires_unforced_route_probe_and_matched_training():
    module = _load()
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e50b_observed_route_math_toy_20260726.md"
    ).read_text(encoding="utf-8")
    assert [origin for _, _, origin in module.SOURCES] == [
        "e49aa_calibrated_pairwise_observed",
        "e49ab_all_observed_persistent_pairwise",
        "e50a_consensus_observed",
        "e49ac_confirmed_singleton_observed",
    ]
    assert "64 independent unforced" in protocol
    assert "least eight of ten prompts" in protocol
    assert "one A100 each" in protocol
    assert "exactly three prompt epochs" in protocol
    assert "policy-entropy adaptation disabled" in protocol
