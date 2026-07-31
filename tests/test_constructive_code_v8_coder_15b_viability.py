from pathlib import Path

import evaluate_constructive_code_v3_coder_viability as base
import evaluate_constructive_code_v6_coder_viability as v6
import evaluate_constructive_code_v8_coder_15b_viability as v8


ROOT = Path(__file__).resolve().parents[1]


def test_v8_wraps_exact_v6_development_boundary():
    assert v6.DEVELOPMENT_PROBLEMS == ("359_B", "988_A", "1399_D")
    assert v6.EVALUATION_PROBLEMS == ("361_B", "1294_C", "149_C")
    source = (ROOT / "ops/evaluate_constructive_code_v8_coder_15b_viability.py").read_text()
    assert "constructive-code-v8-coder-15b-viability-v1" in source
    assert "EXPECTED_SEED = 77101" in source
    assert "v6._validate_v6_gate_and_choose_suites" in source
    assert callable(v8.main)
    assert base.EXPECTED_SEED == 77101


def test_v8_batch_freezes_capacity_model_and_runtime_boundary():
    text = (ROOT / "ops/slurm/evaluate_constructive_code_v8_coder_15b_viability.slurm").read_text()
    mkdir_line = next(line for line in text.splitlines() if line.startswith("mkdir -p "))
    assert '"$RUN_ROOT/runtime"' not in mkdir_line
    assert "2e1fd397ee46e1388853d2af2c993145b0f1098a" in text
    assert "--sample-count 64 --prefix-count 16 --seed 77101" in text
    assert "--temperature 1.0 --top-p 1.0 --max-tokens 1024" in text


def test_v8_protocol_is_explicitly_a_separate_15b_result():
    text = (ROOT / "paper/preregistration/constructive_code_v8_coder_15b_capacity_retry_20260730.md").read_text()
    normalized = " ".join(text.split())
    assert "frozen before any 1.5B sample" in text
    assert "does not turn the ConstructiveCode row into a 0.5B result" in normalized
    assert "are not loaded" in normalized
