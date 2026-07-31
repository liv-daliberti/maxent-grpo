import json
from pathlib import Path

import materialize_constructive_code_v7_sft as materialize
import evaluate_constructive_code_v3_coder_viability as base_eval
import evaluate_constructive_code_v7_coder_viability as evaluator
import train_constructive_code_v7_sft as trainer


ROOT = Path(__file__).resolve().parents[1]


def test_v7_sft_selection_and_split_are_frozen():
    assert materialize.TRAIN == {
        "327_B": "327_b",
        "659_C": "659_c",
        "1283_C": "1283_c",
        "1102_B": "1102_b",
    }
    assert materialize.DEVELOPMENT == ("359_B", "988_A", "1399_D")
    assert materialize.EVALUATION == ("361_B", "1294_C", "149_C")
    protocol = (ROOT / "paper/preregistration/constructive_code_v7_train_only_sft_20260730.md").read_text()
    normalized = " ".join(protocol.split())
    assert "exactly 64 examples" in protocol
    assert "exactly 32 optimizer updates" in protocol
    assert "are never loaded by materialization or SFT" in normalized


def test_materialized_v7_sft_is_train_only_when_present():
    manifest_path = ROOT / "var/data/constructive_code_v7_sft/manifest.json"
    examples_path = ROOT / "var/data/constructive_code_v7_sft/examples.jsonl"
    if not manifest_path.is_file() or not examples_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    examples = [json.loads(line) for line in examples_path.read_text().splitlines()]
    assert manifest["status"] == "pass"
    assert manifest["example_count"] == 64
    assert manifest["development_problem_ids_loaded"] == []
    assert manifest["evaluation_problem_ids_loaded"] == []
    assert manifest["language_model_sampling"] is False
    assert {row["source_problem_id"] for row in examples} == set(materialize.TRAIN)
    assert all(sum(row["source_problem_id"] == task for row in examples) == 16 for task in materialize.TRAIN)


def test_v7_sft_and_post_sft_gate_constants_are_frozen():
    assert trainer.SEED == 77201
    assert trainer.EPOCHS == 4
    assert trainer.EXAMPLES == 64
    assert trainer.ACCUMULATION == 8
    assert trainer.UPDATES == 32
    assert trainer.MAX_LENGTH == 4096
    assert trainer.LEARNING_RATE == 1e-5
    assert base_eval.EXPECTED_SEED == 77101
    assert callable(evaluator.main)
    source = (ROOT / "ops/evaluate_constructive_code_v7_coder_viability.py").read_text()
    assert "EXPECTED_SEED = 77102" in source
    assert "v6.DEVELOPMENT_PROBLEMS" in source
    assert "v6.EVALUATION_PROBLEMS" in source


def test_v7_assistant_only_mask_handles_boundary_token_merge():
    class BoundaryMergingTokenizer:
        def __call__(self, text, **kwargs):
            assert text == "abcde\ncode"
            assert kwargs["return_offsets_mapping"] is True
            return {
                "input_ids": [10, 11, 12, 13],
                "offset_mapping": [(0, 3), (3, 6), (6, 8), (8, 10)],
            }

    input_ids, labels = trainer.encode_assistant_only(
        BoundaryMergingTokenizer(), "abcde", "\ncode"
    )
    assert input_ids == [10, 11, 12, 13]
    assert labels == [-100, 11, 12, 13]


def test_v7_slurm_preserves_absent_runtime_destination_and_frozen_gate():
    text = (ROOT / "ops/slurm/train_constructive_code_v7_sft_and_gate.slurm").read_text()
    mkdir_line = next(line for line in text.splitlines() if line.startswith("mkdir -p "))
    assert '"$RUN_ROOT/runtime"' not in mkdir_line
    assert '"$RUN_ROOT/build"' in mkdir_line
    assert '"$RUN_ROOT/scratch"' in mkdir_line
    assert "--seed 77102" in text
    assert "--sample-count 64 --prefix-count 16" in text
    assert "train_constructive_code_v7_sft.py" in text


def test_v7_launcher_records_token_boundary_repair():
    text = (ROOT / "ops/exp_scaling/launch_constructive_code_v7_sft_and_gate.sh").read_text()
    assert "constructive_code_v7_token_boundary_repair_r1_20260730.md" in text
    assert '"repair_protocol_sha256"' in text
