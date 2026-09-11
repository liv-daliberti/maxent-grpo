from __future__ import annotations

import importlib.util
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "prepare_e49t_declaration_mismatch_calibration.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("e49t_mismatch", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mismatch_calibration_is_byte_identical_math_with_swapped_declaration(
    tmp_path,
):
    output = tmp_path / "mismatch"
    identity = _load().prepare(output)
    public = [
        json.loads(line) for line in (output / "cohort.jsonl").read_text().splitlines()
    ]
    private = {
        row["item_id"]: row
        for row in (
            json.loads(line)
            for line in (output / "private/labels.jsonl").read_text().splitlines()
        )
    }
    assert identity["matched_count"] == identity["mismatched_count"] == 24
    assert len(public) == 48
    grouped = {}
    for row in public:
        label = private[row["item_id"]]
        grouped.setdefault(
            (row["group_id"], label["executed_strategy_id"]), []
        ).append((row, label))
    assert len(grouped) == 24
    for pair in grouped.values():
        assert {label["kind"] for _, label in pair} == {
            "matched_declaration",
            "mismatched_declaration",
        }
        bodies = {
            row["response"].split("\n\n", maxsplit=1)[1] for row, _ in pair
        }
        assert len(bodies) == 1


def test_e49t_training_is_hard_gated_on_mismatch_result():
    launcher = (
        ROOT
        / "ops/exp_scaling/launch_e49t_natural_menu_math_toy_05b.sh"
    ).read_text(encoding="utf-8")
    assert "e49t_declaration_mismatch_calibration_v1" in launcher
    assert 'DECLARATION_CALIBRATION="$DECLARATION_CALIBRATION_DIR/result.json"' in launcher
    assert "e49t_declaration_mismatch_result_v1" in launcher
    assert "declaration-mismatch calibration has not passed" in launcher
    assert "declaration_mismatch_result_sha256" in launcher
    assert "declaration endpoint identity mismatch" in launcher
