from __future__ import annotations

import importlib.util
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "prepare_e49t_route_confusion_calibration.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("prepare_e49t", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e49t_blinded_route_confusion_cohort(tmp_path):
    module = _load()
    output = tmp_path / "cohort"
    module.prepare(output)
    public = [
        json.loads(line)
        for line in (output / "cohort.jsonl").read_text().splitlines()
    ]
    private = [
        json.loads(line)
        for line in (output / "private/labels.jsonl").read_text().splitlines()
    ]
    identity = json.loads((output / "frozen_identity.json").read_text())

    assert len(public) == 60
    assert len(private) == 60
    assert identity["group_count"] == 12
    assert identity["positive_count"] == 48
    assert identity["answer_only_negative_count"] == 12
    assert all("expected_strategy_id" not in row for row in public)
    assert all("<strategy_id>" not in row["response"] for row in public)
    assert all("<action_step" not in row["response"] for row in public)

    by_group = {}
    for row in private:
        by_group.setdefault(row["group_id"], []).append(row)
    assert len(by_group) == 12
    for rows in by_group.values():
        assert len(rows) == 5
        positives = [row for row in rows if row["kind"] == "positive"]
        negatives = [
            row for row in rows if row["kind"] == "answer_only_negative"
        ]
        assert len(positives) == 4
        assert len(negatives) == 1
        assert sorted(
            row["expected_strategy_id"] for row in positives
        ) == ["S1", "S1", "S2", "S2"]
