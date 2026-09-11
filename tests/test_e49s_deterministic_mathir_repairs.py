from __future__ import annotations

import importlib.util
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "certify_e49s_deterministic_repairs.py"
)
spec = importlib.util.spec_from_file_location("e49s_repairs", SCRIPT)
assert spec is not None and spec.loader is not None
e49s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e49s)


def test_all_repair_programs_are_exact_and_deterministic():
    certificates, menus = e49s._execute_repairs()
    assert len(certificates) == 4
    assert set(menus) == set(e49s.REPAIRS)
    for certificate in certificates:
        assert certificate["all_programs_deterministic"] is True
        assert certificate["all_action_combos_exact"] is True
        assert certificate["all_answers_exact"] is True
        assert len(certificate["executions"]) == 2


def test_literal_scans_and_dp_materialize_their_decisive_states():
    divisibility = e49s._divisibility_scan()
    assert divisibility["action_states"][1]["state"]["passing_values"] == [
        60,
        120,
        180,
        240,
        300,
        360,
        420,
        480,
    ]
    assert divisibility["answer"] == "8"

    circular = e49s._circular_scan()
    assert circular["action_states"][0]["state"]["generated_count"] == 5040
    assert (
        circular["action_states"][1]["state"][
            "cyclic_adjacency_passing_count"
        ]
        == 720
    )
    assert len(
        circular["action_states"][1]["state"]["passing_arrangements"]
    ) == 720

    dynamic_program = e49s._binomial_dp()
    vectors = dynamic_program["action_states"][1]["state"][
        "vectors_after_each_island"
    ]
    assert [len(vector) for vector in vectors] == list(range(2, 9))
    assert dynamic_program["answer"] == "448/15625"


def test_frozen_artifact_advances_at_ten_per_split_when_present():
    decision_path = (
        ROOT
        / "var/artifacts/e49s_deterministic_mathir_repairs_v1/"
        "advancement_decision.json"
    )
    if not decision_path.is_file():
        return
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["advance_to_matched_toy_training"] is True
    assert decision["support_counts"] == {
        "train_multi": 10,
        "eval_multi": 10,
        "overall_multi": 20,
    }
