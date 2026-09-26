import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "paper/results/absolute_support_references.json"
MANUSCRIPT = ROOT / "paper/main.tex"


def test_absolute_support_references_are_exact_and_disclosed() -> None:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    assert payload["schema"] == "paper-absolute-support-references-v1"
    policies = payload["uninformed_policies"]
    assert math.isclose(policies["graph_coloring"]["primary"], 1.632851484925304)
    assert math.isclose(policies["pantry_plan"]["primary"], 2.152919212894907)
    assert math.isclose(policies["pantry_plan"]["instruction_following"], 2.7142475267937765)
    assert len(payload["rows"]) == 15
    assert sum(row["replay_drgrpo_distinct8"] > row["frozen_distinct8"] for row in payload["rows"]) == 13
    assert all(row["uninformed_distinct8"] is None for row in payload["rows"] if row["domain"] not in {"graph_coloring", "pantry_plan"})

    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    for token in (
        r"\label{tab:absolute-support-references}",
        "ReplayDr.GRPO exceeds the frozen model in 13 of 15 model--domain comparisons",
        "PantryPlan as an enumerable stress test of support concentration",
        "not a fully success-matched diversity measure",
        "raising its exact reference to",
    ):
        assert token in manuscript
