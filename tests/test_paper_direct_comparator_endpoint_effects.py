from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "paper/figures/direct_comparator_endpoint_effects.json"
PLOTTER = ROOT / (
    "ops/exp_scaling/plot_paper_direct_comparator_endpoint_effects.py"
)
PAPER = ROOT / "paper/main.tex"


def _cells() -> dict[tuple[str, str], dict]:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    return {
        (cell["model"], cell["domain"]): cell
        for cell in payload["cells"]
    }


def test_forest_uses_matched_drgrpo_and_only_five_static_domains() -> None:
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    assert payload["baseline"] == "matched Dr.GRPO"
    assert payload["model_rows"] == [
        "Qwen2.5-0.5B",
        "Falcon3-1B",
        "Qwen2.5-3B",
    ]
    assert payload["domain_order"] == [
        "graph_coloring",
        "countdown",
        "python_factors",
        "mathir",
        "pantry_plan",
    ]
    assert all("point" not in domain for domain in payload["domain_order"])


def test_requested_complete_blocks_and_small_prefix_are_present() -> None:
    cells = _cells()
    qwen_graph_ucpo = cells[("Qwen2.5-0.5B", "graph_coloring")][
        "methods"
    ]["ucpo"]
    falcon_python = cells[("Falcon3-1B", "python_factors")]["methods"]
    assert qwen_graph_ucpo["n"] == 5
    assert qwen_graph_ucpo["seeds"] == [43, 44, 45, 46, 47]
    assert falcon_python["grpo"]["n"] == 5
    assert falcon_python["grpo"]["seeds"] == [55, 56, 57, 58, 59]
    assert falcon_python["rlep_dr"]["n"] == 2
    assert falcon_python["rlep_dr"]["seeds"] == [55, 57]
    qwen_countdown_ucpo = cells[("Qwen2.5-0.5B", "countdown")][
        "methods"
    ]["ucpo"]
    assert qwen_countdown_ucpo["n"] == 5
    assert qwen_countdown_ucpo["seeds"] == [43, 44, 45, 46, 47]
    assert qwen_countdown_ucpo["evidence"] == "balanced_five_seed_terminal"
    assert qwen_countdown_ucpo["summaries"]["adjusted_breadth8"]["student_t_95"][0] > 0
    qwen_mathir_ucpo = cells[("Qwen2.5-0.5B", "mathir")]["methods"]["ucpo"]
    assert qwen_mathir_ucpo["n"] == 5
    assert qwen_mathir_ucpo["seeds"] == [43, 44, 45, 46, 47]
    assert qwen_mathir_ucpo["evidence"] == "balanced_five_seed_terminal"
    assert set(qwen_mathir_ucpo["summaries"]) == {
        "pass8",
        "adjusted_breadth8",
    }
    qwen3_graph_grpo = cells[("Qwen2.5-3B", "graph_coloring")][
        "methods"
    ]["grpo"]
    assert qwen3_graph_grpo["n"] == 5
    assert qwen3_graph_grpo["seeds"] == [70, 71, 72, 73, 74]
    assert qwen3_graph_grpo["evidence"] == "balanced_five_seed_terminal"
    assert set(qwen3_graph_grpo["summaries"]) == {
        "pass8",
        "adjusted_breadth8",
    }
    qwen_countdown_rlep = cells[("Qwen2.5-0.5B", "countdown")][
        "methods"
    ]["rlep_dr"]
    assert qwen_countdown_rlep["n"] == 5
    assert qwen_countdown_rlep["seeds"] == [43, 44, 45, 46, 47]
    qwen3_countdown_grpo = cells[("Qwen2.5-3B", "countdown")][
        "methods"
    ]["grpo"]
    assert qwen3_countdown_grpo["n"] == 5
    assert qwen3_countdown_grpo["seeds"] == [70, 71, 72, 73, 74]
    qwen3_python_grpo = cells[("Qwen2.5-3B", "python_factors")][
        "methods"
    ]["grpo"]
    assert qwen3_python_grpo["n"] == 5
    assert qwen3_python_grpo["seeds"] == [70, 71, 72, 73, 74]
    assert qwen3_python_grpo["evidence"] == "balanced_five_seed_terminal"
    assert set(qwen3_python_grpo["summaries"]) == {
        "pass8",
        "adjusted_breadth8",
    }
    assert qwen3_python_grpo["per_seed"]["72"]["effect"] == {
        "pass8": 0.0,
        "adjusted_breadth8": 0.0,
    }
    qwen3_mathir_grpo = cells[("Qwen2.5-3B", "mathir")]["methods"]["grpo"]
    assert qwen3_mathir_grpo["n"] == 5
    assert qwen3_mathir_grpo["seeds"] == [70, 71, 72, 73, 74]
    assert qwen3_mathir_grpo["evidence"] == "balanced_five_seed_terminal"
    assert set(qwen3_mathir_grpo["summaries"]) == {
        "pass8",
        "adjusted_breadth8",
    }
    falcon_pantry_rlep = cells[("Falcon3-1B", "pantry_plan")]["methods"][
        "rlep_dr"
    ]
    assert falcon_pantry_rlep["n"] == 5
    assert falcon_pantry_rlep["seeds"] == [55, 56, 57, 58, 59]
    assert falcon_pantry_rlep["evidence"] == "balanced_five_seed_terminal"


def test_prefixes_have_raw_pairs_only_and_balanced_blocks_have_intervals() -> None:
    cells = _cells()
    complete = cells[("Falcon3-1B", "python_factors")]["methods"]["grpo"]
    prefix = cells[("Falcon3-1B", "python_factors")]["methods"]["rlep_dr"]
    assert set(complete["summaries"]) == {"pass8", "adjusted_breadth8"}
    assert "summaries" not in prefix
    assert prefix["evidence"] == "exact_terminal_prefix"


def test_plot_and_paper_use_the_existing_endpoint_forest_grammar() -> None:
    plotter = PLOTTER.read_text(encoding="utf-8")
    paper = PAPER.read_text(encoding="utf-8")
    paper_words = " ".join(paper.split())
    assert "mirrors ``cross_scale_terminal_endpoint_effects``" in plotter
    assert 'f"n: {counts}"' in plotter
    assert "diamonds and intervals " in plotter
    assert "require all five registered seeds" in plotter
    assert "{figures/direct_comparator_endpoint_effects.pdf}" in paper
    # Coverage now lives in the consolidated comparator table; the figure
    # caption states the pairing/interval rules instead of repeating the table.
    for method, coverage in (("GRPO", "75/75"), ("UCPO", "50/50"),
                             ("Sparse RLEP-Dr", "47/50")):
        row = re.search(r"\n\s*" + re.escape(method) + r"\s*&(?P<row>.*?)(?=\\\\)",
                        paper, flags=re.DOTALL)
        assert row is not None
        assert coverage in row.group("row")
        if method != "GRPO":
            assert "Level 1, 0.5B and 1B" in row.group("row")
    assert "same-seed Dr.GRPO control" in paper_words
    assert "intervals appear only for complete five-seed blocks" in paper_words
    assert "unsupported comparisons remain blank" in paper_words
    assert "full five-seed 3B plain-GRPO blocks likewise have intervals including zero" in paper_words
    assert "Countdown UCPO raises" in paper_words
    assert "its MathIR effects are inconclusive" in paper_words
    assert "Falcon Pantry effects are inconclusive" in paper_words
