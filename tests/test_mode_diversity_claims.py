"""Every PMD number in the manuscript must still follow from the payloads.

These payloads are regenerated often -- a renderer edit, a rename, a support-bar
change -- and each regeneration can silently move a number the prose states in
words. The manuscript assertions here are the tripwire: if a rebuild changes a
reported quantity, the sentence quoting it fails rather than going stale.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper/main.tex"
MODELS = ("05b", "3b", "7b", "14b")
# The appendix scale table's own ladder and levels. build_mode_diversity_table
# counts the scale trend over these, so a claim about that count is only
# reproducible when this test reads the same scope.
SCALE_LADDER = ("05b", "3b", "7b", "14b", "qwen32b", "qwen72b")
SCALE_LEVELS = ("level1", "level2", "level3", "level4")


def _payload(name: str) -> dict:
    return json.loads((ROOT / "paper/results" / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def grid() -> dict:
    return _payload("mode_diversity_base_grid.json")


@pytest.fixture(scope="module")
def training() -> dict:
    return _payload("mode_diversity_training.json")


@pytest.fixture(scope="module")
def hosted() -> dict:
    return _payload("mode_diversity_hosted.json")


@pytest.fixture(scope="module")
def cohort() -> dict:
    return _payload("mode_diversity_hosted_cohort.json")


@pytest.fixture(scope="module")
def manuscript() -> str:
    return " ".join(MANUSCRIPT.read_text(encoding="utf-8").split())


def _reportable(grid: dict) -> list[dict]:
    return [cell for cell in grid["cells"] if cell["reportable"]]


def test_support_counts_in_the_prose_are_generated_not_hand_written(grid):
    """The grid grows as cells finish, so the prose reads these from the payload.

    Asserting fixed totals here would just relocate the staleness into the test;
    what must hold is that the generated macros agree with the payload and that
    the manuscript spends them rather than hard-coding numbers beside them.
    """
    import re
    coverage = (ROOT / "paper/results/mode_diversity_coverage.tex").read_text(encoding="utf-8")
    macros = dict(re.findall(r"\\newcommand\{\\(MD\w+)\}\{([^}]*)\}", coverage))
    cells = grid["cells"]
    reportable = _reportable(grid)
    assert int(macros["MDcells"]) == len(cells)
    assert int(macros["MDreportable"]) == len(reportable)
    assert int(macros["MDgaps"]) == len(cells) - len(reportable)

    body = MANUSCRIPT.read_text(encoding="utf-8")
    assert r"\input{results/mode_diversity_coverage.tex}" in body
    # MDgaps is generated for the record but the prose no longer quotes it;
    # what must hold is that every count the prose does quote is a macro.
    for macro in ("MDcells", "MDreportable", "MDdistinctcorr", "MDpmdcorr"):
        assert "\\" + macro in body, f"{macro} is generated but never used"


def test_scale_raises_correctness_and_lowers_diversity(grid, manuscript):
    """The headline reversal: bigger models move right, not up."""
    reportable = _reportable(grid)
    means = {
        model: (
            statistics.fmean(c["pass8"] for c in reportable if c["model_label"] == model),
            statistics.fmean(c["pmd"] for c in reportable if c["model_label"] == model),
        )
        for model in MODELS
    }
    # The grid is still filling, so the levels move; what must hold is the
    # direction that the caption claims -- larger buys correctness, not breadth.
    assert means["14b"][0] > means["05b"][0], "pass@8 must rise with scale"
    assert means["14b"][1] < means["05b"][1], "PMD must fall with scale"
    # Table 4 is the Qwen table, so its scale claim is scoped to Qwen; pooling
    # families here would compare parameter count across different pretraining.

    declining = total = 0
    for domain in sorted({c["domain"] for c in reportable}):
        for level in SCALE_LEVELS:
            block = {c["model_label"]: c for c in reportable
                     if c["domain"] == domain and c["level"] == level
                     and c["model_label"] in SCALE_LADDER}
            present = [m for m in SCALE_LADDER if m in block]
            if len(present) < 2:
                continue
            total += 1
            declining += block[present[-1]]["pmd"] < block[present[0]]["pmd"]
    # The ladder keeps growing, so the pair is read from the generated macros
    # rather than written here; what this test pins is that they reconstruct
    # from the grid and that the prose quotes them instead of a literal.
    import re
    coverage = (ROOT / "paper/results/mode_diversity_coverage.tex").read_text(encoding="utf-8")
    macros = dict(re.findall(r"\\newcommand\{\\(MD\w+)\}\{([^}]*)\}", coverage))
    assert int(macros["MDscalefall"]) == declining
    assert int(macros["MDscalesupport"]) == total
    assert declining * 2 > total, "the scale trend must still be a majority decline"
    assert r"\MDscalefall" in manuscript and r"\MDscalesupport" in manuscript


def test_python_is_the_most_collapsed_domain_not_the_broadest(grid, manuscript):
    """distinct@8 ranked Python highest; PMD ranks it lowest."""
    reportable = _reportable(grid)
    seven_b = [c for c in reportable if c["model_label"] == "7b"]
    python = [c for c in seven_b if c["domain"] == "python_factors"]
    graph = [c for c in seven_b if c["domain"] == "graph_coloring"]
    import re
    coverage = (ROOT / "paper/results/mode_diversity_coverage.tex").read_text(encoding="utf-8")
    macros = dict(re.findall(r"\\newcommand\{\\(MD\w+)\}\{([^}]*)\}", coverage))
    def shown(name):
        return float("0" + macros[name]) if macros[name].startswith(".") else float(macros[name])
    assert round(min(c["pass8"] for c in python), 3) == shown("MDsevenbPypasslo")
    assert round(max(c["pmd"] for c in python), 3) == shown("MDsevenbPypmdhi")
    assert max(c["pmd"] for c in python) < min(c["pmd"] for c in graph)
    assert max(c["distinct8"] for c in graph) < max(c["distinct8"] for c in python), (
        "the reversal only matters while distinct@8 still ranks Python above Graph")
    for macro in ("MDsevenbPypasslo", "MDsevenbPypmdhi", "MDsevenbGraphpmdlo"):
        assert "\\" + macro in MANUSCRIPT.read_text(encoding="utf-8"), (
            f"{macro} is generated but the caption no longer spends it")


def test_replay_beats_its_control_on_the_conditional_axis(training, manuscript):
    arms = {(a["level"], a["scale"], a["domain"], a["method"]): a for a in training["arms"]}
    wins = measurable = 0
    for base, replay in (("drgrpo", "replay_drgrpo"), ("maxrl", "replay_maxrl")):
        for level, scale, domain, method in list(arms):
            if method != base:
                continue
            control = arms[(level, scale, domain, base)]
            treated = arms.get((level, scale, domain, replay))
            if treated is None or not (control["terminal_reportable"]
                                       and treated["terminal_reportable"]):
                continue
            measurable += 1
            wins += treated["pmd_after"] > control["pmd_after"]
    # The payload grows as excluded cells are recovered, so the count is read
    # back from the prose rather than frozen here; what must hold is that the
    # sentence names the number of comparisons replay does not win.
    losses = measurable - wins
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    assert measurable >= 31, "the conditional-axis comparison lost coverage"
    assert wins * 5 > measurable * 4, "replay must still win the large majority"
    assert f"all but {words[losses]} of the comparisons" in manuscript


def test_verifier_only_training_collapses_diversity_at_every_scale(training):
    """The precheck percentages quoted beside the extra-mode declines."""
    declines = {}
    for scale in ("qwen05b", "falcon1b", "qwen3b"):
        for method in ("grpo", "drgrpo"):
            arms = [a for a in training["arms"]
                    if a["scale"] == scale and a["method"] == method and a["reportable"]]
            if not arms:
                continue
            before = statistics.fmean(a["pmd_before"] for a in arms)
            after = statistics.fmean(a["pmd_after"] for a in arms)
            declines[(scale, method)] = round((before - after) / before * 100, 1)
    assert declines[("qwen05b", "grpo")] == 98.5
    assert declines[("qwen05b", "drgrpo")] == 99.8
    larger = [v for (scale, _), v in declines.items() if scale != "qwen05b"]
    assert (min(larger), max(larger)) == (50.8, 65.1)
    assert all(value > 0 for value in declines.values()), "every arm must decline"


def test_collapse_universals_hold_without_exception(training):
    """Two universals the paper relies on, whether or not the abstract spells
    them out: PantryPlan concentrates under every verifier-only objective at
    every scale, and Graph concentrates under Dr.GRPO at every scale."""
    arms = {(a["level"], a["scale"], a["domain"], a["method"]): a for a in training["arms"]}
    scales = ("qwen05b", "falcon1b", "qwen3b")

    pantry = [arms[("level1", scale, "pantry_plan", method)]
              for scale in scales for method in ("grpo", "drgrpo", "maxrl")
              if arms.get(("level1", scale, "pantry_plan", method), {}).get("reportable")]
    assert len(pantry) == 9, "the claim covers every scale and verifier-only objective"
    assert all(a["pmd_delta"] < 0 for a in pantry), "PantryPlan must fall in all nine"

    graph = [arms[("level1", scale, "graph_coloring", "drgrpo")] for scale in scales]
    assert all(a["reportable"] and a["pmd_delta"] < 0 for a in graph)

    # "reaching zero at 0.5B and 1B" is a literal zero, not a rounded one.
    for scale in ("qwen05b", "falcon1b"):
        zeroed = [a for a in pantry if a["scale"] == scale and a["pmd_after"] == 0.0]
        assert zeroed, f"no PantryPlan arm reaches exactly zero at {scale}"




def test_hosted_deployments_are_all_concentrated(hosted, cohort, manuscript):
    cells = {(c["level"], c["domain"]): c for c in hosted["cells"]}
    assert round(cells[(3, "graph_coloring")]["pmd"], 3) == 0.049
    assert cells[(2, "mathir")]["pmd"] == 0.0 and cells[(3, "mathir")]["pmd"] == 0.0
    assert cells[(2, "python_factors")]["defined_prompts"] == 14
    assert cells[(3, "python_factors")]["defined_prompts"] == 15

    macros = [m["macro_pmd"] for m in cohort["models"]]
    assert len(macros) == 7
    assert (round(min(macros), 3), round(max(macros), 3)) == (0.142, 0.311)
    assert all(1 / (1 - value) < 1.5 for value in macros), "none may reach 1.5 effective modes"
    for token in ("$.142$", "$.311$", "$1.5$"):
        assert token in manuscript, f"{token} no longer appears in the manuscript"


def test_pooled_hosted_estimate_still_matches_the_published_collision(hosted):
    """The builder's cross-check, asserted here so a rebuild cannot drop it."""
    checked = 0
    for cell in hosted["cells"]:
        published = cell["published_correct_pair_collision"]
        if published is None or cell["pmd_pair_pooled"] is None:
            continue
        assert abs((1 - published) - cell["pmd_pair_pooled"]) < 1e-9
        checked += 1
    assert checked >= 10
