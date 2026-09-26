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
# reproducible when this test reads the same scope. Both axes grow as the grid
# fills -- qwen15b and Level 5 each arrived after this test was first written --
# so they are resolved from the builder rather than repeated here, which would
# relocate the staleness into the test exactly as the docstring warns.
def _scale_axes() -> tuple[tuple[str, ...], tuple[str, ...]]:
    import sys
    sys.path.insert(0, str(ROOT / "ops"))
    import build_mode_diversity_table as builder
    return builder.grid_axes(_payload("mode_diversity_base_grid.json")["cells"])


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
    # The 2026-09-17 payload carries both readings: macro_pmd over whatever
    # cells a deployment reports, and macro_pmd_common_cells over the thirteen
    # every deployment reports. The prose quotes the second, so the test has to
    # read the second; the older file has only the first.
    return _payload("mode_diversity_hosted_cohort_20260917.json")


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

    # The gap composition and the definedness correlation are quoted in the
    # support paragraph, so they are pinned to the payload like the counts are.
    from collections import Counter
    gaps = Counter(c["domain"] for c in cells if not c["reportable"])
    assert int(macros["MDtopgapcount"]) == gaps.most_common(1)[0][1]

    def _corr(xs, ys):
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** .5
        return num / den

    def _fmt(v):
        text = f"{v:.3f}"
        return text.replace("0.", ".", 1) if text.startswith(("0.", "-0.")) else text

    defined = _corr([c["pass8"] for c in cells],
                    [c["defined_prompts"] / c["prompts"] for c in cells])
    assert macros["MDdefinedcorr"] == _fmt(defined)

    body = MANUSCRIPT.read_text(encoding="utf-8")
    assert r"\input{results/mode_diversity_coverage.tex}" in body
    # Every count and coefficient the prose quotes must be a macro, so a rebuild
    # moves the sentence with the payload instead of leaving it stale.
    for macro in ("MDcells", "MDreportable", "MDgaps", "MDtopgapcount",
                  "MDdistinctcorr", "MDpmdcorr", "MDdefinedcorr"):
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

    scale_ladder, scale_levels = _scale_axes()
    declining = total = 0
    for domain in sorted({c["domain"] for c in reportable}):
        for level in scale_levels:
            block = {c["model_label"]: c for c in reportable
                     if c["domain"] == domain and c["level"] == level
                     and c["model_label"] in scale_ladder}
            present = [m for m in scale_ladder if m in block]
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
    assert measurable >= 31, "the conditional-axis comparison lost coverage"
    assert wins * 5 > measurable * 4, "replay must still win the large majority"

    # The prose used to spell this out as "all but three of the comparisons".
    # It now states it through \PMDabove/\PMDblocks, which are generated by
    # build_pmd_retention_matrix.py on the paired support bar of 20 rather than
    # the whole-cell bar of 30 counted above, so the two totals differ by
    # construction and neither is a restatement of the other. What must hold is
    # that the sentence still spends the macros instead of a frozen number.
    import re
    macros = dict(re.findall(r"\\newcommand\{\\(PMD\w+)\}\{([^}]*)\}",
                             (ROOT / "paper/results/pmd_retention_macros.tex")
                             .read_text(encoding="utf-8")))
    above, blocks = int(macros["PMDabove"]), int(macros["PMDblocks"])
    assert 0 < above <= blocks, "the retention macros are not a sub-count"
    assert above * 5 > blocks * 4, "replay must still win the large majority there too"
    for macro in ("PMDabove", "PMDblocks"):
        assert "\\" + macro in MANUSCRIPT.read_text(encoding="utf-8"), (
            f"{macro} is generated but the prose no longer spends it")


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

    # The manuscript reports the common-cell macro, because averaging each
    # deployment over whatever it can report ranks them against different sets
    # of cells. Both readings are checked; only the quoted one is asserted to
    # appear in the prose.
    common = [m["macro_pmd_common_cells"] for m in cohort["models"]]
    own = [m["macro_pmd"] for m in cohort["models"]]
    assert len(common) == 7 and len(own) == 7
    assert all(1 / (1 - value) < 1.5 for value in common), "none may reach 1.5 effective modes"
    assert all(1 / (1 - value) < 1.5 for value in own), "nor on the per-deployment reading"
    lo, hi = round(min(common), 3), round(max(common), 3)
    for token in (f"${lo:.3f}$".replace("0.", "."), f"${hi:.3f}$".replace("0.", "."), "$1.5$"):
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
