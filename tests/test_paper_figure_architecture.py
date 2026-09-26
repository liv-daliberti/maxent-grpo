"""Contracts connecting the experiment matrix to separated paper figures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
EXP = OPS / "exp_scaling"


def load(name: str, path: Path):
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_every_paper_method_has_one_visual_identity():
    matrix = load("paper_matrix_for_style_test", EXP / "paper_matrix.py")
    visuals = load("paper_method_style_under_test", OPS / "paper_method_style.py")

    assert set(visuals.METHOD_STYLE) == {
        method.key for method in matrix.METHODS
    }
    for key, visual in visuals.METHOD_STYLE.items():
        assert visual["label"]
        assert visual["color"]
        assert visual["linestyle"]
        assert visual["marker"]
        assert visuals.method_style(key) is not visual
    assert visuals.METHOD_STYLE["grpo"]["label"] == "GRPO"
    assert visuals.METHOD_STYLE["drgrpo"]["label"] == "Dr.GRPO"
    assert (
        visuals.METHOD_STYLE["grpo"]["color"]
        != visuals.METHOD_STYLE["drgrpo"]["color"]
    )
    assert (
        visuals.METHOD_STYLE["replay_grpo"]["label"]
        == "ReplayDr.GRPO (ours)"
    )
    assert len({item["label"] for item in visuals.METHOD_STYLE.values()}) == 10


def test_legacy_preview_chunks_never_exceed_three_panels():
    figures = load(
        "paper_comparison_figures_under_test",
        EXP / "plot_paper_comparison_families.py",
    )
    snapshot = {
        "domains": [
            "graph_coloring",
            "countdown",
            "python_factors",
            "mathir",
            "pantry_plan",
        ]
    }
    core = figures.COMPARISON_BY_KEY["core_retention"]
    chunks = figures.domain_chunks(snapshot, core, 3)

    assert [len(chunk) for chunk in chunks] == [3, 2]
    assert chunks[-1] == ["mathir", "pantry_plan"]


def test_comparisons_use_only_the_five_static_domains():
    figures = load(
        "paper_comparison_figures_static_domain_test",
        EXP / "plot_paper_comparison_families.py",
    )
    snapshot = {"domains": list(figures.STATIC_DOMAIN_COLUMNS)}
    assert all(not hasattr(comparison, "include_tour") for comparison in figures.COMPARISONS)
    for comparison in figures.COMPARISONS:
        chunks = figures.domain_chunks(snapshot, comparison, 3)
        assert [domain for chunk in chunks for domain in chunk] == list(
            figures.STATIC_DOMAIN_COLUMNS
        )


def terminal_cells(figures, comparison_key: str, domain: str):
    comparison = figures.COMPARISON_BY_KEY[comparison_key]
    scale = figures.SCALE_BY_KEY["qwen05b"]
    return {
        figures.matrix.CellKey(method, scale.key, domain, seed): SimpleNamespace(
            status="terminal"
        )
        for method in comparison.methods
        for seed in figures.matrix.SCALE_BY_KEY[scale.key].seeds
    }


def test_available_mode_has_no_minimum_seed_or_method_gate():
    figures = load(
        "paper_comparison_figures_terminal_test",
        EXP / "plot_paper_comparison_families.py",
    )
    comparison = figures.COMPARISON_BY_KEY["core_retention"]
    scale = figures.SCALE_BY_KEY["qwen05b"]
    cells = terminal_cells(figures, comparison.key, "graph_coloring")

    assert figures.terminal_domains(cells, scale, comparison) == [
        "graph_coloring"
    ]
    cells.pop(next(iter(cells)))
    assert figures.terminal_domains(cells, scale, comparison) == [
        "graph_coloring"
    ]
    cells.clear()
    assert figures.terminal_domains(cells, scale, comparison) == []


def test_terminal_factorial_requires_all_four_canonical_arms():
    figures = load(
        "paper_comparison_figures_factorial_test",
        EXP / "plot_paper_comparison_families.py",
    )
    comparison = figures.COMPARISON_BY_KEY["fixed_semantic_factorial"]
    scale = figures.SCALE_BY_KEY["qwen05b"]
    cells = terminal_cells(figures, comparison.key, "pantry_plan")

    assert comparison.methods == (
        "drgrpo",
        "semantic_maxent",
        "replay_grpo",
        "replay_semantic_maxent",
    )
    assert figures.terminal_domains(cells, scale, comparison) == ["pantry_plan"]


def test_cross_scale_factorial_terminal_availability_is_complete():
    figures = load(
        "paper_comparison_figures_cross_scale_test",
        EXP / "plot_paper_comparison_families.py",
    )
    comparison = figures.COMPARISON_BY_KEY["fixed_semantic_factorial"]
    cells = {}
    terminal = {
        "qwen05b": (
            "graph_coloring",
            "countdown",
            "python_factors",
            "mathir",
            "pantry_plan",
        ),
        "falcon1b": ("graph_coloring", "mathir"),
    }
    for scale_key, domains in terminal.items():
        for domain in domains:
            for method in comparison.methods:
                for seed in figures.matrix.SCALE_BY_KEY[scale_key].seeds:
                    key = figures.matrix.CellKey(method, scale_key, domain, seed)
                    cells[key] = SimpleNamespace(status="terminal")
    for domain in figures.STATIC_DOMAIN_COLUMNS:
        for method in ("drgrpo", "replay_grpo", "replay_semantic_maxent"):
            key = figures.matrix.CellKey(method, "qwen3b", domain, 70)
            cells[key] = SimpleNamespace(status="terminal")

    rows = figures.fixed_semantic_cross_scale_rows(cells)
    assert [(scale.key, domains) for scale, domains in rows] == [
        (
            "qwen05b",
            [
                "graph_coloring",
                "countdown",
                "python_factors",
                "mathir",
                "pantry_plan",
            ],
        ),
        ("falcon1b", ["graph_coloring", "mathir"]),
        (
            "qwen3b",
            [
                "graph_coloring",
                "countdown",
                "python_factors",
                "mathir",
                "pantry_plan",
            ],
        ),
    ]
    assert figures.STATIC_DOMAIN_COLUMNS == (
        "graph_coloring",
        "countdown",
        "python_factors",
        "mathir",
        "pantry_plan",
    )
    source = (EXP / "plot_paper_comparison_families.py").read_text()
    assert "figure.add_gridspec(\n        len(MODEL_ORDER)," in source
    assert figures.MODEL_ORDER == ("qwen05b", "falcon1b", "qwen3b")
    assert '"blank_rule"' in source


def test_compiled_static_trajectories_use_domain_columns_and_model_rows():
    strips = load(
        "paper_aligned_domain_strips_test",
        EXP / "plot_paper_aligned_domain_strips.py",
    )
    assert strips.DOMAIN_ORDER == (
        "graph_coloring",
        "countdown",
        "python_factors",
        "mathir",
        "pantry_plan",
    )
    assert set(strips.OUTPUTS) == {
        "fixed", "core_falcon", "core_pass8", "core_mean8",
        "adaptive", "replay_dose", "ucpo",
    }
    assert strips.MODEL_ORDER == ("qwen05b", "falcon1b", "qwen3b")
    source = (EXP / "plot_paper_aligned_domain_strips.py").read_text()
    assert "len(model_order), len(DOMAIN_ORDER)" in source
    assert "model_order: tuple[str, ...] = MODEL_ORDER" in source
    assert strips.DIRECT_MODEL_ORDER == ("qwen05b", "falcon1b")
    assert "1, len(DOMAIN_ORDER)" not in source
    assert "def strip_axes(" not in source
    assert "pending(" not in source
    assert "awaiting scientific checkpoint" not in source
    assert 'style.style_axis(\n                axis, grid="both"' in source
    assert "RLEP_LEDGER" in source
    assert '"rlep_dr"' in source
    frontier = load(
        "paper_inference_frontier_row_order_test",
        EXP / "plot_paper_inference_frontier.py",
    )
    assert tuple(scale["key"] for scale in frontier.SCALE_SPECS) == (
        "qwen05b", "falcon1b", "qwen3b",
    )
    assert frontier.SCALE_SPECS[2]["control_ledger"].name == (
        "e80r1_qwen3b_aligned_verified_replay_jobs.json"
    )
    assert frontier.SCALE_SPECS[2]["xmode_ledger"].name == (
        "e92_qwen3b_adaptive_semantic_maxent_jobs.json"
    )

    manuscript = (ROOT / "paper/main.tex").read_text()
    # The supporting direct family has modes and accuracy views; old semantic
    # and campaign-monitor strips remain provenance outside the manuscript.
    compiled = (
        "direct_baseline_learning_curves_static_strip.pdf",
        "direct_baseline_learning_curves_pass8.pdf",
    )
    assert set(strips.DIRECT_OUTPUTS) == {"distinct8", "pass8"}
    assert all(manuscript.count(stem) == 1 for stem in compiled)
    for obsolete in (
        "core_retention_falcon1b_part1.pdf",
        "core_retention_falcon1b_part2.pdf",
        "fixed_semantic_factorial_cross_scale.pdf",
        "adaptive_semantic_replay_qwen05b_part1.pdf",
        "adaptive_semantic_replay_falcon1b_part1.pdf",
        "replay_dose_qwen05b_progress_20260813_part1.pdf",
        "replay_dose_qwen05b_progress_20260813_part2.pdf",
        "ucpo_interim_learning_curves.pdf",
    ):
        assert obsolete not in manuscript


def test_cross_scale_endpoint_forest_keeps_every_available_pair_row():
    effects = load(
        "paper_cross_scale_endpoint_effects_test",
        EXP / "plot_paper_cross_scale_endpoint_effects.py",
    )
    def arm(value, n=5):
        return {
            "per_seed": {
                str(seed): {
                    "pass8": value + seed / 10000,
                    "distinct8": value + 0.5 + seed / 10000,
                }
                for seed in range(n)
            },
        }
    families = {}
    for model, domain_count in (
        ("Qwen2.5-0.5B", 5), ("Falcon3-1B", 5), ("Qwen2.5-3B", 0)
    ):
        families[model] = {"domains": {}}
        for domain in effects.DOMAIN_ORDER[:domain_count]:
            families[model]["domains"][domain] = {
                "training_pass": 8.0,
                "methods": {"control": arm(0.1), "replay": arm(0.4)},
            }
    # The four-seed Qwen-3B cell is retained with its exact denominator.
    families["Qwen2.5-3B"]["domains"]["graph_coloring"] = {
        "training_pass": 8.0,
        "methods": {"control": arm(0.1, 4), "replay": arm(0.4, 4)},
    }

    rows = effects._terminal_rows(
        {"schema": "paper-core-terminal-endpoints-v1", "models": families}
    )
    assert len(rows) == 11
    assert [row["model"] for row in rows] == [
        "Qwen2.5-0.5B"
    ] * 5 + ["Falcon3-1B"] * 5 + ["Qwen2.5-3B"]
    assert [len(row["seeds"]) for row in rows] == [5] * 10 + [4]
    assert effects.MODELS == ("Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B")
    assert all(
        set(row["per_seed_endpoints"]) == {str(seed) for seed in row["seeds"]}
        for row in rows
    )
    assert effects.DEFAULT_FRONTIER_OUTPUT.name == (
        "terminal_pass8_distinct8_frontier"
    )
    source = (EXP / "plot_paper_cross_scale_endpoint_effects.py").read_text()
    assert "len(MODELS),\n        len(DOMAIN_ORDER)," in source
    assert '"blank_rule"' in source
    assert '"x_metric": "pass@8"' in source
    assert '"y_metric": "distinct@8"' in source

    manuscript = (ROOT / "paper/main.tex").read_text()
    results = manuscript.split(r"\section{Results}", 1)[1].split(
        r"\section{Conclusion}", 1
    )[0]
    # The omnibus frontier is retained provenance. The compiled main result
    # now separates retention from the supporting direct-comparator forest.
    assert "figures/experiment1_retention_comparator_matrix.pdf" in results
    assert r"\label{fig:cross-scale-terminal-effects}" in results
    assert "figures/terminal_pass8_distinct8_frontier.pdf" not in manuscript
    assert manuscript.index("figures/direct_comparator_endpoint_effects.pdf") > manuscript.index(
        r"\appendix"
    )




def test_plain_grpo_reports_every_available_falcon_block():
    effects = load(
        "paper_cross_scale_endpoint_effects_grpo_test",
        EXP / "plot_paper_cross_scale_endpoint_effects.py",
    )
    core = json.loads(
        (ROOT / "paper/results/core_terminal_endpoints.json").read_text()
    )
    plain_grpo = json.loads(
        (
            ROOT / "paper/results/e95_falcon_plain_grpo_reportable.json"
        ).read_text()
    )
    rows = effects._terminal_rows(core, plain_grpo)
    grpo_rows = [
        row for row in rows if row["grpo_per_seed_effects"]
    ]
    assert [
        (row["model"], row["domain"]) for row in grpo_rows
    ] == [
        ("Falcon3-1B", domain) for domain in effects.DOMAIN_ORDER
    ]
    # This historical three-method overlay uses the replay/control row's
    # paired seeds for effects, while retaining independently valid endpoints.
    assert all(
        set(row["endpoint_summaries"]) == {"grpo", "control", "replay"}
        and set(row["grpo_per_seed_effects"]) == {
            seed for seed, endpoint in row["per_seed_endpoints"].items()
            if "grpo" in endpoint and int(seed) in row["seeds"]
        }
        for row in grpo_rows
    )
    countdown = next(row for row in grpo_rows if row["domain"] == "countdown")
    assert countdown["seeds"] == [55, 56, 57, 58]
    assert set(countdown["per_seed_endpoints"]["59"]) == {"control", "grpo"}
    assert countdown["endpoint_summaries"]["grpo"]["pass8"]["n"] == 5
    assert countdown["endpoint_summaries"]["replay"]["pass8"]["n"] == 4

    # The current compiled direct comparison uses all five valid GRPO/control
    # pairs; a ReplayDr.GRPO exclusion does not remove the independent GRPO pair.
    direct = json.loads((ROOT / "paper/figures/direct_comparator_endpoint_effects.json").read_text())
    falcon = [cell for cell in direct["cells"] if cell["model"] == "Falcon3-1B"]
    assert len(falcon) == 5
    assert all(cell["methods"]["grpo"]["seeds"] == [55, 56, 57, 58, 59]
               for cell in falcon)


def test_main_frontier_uses_requested_methods_and_exact_partial_prefixes():
    effects = load(
        "paper_cross_scale_endpoint_frontier_test",
        EXP / "plot_paper_cross_scale_endpoint_effects.py",
    )
    core = json.loads(
        (ROOT / "paper/results/core_terminal_endpoints.json").read_text()
    )
    plain_grpo = json.loads(
        (ROOT / "paper/results/e95_falcon_plain_grpo_reportable.json").read_text()
    )
    direct_comparators = json.loads(
        (
            ROOT / "paper/figures/direct_comparator_endpoint_effects.json"
        ).read_text()
    )
    rows = effects._terminal_rows(core, plain_grpo)
    cells = effects._frontier_cells(rows, direct_comparators)
    by_key = {
        (cell["model"], cell["domain"]): cell for cell in cells
    }

    assert len(cells) == 15
    assert effects.FRONTIER_METHODS == (
        "drgrpo",
        "grpo",
        "replay_grpo",
        "ucpo",
        "rlep_dr",
    )
    current_methods = set(effects.FRONTIER_METHODS)
    for model in ("Qwen2.5-0.5B", "Falcon3-1B"):
        assert all(
            set(by_key[(model, domain)]["methods"]) == current_methods
            for domain in effects.DOMAIN_ORDER
        )

    qwen3_methods = {"drgrpo", "grpo", "replay_grpo"}
    for domain in effects.DOMAIN_ORDER:
        cell = by_key[("Qwen2.5-3B", domain)]
        assert set(cell["methods"]) == qwen3_methods
        assert set(cell["blank_methods"]) == {"ucpo", "rlep_dr"}
        for method in qwen3_methods:
            assert cell["methods"][method]["seeds"] == [70, 71, 72, 73, 74]

    falcon_python = by_key[("Falcon3-1B", "python_factors")]["methods"]
    assert falcon_python["rlep_dr"]["seeds"] == [55, 57]
    assert falcon_python["rlep_dr"]["n"] == 2
    assert all(
        "summary" in record
        for cell in cells
        for record in cell["methods"].values()
    )
    assert not {
        "semantic_maxent",
        "adaptive_semantic_maxent",
        "replay_semantic_maxent",
        "adaptive_semantic_replay",
    } & current_methods


def test_replay_dose_progress_restricts_to_exact_terminal_subsets():
    progress = load(
        "paper_replay_dose_progress_test",
        EXP / "plot_paper_replay_dose_progress.py",
    )
    selected_seeds = {
        "graph_coloring": [43, 44, 45, 46, 47],
        "countdown": [43, 44, 45, 46, 47],
        "python_factors": [43, 44],
        "mathir": [43, 44, 45, 46, 47],
        "pantry_plan": [43, 44, 45, 46],
    }
    snapshot = {
        "curves": {
            domain: {
                "control": {seed: {8: float(seed)} for seed in range(43, 48)},
                "replay": {seed: {8: float(seed)} for seed in range(43, 48)},
            }
            for domain in selected_seeds
        },
        "semantic_arms": [
            {
                "arm": "bank_normalized_replay",
                "curves": {
                    domain: {
                        seed: {8: float(seed)}
                        for seed in selected_seeds[domain]
                    }
                    for domain in selected_seeds
                },
            }
        ],
    }
    progress._restrict_snapshot(
        snapshot,
        list(selected_seeds),
        selected_seeds,
    )
    for domain, seeds in selected_seeds.items():
        for arm in ("control", "replay"):
            assert list(snapshot["curves"][domain][arm]) == seeds
        assert list(snapshot["semantic_arms"][0]["curves"][domain]) == seeds


def test_replay_mechanism_renderer_requires_the_full_terminal_block():
    telemetry = load(
        "paper_replay_mechanism_telemetry_test",
        EXP / "plot_paper_replay_mechanism_telemetry.py",
    )
    assert telemetry.DOMAINS == (
        "graph_coloring",
        "countdown",
        "python_factors",
        "mathir",
        "pantry_plan",
    )
    assert telemetry.SEEDS == (43, 44, 45, 46, 47)
    assert telemetry.BIN_WIDTH == 0.25


def test_comparison_renderer_uses_canonical_method_identities():
    figures = load(
        "paper_comparison_figures_style_test",
        EXP / "plot_paper_comparison_families.py",
    )
    visuals = load(
        "paper_method_style_for_renderer_test",
        OPS / "paper_method_style.py",
    )

    assert set(figures.PANEL_ARM_STYLE) == set(figures.ARM_TO_METHOD)
    for arm, method in figures.ARM_TO_METHOD.items():
        visual = visuals.method_style(method)
        assert figures.PANEL_ARM_STYLE[arm] == (
            visual["color"],
            visual["linestyle"],
            visual["label"],
        )
