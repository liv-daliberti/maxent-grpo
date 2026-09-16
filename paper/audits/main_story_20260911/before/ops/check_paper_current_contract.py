#!/usr/bin/env python3
"""Fail-closed contract for the current ModeBench/ReplayMaxRL paper story."""
from __future__ import annotations

import hashlib
import json
import re
import runpy
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper/main.tex"
PDF = ROOT / "paper/main.pdf"
MAKEFILE = ROOT / "paper/Makefile"
LINE_FILL_CHECKER = ROOT / "ops/check_paper_line_fill.py"
E118 = ROOT / "paper/figures/e118_all_scale_factorial_progress.json"
LEVELS = ROOT / "paper/figures/modebench_level_admission.json"
FIG4 = ROOT / "paper/figures/experiment1_retention_comparator_matrix.json"
E120 = ROOT / "paper/results/e120_frequency_progress.json"
ABSOLUTE_REFERENCES = ROOT / "paper/results/absolute_support_references.json"
CORE = ROOT / "paper/results/core_terminal_endpoints.json"

MAIN_FIGURES = (
    "modecollapse_story",
    "modebench_examples",
    "verified_support_story",
    "experiment1_retention_comparator_matrix",
    "e118_all_scale_factorial_progress",
    "modebench_level_admission",
)
APPENDIX_FIGURES = (
    "baseline_collapse_precheck",
    "e118_scale_extensions_appendix",
    "direct_comparator_endpoint_effects",
    "direct_baseline_learning_curves_static_strip",
    "factorial_training_curves_pass8",
    "factorial_training_curves_distinct8",
    "level2_factorial_training_curves",
    "direct_baseline_learning_curves_pass8",
    "e121_fixed_bank_survival",
)
RETIRED = (
    "sustained_auc_effects_qwen05b",
    "verified_support_discovery_two_scale_effects",
    "replay_mechanism_telemetry_qwen05b",
    "terminal_pass8_distinct8_frontier",
    "replay_maxrl_qwen05b",
)
DOMAINS = {
    "graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan"
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(f"Current paper contract failed: {message}")


def normalized(text: str) -> str:
    return " ".join(text.split())


def check_training_curves(main_figure: dict) -> None:
    """Reconstruct plotted evidence from the retained draw-level snapshot."""
    path = ROOT / "paper/results/training_curve_snapshot_20260911.json"
    raw = path.read_bytes()
    snapshot = json.loads(raw)
    collector = runpy.run_path(str(ROOT / "ops/exp_scaling/build_paper_training_curve_snapshot.py"))
    validation = collector["validate_snapshot"](snapshot, main_figure)
    require(validation == {"registered_cells": 400, "panels": 20,
                           "terminal_admitted_cells": 375},
            "training snapshot no longer matches the frozen main-paper population")
    for name, expected in snapshot["source_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                f"training snapshot source changed: {name}")
    renderer_path = ROOT / "ops/exp_scaling/plot_paper_training_curves.py"
    renderer = runpy.run_path(str(renderer_path))
    source_hash = hashlib.sha256(raw).hexdigest()
    for level, metric, stem in (
        ("level1", "pass8", "factorial_training_curves_pass8"),
        ("level1", "distinct8", "factorial_training_curves_distinct8"),
        ("level2", None, "level2_factorial_training_curves"),
    ):
        retained = json.loads((ROOT / "paper/figures" / (stem + ".json")).read_text())
        figure, expected = renderer["build_figure"](snapshot, level=level, metric=metric)
        renderer["plt"].close(figure)
        expected.update(source_snapshot=str(path.relative_to(ROOT)), source_sha256=source_hash,
                        builder={"path": str(renderer_path.relative_to(ROOT)),
                                 "sha256": hashlib.sha256(renderer_path.read_bytes()).hexdigest()})
        require(retained == expected,
                f"training figure values, cohorts, gaps, or source binding drifted: {stem}")
    direct_modes = json.loads((ROOT / "paper/figures/direct_baseline_learning_curves_static_strip.json").read_text())
    direct_accuracy = json.loads((ROOT / "paper/figures/direct_baseline_learning_curves_pass8.json").read_text())
    for field in ("cells", "plotted_cells", "numerical_snapshot_sha256",
                  "checkpoint_grid_training_passes", "model_rows"):
        require(direct_modes[field] == direct_accuracy[field],
                f"direct modes/accuracy figures use different {field}")
    require(direct_modes["plot_metric"] == "distinct8"
            and direct_accuracy["plot_metric"] == "pass8"
            and direct_modes["model_rows"] == ["qwen05b", "falcon1b"]
            and set(direct_modes["plotted_cells"]) == {
                f"{scale}/{domain}" for scale in ("qwen05b", "falcon1b") for domain in DOMAINS},
            "UCPO/RLEP training figures must show both metrics at 0.5B and 1B only")


def check_hosted_comparison(appendix: str) -> None:
    stem = "frontier_comparison_20260911"
    include = f"\\input{{results/{stem}_appendix.tex}}"
    require(appendix.count(include) == 1, "hosted comparison appendix is missing or duplicated")
    supplement = (ROOT / "paper/results" / (stem + "_appendix.tex")).read_text()
    require(supplement.count(f"\\input{{results/{stem}_graph_figure.tex}}") == 1,
            "hosted Graph figure include is missing or duplicated")
    graphic = (ROOT / "paper/results" / (stem + "_graph_figure.tex")).read_text()
    require(graphic.count(f"figures/{stem}_graph.pdf") == 1,
            "hosted Graph PDF is missing or duplicated")
    record = json.loads((ROOT / "paper/results" / (stem + ".json")).read_text())
    require(record["completed_model_count"] >= 2, "hosted comparison needs two audited deployments")
    require(all(run.get("provider_outcomes") is not None for run in record["models"]),
            "hosted comparison lacks native refusal/filter outcome audits")
    exporter = runpy.run_path(str(ROOT / "ops/build_frontier_paper_comparison.py"))
    expected = exporter["build_record"](
        [ROOT / run["run_directory"] for run in record["models"]],
        ROOT / record["protocol"]["path"],
        {run["model"]: ROOT / run["provider_outcomes"]["path"] for run in record["models"]},
    )
    require(record == expected, "hosted source, audit, prompt, provider, or graded evidence changed")
    for secondary, suffix in ((False, "strict"), (True, "normalized")):
        require((ROOT / "paper/results" / f"{stem}_{suffix}_overview_rows.tex").read_text()
                == exporter["overview_table"](record, secondary), "hosted level table drifted")
        require((ROOT / "paper/results" / f"{stem}_{suffix}_cells.tex").read_text()
                == exporter["cell_tables"](record, secondary), "hosted domain table drifted")
    figure_path = ROOT / "paper/figures" / (stem + "_graph.pdf")
    figure = json.loads(figure_path.with_suffix(".json").read_text())
    require(figure["pdf_sha256"] == hashlib.sha256(figure_path.read_bytes()).hexdigest(),
            "hosted figure PDF differs from its data record")
    require(figure["models"] == [{"model": run["model"],
                                  "cells": [run["cells"][f"level{level}/graph_coloring"]
                                            for level in (1, 2, 3)]}
                                 for run in record["models"]],
            "hosted figure observations differ from admitted summaries")


def main() -> None:
    manuscript = TEX.read_text(encoding="utf-8")
    main_body, appendix = manuscript.split(r"\appendix", 1)
    makefile = MAKEFILE.read_text(encoding="utf-8")
    check_hosted_comparison(appendix)
    require(LINE_FILL_CHECKER.is_file(), "rendered line-fill checker is missing")
    require(r"\enforcehalffinalproseline" in manuscript
            and r"\AtBeginEnvironment{abstract}{\enforcehalffinalproseline}" in manuscript,
            "TeX paragraph-ending control is missing")
    require("python ../ops/check_paper_line_fill.py --pdf $(PAPER).pdf" in makefile,
            "rendered line-fill checker is not a build gate")

    abstract = manuscript.split(r"\begin{abstract}", 1)[1].split(
        r"\end{abstract}", 1
    )[0]
    abstract_plain = re.sub(r"\\[A-Za-z]+", "", abstract)
    words = re.findall(r"[A-Za-z0-9@.+-]+", abstract_plain)
    require(len(words) <= 275, f"abstract has {len(words)} words")

    ordered = (
        r"\section{Introduction}",
        r"\section{Related Work}",
        r"\section{Correctness and Verified Support}",
        r"\section{\mb: Executable Modes Across Levels}",
        r"\section{ReplayMaxRL: Explore and Preserve Verified Modes}",
        r"\section{Experimental Design}",
        r"\section{Results}",
        r"\section{Conclusion}",
    )
    positions = [manuscript.index(token) for token in ordered]
    require(positions == sorted(positions), "main sections are out of order")

    for stem in MAIN_FIGURES:
        token = f"figures/{stem}.pdf"
        require(main_body.count(token) == 1, f"main figure {stem} not compiled once")
    for stem in APPENDIX_FIGURES:
        token = f"figures/{stem}.pdf"
        require(appendix.count(token) == 1, f"appendix figure {stem} not compiled once")
    for stem in RETIRED:
        require(f"figures/{stem}.pdf" not in manuscript, f"retired figure {stem} returned")

    figure_blocks = re.findall(
        r"\\begin\{figure\}.*?\\end\{figure\}", manuscript, flags=re.DOTALL
    )
    for i, block in enumerate(figure_blocks, 1):
        require(block.index(r"\includegraphics") < block.index(r"\caption{"),
                f"figure {i} caption precedes its image")

    results = manuscript.split(r"\section{Results}", 1)[1].split(
        r"\long\def\crossscaletablematerial", 1
    )[0]
    result_headings = (
        r"\subsection{Experiment 1:",
        r"\subsection{Experiment 2:",
        r"\subsection{Experiment 3:",
    )
    rpos = [results.index(x) for x in result_headings]
    require(rpos == sorted(rpos), "Results do not follow Experiments 1--3")
    # Each experiment leads with its finding; repeated closing recaps are not
    # required after the reader-first prose cut.
    for n, finding in enumerate((
        "Replay improves success and sampled support across scales.",
        "Stronger fresh-rollout optimization does not eliminate the value of memory at either completed scale.",
        "Replay improves both terminal means at both difficulty levels.",
    ), 1):
        require(normalized(finding) in normalized(results),
                f"Experiment {n} lacks its reader-facing finding")

    # The precheck is a figure over all three scales, not a Qwen-0.5B table, so
    # the contract is now on the claim rather than on six typeset numbers: the
    # scale-specific removal figures have to keep matching the frozen builder
    # output, and the two statements that stop the figure being read as a
    # universal collapse law -- that pass@8 itself falls in some cells, and that
    # Qwen2.5-3B ends above its pass-0 distinct@8 -- have to stay in the text.
    precheck = json.loads(
        (ROOT / "paper/results/baseline_collapse_precheck.json").read_text(
            encoding="utf-8"
        )
    )
    # Matched against the whitespace-normalized body: these phrases wrap across
    # source lines, and a reflow is not a contract violation.
    for token in (
        r"\textbf{Problem precheck: collapse without an intervention.}",
        "150 registered runs across 30 model--domain--method blocks of five seeds",
        "falls in 10 of the 30 domain--method--scale comparisons",
        "ends above its pass-0 reference at Qwen2.5-3B",
        "precheck characterizes the failure pattern before intervention comparison",
    ):
        require(normalized(token) in normalized(main_body),
                f"baseline-collapse precheck missing {token!r}")
    # The main text quotes the complete precheck and points to the audit grid;
    # keep the space-heavy per-cell visualization with its appendix analysis.
    for token in (r"\label{fig:baseline-precheck}", r"figures/baseline_collapse_precheck.pdf"):
        require(token in appendix,
                f"appendix baseline-collapse figure missing {token!r}")
        require(token not in main_body,
                f"baseline-collapse figure is duplicated into the main body: {token!r}")

    for arm, scales in (("drgrpo", ("qwen05b",)), ("grpo", ("qwen05b",))):
        for scale in scales:
            macro = precheck["arms"][arm]["scales"][scale]["macro"]
            removed = 100 * (1 - macro["extra_modes_retained_fraction"])
            require(f"{removed:.1f}\\%" in main_body,
                    f"main text does not quote the {arm}/{scale} removal {removed:.1f}%")
    require(
        all(
            precheck["arms"][arm]["scales"][scale]["macro"]["complete"]
            for arm in ("drgrpo", "grpo")
            for scale in ("qwen05b", "falcon1b", "qwen3b")
        ),
        "the precheck figure claims all three scales but a block is incomplete",
    )
    require(all(normalized(token) in normalized(appendix) for token in (
        "150 complete runs", "passes 0 and 8", "all three models and five domains",
        "Initial evaluations are measured within each method.", "E80-R1 seeds 71--74",
    )), "baseline-collapse precheck population or initial-reference disclosure is missing")
    require(r"\label{tab:baseline-collapse-precheck}" in appendix,
            "the precheck macro table is missing from the appendix")

    absolute = json.loads(ABSOLUTE_REFERENCES.read_text(encoding="utf-8"))
    require(absolute.get("schema") == "paper-absolute-support-references-v1",
            "wrong absolute-support reference schema")
    references = absolute["uninformed_policies"]
    require(abs(references["graph_coloring"]["primary"] - 1.632851484925304) < 1e-12,
            "Graph uninformed distinct@8 reference drifted")
    require(abs(references["pantry_plan"]["primary"] - 2.152919212894907) < 1e-12,
            "Pantry bit-uniform distinct@8 reference drifted")
    require(abs(references["pantry_plan"]["instruction_following"] - 2.7142475267937765) < 1e-12,
            "Pantry instruction-following reference drifted")
    require(sum(row["replay_drgrpo_distinct8"] > row["frozen_distinct8"] for row in absolute["rows"]) == 13,
            "Replay-over-frozen cell count drifted")
    for token in (r"\label{tab:absolute-support-references}",
                  "Pantry's bit-uniform reference exceeds every final model",
                  "support-concentration stress test rather than evidence of complete support coverage"):
        require(normalized(token) in normalized(manuscript),
                f"absolute-reference disclosure missing {token!r}")

    for token in (
        r"\label{lem:maxrl-mean}",
        r"\label{thm:grpo-collapse}",
        "Thus these GRPO, Dr.GRPO, and binary MaxRL mean flows converge to a single correct execution mode",
        r"\label{thm:replay-retention}",
        r"p_b(t)\ge \exp(-kC_T)>0",
        r"\label{cor:replay-no-collapse}",
        r"q(t)\longrightarrow u",
        "Theory studies idealized mean flows.",
        "not a new general theorem about replicator dynamics",
        r"e^{-2560}",
        "full-coverage corollary therefore cannot supply a domain-wide guarantee there",
        r"\label{app:fixed-bank-survival}",
        "The fixed-bank study tracks Qwen2.5-0.5B ReplayDr.GRPO",
        "score surrogates, not exact probabilities of sampling canonical modes",
        "Without a matched no-replay fixed-bank arm",
        "supplies no probability floor for an unbanked key",
    ):
        require(normalized(token) in normalized(manuscript), f"proof contract missing {token!r}")

    e121_path = ROOT / "paper/results/e121_fixed_bank_survival.json"
    e121 = json.loads(e121_path.read_text())
    require(e121.get("schema") == "e121-fixed-bank-survival-paper-v1"
            and e121.get("registered_seeds") == [43, 44, 45, 46, 47]
            and e121.get("freeze_step") == 384
            and e121.get("final_optimizer_update") == 3072,
            "fixed-bank score analysis has the wrong registered population")
    e121_script = ROOT / "ops/exp_scaling/build_paper_e121_survival.py"
    require(e121["provenance"]["analysis_script_sha256"]
            == hashlib.sha256(e121_script.read_bytes()).hexdigest(),
            "fixed-bank analysis builder changed; regenerate results")
    audit_path = ROOT / e121["provenance"]["audit"]
    require(e121["provenance"]["audit_sha256"]
            == hashlib.sha256(audit_path.read_bytes()).hexdigest()
            and json.loads(audit_path.read_text()).get("passed") is True,
            "fixed-bank result is not bound to a passing coverage audit")
    require([row["seed"] for row in e121["runs"]] == [43, 44, 45, 46, 47]
            and all(row["finite_at_every_scheduled_observation_fraction"] == 1
                    and row["visit_count_min"] >= 2 for row in e121["runs"]),
            "fixed-bank result omits a seed or fails registered score coverage")
    e121_analysis = runpy.run_path(str(e121_script))
    all_identities = []
    for row in e121["runs"]:
        identities = row["identities"]
        require(len(identities) == row["identity_count"],
                "fixed-bank reported identity count differs from its population")
        expected = e121_analysis["named_statistics"](
            e121_analysis["array_from_identities"](identities))
        require(row["statistics"] == expected,
                "fixed-bank seed summaries differ from identity-level changes")
        all_identities.extend(identities)
    require(e121["pooled"]["identity_count"] == len(all_identities)
            and e121["pooled"]["statistics"] == e121_analysis["named_statistics"](
                e121_analysis["array_from_identities"](all_identities))
            and e121["bootstrap"]["draws"] == 10000
            and e121["bootstrap"]["rng_seed"] == 121,
            "fixed-bank pooled summaries or registered bootstrap settings drifted")

    require(main_body.count("Semantic-MaxEnt") == 1,
            "main body should contain Semantic-MaxEnt only in Figure 4")
    require(r"\label{app:semantic-maxent}" in appendix
            and r"\label{app:semantic-estimator}" in appendix,
            "the fixed-semantic comparator definition is missing")
    require("figures/verified_support_discovery_two_scale_effects.pdf" not in appendix,
            "the archived bundled semantic experiment returned")

    # Hyperlink destinations are metadata; their visible labels remain subject
    # to the same reader-facing cohort-code rule.
    reader_main_body = re.sub(r"\\href\{[^{}]*\}", "", main_body)
    require(not re.search(r"(?<![A-Za-z])E\d{2,}(?:-R\d+)*(?!\d)", reader_main_body),
            "reader-facing main body contains an internal cohort code")
    require(r"\label{tab:experiment-map}" in appendix
            and r"\label{tab:direct-comparators}" in manuscript,
            "current study and comparator coverage tables are missing")
    require("50/50" in manuscript and "47/50" in manuscript,
            "UCPO/RLEP coverage must use the selected two-model scope")
    for phrase in ("Still needed", "finish PantryPlan", "finish both Qwen-3B blocks",
                   "Newly completed Qwen2.5-3B Python factorial", "Dated endpoint update:"):
        require(phrase not in manuscript, f"archived campaign chronology returned: {phrase}")
    for name in ("latest_results_20260911_table_body.tex",
                 "latest_results_20260911_effects_table_body.tex",
                 "e118_qwen3b_python_terminal_table_body.tex",
                 "e120_primary_breadth_seeds_table_body.tex",
                 "e120_frequency_progress_table_body.tex",
                 "e120_falcon_graph_20260906_table_body.tex",
                 "e120_falcon_graph_20260906_seeds_table_body.tex",
                 "e121_fixed_bank_survival_sequence_table_body.tex",
                 "e121_fixed_bank_survival_bootstrap_table_body.tex"):
        require(f"\\input{{results/{name}}}" not in manuscript,
                f"archived duplicate table returned: {name}")

    core = json.loads(CORE.read_text(encoding="utf-8"))
    exclusions = core.get("exclusions", [])
    require(len(exclusions) == 1 and exclusions[0].get("job_id") == 30269051
            and exclusions[0].get("outcome_value_selected") is False,
            "registered source exclusion is missing or was replaced by a selected retry")
    paired_count = 0
    for model, model_record in core["models"].items():
        for domain, domain_record in model_record["domains"].items():
            methods = domain_record["methods"]
            control = methods["control"]["per_seed"]
            replay = methods["replay"]["per_seed"]
            pairs = set(control) & set(replay)
            expected = set(control)
            if model == "Falcon3-1B" and domain == "countdown":
                expected = expected - {"59"}
                require("59" not in replay, "ambiguous replay endpoint was reintroduced")
            require(pairs == expected, f"unexpected primary source loss in {model}/{domain}")
            paired_count += len(pairs)
    require(paired_count == 74, "primary evidence must contain 74 admissible pairs")
    require(r"\label{app:source-integrity}" in appendix,
            "source exclusion is not disclosed in the manuscript")

    fig4 = json.loads(FIG4.read_text(encoding="utf-8"))
    require(
        fig4.get("schema") == "paper-experiment1-retention-comparator-matrix-v4",
        "wrong Experiment 1 composite schema",
    )
    require(
        fig4.get("estimands") == {
            "pass8": "method minus matched Dr.GRPO terminal pass@8",
            "distinct8": "method minus matched Dr.GRPO terminal distinct@8",
        },
        "Figure 4 does not show the two registered co-primary endpoints",
    )
    require(
        "not encoded as cell-level significance decisions"
        in fig4.get("uncertainty", ""),
        "Figure 4 reintroduced cell-level significance encoding",
    )
    omnibus = fig4.get("omnibus_consistency_checks", {})
    require(
        omnibus.get("status") == "post-hoc omnibus consistency check"
        and omnibus.get("magnitude_pooling") is False
        and omnibus.get("family") == ["pass8", "distinct8"],
        "Figure 4 omnibus-test family drifted",
    )
    for endpoint in ("pass8", "distinct8"):
        test = omnibus.get("tests", {}).get(endpoint, {})
        require(
            test.get("positive") == 14
            and test.get("negative") == 0
            and test.get("ties") == 0
            and abs(test.get("two_sided_exact_sign_p", 0.0) - 0.0001220703125)
            < 1e-15
            and abs(
                test.get("holm_adjusted_p_across_two_coprimary_endpoints", 0.0)
                - 0.000244140625
            )
            < 1e-15,
            f"Figure 4 {endpoint} omnibus statistics drifted",
        )
    figure4_columns = [
        "graph_coloring", "countdown", "python_factors",
        "mathir", "pantry_plan", "average",
    ]
    average_definition = (
        "post-hoc descriptive unweighted macro-average across five "
        "domains within paired seed; excluded from inference"
    )

    panel_a = fig4.get("panel_a", {})
    require(
        panel_a.get("models")
        == ["Qwen2.5-0.5B", "Falcon3-1B", "Qwen2.5-3B"]
        and set(panel_a.get("domains", [])) == DOMAINS
        and panel_a.get("display_columns") == figure4_columns
        and panel_a.get("average_definition") == average_definition,
        "Figure 4A does not contain the complete cross-scale retention grid",
    )
    for model in panel_a["models"]:
        for domain in DOMAINS | {"average"}:
            require(
                panel_a["cells"][model][domain]["n"]
                == (4 if model == "Falcon3-1B" and domain in {"countdown", "average"} else 5),
                f"Figure 4A denominator drifted for {model}/{domain}",
            )
    panel_b = fig4.get("panel_b", {})
    require(
        set(panel_b.get("methods", []))
        == {
            "before_training", "replay_drgrpo", "maxrl", "ucpo",
            "rlep_dr", "semantic_maxent", "grpo",
        }
        and set(panel_b.get("domains", [])) == DOMAINS
        and panel_b.get("display_columns") == figure4_columns
        and panel_b.get("average_definition") == average_definition,
        "Figure 4B lacks a requested direct alternative",
    )
    for method in panel_b["methods"]:
        for domain in DOMAINS | {"average"}:
            require(
                panel_b["cells"][method][domain]["n"] == 5,
                f"Figure 4B denominator drifted for {method}/{domain}",
            )

    e118 = json.loads(E118.read_text(encoding="utf-8"))
    require(e118.get("schema") == "e118-all-scale-terminal-progress-v6",
            "wrong E118 figure schema")
    require(e118.get("target_step") == 3072, "wrong E118 terminal step")
    require(e118.get("before_training_step") == 0,
            "Figure 5 lacks the shared before-training checkpoint")
    display = e118.get("display_contract", {})
    require(
        display.get("tracks") == {
            "upper": "Untrained to MaxRL to ReplayMaxRL",
            "lower": "Untrained to Dr.GRPO to ReplayDr.GRPO",
        }
        and "restricted to each track's paired seeds"
        in display.get("untrained_reference", ""),
        "Figure 5 does not retain matched Untrained references on both tracks",
    )
    model_scales = ["qwen05b", "falcon1b", "qwen3b"]
    require(e118.get("main_figure_scales") == model_scales,
            "Figure 5 must show all three registered model scales")
    for field in ("main_figure_tracks", "main_cross_domain_tracks"):
        require(e118.get(field) == {
            scale: ["maxrl", "drgrpo"] for scale in model_scales
        }, "Figure 5 must display both replay tracks directly in every model row")
    require(e118.get("main_domain_tracks") == {},
            "Figure 5 main layout must use model rows rather than a separate domain panel")
    require(e118.get("appendix_figure_scales") == model_scales,
            "E118 appendix panels must cover all three registered models")
    require(e118.get("incomplete_scales") == ["qwen3b"],
            "pending Qwen3B status is not explicit")
    domain_definition = "equal domain average of domain-specific paired-seed means"
    require(
        display.get("main_figure")
        == ("three-model cross-domain pass@8 and distinct@8 with MaxRL/replay and "
            "Dr.GRPO/replay tracks; Qwen2.5-3B MaxRL uses descriptive equal-domain means "
            "of available within-domain paired seeds")
        and display.get("main_panels") == {
            "A": "cross-domain pass@8", "B": "cross-domain distinct@8",
        }
        and display.get("appendix_figure")
        == "all three models across five domains and both terminal metrics"
        and "no pooled seed paths or intervals" in display.get("qwen3b_display", "")
        and display.get("qwen3b_maxrl_average") == domain_definition
        and display.get("dashed_connector")
        == "partial or domain-specific paired seeds with exact counts and no interval",
        "Figure 5 model-row layout or descriptive-domain display drifted",
    )
    available = e118.get("descriptive_available_domain_average", {})
    require(set(available) == {"qwen3b"},
            "descriptive domain-specific averaging must be separate from complete model tracks")
    qwen3_counts = {
        domain: len(e118["cells"]["qwen3b"][domain]["matched_seeds"])
        for domain in DOMAINS
    }
    require(display.get("qwen3b_maxrl_counts") == qwen3_counts and min(qwen3_counts.values()) > 0,
            "Qwen3B MaxRL descriptive summary must retain all five domains with exact paired counts")
    for metric in ("pass8", "distinct8"):
        require(set(available["qwen3b"][metric]) == {"before_training", "maxrl", "replay_maxrl"},
                "Qwen3B descriptive summary must retain its matched initial/MaxRL/replay track")
        for method, summary in available["qwen3b"][metric].items():
            expected_per_domain = {}
            for domain in DOMAINS:
                cell = e118["cells"]["qwen3b"][domain]
                seeds = cell["method_seeds"][method]
                require(seeds == cell["matched_seeds"],
                        f"Qwen3B descriptive {domain}/{method} uses unmatched seeds")
                expected_per_domain[domain] = dict(zip(
                    map(str, seeds), cell["methods"][method][metric], strict=True,
                ))
            expected_means = {
                domain: sum(values.values()) / len(values)
                for domain, values in expected_per_domain.items()
            }
            require(summary.get("n_domains") == 5
                    and summary.get("definition") == domain_definition
                    and summary.get("domain_seed_counts") == qwen3_counts
                    and summary.get("domain_weights") == {domain: .2 for domain in DOMAINS}
                    and summary.get("per_domain_per_seed") == expected_per_domain
                    and all(abs(summary["domain_means"][domain] - value) < 1e-12
                            for domain, value in expected_means.items())
                    and abs(summary["mean"] - sum(expected_means.values()) / 5) < 1e-12,
                    f"Qwen3B {method}/{metric} is not the equally weighted five-domain paired mean")
            require(not ({"per_seed", "seeds", "n", "student_t_95", "standard_error", "sem"} & set(summary)),
                    "Qwen3B descriptive domain mean invents a common seed cohort or interval")
    qwen = e118.get("cells", {}).get("qwen05b", {})
    require(set(qwen) == DOMAINS, "Qwen E118 domain grid is incomplete")
    for domain, cell in qwen.items():
        require(len(cell.get("matched_seeds", [])) == 5,
                f"Qwen E118 {domain} is not a complete five-seed block")
        require(set(cell.get("methods", {})) == {
                    "before_training", "before_training_drgrpo", "drgrpo", "replay_drgrpo",
                    "maxrl", "replay_maxrl",
                }, f"Qwen E118 {domain} lacks a before/after factorial arm")
        for record in cell.get("methods", {}).values():
            require(all(len(values) == 5 for values in record.values()),
                    f"Qwen E118 {domain} method denominator drifted")
    falcon = e118.get("cells", {}).get("falcon1b", {})
    require(set(falcon) == DOMAINS, "Falcon E118 domain grid is incomplete")
    for domain, cell in falcon.items():
        require(len(cell.get("matched_seeds", [])) == 5
                and cell.get("replay_maxrl_minus_maxrl", {}).get("status")
                == "complete paired block",
                f"Falcon E118 {domain} is not a complete five-seed block")
    qwen3 = e118.get("cells", {}).get("qwen3b", {})
    require(set(qwen3) == DOMAINS, "Qwen3B E118 domain grid is incomplete")
    require(qwen3["python_factors"].get("matched_seeds", []) == [70, 71, 72, 73, 74],
            "Qwen3B progress record lost the complete five-seed Python block")
    check_training_curves(e118)
    latest_path = ROOT / "paper/results/latest_results_20260911.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    require(latest.get("schema") == "paper-latest-terminal-status-v2"
            and latest.get("analysis_date") == "2026-09-11",
            "latest terminal census does not describe the September 11 update")
    source_audit = latest["source_audit"]
    audit_path = ROOT / source_audit["path"]
    require(hashlib.sha256(audit_path.read_bytes()).hexdigest() == source_audit["sha256"],
            "latest terminal census audit hash drifted")
    latest_qwen3 = {block["domain"]: block for block in latest["campaigns"]["e118"]["blocks"]
                   if block["model_key"] == "qwen3b"}
    require(all(qwen3[domain]["matched_seeds"] == latest_qwen3[domain]["paired_seeds"]
                for domain in DOMAINS),
            "Qwen3B Figure 5 terminal prefixes differ from the dated audit")
    require(e118.get("endpoint_snapshot", {}).get("sha256") == source_audit["sha256"],
            "Figure 5 is not bound to the latest dated endpoint audit")
    # Reproduce all current result tables from the same frozen endpoint census.
    digest_builder = runpy.run_path(str(ROOT / "ops/exp_scaling/build_paper_current_campaign_results.py"))
    digest_result = digest_builder["prepare"](latest, latest_path)
    digest_result["findings"] = digest_builder["findings"](digest_result)
    digest_stem = "current_campaign_results_20260911"
    require(json.loads((ROOT / "paper/results" / (digest_stem + ".json")).read_text()) == digest_result,
            "current campaign results differ from their audited snapshot")
    for campaign in ("e118", "e119", "e120"):
        name = f"{digest_stem}_{campaign}_table_body.tex"
        require((ROOT / "paper/results" / name).read_text() == digest_builder["render_tex"](digest_result, campaign),
                f"current campaign table is stale: {name}")
        require(f"\\input{{results/{name}}}" in appendix,
                f"current campaign table is not compiled: {name}")
    factorial_builder = runpy.run_path(str(ROOT / "ops/exp_scaling/build_paper_level2_factorial_contrasts.py"))
    factorial = json.loads((ROOT / "paper/results/level2_factorial_contrasts_20260911.json").read_text())
    expected_factorial = factorial_builder["build"](json.loads(audit_path.read_text()))
    require(all(factorial[key] == value for key, value in expected_factorial.items())
            and factorial["source_audit"] == source_audit,
            "Level-2 factorial contrasts differ from the common four-arm audited seeds")
    factorial_name = "level2_factorial_contrasts_20260911_table_body.tex"
    require((ROOT / "paper/results" / factorial_name).read_text() == factorial_builder["render_table"](factorial)
            and f"\\input{{results/{factorial_name}}}" in appendix,
            "Level-2 factorial table is stale or not compiled")

    initial_pass = [
        value
        for cell in qwen.values()
        for value in cell["methods"]["before_training"]["pass8"]
    ]
    initial_distinct = [
        value
        for cell in qwen.values()
        for value in cell["methods"]["before_training"]["distinct8"]
    ]
    require(len(initial_pass) == len(initial_distinct) == 25
            and abs(sum(initial_pass) / 25 - .33484375) < 1e-12
            and abs(sum(initial_distinct) / 25 - .663671875) < 1e-12,
            "Figure 5 before-training Qwen reference drifted")

    for scale, cells in e118.get("cells", {}).items():
        for domain, cell in cells.items():
            for method, record in cell.get("methods", {}).items():
                seeds = cell.get("method_seeds", {}).get(method)
                require(seeds is not None and all(len(values) == len(seeds) for values in record.values()),
                        f"E118 method-specific denominator mismatch in {scale}/{domain}/{method}")
    for metric in ("pass8", "distinct8"):
        means = e118["absolute_cross_domain_average"]["falcon1b"][metric]
        for method in ("before_training_drgrpo", "drgrpo", "replay_drgrpo"):
            require(set(means[method]["per_seed"]) == {"55", "56", "57", "58"},
                    f"Falcon {method} macro contains an inadmissible or unmatched seed")
            require("student_t_95" not in means[method], "partial track has a five-seed interval")
        for method in ("maxrl", "replay_maxrl"):
            require(set(means[method]["per_seed"]) == {"55", "56", "57", "58", "59"},
                    f"valid Falcon {method} seed was lost")

    # E80-R1 is complete even though the E118 Qwen3B MaxRL grid is incomplete.
    for metric in ("pass8", "distinct8"):
        means = e118["absolute_cross_domain_average"]["qwen3b"][metric]
        require(set(means) == {"before_training_drgrpo", "drgrpo", "replay_drgrpo"},
                "Qwen3B complete-track record must exclude incomplete factorial aggregates")
        for method, arm in (("drgrpo", "control"), ("replay_drgrpo", "replay")):
            expected = {
                str(seed): sum(
                    core["models"]["Qwen2.5-3B"]["domains"][domain]
                    ["methods"][arm]["per_seed"][str(seed)][metric]
                    for domain in DOMAINS
                ) / 5
                for seed in range(70, 75)
            }
            require(means[method]["n"] == 5 and all(
                abs(means[method]["per_seed"][seed] - value) < 1e-12
                for seed, value in expected.items()
            ) and set(means[method]["per_seed"]) == set(expected),
                    f"Qwen3B {method} Figure 5 average is not the five-seed E80-R1 mean")
        initial = means["before_training_drgrpo"]
        expected_initial = {
            str(seed): sum(
                precheck["arms"]["drgrpo"]["scales"]["qwen3b"]["domains"][domain]
                ["per_seed"][str(seed)]["pass0"][metric]
                for domain in DOMAINS
            ) / 5
            for seed in range(70, 75)
        }
        require(initial["n"] == 5 and set(initial["per_seed"]) == set(expected_initial)
                and all(abs(initial["per_seed"][seed] - value) < 1e-12
                        for seed, value in expected_initial.items()),
                "Qwen3B Figure 5 initial reference drifted")

    # The completed Graph and Python blocks each retain their own five seeds.
    # Completeness counts remain distinct from the descriptive 3B mean.
    all_cells = [cell for cells in e118["cells"].values() for cell in cells.values()]
    pair_blocks = sum(len(cell.get("replay_maxrl_minus_maxrl", {}).get("per_seed", {})) == 5
                      for cell in all_cells)
    four_arm_blocks = sum(
        len(set.intersection(*(set(cell.get("method_seeds", {}).get(arm, []))
                               for arm in ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")))) == 5
        for cell in all_cells)
    require(pair_blocks == 12 and four_arm_blocks == 11,
            "E118 pair/four-arm complete-block counts drifted")
    python3b = e118["cells"]["qwen3b"]["python_factors"]
    require(python3b["matched_seeds"] == [70, 71, 72, 73, 74],
            "Qwen3B Python does not retain all five registered seeds")
    # Reproduce the frozen E120 analysis and table bodies without refreshing
    # a campaign or silently accepting a stale generated analysis.
    analysis_path = ROOT / "ops/exp_scaling/build_paper_e120_primary_breadth.py"
    analysis = runpy.run_path(str(analysis_path))
    expected_primary = analysis["build"]()
    primary_path = ROOT / "paper/results/e120_primary_breadth.json"
    require(json.loads(primary_path.read_text()) == expected_primary,
            "E120 primary estimates/provenance differ from the frozen source")
    rendered = analysis["render_tables"](expected_primary["rows"])
    for name, expected in zip(("e120_primary_breadth_table_body.tex",
                               "e120_primary_breadth_seeds_table_body.tex"), rendered):
        require((ROOT / "paper/results" / name).read_text() == expected,
                f"E120 generated table is stale: {name}")
    require(r"\label{app:frequency-replay-progress}" in appendix
            and "$+.318$" in appendix and "$+.010$" in appendix,
            "the registered E120 aggregate is missing from the consolidated ablation")

    levels = json.loads(LEVELS.read_text(encoding="utf-8"))
    rows = levels.get("admission_rows", [])
    require(levels.get("schema") == "modebench-level-admission-and-terminal-v4"
            and levels.get("target_step") == 3072 and len(rows) == 5,
            "wrong matched-level admission/progress record")
    require(all(.10 <= row["level2_pass8"] <= .90
                and row["level2_pass8"] < row["level1_pass8"] for row in rows),
            "Level 2 no longer passes the harder-but-solvable admission gate")
    partial = levels.get("partial_treatment", {})
    require(partial.get("domain") == "graph_coloring"
            and partial.get("matched_seeds") == [43, 44, 45, 46, 47]
            and partial.get("n") == 5
            and partial.get("complete_block") is True
            and set(partial.get("methods", {}))
            == {"drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl"},
            "Level-2 Graph prefix is not the exact complete four-arm block")
    for method in partial["methods"].values():
        require(set(method["per_seed"]) == {"43", "44", "45", "46", "47"},
                "Level-2 Graph method denominator drifted")
        for metric in ("pass8", "distinct8"):
            values = [row[metric] for row in method["per_seed"].values()]
            require(abs(method["means"][metric] - sum(values) / 5) < 1e-12,
                    "Level-2 Graph complete-block mean drifted")

    comparison_snapshot = ROOT / "paper/results/modebench_level_comparison_snapshot.json"
    frozen_levels = json.loads(comparison_snapshot.read_text(encoding="utf-8"))
    require(frozen_levels.get("schema") == "modebench-level-comparison-frozen-snapshot-v1",
            "wrong frozen Level1/Level2 snapshot")
    require(levels.get("sources", {}).get(str(comparison_snapshot.resolve()))
            == hashlib.sha256(comparison_snapshot.read_bytes()).hexdigest(),
            "Figure6 is not bound to its frozen comparison snapshot")
    level_analysis = runpy.run_path(
        str(ROOT / "ops/exp_scaling/build_paper_modebench_level_comparison.py")
    )
    expected_interim = level_analysis["build_interim_comparison"](
        frozen_levels["evaluations"], frozen_levels["availability"]
    )
    require(levels.get("interim_comparison") == expected_interim,
            "Figure6 interim selection, pairing, or equal-domain means drifted")
    require(levels.get("terminal_progress_by_domain")
            == level_analysis["build_terminal_progress"](frozen_levels["level2_terminal_evaluations"]),
            "Figure6 terminal coverage is not the frozen exact-step record")
    require(set(expected_interim["domains"]) == DOMAINS
            and expected_interim["model"] == "Qwen2.5-0.5B-Instruct",
            "Figure6 interim audit scope lost a domain or model")
    expected_terminal = level_analysis["build_terminal_comparison"](
        frozen_levels["terminal_evaluations"], frozen_levels["terminal_admission"]
    )
    require(levels.get("terminal_comparison") == expected_terminal,
            "Figure6 terminal cohort, seed pairing, or means drifted")
    require(levels.get("sources", {}).get(str(audit_path.resolve()))
            == hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "Figure6 terminal values are not bound to the dated endpoint census")

    if PDF.is_file() and PDF.stat().st_mtime >= TEX.stat().st_mtime:
        page9 = subprocess.run(
            ["pdftotext", "-f", "9", "-l", "9", str(PDF), "-"],
            check=True, capture_output=True, text=True,
        ).stdout
        page10 = subprocess.run(
            ["pdftotext", "-f", "10", "-l", "10", str(PDF), "-"],
            check=True, capture_output=True, text=True,
        ).stdout
        page11 = subprocess.run(
            ["pdftotext", "-f", "11", "-l", "11", str(PDF), "-"],
            check=True, capture_output=True, text=True,
        ).stdout
        page9_compact = re.sub(r"\s+", "", page9.upper())
        page10_compact = re.sub(r"\s+", "", page10.upper())
        page11_compact = re.sub(r"\s+", "", page11.upper())
        aux = (ROOT / "paper/main.aux").read_text()
        conclusion = re.search(r"\\newlabel\{sec:conclusion\}\{\{[^}]*\}\{(\d+)\}", aux)
        require(conclusion is not None and int(conclusion.group(1)) <= 9,
                "conclusion exceeds the nine-page main-text limit")
        require(
            "REFERENCES" in page10_compact or "REFERENCES" in page11_compact,
            "references do not begin after the main body",
        )
        log = ROOT / "paper/main.log"
        if log.is_file():
            require("Overfull" not in log.read_text(errors="replace"),
                    "LaTeX reports an overfull box")

    print("Current paper contract passed: three experiments, six main figures, proof chain, and evidence gates.")


if __name__ == "__main__":
    main()
