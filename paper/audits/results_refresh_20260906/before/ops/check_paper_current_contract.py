#!/usr/bin/env python3
"""Fail-closed contract for the current ModeBench/ReplayMaxRL paper story."""
from __future__ import annotations

import json
import re
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
    "sustained_auc_effects_qwen05b",
    "direct_comparator_endpoint_effects",
    "direct_baseline_learning_curves_static_strip",
    "verified_support_discovery_two_scale_effects",
    "replay_mechanism_telemetry_qwen05b",
)
RETIRED = (
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


def main() -> None:
    manuscript = TEX.read_text(encoding="utf-8")
    main_body, appendix = manuscript.split(r"\appendix", 1)
    makefile = MAKEFILE.read_text(encoding="utf-8")
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
        "Replay improves the chance of solving harder Graph.",
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
    require("E78, E79," in appendix and "E95 at all three scales" in appendix,
            "baseline-collapse precheck provenance is missing")
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
                  "PantryPlan as an enumerable stress test of support concentration",
                  "rather than a fully success-matched diversity measure"):
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
        "not claimed as a new replicator-dynamics result",
        r"\exp(-2560)",
        "inapplicable to Python factors as a domain-wide guarantee",
        "individual-mode prediction is not tested by the experiments in this paper",
        "freezing a Graph bank after one pass and recording per-exemplar score trajectories",
        "no E121 empirical result is reported here",
    ):
        require(normalized(token) in normalized(manuscript), f"proof contract missing {token!r}")

    require(main_body.count("Semantic-MaxEnt") == 1,
            "main body should contain Semantic-MaxEnt only in Figure 4")
    require(r"\section{Semantic MaxEnt Baseline}" in appendix,
            "Semantic-MaxEnt appendix is missing")
    semantic_graphics = [x for x in re.findall(r"figures/[^}]+\.pdf", appendix)
                         if "semantic" in x.lower() or "verified_support_discovery" in x]
    require(semantic_graphics == ["figures/verified_support_discovery_two_scale_effects.pdf"],
            "Semantic-MaxEnt must have exactly one compiled appendix comparison")

    require(not re.search(r"(?<![A-Za-z])E\d{2,}(?:-R\d+)*(?!\d)", main_body),
            "reader-facing main body contains an internal cohort code")
    for token in (
        "14/15 blocks", "10/15 blocks", "1/5 blocks; Graph complete",
        "five Qwen-3B blocks; full factorial has nine complete blocks",
        "finish the other four domains",
        "5/9 blocks; 25/45 cells",
        "larger-model blocks absent from the frozen E120 snapshot",
    ):
        require(token in manuscript, f"experiment status table missing {token!r}")

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
    require(e118.get("schema") == "e118-all-scale-terminal-progress-v4",
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
        "Figure 5 does not display Untrained on both factorial tracks",
    )
    require(e118.get("main_figure_scales") == ["qwen05b", "falcon1b"],
            "Figure 5 must show both completed-scale cross-domain averages")
    require(e118.get("appendix_figure_scales") == ["qwen05b", "falcon1b"],
            "E118 per-domain panels are not assigned to the appendix")
    require(e118.get("incomplete_scales") == ["qwen3b"],
            "pending Qwen3B status is not explicit")
    require(
        display.get("main_figure")
        == "Qwen2.5-0.5B and Falcon3-1B cross-domain averages only"
        and display.get("appendix_figure")
        == "Qwen2.5-0.5B and Falcon3-1B per-domain panels"
        and "machine-readable progress only"
        in display.get("qwen3b_display", ""),
        "Figure 5 summary/detail split drifted",
    )
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
    require(len(qwen3["python_factors"].get("matched_seeds", [])) == 2,
            "Qwen3B progress record lost the exact two-seed Python prefix")
    require(all(not qwen3[domain].get("matched_seeds")
                for domain in DOMAINS - {"python_factors"}),
            "Qwen3B appendix contains an unsupported terminal domain")
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

    levels = json.loads(LEVELS.read_text(encoding="utf-8"))
    rows = levels.get("admission_rows", [])
    require(levels.get("schema") == "modebench-level-admission-and-e119-progress-v2"
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
        require("CONCLUSION" in page9_compact, "conclusion does not fit on page 9")
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
