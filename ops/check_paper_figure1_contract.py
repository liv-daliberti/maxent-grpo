#!/usr/bin/env python3
"""Fail closed if any manuscript figure regresses from its accepted contract."""
from __future__ import annotations
import ast
import json
import re
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ops/plot_paper_collapse_toy.py"
AUDIT = ROOT / "var/artifacts/paper_graph_collapse_toy.json"
MANUSCRIPT = ROOT / "paper/main.tex"
EXAMPLES_SOURCE = ROOT / "ops/plot_paper_modebench_examples.py"
EXAMPLES_PDF = ROOT / "paper/figures/modebench_examples.pdf"
MECHANISM_SOURCE = ROOT / "ops/plot_paper_xdr_mechanism.py"
MECHANISM_PDF = ROOT / "paper/figures/xdr_mechanism.pdf"
MAIN_PDF = ROOT / "paper/main.pdf"
HEADLINE_SOURCE = ROOT / "ops/plot_paper_modecollapse.py"
APPENDIX_SOURCE = ROOT / "ops/exp_scaling/plot_e70_clean_05b_wide_live.py"
HEADLINE_PDF = ROOT / "paper/figures/modecollapse_training.pdf"
APPENDIX_PDF = (
    ROOT
    / "paper/figures/e68_e58_vs_grpo_05b_12ep_terminal_provenance.pdf"
)
APPENDIX_PROVENANCE = (
    ROOT / "var/artifacts/clean_05b_eight_environment_figure_provenance.json"
)
E70_TERMINAL_AUDIT = ROOT / "var/artifacts/e70_clean_stage_a_05b_audit_latest.json"
PANTRY_TERMINAL_AUDIT = ROOT / "var/artifacts/pantry_stage_b_05b_12pass_audit.json"

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"Paper figure contract failed: {message}")


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise SystemExit(f"Paper figure contract failed: missing assignment {name}")


def pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout

def main() -> None:
    source = SOURCE.read_text()
    manuscript = MANUSCRIPT.read_text()
    audit = json.loads(AUDIT.read_text())
    example_source = EXAMPLES_SOURCE.read_text()
    require("excess@" not in manuscript, "excess@K remains in manuscript")
    require(r"\paragraph{" not in manuscript, "compact bold headings regressed")
    require(
        r"\usepackage{iclr2026_conference,times}" in manuscript,
        "paper is not using the ICLR 2026 template",
    )
    require(
        r"\bibliographystyle{iclr2026_conference}" in manuscript,
        "paper is not using the ICLR 2026 bibliography style",
    )
    require("neurips_2025" not in manuscript, "NeurIPS template remains in manuscript")
    abstract = manuscript.split(r"\begin{abstract}", 1)[1].split(
        r"\end{abstract}", 1
    )[0]
    abstract_plain = re.sub(r"\\[A-Za-z]+", "", abstract)
    abstract_words = re.findall(r"[A-Za-z0-9@.+-]+", abstract_plain)
    require(len(abstract_words) <= 130, f"abstract has {len(abstract_words)} words")
    introduction = manuscript.split(r"\section{Introduction}", 1)[1].split(
        r"\section{Mode Collapse under GRPO}", 1
    )[0]
    introduction_argument = introduction.split(
        r"\noindent\textbf{Summary of contributions.}", 1
    )[0]
    introduction_argument = re.sub(
        r"\\begin{figure}.*?\\end{figure}",
        "",
        introduction_argument,
        flags=re.DOTALL,
    )
    introduction_paragraphs = [
        block.strip()
        for block in re.split(r"\n\s*\n", introduction_argument)
        if block.strip() and not block.strip().startswith(r"\label{")
    ]
    five_point_tokens = (
        "breaks that trade",
        "Breadth is what test-time scaling consumes",
        "Useful breadth is difficult",
        "Prior work documents",
        r"We introduce \mb{}",
    )
    require(
        len(introduction_paragraphs) == 5,
        "Introduction must contain exactly five argumentative paragraphs "
        f"before contributions, got {len(introduction_paragraphs)}",
    )
    for index, (paragraph, token) in enumerate(
        zip(introduction_paragraphs, five_point_tokens), start=1
    ):
        require(
            token in paragraph,
            f"Introduction paragraph {index} is missing five-point token {token!r}",
        )
    for forbidden in (r"V(y,s_x)", r"D_K(x)", r"P_K(x)="):
        require(forbidden not in introduction, f"Introduction contains formal token {forbidden!r}")
    for token in (
        "breaks that trade", "Breadth is what test-time scaling consumes",
        "Useful breadth is difficult", "Prior work documents",
        r"We introduce \mb{}", "Summary of contributions",
    ):
        require(token in introduction, f"Introduction structure missing {token!r}")
    for token in (
        r"\begin{list}{\textbullet}",
        r"\setlength{\leftmargin}{1.15em}",
        r"\setlength{\labelwidth}{.65em}",
    ):
        require(token in introduction, f"Contribution list alignment missing {token!r}")
    for token in (
        "flattens what it means to be right",
        "Exact domains make this loss countable",
        "correct-answer effect",
        "sorensen2024pluralistic",
        "park2024diversitythought",
        "wang2025flatten",
    ):
        require(token in introduction, f"Introduction stakes missing {token!r}")
    conclusion = manuscript.split("Conclusion.", 1)[1].split(
        "bibliographystyle", 1
    )[0]
    for token in (
        "The verified world is a microscope",
        "Preserving the diversity",
        "is therefore option value",
        "doshi2024creative",
        "anderson2024homogenization",
        "These studies show that quality and a narrower",
        "transfers to open-ended tasks",
    ):
        require(token in conclusion, f"Conclusion stakes missing {token!r}")
    related = manuscript.split(r"\section{Related Work}", 1)[1].split(
        r"\section{Experimental Design and Evidence Policy}", 1
    )[0]
    require(
        related.count(r"\noindent\textbf{") == 4,
        "Related Work must contain exactly four bold areas",
    )
    for token in (
        "Mode-covering objectives and archives",
        "bengio2023gflownet", "hu2024amortizing",
        "lehman2011novelty", "mouret2015illuminating",
        "yue2025rlvrlimit", "kirk2024understanding",
    ):
        require(token in related, f"Related Work coverage missing {token!r}")
    main_body = manuscript.split(r"\appendix", 1)[0]
    appendix_labels = (
        "app:theory", "app:python", "app:prompts", "app:data", "app:algorithm",
        "app:per-seed", "app:terminal-mathir", "app:ablations",
        "app:decoding", "app:telemetry", "app:cross-family",
        "app:reproducibility",
    )
    for label in appendix_labels:
        require(
            rf"\ref{{{label}}}" in main_body,
            f"appendix {label!r} is not tied to the main body",
        )
    require(
        manuscript.count(r"\noindent\emph{Main-body link.}")
        == len(appendix_labels),
        "every appendix must begin with one explicit main-body link",
    )
    algorithm_appendix = manuscript.split(r"\section{Algorithmic Details}", 1)[1].split(
        r"\section{Per-Seed Pass-12 Results}", 1
    )[0]
    for token in (
        r"\usepackage{algorithm}", r"\usepackage{algorithmic}",
        r"\begin{algorithm}[H]", r"\begin{algorithmic}[1]",
        r"\label{alg:xdr-update}", r"\mathcal T_x", r"\mathcal R_x",
        r"\textsc{NextBank}", r"w_{\mathrm{rep}}\gets(G-1)/G^2",
        "atomic state",
    ):
        require(token in manuscript, f"formal algorithm contract missing {token!r}")
    require(
        r"\begin{enumerate}" not in algorithm_appendix,
        "Algorithmic Details regressed to a prose numbered list",
    )
    # The manuscript reports one completed five-domain surface, so it names
    # components and frozen revisions rather than internal cohort codes. Guard
    # the evidence-typing language, and forbid the codes from returning.
    for token in (
        "Component Necessity Ablations",
        "Only the separated-support actuator row is a completed causal",
        "sole terminal causal performance",
        "randomized factorial study",
        "immutable frozen record",
    ):
        require(token in manuscript, f"ablation contract missing {token!r}")
    for forbidden in ("Stage-A", "Stage A", "Stage-B", "Stage B", "E-Series"):
        require(
            forbidden not in manuscript,
            f"manuscript reintroduced staged-campaign language {forbidden!r}",
        )
    for match in re.finditer(r"(?<![A-Za-z])E\d\d(?![\d])", manuscript):
        # Artifact and preregistration filenames legitimately keep their cohort
        # names; running prose must not.
        line = manuscript[: match.start()].rsplit("\n", 1)[-1]
        require(
            r"\path{" in line,
            f"manuscript prose reintroduced cohort code {match.group(0)!r}",
        )
    for token in (
        "eval_multi_answer-10000-0",
        "eval-15100-93",
        "multi_answer-15900-61",
        "def load_and_validate(",
        'graph_answers == ("113", "131")',
        '"outputs": (3, 41, 13, 3)',
        '"answer": "6 + 3 + 9"',
        '{"answer": "F;E"',
        "eval-94002-high_fiber_snack-19",
        "navel_orange=75;sunflower_seeds=50",
        "grape_tomatoes=75;almonds=50",
        "def draw_mini_graph(",
        # One hue per operation, so the two Countdown keys are
        # distinguishable by which operations they execute.
        "OPERATOR_COLORS = {",
        '"mul": "#C2410C"',
        '"div": "#B45309"',
        '"add": "#0F766E"',
        '"C: add 9 · F: ×2 · E: add 18"',
        # MathIR names the actions its key ran, as PantryPlan does.
        'mathir_menu = {"C": "add 9", "F": "×2", "E": "add 18"}',
        "NODE_PAINTS = {1: \"#C9D6E2\", 2: \"#6E8599\", 3: \"#263D51\"}",
        "INGREDIENT_COLORS = {",
    ):
        require(token in example_source, f"ModeBench example source missing {token!r}")
    for token in ("6.27", "4.52", "229.44", "5.00", "18.19", r"\renewcommand{\arraystretch}{0.92}"):
        require(token in manuscript, f"Table 1 contract missing {token!r}")
    require(
        manuscript.index(r"\label{tab:tasks}")
        < manuscript.index(r"\label{fig:modebench-examples}"),
        "Figure 2 must appear after Table 1",
    )
    for token in (
        "def render_method_trajectory(", 'letter="B"', 'title="GRPO"',
        'letter="C"', 'title="x-mode GRPO"',
        "steps = [0, 48, 96, 192, 384, 576, 768]",
        "width_ratios=[4.35, 0.65, 3.74, 0.17, 3.74]",
        "FONT = 17.0",
    ):
        require(token in source, f"missing source token {token!r}")
    # Panel C is the xGRPO trajectory beside its control, never a return of the
    # withdrawn paired-bar panel or of the fifth training epoch.
    for forbidden in ("render_paired_trajectory", "Paired trajectories",
                      "Step 960", "epoch 5"):
        require(forbidden not in source, f"forbidden source token {forbidden!r}")
    require(
        re.search(r"fontsize=(?!FONT)", source) is None,
        "figure 1 text must take its size from FONT, not a literal",
    )
    require(audit["schema"] == "paper_graph_collapse_toy_v16", "wrong audit schema")
    require(audit["layout_contract"] == {
        "panels": ["A", "B", "C"],
        "panel_titles": ["One prompt, many answers", "GRPO", "x-mode GRPO"],
        "steps": [0, 48, 96, 192, 384, 576, 768],
        "end_epoch": 4, "paired_bars": False,
    }, "wrong panel layout contract")
    expected = {"0", "48", "96", "192", "384", "576", "768"}
    require(set(audit["drgrpo_trajectory"]) == expected, "wrong Dr.GRPO checkpoints")
    require(set(audit["xdrgrpo_trajectory"]) == expected, "wrong xDr.GRPO checkpoints")
    require(audit["drgrpo_trajectory"]["768"]["distinct"] == 1, "Dr endpoint changed")
    require(audit["xdrgrpo_trajectory"]["768"]["distinct"] == 6, "xDr endpoint changed")
    for token in (r"(B--C) Four fixed-seed",
                  "contracts to one mode", "retains six",
                  "marginal value of sampling"):
        require(token in manuscript, f"missing manuscript token {token!r}")

    expected_metrics = {
        "greedy": "neutral pass@1",
        "pass8": "neutral pass@8",
        "distinct8": "mean # distinct correct@8",
        "online_canonical_tracked_outcomes": (
            "cumulative verified discoveries"
        ),
    }
    headline_metrics = literal_assignment(
        HEADLINE_SOURCE,
        "E68_TRAIN_METRICS",
    )
    require(
        list(headline_metrics) == list(expected_metrics),
        f"wrong headline metric keys: {list(headline_metrics)}",
    )
    require(
        [row["title"] for row in headline_metrics.values()]
        == list(expected_metrics.values()),
        "wrong headline metric titles",
    )

    headline_source = HEADLINE_SOURCE.read_text(encoding="utf-8")
    appendix_source = APPENDIX_SOURCE.read_text(encoding="utf-8")
    e70_terminal = json.loads(E70_TERMINAL_AUDIT.read_text(encoding="utf-8"))
    require(
        e70_terminal.get("status") == "pass"
        and e70_terminal.get("summary", {}).get("terminal_runs") == 40,
        "E70 Stage A is not a passing 40-run terminal cohort",
    )
    pantry_terminal = json.loads(
        PANTRY_TERMINAL_AUDIT.read_text(encoding="utf-8")
    )
    require(
        pantry_terminal.get("status") == "pass"
        and pantry_terminal.get("decision") == "pantry_terminal_eligible",
        "PantryPlan Stage B is not terminal-eligible",
    )
    for token in (
        "e68_plot._series",
        "def _complete_mean(",
        "e68_plot._set_y_limits",
        "e68_plot._style_axis",
        "ls=_seed_style(seed)",
        "lw=1.15",
        "lw=2.8",
        "alpha=0.68",
        "alpha=0.10",
        "PAPER_SEEDS = (43, 44, 45, 46, 47)",
        "ppe71_scale384_05b_12pass_scaling_curve.json",
    ):
        require(token in headline_source, f"headline lost E68 style token {token!r}")
    for token in (
        "linestyle=SEED_STYLES[seed]",
        "linewidth=1.15",
        "linewidth=2.8",
        "alpha=0.68",
        "alpha=0.10",
        'markerfacecolor=color if arm == CONTROL else "none"',
        "axis.tick_params(length=2.5, width=0.7, labelsize=7.5)",
        "axis.set_ylim(0.0, high * 1.08 if high > 0 else 1.0)",
        "figure = plt.figure(figsize=(9.2, 12.0))",
        "figure.add_gridspec(",
        "outer_cell.subgridspec(2, 2",
        "axis.set_box_aspect(0.80)",
    ):
        require(token in appendix_source, f"appendix lost E68 style token {token!r}")

    expected_panels = (
        ("greedy", "neutral pass@1", False),
        ("pass8", "neutral pass@8", False),
        ("distinct8", "mean # distinct correct@8", True),
        (
            "online_canonical_tracked_outcomes",
            "cumulative verified discoveries",
            True,
        ),
    )
    appendix_panels = literal_assignment(APPENDIX_SOURCE, "PANELS")
    require(appendix_panels == expected_panels, "wrong appendix panels")
    if APPENDIX_PROVENANCE.is_file():
        provenance = json.loads(APPENDIX_PROVENANCE.read_text(encoding="utf-8"))
        require(
            provenance["panels"] == [row[0] for row in expected_panels],
            "wrong appendix provenance panels",
        )

        require(
            provenance["layout"]
            == {
                "domain_card_grid": [4, 2],
                "panels_per_card": [2, 2],
                "figure_inches": [9.2, 12.0],
                "panel_box_aspect": 0.80,
            },
            "wrong appendix card layout",
        )
    for token in (
        r"\begin{figure}[p]",
        r"height=.84\textheight",
        r"Each domain is a $2\times2$ card",
        r"\label{fig:clean-cohort}",
    ):
        require(token in manuscript, f"manuscript lost full-page Figure 5 token {token!r}")
    forbidden_metrics = (
        "neutral mean@8",
        "valid coverage@8",
        "open-set predictive entropy EMA",
        "verified replay KL",
        "new verified outcome fraction",
        "mean verified support per prompt",
        "excess@8",
    )
    for path, expected_repetitions in (
        (HEADLINE_PDF, 5),
        (APPENDIX_PDF, 8),
    ):
        text = pdf_text(path)
        for title in expected_metrics.values():
            require(
                text.count(title) == expected_repetitions,
                f"{path.name}: expected {expected_repetitions} rendered "
                f"instances of {title!r}, got {text.count(title)}",
            )
        for forbidden in forbidden_metrics:
            require(
                forbidden not in text,
                f"{path.name}: forbidden rendered metric {forbidden!r}",
            )
    example_text = pdf_text(EXAMPLES_PDF)
    for token in (
        "Graph coloring", "Countdown", "Python factors", "MathIR", "PantryPlan",
        "(6 × 9) / 3", "6 + 3 + 9", "canonical keys",
        "navel_orange=75", "sunflower_seeds=50", "grape_tomatoes=75", "almonds=50",
        "C;F", "F;E",
    ):
        require(token in example_text, f"ModeBench example missing {token!r}")
    # Both keys of every domain must render and differ. Graph and PantryPlan
    # keys are drawn rather than set in type -- paint chips and pictograms --
    # so those two are held by the renderer's own validated structures.
    for first, second in (
        ("div(mul(6,9),3)", "add(9,add(3,6))"),
        ("[2, 2, 7, 3]", "[3, 41, 13, 3]"),
        ("x/2 = 8 → x = 16", "x − 18 = −2 → x = 16"),
    ):
        require(first != second, "paired keys must differ")
        require(first in example_text, f"Figure 2 missing key {first!r}")
        require(second in example_text, f"Figure 2 missing key {second!r}")
    for token in (
        'graph_modes = ((2, 1, 1, 1, 3, 2), (2, 1, 3, 1, 1, 2))',
        '"key_kind": "paints"',
        '"key_kind": "icons"',
        "def draw_paint_row(",
        "def draw_icon_row(",
    ):
        require(token in example_source, f"Figure 2 source missing {token!r}")
    require(
        example_source.count('"span": 1') == 4 and example_source.count('"span": 2') == 1,
        "Figure 2 must stay a two-column grid with one full-width row",
    )
    require(
        manuscript.count(r"figures/modebench_examples.pdf") == 1
        and "modebench_pantry_example" not in manuscript,
        "Figure 2 must be one figure, not a split pair",
    )
    mechanism_source = MECHANISM_SOURCE.read_text(encoding="utf-8")
    # The mechanism figure sets each column title on two lines, so the rendered
    # text is compared with whitespace collapsed rather than line by line.
    mechanism_text = re.sub(r"\s+", " ", pdf_text(MECHANISM_PDF))
    for source_token, rendered in (
        ('("Discover", "verified modes")', "Discover verified modes"),
        ('("Retain every", "discovery")', "Retain every discovery"),
        ('("Preserve mass", "+ rebalance")', "Preserve mass + rebalance"),
        ('("Self-referenced", "control")', "Self-referenced control"),
        ('("Singleton", "escape")', "Singleton escape"),
        ("never enters PPO", "never enters PPO"),
    ):
        require(source_token in mechanism_source, f"mechanism source missing {source_token!r}")
        require(rendered in mechanism_text, f"mechanism PDF missing {rendered!r}")
    require(
        mechanism_source.count("COLUMN_SPECS") >= 1
        and len(re.findall(r'\(\("[A-E]", "[A-Z 0-9]+"\)', mechanism_source)) == 5,
        "mechanism figure must stay one row of five columns",
    )
    if MAIN_PDF.is_file() and MAIN_PDF.stat().st_mtime >= MANUSCRIPT.stat().st_mtime:
        page_one = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "1", str(MAIN_PDF), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        require("Figure 1:" in page_one, "Figure 1 is not on page 1")
        first_nine = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "9", str(MAIN_PDF), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        # ICLR counts main text through the Conclusion against the nine-page
        # limit.  The Ethics and Reproducibility statements sit after it, do
        # not count, and must precede the references.
        page_ten = subprocess.run(
            ["pdftotext", "-f", "10", "-l", "10", str(MAIN_PDF), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        page_eleven = subprocess.run(
            ["pdftotext", "-f", "11", "-l", "11", str(MAIN_PDF), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        require("Conclusion." in first_nine, "main content exceeds nine pages")
        require(
            re.search(r"(?mi)^R\s*EFERENCES\s*$", first_nine) is None,
            "References must start on a fresh page after the main body",
        )
        # "Conclusion. in first_nine" alone passes even when the paragraph
        # spills; require page 10 to *open* with the Ethics heading so no main
        # text can leak past the nine-page limit unnoticed.
        # The template prints margin line numbers, which pdftotext emits as
        # bare-integer lines; skip them to reach the first real content line.
        page_ten_first = next(
            (
                line.strip()
                for line in page_ten.splitlines()
                if line.strip() and not line.strip().isdigit()
            ),
            "",
        )
        require(
            re.fullmatch(r"E\s*THICS\s+S\s*TATEMENT", page_ten_first, re.I)
            is not None,
            "main text spills past page 9: page 10 opens with "
            f"{page_ten_first!r}, not the Ethics Statement",
        )
        for heading, label in (
            (r"(?mi)^E\s*THICS\s+S\s*TATEMENT\s*$", "Ethics Statement"),
            (
                r"(?mi)^R\s*EPRODUCIBILITY\s+S\s*TATEMENT\s*$",
                "Reproducibility Statement",
            ),
        ):
            require(
                re.search(heading, page_ten) is not None,
                f"{label} must sit on page 10, after the nine-page main body",
            )
        require(
            re.search(r"(?mi)^R\s*EFERENCES\s*$", page_eleven) is not None,
            "References do not begin on page 11",
        )
    print(
        "Paper figure contracts: PASS "
        f"({len(abstract_words)}-word abstract, max 130; Figure 1 page 1; "
        "nine-page main body; statements page 10; References page 11; "
        "paired examples A-E; three-stage mechanism A-C; E68 metric grids)"
    )
if __name__ == "__main__":
    main()
