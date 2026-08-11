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
MECHANISM_SOURCE = ROOT / "ops/plot_paper_verified_replay_mechanism.py"
MECHANISM_PDF = ROOT / "paper/figures/verified_replay_mechanism.pdf"
MAIN_PDF = ROOT / "paper/main.pdf"
INTERIM_FIGURE4 = ROOT / "paper/figures/figure4_interim_20260806.json"
INTERIM_TABLE = ROOT / "paper/results/figure4_interim_20260806_table.json"
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


B1B_SUMMARY = ROOT / "var/artifacts/e72_b1b_summary.json"
E73_SUMMARY = ROOT / "var/artifacts/e73_falcon_cross_family_summary_v2.json"
REHEARSAL_TABLE_ROWS = {
    "Graph coloring": ("graph_coloring", "E"),
    "Countdown": ("countdown", "E"),
    "Python factors": ("python_factors", "rehearsal better"),
    "MathIR action menu": ("mathir", "E"),
    "PantryPlan": ("pantry_plan", "balance better"),
}


def check_rehearsal_table(manuscript: str) -> None:
    """Bind the clean rehearsal/replay-plus-balance table to its artifact."""
    if not B1B_SUMMARY.is_file():
        return
    summary = json.loads(B1B_SUMMARY.read_text())
    require(
        summary.get("all_25_cells_terminal_and_valid") is True,
        "rehearsal table is present without 25 valid terminal cells",
    )
    by_domain = {entry["domain"]: entry for entry in summary["domains"]}
    section = manuscript.split(r"\label{tab:rehearsal-ablation}", 1)[0]
    section = section.rsplit(r"\begin{tabular}", 1)[-1]

    def decimal(value: float) -> str:
        rendered = f"{value:.3f}"
        return rendered[1:] if rendered.startswith("0.") else rendered

    for title, (domain, effect) in REHEARSAL_TABLE_ROWS.items():
        entry = by_domain.get(domain)
        require(
            entry is not None and entry.get("reportable"),
            f"rehearsal table shows {title} but its domain is withheld",
        )
        rows = re.findall(
            rf"^\s*{re.escape(title)}\s*&.*?\\\\",
            section,
            flags=re.MULTILINE | re.DOTALL,
        )
        require(len(rows) == 1, f"rehearsal table is missing {title}")
        cells = [cell.strip() for cell in rows[0].split("&")]
        expected = (
            decimal(entry["means"]["drgrpo"]["distinct8"]),
            decimal(entry["means"]["b1b"]["distinct8"]),
            decimal(entry["means"]["b1a"]["distinct8"]),
            effect,
        )
        printed = (
            cells[1],
            cells[2],
            cells[3],
            cells[4].split(r"\\", 1)[0].strip(),
        )
        require(
            printed == expected,
            f"rehearsal table {title} shows {printed!r}, artifact says {expected!r}",
        )


def check_cross_family_table(manuscript: str) -> None:
    'Bind the terminal Falcon table to all fifty audited cells.'
    if not E73_SUMMARY.is_file():
        return
    summary = json.loads(E73_SUMMARY.read_text())
    require(
        summary.get("all_50_cells_terminal_and_valid") is True,
        "cross-family table is present without 50 valid terminal cells",
    )
    outcomes = summary.get("registered_directional_outcomes", {})
    require(
        outcomes.get("distinct_positive_domains") == 5
        and outcomes.get("pass8_nonnegative_domains") == 5
        and outcomes.get("distinct_paired_wins") == 25
        and outcomes.get("pass8_paired_wins") == 25,
        "cross-family directional claim does not match the terminal artifact",
    )
    section = manuscript.split(r"\label{tab:cross-family}", 1)[0]
    section = section.rsplit(r"\begin{tabular}", 1)[-1]
    titles = {
        "Graph coloring": "graph_coloring",
        "Countdown": "countdown",
        "Python factors": "python_factors",
        "MathIR": "mathir",
        "PantryPlan": "pantry_plan",
    }

    def clean(cell: str) -> str:
        cell = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", cell)
        return " ".join(cell.split())

    def decimal3(value: float) -> str:
        rendered = f"{value:.3f}"
        return rendered[1:] if rendered.startswith("0.") else rendered

    for title, domain in titles.items():
        entry = summary["domains"].get(domain)
        require(
            entry is not None and entry.get("complete")
            and entry.get("common_endpoint") == 4608,
            f"cross-family table shows {title} without a terminal domain",
        )
        match = re.search(
            rf"^\s*{re.escape(title)}\s*&\s*matched Dr\.GRPO\s*&"
            rf"(.*?)\\\\\s*&\s*\\historicalt\{{\}}\s*&(.*?)\\\\",
            section,
            flags=re.MULTILINE | re.DOTALL,
        )
        require(match is not None, f"cross-family table is missing {title}")
        printed = tuple(
            clean(cell)
            for group in match.groups()
            for cell in group.split("&")
        )
        expected_cells: list[str] = []
        for arm in ("drgrpo", "xgrpo"):
            means = entry["means"][arm]
            passed = means["pass_at_8"]
            expected_cells.extend(
                (
                    decimal3(means["pass_at_1"]),
                    decimal3(means["mean_at_8"]),
                    decimal3(passed),
                    f"{means['distinct_at_8']:.2f}",
                    f"{means['distinct_at_8'] / passed:.2f}" if passed else "--",
                )
            )
        expected = tuple(expected_cells)
        require(
            printed == expected,
            f"cross-family table {title} shows {printed!r}, "
            f"artifact says {expected!r}",
        )


INTERIM_DOMAIN_TITLES = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
    "point_maze": "PointMaze",
}


def check_interim_replay_table(manuscript: str) -> None:
    """Bind the explicitly interim main-text table to its dated snapshot."""

    require(INTERIM_FIGURE4.is_file(), "dated interim Figure 4 JSON is missing")
    figure_payload = json.loads(INTERIM_FIGURE4.read_text())
    require(
        figure_payload.get("schema") == "figure4_multimodel_interim_v2",
        "dated interim Figure 4 has the wrong schema",
    )
    require(
        figure_payload.get("generated_at") == "2026-08-11T17:59:49.434939+00:00",
        "dated interim Figure 4 timestamp drifted",
    )
    require(
        r"\label{fig:interim-replay}" in manuscript
        and r"\label{tab:interim-replay}" in manuscript,
        "interim figure or table is absent from the manuscript",
    )
    require(INTERIM_TABLE.is_file(), "dated interim four-metric table JSON is missing")
    payload = json.loads(INTERIM_TABLE.read_text())
    require(
        payload.get("schema") == "figure4_interim_four_metric_table_v1",
        "dated interim table has the wrong schema",
    )
    require(
        payload.get("figure_snapshot_generated_at") == figure_payload.get("generated_at"),
        "dated interim table is not bound to the frozen Figure 4 snapshot",
    )
    table = manuscript.split(r"\label{tab:interim-replay}", 1)[0]
    table = table.rsplit(r"\begin{tabular}", 1)[-1]
    blocks = {
        "Qwen2.5-0.5B": table.split(r"\qwenmark{}2.5-0.5B", 1)[1].split(
            "Falcon3-1B", 1
        )[0],
        "Falcon3-1B": table.split("Falcon3-1B", 1)[1].split(
            r"\qwenmark{}2.5-3B", 1
        )[0],
        "Qwen2.5-3B": table.split(r"\qwenmark{}2.5-3B", 1)[1],
    }

    def decimal3(value: float) -> str:
        rendered = f"{value:.3f}"
        if rendered.startswith("0."):
            return rendered[1:]
        if rendered.startswith("-0."):
            return "-" + rendered[2:]
        return rendered

    expected_rows = 0
    for family, family_record in payload["families"].items():
        for domain, domain_record in family_record["domains"].items():
            expected_rows += 1
            title = INTERIM_DOMAIN_TITLES[domain]
            match = re.search(
                rf"&\s*{re.escape(title)}\s*&(.*?)\\\\",
                blocks[family],
            )
            require(match is not None, f"interim table is missing {family}/{title}")
            raw_cells = tuple(
                " ".join(cell.split()) for cell in match.group(1).split("&")
            )
            highlighted = tuple(r"\bettercell{" in cell for cell in raw_cells)
            printed = tuple(
                re.sub(r"\\bettercell\{([^{}]+)\}", r"\1", cell)
                for cell in raw_cells
            )
            expected_cells = [
                f"{len(domain_record['paired_seeds'])}@{float(domain_record['pass']):.1f}"
            ]
            for arm in ("control", "replay"):
                means = domain_record["arms"][arm]["means"]
                for metric in ("pass1", "pass8", "mean8", "distinct8"):
                    value = means.get(metric)
                    expected_cells.append("--" if value is None else decimal3(float(value)))
            expected = tuple(expected_cells)
            require(
                not any(highlighted[:5]),
                f"interim table shades a non-treatment cell in {family}/{title}",
            )
            control_means = domain_record["arms"]["control"]["means"]
            replay_means = domain_record["arms"]["replay"]["means"]
            for metric_index, metric in enumerate(
                ("pass1", "pass8", "mean8", "distinct8")
            ):
                control_value = control_means.get(metric)
                replay_value = replay_means.get(metric)
                should_highlight = (
                    control_value is not None
                    and replay_value is not None
                    and float(replay_value) > float(control_value)
                )
                require(
                    highlighted[5 + metric_index] == should_highlight,
                    f"interim table highlight is wrong for {family}/{title}/{metric}",
                )
            require(
                printed == expected,
                f"interim table {family}/{title} shows {printed!r}, "
                f"snapshot says {expected!r}",
            )
    # The loop above proves every snapshot panel appears in the table. Count
    # the printed rows to prove the converse, so a panel that leaves the
    # snapshot cannot survive as a stale row. Binding this to the snapshot
    # rather than to a fixed count lets newly evaluable panels through, which
    # is the whole point of an interim table.
    printed_rows = sum(len(re.findall(r"\\\\", block)) for block in blocks.values())
    require(
        printed_rows == expected_rows,
        f"interim table prints {printed_rows} rows, snapshot has {expected_rows}",
    )

def main() -> None:
    source = SOURCE.read_text()
    manuscript = MANUSCRIPT.read_text()
    audit = json.loads(AUDIT.read_text())
    dated_snapshot = (
        r"\label{fig:interim-replay}" in manuscript
        and r"\textbf{Snapshot boundary.}" in manuscript
    )
    main_page_limit = 10 if dated_snapshot else 9
    page_contract = "snapshot limit 10" if dated_snapshot else "final limit 9"
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
        "Inference-time scaling requires genuine solution diversity",
        "Useful response breadth",
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
        "breaks that trade", "Inference-time scaling requires genuine solution diversity",
        "Useful response breadth", "Prior work documents",
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
        "validator-produced canonical execution",
        "same validator determines correctness",
        r"\xmode{} adds one intervention",
    ):
        require(token in introduction, f"Introduction contract missing {token!r}")
    for forbidden in (
        "flattens what it means to be right",
        "plurality matters everywhere",
        "correct-answer effect",
        "open-set executable task set",
        "semantic identity binds correctness",
        "common retention mechanism",
    ):
        require(
            forbidden not in introduction,
            f"Introduction retains superseded wording {forbidden!r}",
        )
    # Anchored on the section, not on a "Conclusion." run-in: discussion,
    # limitations, and the closing argument are now bolded paragraphs of one
    # Conclusion section, which spends two section rules on text instead.
    require(
        r"\section{Conclusion}" in manuscript,
        "manuscript lost its Conclusion section",
    )
    conclusion = manuscript.split(r"\section{Conclusion}", 1)[1].split(
        "bibliographystyle", 1
    )[0]
    conclusion_flat = " ".join(conclusion.split())
    for token in (
        "Capacity-16 banks can omit later discoveries",
        "fixed $.10$ replay dose is neither ablated nor claimed optimal",
        "conditional post-discovery retention guarantee",
        "no neural performance claim",
        "More verified modes are not inherently better",
    ):
        require(token in conclusion_flat, f"Conclusion scope missing {token!r}")
    related = manuscript.split(r"\section{Related Work}", 1)[1].split(
        r"\section{Experimental Design and Evidence Policy}", 1
    )[0]
    require(
        related.count(r"\noindent\textbf{") == 3,
        "Related Work must contain exactly three bold areas",
    )
    for token in (
        "Experience replay, archives, and mode-covering objectives",
        "bengio2023gflownet", "hu2024amortizing",
        "lehman2011novelty", "mouret2015illuminating",
        "yue2025rlvrlimit", "kirk2024understanding",
    ):
        require(token in related, f"Related Work coverage missing {token!r}")
    main_body = manuscript.split(r"\appendix", 1)[0]
    appendix_labels = (
        "app:theory", "app:python", "app:prompts", "app:data", "app:algorithm",
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
    check_rehearsal_table(manuscript)
    check_interim_replay_table(manuscript)
    check_cross_family_table(manuscript)
    algorithm_appendix = manuscript.split(r"\section{Algorithmic Details}", 1)[1].split(
        r"\section{Why x-Mode GRPO Is Minimal}", 1
    )[0]
    for token in (
        r"\usepackage{algorithm}", r"\usepackage{algorithmic}",
        r"\begin{algorithm}[H]", r"\begin{algorithmic}[1]",
        r"\label{alg:verified-replay-update}", r"\mathcal B_x", r"\tau",
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
    # These tokens span line breaks, so they are matched against a
    # whitespace-normalized copy: rewrapping a paragraph is not a regression,
    # and pinning the exact wrap made it look like one twice.
    collapsed = " ".join(manuscript.split())
    for token in (
        "Why x-Mode GRPO Is Minimal",
        "Uniform verified replay",
        "Deterministic recurrent schedule",
        "Exact-zero replay control",
        "No effect is pooled across domains",
        "historical provenance",
        r"not components of \xmode{}",
    ):
        require(
            " ".join(token.split()) in collapsed,
            f"ablation contract missing {token!r}",
        )
    for forbidden in (
        "sole terminal causal performance",
        "Only the separated-support actuator row is a completed causal",
    ):
        require(
            " ".join(forbidden.split()) not in collapsed,
            f"ablation contract regressed to single-contrast language {forbidden!r}",
        )
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
        '"mul": "#20068f"',
        '"div": "#7a02a8"',
        '"add": "#bc3587"',
        # MathIR names the actions its key ran, as PantryPlan does, and prints
        # them from the released menu so the panel shows a legal action neither
        # answer selects rather than only the ones they took.
        'mathir_elided = ("A", "B")',
        '"D": {"action": "sub(c)", "name": "add 1"}',
        'assert sorted(set(mathir_menu) - selected) == ["D"]',
        "NODE_PAINTS = {1: \"#e16462\", 2: \"#9e199d\", 3: \"#2f0596\"}",
        "INGREDIENT_COLORS = {",
    ):
        require(token in example_source, f"ModeBench example source missing {token!r}")
    for token in ("6.27", "4.52", "229.44", "5.00", "18.19", r"\renewcommand{\arraystretch}{1.08}"):
        require(token in manuscript, f"Table 1 contract missing {token!r}")
    # The domain specification table now sits beside the audits that verify it,
    # so the main body must still carry the two numbers a reader needs to size
    # the claim, and the table must be in the appendix rather than before the
    # examples figure.
    require(
        "trains on 384 prompts and evaluates on a fixed 128-prompt split"
        in " ".join(manuscript.split()),
        "main body lost the training/evaluation split sizes",
    )
    require(
        manuscript.index(r"\label{tab:tasks}")
        > manuscript.index(r"\label{fig:modebench-examples}"),
        "the domain specification table should follow the examples figure",
    )
    for token in (
        "def render_method_trajectory(", 'letter="B"', 'title="GRPO"',
        'letter="C"', 'title="historical treatment"',
        "DISPLAY_STEPS = [0, 48, 96, 192, 384, 768, 1152]",
        # The five column proportions, still pinned to these exact numbers.
        # They moved from the add_gridspec call into a named list because the
        # B/C card and its key are now centred on the block those ratios
        # define, and deriving that from the same list keeps the two in step.
        "ratios = [4.05, 1.32, 4.02, 0.17, 4.02]",
        # The story figure is drawn on an oversized canvas and scaled to
        # \linewidth, so its type size is derived rather than literal;
        # pin the derivation, which is what keeps it legible in print.
        "FONT = style.font_for_canvas(CANVAS_WIDTH)",
    ):
        require(token in source, f"missing source token {token!r}")
    # Panel C is the historical-treatment trajectory beside its control, never a return of the
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
        # Panel A is titled with the prompt question itself, so the figure no
        # longer carries a separate banner repeating it.
        "panel_titles": [
            "How can we color the three uncolored nodes so that connected "
            "nodes get different colors?",
            "GRPO",
            "historical treatment",
        ],
        "steps": [0, 48, 96, 192, 384, 768, 1152],
        "end_epoch": 6, "paired_bars": False,
    }, "wrong panel layout contract")
    expected = {"0", "48", "96", "192", "384", "768", "1152"}
    require(set(audit["drgrpo_trajectory"]) == expected, "wrong Dr.GRPO checkpoints")
    require(set(audit["xdrgrpo_trajectory"]) == expected, "wrong xDr.GRPO checkpoints")
    # Pinned at the window's own last checkpoint rather than a hardcoded 768,
    # which stopped being the endpoint when the window moved to six epochs and
    # would have kept passing while checking an interior point.
    endpoint = max(expected, key=int)
    require(audit["drgrpo_trajectory"][endpoint]["distinct"] == 1, "Dr endpoint changed")
    require(audit["xdrgrpo_trajectory"][endpoint]["distinct"] == 7, "xDr endpoint changed")
    for token in (r"(B--C) Four fixed-seed",
                  "contracts to one mode", "retains seven",
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
    # Line weights, seed styling, and alphas moved into ops/paper_style.py so
    # every figure shares one specification. The guard follows them there: both
    # figure sources must draw from that module, and the module must still
    # define the constants, rather than each figure pinning its own literals.
    style_source = (ROOT / "ops/paper_style.py").read_text()
    for token in (
        "MEAN_LW = 1.35",
        "SEED_LW = 0.5",
        "BAND_ALPHA = 0.13",
        "CONTROL = ",
        "METHOD = ",
        "def style_axis(",
    ):
        require(token in style_source, f"shared paper style lost {token!r}")
    for token in (
        "e68_plot._series",
        "def _complete_mean(",
        "e68_plot._set_y_limits",
        "import paper_style as style",
        "style.style_axis(",
        "PAPER_SEEDS = (43, 44, 45, 46, 47)",
        "ppe71_scale384_05b_12pass_scaling_curve.json",
    ):
        require(token in headline_source, f"headline lost E68 style token {token!r}")
    for token in (
        "import paper_style as style",
        "axis.set_ylim(0.0, high * 1.08 if high > 0 else 1.0)",
        # Canvas geometry is now derived from the shared column width rather
        # than a literal, so pin the derivation and the grid, not the inches.
        "figsize=(style.WIDTH,",
        "figure.add_gridspec(",
        "outer_cell.subgridspec(1, len(PANELS)",
        "axis.set_box_aspect(0.62)",
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
    # The full-page cohort surface was removed from the manuscript: it was
    # visually redundant with the training-dynamics grid and read as an
    # internal monitoring panel --- timestamped, captioned in "available-seed
    # mean" and "latest mean" language --- rather than as a result. The
    # generator and its provenance are retained and still checked above, so the
    # surface can be regenerated for audit, but it must not return to the paper
    # in live-dashboard form.
    for forbidden in (
        r"\label{fig:clean-cohort}",
        r"height=.84\textheight",
        "available-seed mean",
        "latest mean",
    ):
        require(
            forbidden not in manuscript,
            f"manuscript reintroduced the removed cohort dashboard {forbidden!r}",
        )
    forbidden_metrics = (
        "neutral mean@8",
        "valid coverage@8",
        "open-set predictive entropy EMA",
        "verified replay KL",
        "new verified outcome fraction",
        "mean verified support per prompt",
        "excess@8",
    )
    # Only the headline surface is rendered into the manuscript now; the
    # appendix cohort dashboard was removed, so its PDF is no longer a paper
    # artifact to hold to the manuscript's metric vocabulary.
    for path, expected_repetitions in ((HEADLINE_PDF, 5),):
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
        "PointMaze", "corridor_0+", "corridor_1+",
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
        '"glyph": "maze"',
        "def draw_paint_row(",
        "def draw_icon_row(",
        "def draw_mini_maze(",
    ):
        require(token in example_source, f"Figure 2 source missing {token!r}")
    require(
        example_source.count('"span": 1') == 6
        and example_source.count('"span": 2') == 0,
        "Figure 2 must stay six equal-width environment cards on a three-column grid",
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
        ('("Verify", "outputs")', "Verify outputs"),
        ('("Store", "exemplars")', "Store exemplars"),
        ('("Revisit", "recurrently")', "Revisit recurrently"),
        ('("Replay", "uniformly")', "Replay uniformly"),
        ("every banked mode", "every banked mode"),
    ):
        require(source_token in mechanism_source, f"mechanism source missing {source_token!r}")
        require(rendered in mechanism_text, f"mechanism PDF missing {rendered!r}")
    require(
        mechanism_source.count("COLUMN_SPECS") >= 1
        and len(re.findall(r'\(\("[A-D]", "[A-Z 0-9]+"\)', mechanism_source)) == 4,
        "mechanism figure must stay one row of four columns",
    )
    if MAIN_PDF.is_file() and MAIN_PDF.stat().st_mtime >= MANUSCRIPT.stat().st_mtime:
        page_one = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "1", str(MAIN_PDF), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        require("Figure 1:" in page_one, "Figure 1 is not on page 1")
        # ICLR counts main text through the Conclusion against nine pages.  A
        # source-labeled interim snapshot may temporarily use page 10; deleting
        # either the snapshot or its over-length disclosure restores the strict
        # final limit automatically.
        page_text = {
            page: subprocess.run(
                [
                    "pdftotext", "-f", str(page), "-l", str(page),
                    str(MAIN_PDF), "-",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            for page in range(1, 13)
        }

        def heading_page(pattern: str) -> int | None:
            return next(
                (
                    page
                    for page, text in page_text.items()
                    if re.search(pattern, text) is not None
                ),
                None,
            )

        conclusion_page = heading_page(r"(?mi)^\s*(?:\d+\s+)?C\s*ONCLUSION\s*$")
        ethics_page = heading_page(r"(?mi)^\s*E\s*THICS\s+S\s*TATEMENT\s*$")
        reproducibility_page = heading_page(
            r"(?mi)^\s*R\s*EPRODUCIBILITY\s+S\s*TATEMENT\s*$"
        )
        references_page = heading_page(r"(?mi)^\s*R\s*EFERENCES\s*$")
        require(
            conclusion_page is not None and conclusion_page <= main_page_limit,
            f"main content exceeds the {page_contract}",
        )
        require(
            ethics_page is not None and ethics_page <= main_page_limit + 1,
            "Ethics Statement begins after the allowed main-body boundary",
        )
        if ethics_page == main_page_limit + 1:
            page_after_main_first = next(
                (
                    line.strip()
                    for line in page_text[main_page_limit + 1].splitlines()
                    if line.strip() and not line.strip().isdigit()
                ),
                "",
            )
            require(
                re.fullmatch(
                    r"E\s*THICS\s+S\s*TATEMENT", page_after_main_first, re.I
                ) is not None,
                f"main text spills past page {main_page_limit}: the next page "
                f"opens with {page_after_main_first!r}, not the Ethics Statement",
            )
        require(
            reproducibility_page is not None
            and ethics_page is not None
            and reproducibility_page >= ethics_page,
            "Reproducibility Statement must follow the Ethics Statement",
        )
        require(
            references_page is not None
            and reproducibility_page is not None
            and references_page > reproducibility_page,
            "References must start on a fresh page after the statements",
        )
    print(
        "Paper figure contracts: PASS "
        f"({len(abstract_words)}-word abstract, max 130; Figure 1 page 1; "
        f"main body satisfies {page_contract}; statements precede References; "
        "paired examples A-F; four-stage verified replay A-D; audit metric grids)"
    )
if __name__ == "__main__":
    main()
