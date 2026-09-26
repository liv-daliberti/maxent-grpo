#!/usr/bin/env python3
"""Summarize the complete hosted figure cohorts by model and benchmark level.

Pass@8 is empirical prompt success, read from audited per-prompt counts. The
source display retains every draw, the frozen formatting normalizer, and the
complete alternative Python prompt cohort used for Claude Opus 5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from ops.plot_paper_hosted_breadth import (
        DEFAULT_SOURCE, DOMAIN_ORDER, DRAWS, LEVELS, PROMPTS, ROOT,
        build_display_data,
    )
except ModuleNotFoundError:  # Support direct execution from any working directory.
    from plot_paper_hosted_breadth import (
        DEFAULT_SOURCE, DOMAIN_ORDER, DRAWS, LEVELS, PROMPTS, ROOT,
        build_display_data,
    )

DEFAULT_OUTPUT = ROOT / "paper/results/hosted_level_averages_20260911"
PYTHON = "python_factors"
#: The four domains every deployment answers under one common prompt. A macro
#: over these holds task wording fixed across deployments; the five-domain
#: macro includes revised Python wording for Opus 5.
PARITY_DOMAINS = tuple(domain for domain in DOMAIN_ORDER if domain != PYTHON)
DOMAIN_TITLES = {"graph_coloring": "Graph", "countdown": "Countdown",
                 "python_factors": "Python", "mathir": "MathIR",
                 "pantry_plan": "Pantry"}


def _count(value: Any, field: str) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or value != int(value)):
        raise ValueError(f"{field}: expected a nonnegative integer count")
    return int(value)


def _cell_counts(cell: dict[str, Any]) -> dict[str, int]:
    counts = {field: _count(cell["counts"][field], field) for field in (
        "prompts", "responses", "correct_responses", "distinct_correct_modes",
    )}
    if counts["prompts"] != PROMPTS or counts["responses"] != PROMPTS * DRAWS:
        raise ValueError("Every cell must retain all 128 prompts and 1,024 responses")
    raw = cell["provenance"]["source_counts"]
    condition = cell["provenance"]["condition"]
    if condition == "original_benchmark_prompt":
        successful = raw["prompts_with_correct"]
    elif condition == "plain_direct_expression_no_system":
        if cell["model"] != "claude-opus-5" or cell["domain"] != "python_factors":
            raise ValueError("Only Opus 5 Python can use the alternative prompt cohort")
        successful = raw["normalized"]["prompts_with_correct_answer"]
    else:
        raise ValueError(f"Unexpected hosted prompt condition: {condition}")
    counts["prompts_with_correct"] = _count(successful, "prompts_with_correct")
    success = counts["prompts_with_correct"]
    if (success > counts["prompts"]
            or not success <= counts["distinct_correct_modes"]
            <= counts["correct_responses"] <= success * DRAWS):
        raise ValueError("Prompt-success counts are inconsistent with verified draw and mode counts")
    for metric, numerator, denominator in (
        ("accuracy", "correct_responses", "responses"),
        ("distinct8", "distinct_correct_modes", "prompts"),
    ):
        if not math.isclose(cell["metrics"][metric], counts[numerator] / counts[denominator],
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{metric}: metric does not match complete-cohort counts")
    return counts


def _aggregate(counts_by_domain: dict[str, dict[str, int]],
               domains: tuple[str, ...]) -> dict[str, Any]:
    """Equal-size domain cohorts, so summed counts are their equal-weight mean."""
    selected = [counts_by_domain[domain] for domain in domains]
    if len({tuple(sorted(item)) for item in selected}) != 1:
        raise ValueError("Domain cohorts carry different count fields")
    counts = {field: sum(item[field] for item in selected) for field in selected[0]}
    counts["domains"] = len(domains)
    return {"counts": counts,
            "metrics": {"pass8": counts["prompts_with_correct"] / counts["prompts"],
                        "distinct8": counts["distinct_correct_modes"] / counts["prompts"]}}


def build_level_averages(display: dict[str, Any]) -> list[dict[str, Any]]:
    """Average five equally sized domain cohorts, retaining zero-success prompts."""
    models = display["models"]
    model_ids = [model["model"] for model in models]
    if not model_ids or len(model_ids) != len(set(model_ids)):
        raise ValueError("Expected nonempty, distinct admitted model identities")
    if display["domains"] != list(DOMAIN_ORDER) or display["levels"] != list(LEVELS):
        raise ValueError("Expected the five original domains and all three levels")
    expected = {(model, domain, level) for model in model_ids
                for domain in DOMAIN_ORDER for level in LEVELS}
    cells = display["cells"]
    indexed = {(cell["model"], cell["domain"], cell["level"]): cell for cell in cells}
    if len(cells) != len(expected) or set(indexed) != expected:
        raise ValueError("Expected exactly one complete cell per model, domain, and level")
    if (display["sampling"]["prompts_per_cell"] != PROMPTS
            or display["sampling"]["draws_per_prompt"] != DRAWS
            or display["sampling"]["cell_count"] != len(expected)
            or display["sampling"]["displayed_responses"] != len(expected) * PROMPTS * DRAWS):
        raise ValueError("Hosted sampling totals do not match the complete domain-level grid")
    companions = {(cell["model"], cell["domain"], cell["level"]): cell
                  for cell in display["companion_cells"]}
    if any(domain != PYTHON for _, domain, _ in companions):
        raise ValueError("A substituted cell now lies outside the Python domain")
    rows = []
    for model in models:
        levels = {}
        for level in LEVELS:
            selected = [indexed[(model["model"], domain, level)] for domain in DOMAIN_ORDER]
            if any(cell["label"] != model["label"] for cell in selected):
                raise ValueError("Cell and model labels must agree")
            counts_by_domain = {domain: _cell_counts(cell)
                                for domain, cell in zip(DOMAIN_ORDER, selected)}
            levels[str(level)] = {
                **_aggregate(counts_by_domain, DOMAIN_ORDER),
                "source_cells": [cell["provenance"]["source_cell"] for cell in selected],
                # Python is the one domain whose condition is not common to every
                # deployment, so the same macro without it is carried beside the
                # five-domain reading rather than derived by a reader.
                "excluding_python": {
                    **_aggregate(counts_by_domain, PARITY_DOMAINS),
                    "excluded_domains": [PYTHON],
                },
            }
            replaced = companions.get((model["model"], PYTHON, level))
            if replaced is not None:
                levels[str(level)]["replaced_condition"] = {
                    **_aggregate({**counts_by_domain, PYTHON: _cell_counts(replaced)},
                                 DOMAIN_ORDER),
                    "condition": replaced["provenance"]["condition"],
                    "source_cell": replaced["provenance"]["source_cell"],
                }
        rows.append({"model": model["model"], "label": model["label"], "levels": levels})
    return rows


#: PCMD uses the strict, original-wording cohort, independently of the normalized
#: grading and revised Opus 5 Python condition used for success and mode counts.
#: Its per-level means must match the corresponding cohort-table columns.
#: The separate neutral-wording comparison provides descriptive sensitivity
#: estimates for its own six cells, not bounds on other wording changes.
WORDING_RECORD = ROOT / "paper/results/modebench_discovery_curves_20260911.json"
WORDING_DEPLOYMENTS = {"gpt56sol": "gpt-5.6-sol", "gpt54": "gpt-5.4", "grok43": "grok-4.3"}
WORDING_ARMS = ("original", "neutral")
WORDING_LEVELS = (2, 3)

PCMD_RECORD = ROOT / "paper/results/mode_diversity_hosted_cohort_20260917.json"
PCMD_BODY = ROOT / "paper/results/mode_diversity_hosted_cohort_table_body.tex"


#: Small counts are written as words in this paper, and the caption below is
#: generated, so the word comes from the record rather than from the sentence.
WORDS = ("no", "one", "two", "three", "four", "five", "six",
         "seven", "eight", "nine", "ten")


def _signed(value: float) -> str:
    return ("+" if value >= 0 else "-") + _pcmd(abs(value))


def _pcmd(value: float) -> str:
    text = f"{value:.3f}"
    return text[1:] if text.startswith("0.") else text


def build_pcmd_levels(substituted_cohorts: set[str]) -> dict[str, Any]:
    """Per-level macro PCMD for each deployment, as the cohort table reports it."""
    payload = json.loads(PCMD_RECORD.read_text(encoding="utf-8"))
    if payload.get("schema") != "paper-mode-diversity-hosted-cohort-v1":
        raise ValueError("hosted PCMD cohort record schema drifted")
    # The success-conditional axis must stay on one protocol for the whole
    # cohort: it is the axis the paper's diversity claims rest on, so a
    # substituted prompt cohort may not reach it even indirectly.
    if not substituted_cohorts:
        raise ValueError("Expected at least one substituted prompt cohort to exclude")
    for model in payload["models"]:
        samples = (ROOT / model["samples"]["path"]).resolve()
        if any(samples.is_relative_to(cohort) for cohort in substituted_cohorts):
            raise ValueError(f"{model['model']}: PCMD reads a substituted prompt cohort")
    body = PCMD_BODY.read_text(encoding="utf-8")
    rows = {}
    for model in payload["models"]:
        levels = {}
        for level in LEVELS:
            reported = [cell for cell in model["cells"]
                        if cell["level"] == level and cell["reportable"]]
            cells = [cell["pmd"] for cell in reported]
            levels[str(level)] = {
                "macro_pcmd": sum(cells) / len(cells) if cells else None,
                "domains": len(cells),
                "domains_without_support": sorted(
                    set(DOMAIN_ORDER) - {cell["domain"] for cell in reported}),
            }
        # The same three numbers are already printed in the cohort table; if the
        # two displays ever part company, this is where it stops.
        printed = [_pcmd(levels[str(level)]["macro_pcmd"]) if levels[str(level)]["domains"]
                   else "---" for level in LEVELS]
        line = next((row for row in body.splitlines()
                     if row.startswith(model["label"] + " &")), None)
        columns = [] if line is None else [
            part.strip() for part in line.rstrip().removesuffix(r"\\").split("&")]
        # The per-level macros are the three columns after the label, the macro
        # and its effective-mode count. Read them by that position rather than
        # from the end of the row: the table carries trailing sensitivity
        # columns, and slicing from the end silently compared against those.
        if len(columns) < 6:
            raise ValueError(f"{model['model']}: cohort table row is too short to carry "
                             "the per-level macros")
        if columns[3:6] != printed:
            raise ValueError(f"{model['model']}: per-level PCMD differs from the cohort table")
        rows[model["model"]] = {"label": model["label"], "levels": levels}
    return {"record": _binding(PCMD_RECORD), "table_body": _binding(PCMD_BODY),
            "aggregation": "Equal-weight mean of the domain cells that clear the "
                           "thirty-prompt support bar within each deployment and level.",
            "prompt_condition": "Original benchmark prompt for every deployment and domain.",
            "definition": payload["definition"], "deployments": rows}


def build_wording_sensitivity(displayed: set[str]) -> dict[str, Any]:
    """Pooled Python \pmd{} under two wordings, for the deployments that answered both."""
    payload = json.loads(WORDING_RECORD.read_text(encoding="utf-8"))
    if payload.get("experiment_status") != "complete":
        raise ValueError("The wording cohort is not a completed measurement")
    protocol = payload["protocol"]
    if tuple(protocol["arms"]) != WORDING_ARMS:
        raise ValueError("The wording cohort no longer carries the two compared arms")
    hosted = {model["model_id"]: model for model in payload["models"]
              if model.get("family") == "frontier"}
    if set(hosted) != set(WORDING_DEPLOYMENTS):
        raise ValueError("The hosted wording cohort changed which deployments it covers")
    if not set(WORDING_DEPLOYMENTS.values()) <= displayed:
        raise ValueError("A wording deployment is absent from the displayed cohort")
    rows, shifts, observed = {}, [], []
    for model_id, model in hosted.items():
        cells = model["analyses"]["normalized_secondary"]["cells"]
        levels = {}
        for level in WORDING_LEVELS:
            counts = cells[f"level{level}/{PYTHON}"]["counts"]
            arms = {}
            for arm in WORDING_ARMS:
                pairs = _count(counts[arm]["correct_pairs"], "correct_pairs")
                colliding = _count(counts[arm]["colliding_correct_pairs"], "colliding_correct_pairs")
                if not colliding <= pairs or not pairs:
                    raise ValueError(f"{model_id} level {level} {arm}: no estimable correct pairs")
                arms[arm] = 1 - colliding / pairs
            shift = arms["neutral"] - arms["original"]
            shifts.append(shift)
            observed.extend(arms.values())
            levels[str(level)] = {**arms, "shift": shift}
        rows[WORDING_DEPLOYMENTS[model_id]] = {"cohort_id": model_id, "levels": levels}
    return {"record": _binding(WORDING_RECORD),
            "metric": "Pooled correct-pair PCMD over the correct pairs of each cell.",
            "population": {"prompts_per_cell": protocol["prompts_per_cell"], "draws_per_prompt": 64,
                           "domain": PYTHON, "levels": list(WORDING_LEVELS), "arms": list(WORDING_ARMS)},
            "deployments": rows, "cells": len(shifts),
            "shift_range": [min(shifts), max(shifts)], "highest_pcmd": max(observed)}


def _binding(path: Path) -> dict[str, str]:
    path = Path(path).resolve()
    try:
        label = str(path.relative_to(ROOT))
    except ValueError:
        label = str(path)
    return {"path": label, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_record(source_path: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    source_path = Path(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    display = build_display_data(source)
    original_models = {model["model"]: model for model in source["models"]}
    for cell in display["cells"]:
        counts = _cell_counts(cell)
        if cell["provenance"]["condition"] == "original_benchmark_prompt":
            original = original_models[cell["model"]]["normalized_secondary"]["cells"][
                f"level{cell['level']}/{cell['domain']}"]
            supplied = original["metrics"]["pass8"]["estimate"]
            expected = counts["prompts_with_correct"] / counts["prompts"]
            if (isinstance(supplied, bool) or not isinstance(supplied, (int, float))
                    or not math.isclose(supplied, expected, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("pass8: metric does not match empirical prompt-success counts")
    concentration = build_pcmd_levels(
        {Path(cell["provenance"]["cohort"]).resolve() for cell in display["cells"]
         if cell["provenance"]["condition"] != "original_benchmark_prompt"})
    rows = build_level_averages(display)
    if {row["model"] for row in rows} != set(concentration["deployments"]):
        raise ValueError("Every displayed deployment must carry a PCMD reading")
    if any(concentration["deployments"][row["model"]]["label"] != row["label"] for row in rows):
        raise ValueError("PCMD and level-average labels disagree for a deployment")
    return {
        "schema": "hosted-level-averages-v1",
        "source": _binding(source_path),
        "renderer": _binding(Path(__file__)),
        "cohort_builder": _binding(ROOT / "ops/plot_paper_hosted_breadth.py"),
        "aggregation": "Equal-domain arithmetic mean over all five domains within each model and level; 128 prompts per domain, including zero-success prompts.",
        "metric_definitions": {
            "pass8": "Fraction of prompts with at least one verified response in the eight observed draws, computed from empirical prompt-success counts.",
            "distinct8": "Mean number of distinct verified canonical modes among eight responses per prompt, including zero for prompts without a verified response.",
        },
        "display": display,
        "rows": rows,
        "concentration": concentration,
        "wording_sensitivity": build_wording_sensitivity({row["model"] for row in rows}),
    }


def _tex(value: str) -> str:
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in value)


def _reduced(record: dict[str, Any]) -> str:
    """How many displayed cells rest on fewer than all five domains."""
    marked = [level for deployment in record["concentration"]["deployments"].values()
              for level in deployment["levels"].values()
              if level["macro_pcmd"] is not None and level["domains"] != len(DOMAIN_ORDER)]
    if any(level["domains_without_support"] != ["python_factors"] for level in marked):
        raise ValueError("A reduced PCMD cell now misses a domain other than Python")
    count = len(marked)
    word = WORDS[count] if count < len(WORDS) else str(count)
    return f"{word} cell{'' if count == 1 else 's'}"


def _substituted(record: dict[str, Any]) -> str:
    """How many displayed cells rest on a prompt the rest of the cohort did not answer."""
    marked = [(row["label"], level) for row in record["rows"] for level in LEVELS
              if "replaced_condition" in row["levels"][str(level)]]
    labels = sorted({label for label, _ in marked})
    if len(labels) != 1:
        raise ValueError("The substituted-prompt caption assumes a single deployment")
    count = len(marked)
    word = WORDS[count] if count < len(WORDS) else str(count)
    return f"the {word} {_tex(labels[0])} cell{'' if count == 1 else 's'}"


def render_table(record: dict[str, Any]) -> str:
    lines = [
        "% Generated by ops/build_paper_hosted_level_averages.py; do not edit.",
        r"\begin{table}[H]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        # Ten columns of seven deployments do not fit the text block; the body
        # is scaled to it rather than bled past it, as elsewhere in this paper.
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lrrrrrrrrr}",
        r"    \toprule",
        r"    & \multicolumn{3}{c}{Level 1} & \multicolumn{3}{c}{Level 2} & \multicolumn{3}{c}{Level 3} \\",
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"    Model" + r" & \texttt{pass@8} (\%) & \# modes & \pmd{}" * 3 + r" \\",
        r"    \midrule",
    ]
    for row in record["rows"]:
        icon = record["display"]["icons"][row["model"]]
        icon_path = ROOT / icon["path"]
        if hashlib.sha256(icon_path.read_bytes()).hexdigest() != icon["sha256"]:
            raise ValueError(f"{row['model']}: icon differs from its source binding")
        relative_icon = icon_path.relative_to(ROOT / "paper").as_posix()
        model_label = (r"\raisebox{-0.15em}{\includegraphics[width=1.05em,height=1.05em,keepaspectratio]{"
                       + relative_icon + r"}}\hspace{0.4em}" + _tex(row["label"]))
        values = [model_label]
        concentration = record["concentration"]["deployments"][row["model"]]["levels"]
        for level in LEVELS:
            metrics = row["levels"][str(level)]["metrics"]
            # Two columns of this row rest on a prompt the other deployments did
            # not answer; the mark is derived from the cell sources, not asserted.
            substituted = "replaced_condition" in row["levels"][str(level)]
            # \textddagger is unavailable in T1; the math symbol is not.
            mark = r"\textsuperscript{\ensuremath{\ddagger}}" if substituted else ""
            values.extend((f"{100 * metrics['pass8']:.1f}" + mark,
                           f"{metrics['distinct8']:.2f}" + mark))
            cell = concentration[str(level)]
            if cell["macro_pcmd"] is None:
                values.append("---")
            else:
                # A cell below the reporting threshold is excluded from the
                # level mean; formatting failures and refusals have different causes.
                mark = "" if cell["domains"] == len(DOMAIN_ORDER) else r"\textsuperscript{\textdagger}"
                values.append(_pcmd(cell["macro_pcmd"]) + mark)
        lines.append("    " + " & ".join(values) + r" \\")
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}}",
        r"  \caption{\textbf{High prompt success coexists with few verified solution modes.}",
        r"  Each deployment requests medium reasoning effort, with 128 prompts per domain",
        r"  and level and eight responses per prompt. \texttt{pass@8} and \# modes use",
        r"  formatting-normalized grading and average the five domains equally:",
        r"  \texttt{pass@8} is empirical prompt success; \# modes is mean",
        r"  \texttt{distinct@8}, including zero for prompts without a verified response.",
        f"  \\textsuperscript{{\\ensuremath{{\\ddagger}}}} marks {_substituted(record)} using",
        r"  revised Python wording without a system message (App.~\ref{app:hosted-python-prompt}).",
        r"  \pmd{} instead uses strict grading and the original wording throughout,",
        r"  averaging per-prompt diversity over prompts with at least two verified draws",
        r"  and then equally over reportable domains. These are the per-level columns of",
        r"  Table~\ref{tab:hosted-cohort-mode-diversity}, not its common-cell macro.",
        f"  \\textsuperscript{{\\textdagger}} marks {_reduced(record)} omitting Python because fewer",
        r"  than thirty prompts qualify: GPT-5.6 Sol has strict-formatting failures;",
        r"  Opus 5 has provider refusals. All entries are point estimates.",
        r"  Table~\ref{tab:hosted-level-averages-parity} gives normalized success and mode",
        r"  counts on the four domains with common wording.}",
        r"  \label{tab:hosted-level-averages-medium}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def _readings(record: dict[str, Any]) -> dict[str, dict[int, dict[str, float]]]:
    """Verified-mode averages per deployment under each of the three readings."""
    readings: dict[str, dict[int, dict[str, float]]] = {
        "five_domains": {}, "four_domains": {}, "five_domains_original_python": {}}
    for level in LEVELS:
        for name, pointer in (("five_domains", ()),
                              ("four_domains", ("excluding_python",)),
                              ("five_domains_original_python", ("replaced_condition",))):
            values = {}
            for row in record["rows"]:
                cell = row["levels"][str(level)]
                for step in pointer:
                    if step == "replaced_condition":
                        # Deployments without a revised condition already use
                        # original Python wording; retain them in the ranking.
                        cell = cell.get(step, cell)
                    else:
                        cell = cell.get(step) if isinstance(cell, dict) else None
                    if cell is None:
                        break
                if cell is not None:
                    values[row["label"]] = cell["metrics"]["distinct8"]
            readings[name][level] = values
    return readings


def _least_varied(record: dict[str, Any], readings: dict[str, Any]) -> str:
    """The substituted deployment must stay lowest, or this claim is not made."""
    substituted = sorted({row["label"] for row in record["rows"] for level in LEVELS
                          if "replaced_condition" in row["levels"][str(level)]})
    if len(substituted) != 1:
        raise ValueError("The parity reading assumes a single substituted deployment")
    label = substituted[0]
    for name, by_level in readings.items():
        for level, values in by_level.items():
            if not values:
                raise ValueError(f"{name} level {level}: no deployment carries this reading")
            if label not in values or min(values, key=values.get) != label:
                raise ValueError(
                    f"{label} is no longer the least varied deployment under {name} "
                    f"at Level {level}; the parity paragraph must be rewritten")
    return label


def _series(values: dict[int, dict[str, float]], label: str) -> str:
    return "/".join(f"{values[level][label]:.2f}" for level in LEVELS)


def render_parity_table(record: dict[str, Any]) -> str:
    """The same comparison over the four domains every deployment answers alike."""
    readings = _readings(record)
    label = _least_varied(record, readings)
    wording = record["wording_sensitivity"]
    domains = ", ".join(_tex(DOMAIN_TITLES[domain]) for domain in PARITY_DOMAINS[:-1])
    domains += f" and {_tex(DOMAIN_TITLES[PARITY_DOMAINS[-1]])}"
    lines = [
        "% Generated by ops/build_paper_hosted_level_averages.py; do not edit.",
        r"\paragraph{Comparison with common prompts.}",
        f"The accuracy panels of Fig.~\\ref{{fig:hosted-verified-breadth}} and the",
        r"\texttt{pass@8}/\texttt{distinct@8} columns of",
        r"Table~\ref{tab:hosted-level-averages-medium} use normalized grading, with",
        f"revised Python wording for {_tex(label)}. The lower figure panels and the",
        r"table's \pmd{} columns instead use strict grading and original wording.",
        r"Table~\ref{tab:hosted-level-averages-parity}",
        f"compares normalized success and mode counts over {domains},",
        f"the {WORDS[len(PARITY_DOMAINS)]} domains with identical prompts across deployments.",
        f"{_tex(label)} averages {_series(readings['four_domains'], label)} verified modes",
        f"at Levels 1--3, compared with {_series(readings['five_domains'], label)} over five",
        f"domains using revised Python wording and {_series(readings['five_domains_original_python'], label)}",
        r"using original Python wording. It has the lowest mean \texttt{distinct@8} at",
        r"each level in all three comparisons. The revised condition increases its verified",
        r"Python mode count, which also depends on response success; these means do not",
        r"compare diversity at matched accuracy. The ordering of the other deployments",
        r"depends on whether Python is included.",
        "",
        f"A separate Python comparison in App.~\\ref{{app:sampling-budget-effects}} uses",
        f"original and neutral wording on {WORDS[len(wording['deployments'])]} deployments,",
        f"with {wording['population']['prompts_per_cell']} prompts and {wording['population']['draws_per_prompt']}",
        f"responses per prompt at Levels 2 and 3. Pair-pooled diversity $1-C$ rises by",
        f"{_signed(wording['shift_range'][0])} to {_signed(wording['shift_range'][1])} across its",
        f"{WORDS[wording['cells']]} deployment--level cells; the largest value under either",
        f"wording is {_pcmd(wording['highest_pcmd'])} (effective-mode transform",
        f"{1 / (1 - wording['highest_pcmd']):.2f}). These values weight correct pairs equally,",
        r"unlike promptwise \pmd{}. They describe that neutral-wording comparison and",
        r"do not bound the effect of other wordings or the revised no-system condition",
        r"on untested deployments.",
        "",
        r"\begin{table}[H]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{Level 1} & \multicolumn{2}{c}{Level 2} & \multicolumn{2}{c}{Level 3} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"    Model" + r" & \texttt{pass@8} (\%) & \# modes" * 3 + r" \\",
        r"    \midrule",
    ]
    for row in record["rows"]:
        values = [_tex(row["label"])]
        for level in LEVELS:
            metrics = row["levels"][str(level)]["excluding_python"]["metrics"]
            values.extend((f"{100 * metrics['pass8']:.1f}", f"{metrics['distinct8']:.2f}"))
        lines.append("    " + " & ".join(values) + r" \\")
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{\textbf{Few verified modes remain when all deployments share the same prompts.}",
        f"  Equal-domain means over {domains}, with 128 prompts",
        r"  per domain and level and eight responses per prompt, under normalized grading.",
        r"  Python is excluded for every deployment. \texttt{pass@8} is empirical prompt",
        r"  success; \# modes is mean \texttt{distinct@8} over all prompts, including",
        r"  those with no verified response. Entries are point estimates; response",
        r"  accuracy and provider compute are not matched.}",
        r"  \label{tab:hosted-level-averages-parity}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output stem for the .tex and .json artifacts")
    args = parser.parse_args(argv)
    record = build_record(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    args.output.with_suffix(".tex").write_text(render_table(record), encoding="utf-8")
    parity = args.output.with_name(args.output.name + "_parity").with_suffix(".tex")
    parity.write_text(render_parity_table(record), encoding="utf-8")
    print(f"Wrote {args.output.with_suffix('.tex')}, {parity} "
          f"and {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
