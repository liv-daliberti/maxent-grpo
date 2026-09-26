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
    rows = []
    for model in models:
        levels = {}
        for level in LEVELS:
            selected = [indexed[(model["model"], domain, level)] for domain in DOMAIN_ORDER]
            if any(cell["label"] != model["label"] for cell in selected):
                raise ValueError("Cell and model labels must agree")
            counts_by_domain = [_cell_counts(cell) for cell in selected]
            counts = {field: sum(item[field] for item in counts_by_domain)
                      for field in counts_by_domain[0]}
            counts["domains"] = len(DOMAIN_ORDER)
            levels[str(level)] = {
                "counts": counts,
                "metrics": {
                    "pass8": counts["prompts_with_correct"] / counts["prompts"],
                    "distinct8": counts["distinct_correct_modes"] / counts["prompts"],
                },
                "source_cells": [cell["provenance"]["source_cell"] for cell in selected],
            }
        rows.append({"model": model["model"], "label": model["label"], "levels": levels})
    return rows


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
        "rows": build_level_averages(display),
    }


def _tex(value: str) -> str:
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in value)


def render_table(record: dict[str, Any]) -> str:
    lines = [
        "% Generated by ops/build_paper_hosted_level_averages.py; do not edit.",
        r"\begin{table}[H]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{Level 1} & \multicolumn{2}{c}{Level 2} & \multicolumn{2}{c}{Level 3} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"    Model & \texttt{pass@8} (\%) & \# modes & \texttt{pass@8} (\%) & \# modes & \texttt{pass@8} (\%) & \# modes \\",
        r"    \midrule",
    ]
    for row in record["rows"]:
        values = [_tex(row["label"])]
        for level in LEVELS:
            metrics = row["levels"][str(level)]["metrics"]
            values.extend((f"{100 * metrics['pass8']:.1f}", f"{metrics['distinct8']:.2f}"))
        lines.append("    " + " & ".join(values) + r" \\")
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{\textbf{Hosted success and output diversity, averaged by level.}",
        r"  Each level averages five domains equally, using 128 prompts per domain",
        r"  and eight responses per prompt. \texttt{pass@8} is the fraction with any",
        r"  verified response; \# modes is mean \texttt{distinct@8} over all prompts,",
        r"  including failures.}",
        r"  \label{tab:hosted-level-averages}",
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
    print(f"Wrote {args.output.with_suffix('.tex')} and {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
