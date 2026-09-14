#!/usr/bin/env python3
"""Render complete, authenticated matched reasoning-control cohorts for the paper.

This is an offline publication step: it authenticates existing analysis receipts,
rebuilds empirical aggregates from complete prompt counts, and never runs grading
or model inference. Incomplete deployments receive no numerical estimates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from ops.summarize_hosted_reasoning_off import (
        BASE, CONDITION, DOMAINS, DRAWS, LEVELS, PROMPTS, PROMPTS_PER_CELL,
        RESPONSES, ROOT, validate_rows,
    )
except ModuleNotFoundError:
    from summarize_hosted_reasoning_off import (
        BASE, CONDITION, DOMAINS, DRAWS, LEVELS, PROMPTS, PROMPTS_PER_CELL,
        RESPONSES, ROOT, validate_rows,
    )

DEFAULT_SOURCE = BASE / "reasoning_comparison.json"
DEFAULT_OUTPUT = ROOT / "paper/results/hosted_reasoning_off_20260912"
ICON_SOURCE = ROOT / "paper/results/frontier_comparison_20260911.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path | str) -> str:
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def binding(path: Path | str) -> dict[str, str]:
    path = Path(path).resolve()
    label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return {"path": label, "sha256": file_sha(path)}


def count(value: Any, label: str) -> int:
    require(type(value) is int and value >= 0, f"{label}: require a nonnegative integer")
    return value


def validate_cohort(cohort: dict[str, Any]) -> dict[str, Any]:
    """Rebuild all averages from a complete first-32 prompt grid, including failures."""
    prompts = cohort["prompts"]
    validate_rows(prompts)
    for prompt in prompts:
        correct = count(prompt["correct_responses"], "correct responses")
        modes = count(prompt["distinct8"], "distinct modes")
        success = count(prompt["pass8"], "prompt success")
        require(success == int(modes > 0) and success <= modes <= correct <= success * DRAWS,
                "Prompt outcomes are missing or inconsistent")

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {
            "prompts": len(selected), "responses": len(selected) * DRAWS,
            "prompts_with_correct": sum(p["pass8"] for p in selected),
            "distinct_correct_modes": sum(p["distinct8"] for p in selected),
            "correct_responses": sum(p["correct_responses"] for p in selected),
        }
        return {"counts": counts, "metrics": {
            "pass8": counts["prompts_with_correct"] / counts["prompts"],
            "distinct8": counts["distinct_correct_modes"] / counts["prompts"],
        }}

    rebuilt = {
        "cells": {f"level{level}/{domain}": aggregate([
            p for p in prompts if p["level"] == level and p["domain"] == domain])
            for level in LEVELS for domain in DOMAINS},
        "levels": {str(level): aggregate([p for p in prompts if p["level"] == level])
                   for level in LEVELS},
        "overall": aggregate(prompts),
    }
    for field, value in rebuilt.items():
        require(cohort[field] == value, f"{field}: supplied metrics differ from complete prompt counts")
    return rebuilt


def validate_model(model: dict[str, Any]) -> dict[str, Any]:
    require(model["schema"] == "hosted-reasoning-paired-model-summary-v1"
            and model["status"] == "admitted_complete", "Only complete admitted models may be published")
    protocol = model["protocol"]
    require(protocol["schema"] == CONDITION and protocol["all_prompt_bytes_unchanged"] is True
            and protocol["selection_uses_outputs"] is False
            and protocol["prompts_per_cell"] == PROMPTS_PER_CELL
            and protocol["samples_per_prompt"] == DRAWS
            and protocol["planned_terminal_generations"] == RESPONSES
            and protocol["max_output_tokens"] == 8192,
            "Mixed or incomplete reasoning-control protocol")
    controls = model["settings"]["off_request_controls"]
    if model["model"] in ("gpt-5.6-sol", "gpt-5.4"):
        disabled = controls.get("reasoning") == {"effort": "none"}
    elif model["model"].startswith("claude-"):
        disabled = controls.get("thinking") == {"type": "disabled"}
    else:
        disabled = controls.get("reasoning_effort") == "none"
    require(disabled and controls["model"] == model["model"]
            and controls.get("max_output_tokens", controls.get("max_tokens")) == 8192,
            "Off controls or output budget differ from registered condition")
    require("medium" in model["settings"]["on"]["reasoning_effort"], "Baseline is not the medium condition")
    analyses = {}
    for grading in ("strict", "normalized"):
        source = model["analyses"][grading]
        paired = {condition: validate_cohort(source[condition]) for condition in ("on", "off")}
        delta = {str(level): {metric: paired["off"]["levels"][str(level)]["metrics"][metric]
                 - paired["on"]["levels"][str(level)]["metrics"][metric]
                 for metric in ("pass8", "distinct8")} for level in LEVELS}
        require(source["off_minus_on"] == delta, "Paired differences do not match condition-specific counts")
        paired["off_minus_on"] = delta
        analyses[grading] = paired
    return analyses


def build_rows(source: dict[str, Any], labels: dict[str, str]) -> list[dict[str, Any]]:
    require(source["schema"] == "hosted-reasoning-paired-summary-v1", "Wrong paired summary schema")
    models = source["models"]
    admitted = [m["model"] for m in models]
    pending = [m["model"] for m in source["pending_models"]]
    require(admitted and len(admitted + pending) == 7 and len(set(admitted + pending)) == 7,
            "Require seven distinct registered deployment dispositions")
    require(set(admitted + pending) == set(labels), "Unexpected registered model identity")
    require(source["status"] == ("complete" if not pending else "incomplete"), "Summary admission status differs")
    for excluded in source["pending_models"]:
        require(excluded["scores_admitted"] is False and "analyses" not in excluded,
                "Incomplete models cannot carry numerical estimates")
    indexed = {m["model"]: m for m in models}
    rows = []
    for identity in labels:
        if identity not in indexed:
            continue
        model = indexed[identity]
        analyses = validate_model(model)
        normalized = analyses["normalized"]
        levels = {str(level): {
            condition: normalized[condition]["levels"][str(level)] for condition in ("on", "off")
        } for level in LEVELS}
        overall = {condition: normalized[condition]["overall"] for condition in ("on", "off")}
        overall["off_minus_on"] = {metric: overall["off"]["metrics"][metric] - overall["on"]["metrics"][metric]
                                   for metric in ("pass8", "distinct8")}
        rows.append({"model": identity, "label": labels[identity], "levels": levels,
                     "overall": overall, "analyses": analyses, "settings": deepcopy(model["settings"]),
                     "protocol": deepcopy(model["protocol"])})
    return rows


def build_record(source_path: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    source_path = Path(source_path).resolve()
    source = json.loads(source_path.read_text())
    catalog = json.loads(ICON_SOURCE.read_text())
    labels = {m["model"]: m["label"] for m in catalog["models"]}
    rows = build_rows(source, labels)
    checked: dict[Path, str] = {}

    def authenticate(path: Path | str, digest: str) -> None:
        path = Path(path).resolve()
        if path not in checked:
            checked[path] = file_sha(path)
        require(checked[path] == digest, f"Source evidence hash changed: {path}")

    for field in ("registry", "analysis_script", "deployment_amendments"):
        if source.get(field):
            authenticate(source[field]["path"], source[field]["sha256"])
    require(Path(source["analysis_script"]["path"]).resolve() == ROOT / "ops/summarize_hosted_reasoning_off.py",
            "Unrecognized paired admission analyzer")
    registry = json.loads(Path(source["registry"]["path"]).read_text())
    # Resolve the authenticated declarative amendment without invoking the mutable
    # collection runner, whose status-display implementation can change later.
    if source.get("deployment_amendments"):
        amendment = json.loads(Path(source["deployment_amendments"]["path"]).read_text())
        require(amendment["original_experiment_sha256"] == source["registry"]["sha256"],
                "Amendment belongs to another experiment")
        for replacement in amendment["replacements"]:
            matches = [i for i, run in enumerate(registry["runs"]) if run["slug"] == replacement["slug"]]
            require(len(matches) == 1, "Unknown amended deployment")
            index = matches[0]
            original, updated = replacement["original_entry"], replacement["replacement_entry"]
            require(registry["runs"][index] == original and original["model"] == updated["model"]
                    and original["slug"] == updated["slug"], "Amendment changed deployment identity")
            registry["runs"][index] = updated
    require(registry["schema"] == CONDITION and registry["planned_terminal_generations"] == 26880
            and registry["new_on_generations"] == 0, "Registered experiment scope changed")
    registered = {run["model"]: run for run in registry["runs"]}
    require(len(registry["runs"]) == 7 and set(registered) == set(labels), "Registry model inventory differs")
    audits, graders = {}, set()
    for model in source["models"]:
        directory = Path(model["run_directory"]).resolve()
        run = registered[model["model"]]
        require(directory == Path(run["run_directory"]).resolve(), "Admitted deployment differs from registry")
        authenticate(directory / "manifest.json", run["manifest_sha256"])
        marker = directory / "reasoning_comparison.json"
        require(json.loads(marker.read_text()) == model, "Top-level summary differs from the model admission receipt")
        require(model["analysis_script_sha256"] == source["analysis_script"]["sha256"], "Mixed analysis versions")
        evidence = model["evidence_sha256"]
        required = ("manifest.json", "rows.jsonl", "requests.jsonl", "paired_medium_samples.jsonl",
                    "reasoning_condition.json", "samples.jsonl", "audited_primary_samples.jsonl",
                    "normalized_samples.jsonl", "summary.json", "primary_python_regrade_audit.json")
        require(all(str(directory / name) in evidence for name in required), "Admission omits required evidence bindings")
        paths = [Path(path).resolve() for path in evidence if Path(path).resolve() not in checked]
        # Saved admission records bind thousands of small NFS receipts; bounded
        # concurrent reads avoid serial metadata latency without rerunning grades.
        with ThreadPoolExecutor(max_workers=12) as pool:
            checked.update(zip(paths, pool.map(file_sha, paths)))
        for path, digest in evidence.items():
            authenticate(path, digest)
        manifest = json.loads((directory / "manifest.json").read_text())
        require(manifest["experiment_condition"] == CONDITION and manifest["prompt_count"] == PROMPTS,
                "Manifest has a different cohort or reasoning condition")
        require(json.loads((directory / "reasoning_condition.json").read_text()) == model["protocol"],
                "Protocol differs from frozen request preparation")
        for root in (directory, Path(manifest["reference_run"])):
            summary = json.loads((root / "summary.json").read_text())
            secondary = summary["normalized_secondary"]
            graders.add((secondary["normalization_source_sha256"], secondary["frozen_grader_contract_sha256"]))
        audits[model["model"]] = {"source": binding(marker), "authenticated_evidence_files": len(evidence)}
    require(len(graders) == 1, "Published conditions use different frozen graders or normalizers")
    exclusions = deepcopy(source["pending_models"])
    for item in exclusions:
        run = registered[item["model"]]
        directory = Path(item["run_directory"])
        require(directory == Path(run["run_directory"]), "Excluded deployment differs from registry")
        authenticate(directory / "manifest.json", run["manifest_sha256"])
        if "control_violation" in item:
            authenticate(item["control_violation"]["path"], item["control_violation"]["sha256"])
        require(len(list((directory / "sample_receipts").glob("*.json"))) == item["terminal_sample_receipts"],
                "Excluded collection changed; refresh the paired summary before publishing")
    icons = {row["model"]: deepcopy(catalog["model_icons"][row["model"]]) for row in rows}
    for icon in icons.values():
        authenticate(ROOT / icon["path"], icon["sha256"])
        if icon.get("provenance"):
            authenticate(ROOT / icon["provenance"]["path"], icon["provenance"]["sha256"])
    return {
        "schema": "paper-hosted-reasoning-off-v1", "source": binding(source_path),
        "renderer": binding(__file__), "icon_catalog": binding(ICON_SOURCE),
        "analysis_script": deepcopy(source["analysis_script"]), "model_audits": audits,
        "historical_registry_resolver": deepcopy(source["registry_resolver"]),
        "registry_resolution": "Authenticated registry and identity-preserving amendment entries resolved locally. The historical resolver hash is retained for provenance; its live collection-status display has since changed and is not executed here.",
        "normalization_source_sha256": next(iter(graders))[0],
        "frozen_grader_contract_sha256": next(iter(graders))[1],
        "sampling": {"registered_deployments": 7, "admitted_deployments": len(rows),
            "domains": list(DOMAINS), "levels": list(LEVELS), "prompts_per_cell": PROMPTS_PER_CELL,
            "draws_per_prompt": DRAWS, "prompts_per_model_condition": PROMPTS,
            "responses_per_model_condition": RESPONSES, "reasoning_off_responses": len(rows) * RESPONSES},
        "metric_definitions": deepcopy(source["metric_definitions"]),
        "aggregation": source["aggregation"], "rows": rows, "excluded_models": exclusions, "icons": icons,
        "validation": {"all_admitted_evidence_hashes_authenticated": True,
                       "metrics_reconstructed_from_complete_prompt_counts": True, "api_calls": 0},
    }


def tex(value: str) -> str:
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in value)


def model_label(record: dict[str, Any], row: dict[str, Any]) -> str:
    icon = record["icons"][row["model"]]
    path = ROOT / icon["path"]
    require(file_sha(path) == icon["sha256"], "Model icon differs from its source binding")
    return (r"\raisebox{-0.15em}{\includegraphics[width=1.05em,height=1.05em,keepaspectratio]{"
            + path.relative_to(ROOT / "paper").as_posix() + r"}}\hspace{0.4em}" + tex(row["label"]))


def pair_values(value: dict[str, Any]) -> list[str]:
    return [f"{100 * value['metrics']['pass8']:.1f}", f"{value['metrics']['distinct8']:.2f}"]


def render_main_table(record: dict[str, Any]) -> str:
    lines = ["% Generated by ops/build_paper_hosted_reasoning_off.py; do not edit.",
        r"\begin{table}[H]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{lrrrrrr}", r"    \toprule",
        r"    & \multicolumn{2}{c}{Level 1} & \multicolumn{2}{c}{Level 2} & \multicolumn{2}{c}{Level 3} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"    Model & \texttt{pass@8} (\%) & \# modes & \texttt{pass@8} (\%) & \# modes & \texttt{pass@8} (\%) & \# modes \\",
        r"    \midrule"]
    for row in record["rows"]:
        values = [model_label(record, row)]
        for level in LEVELS:
            values.extend(pair_values(row["levels"][str(level)]["off"]))
        lines.append("    " + " & ".join(values) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}",
        r"  \caption{\textbf{Hosted success and output diversity with reasoning disabled.}",
        r"  Each level averages five domains equally: 32 prompts per domain and eight",
        r"  responses per prompt. \texttt{pass@8} is the fraction with any verified",
        r"  response; \# modes is mean \texttt{distinct@8}, including failures.",
        r"  The frozen formatting normalizer applies throughout. Appendix~\ref{app:hosted-reasoning-control}",
        r"  gives matched medium-reasoning results.}",
        r"  \label{tab:hosted-level-averages}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def render_appendix(record: dict[str, Any]) -> str:
    excluded = {item["model"]: item for item in record["excluded_models"]}
    require(len(record["rows"]) == 5 and set(excluded) == {"claude-opus-5", "DeepSeek-V4-Pro"}
            and excluded["claude-opus-5"]["status"] == "blocked_control_violation"
            and excluded["DeepSeek-V4-Pro"]["terminal_sample_receipts"] == 3839,
            "Refresh the snapshot-specific appendix narrative when deployment admission changes")
    require(all(all(row["overall"]["off_minus_on"][metric] < 0 for metric in ("pass8", "distinct8"))
                for row in record["rows"]), "Snapshot interpretation differs from observed comparisons")
    lines = ["% Generated by ops/build_paper_hosted_reasoning_off.py; do not edit.",
        r"\subsection{Matched reasoning-control comparison}", r"\label{app:hosted-reasoning-control}",
        r"We selected original row indices 0--31 in each of five domains and three",
        r"levels, without consulting outputs. Each model and condition therefore has",
        r"480 prompts and 3,840 responses. The medium baseline reuses the original",
        r"audited responses for these exact prompts and eight sample slots; no new",
        r"medium responses were collected. Both conditions preserve the original",
        r"system and user prompt bytes, including the original Python wording.",
        r"The 128-prompt medium results in Table~\ref{tab:hosted-level-averages-medium}",
        r"and Figure~\ref{fig:hosted-verified-breadth} are a separate, larger display;",
        r"the comparison below uses only the matched 32-prompt subset.", "",
        r"Only the provider reasoning control changes, with an 8,192-token output",
        r"budget retained in both conditions. GPT-5.6 Sol and GPT-5.4 request",
        r"\texttt{reasoning.effort=none}; Grok 4.3 and Kimi K3 request",
        r"\texttt{reasoning\_effort=none}. Claude Opus 4.8 requests",
        r"\texttt{thinking.type=disabled} while retaining",
        r"\texttt{output\_config.effort=medium}. These controls denote disabled",
        r"explicit reasoning at the endpoint, without establishing the absence of",
        r"hidden internal computation. Native response receipts are checked for",
        r"control violations. The same frozen formatting normalizer and canonical",
        r"mode grader apply to both conditions; the accompanying JSON retains strict",
        r"grades, domain-level counts, controls, and source hashes.", "",
        r"\begin{table}[H]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{7pt}",
        r"  \begin{tabular}{llrrrr}", r"    \toprule",
        r"    & & \multicolumn{2}{c}{Medium reasoning} & \multicolumn{2}{c}{Reasoning disabled} \\",
        r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"    Model & Level & \texttt{pass@8} (\%) & \# modes & \texttt{pass@8} (\%) & \# modes \\",
        r"    \midrule"]
    for index, row in enumerate(record["rows"]):
        if index:
            lines.append(r"    \addlinespace")
        for level in LEVELS:
            values = [model_label(record, row) if level == 1 else "", str(level)]
            for condition in ("on", "off"):
                values.extend(pair_values(row["levels"][str(level)][condition]))
            lines.append("    " + " & ".join(values) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}",
        r"  \caption{\textbf{Matched medium and disabled reasoning, by level.}",
        r"  Each row averages 160 prompts across five domains. All eight responses",
        r"  and zero-success prompts are retained. \# modes denotes mean",
        r"  \texttt{distinct@8}; estimates use the frozen formatting normalizer.}",
        r"  \label{tab:hosted-reasoning-matched-levels}", r"\end{table}", "",
        r"\begin{table}[H]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{7pt}",
        r"  \begin{tabular}{lrrrr}", r"    \toprule",
        r"    & \multicolumn{2}{c}{Medium reasoning} & \multicolumn{2}{c}{Reasoning disabled} \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"    Model & \texttt{pass@8} (\%) & \# modes & \texttt{pass@8} (\%) & \# modes \\",
        r"    \midrule"]
    for row in record["rows"]:
        values = [model_label(record, row)]
        for condition in ("on", "off"):
            values.extend(pair_values(row["overall"][condition]))
        lines.append("    " + " & ".join(values) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}",
        r"  \caption{\textbf{Matched reasoning-control averages across levels.}",
        r"  Each model and condition uses the same 480 prompts and 3,840 responses,",
        r"  with equal weight on every domain and level.}",
        r"  \label{tab:hosted-reasoning-matched-overall}", r"\end{table}", "",
        r"\paragraph{Incomplete deployments.}",
        r"Five of seven registered deployments passed complete-cohort admission.",
        r"Claude Opus 5 is excluded because a native response returned a thinking",
        r"block and 40 thinking tokens despite an explicit disabled-thinking request.",
        r"DeepSeek V4 Pro is excluded because only 3,839 of 3,840 sample receipts",
        r"were recovered, leaving one unresolved request. Neither partial cohort",
        r"receives a score, and the missing response is not counted as a failure.", "",
        r"\paragraph{Scope of the comparison.}",
        r"Across all five complete deployments, disabling reasoning lowers both",
        r"overall \texttt{pass@8} and observed \texttt{distinct@8}. Reduced success",
        r"also reduces opportunities to observe distinct successful modes, so this",
        r"does not isolate diversity at fixed accuracy. Medium and disabled outputs",
        r"were collected at different times; matching sample slots are not shared",
        r"random seeds. Omitted temperature and \texttt{top\_p} controls retain",
        r"provider defaults, which can vary over time. These are descriptive point",
        r"estimates without uncertainty intervals, not causal claims about training."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output stem for JSON and main/appendix TeX")
    args = parser.parse_args(argv)
    record = build_record(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    Path(str(args.output) + "_main.tex").write_text(render_main_table(record))
    Path(str(args.output) + "_appendix.tex").write_text(render_appendix(record))
    print(f"Wrote {args.output}: {len(record['rows'])} complete deployments, zero model API calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
