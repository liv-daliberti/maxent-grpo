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
        # The success-conditional column is rebuilt separately from the same
        # draws; binding it here puts it under the same drift check as the rest.
        "pcmd_record": {"path": str(PMD_RECORD.relative_to(ROOT)),
                        "sha256": file_sha(PMD_RECORD)} if PMD_RECORD.is_file() else None,
        "validation": {"all_admitted_evidence_hashes_authenticated": True,
                       "metrics_reconstructed_from_complete_prompt_counts": True, "api_calls": 0},
    }


def tex(value: str) -> str:
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in value)


#: The reasoning-off summary names deployments by model id; the PCMD rebuild
#: names them by the artifact directory it read. One map, stated once.
PMD_RECORD = ROOT / "paper/results/hosted_reasoning_pmd.json"
PMD_DIRECTORY = {"gpt-5.6-sol": "gpt56sol", "gpt-5.4": "gpt54",
                 "grok-4.3": "grok43", "FW-Kimi-K3": "kimi_k3",
                 "claude-opus-4-8": "claude_opus48"}


def _deepseek() -> dict:
    """The excluded DeepSeek cohort's paired PCMD, from the same rebuild."""
    payload = json.loads(PMD_RECORD.read_text(encoding="utf-8"))
    return payload["excluded_deployments"]["deepseek_v4_pro"]


def pmd_by_deployment() -> dict:
    """Paired PCMD per model id, or an empty map if the rebuild is absent."""
    if not PMD_RECORD.is_file():
        return {}
    payload = json.loads(PMD_RECORD.read_text(encoding="utf-8"))
    if payload.get("schema") != "paper-hosted-reasoning-pmd-v2":
        raise RuntimeError("hosted reasoning PCMD record schema drifted")
    deployments = payload["deployments"]
    out = {}
    for model, directory in PMD_DIRECTORY.items():
        cell = deployments.get(directory)
        if cell and cell.get("reportable"):
            out[model] = cell
    return out


def _paired_span(scope: str) -> tuple[int, int]:
    """The smallest and largest paired support behind a PCMD column."""
    cells = pmd_by_deployment().values()
    if scope == "total":
        counts = [cell["paired_prompts"] for cell in cells]
    else:
        counts = [level["paired_prompts"] for cell in cells
                  for level in cell[scope].values()]
    return min(counts), max(counts)


def _level_effects() -> dict[str, list[float]]:
    """Each deployment's paired PCMD effect at Levels 1, 2 and 3."""
    return {model: [cell["by_level"][str(level)]["effect"] for level in LEVELS]
            for model, cell in pmd_by_deployment().items()}


def three(value: float) -> str:
    """A PCMD-scale number, written as the paper writes them."""
    return f"{value:.3f}".replace("0.", ".", 1)


def pmd_tex(cell: dict | None, field: str) -> str:
    """One PCMD entry, or a dash where the support bar is not cleared."""
    if cell is None or not cell.get("reportable"):
        return "---"
    return three(cell[field])


def pmd_levels(model: str) -> dict[str, dict]:
    """Paired PCMD by level for one deployment, keyed by level as a string."""
    cell = pmd_by_deployment().get(model)
    return {} if cell is None else cell["by_level"]


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
        r"\begin{table}[H]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{4pt}",
        # Ten columns of five deployments overrun \linewidth by some 46pt at
        # this separation; the body is scaled to the text block rather than bled
        # past it, as the other wide tables in this paper are.
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lrrrrrrrrr}", r"    \toprule",
        r"    & \multicolumn{3}{c}{Level 1} & \multicolumn{3}{c}{Level 2} & \multicolumn{3}{c}{Level 3} \\",
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"    Model" + r" & \texttt{pass@8} (\%) & \# modes & \pmd{}" * 3 + r" \\",
        r"    \midrule"]
    for row in record["rows"]:
        values = [model_label(record, row)]
        levels = pmd_levels(row["model"])
        for level in LEVELS:
            values.extend(pair_values(row["levels"][str(level)]["off"]))
            values.append(pmd_tex(levels.get(str(level)), "reasoning_off"))
        lines.append("    " + " & ".join(values) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}}",
        r"  \caption{\textbf{Hosted success and output diversity with reasoning disabled.}",
        r"  Each level averages five domains equally: 32 prompts per domain and eight",
        r"  responses per prompt. \texttt{pass@8} is the fraction with any verified",
        r"  response; \# modes is mean \texttt{distinct@8}, including failures.",
        r"  Both fall when the model simply succeeds less often, so \pmd{} carries the",
        r"  success-conditional axis beside them: it is the mean probability that two",
        r"  verified draws on the same prompt differ, over the prompts that return two",
        r"  verified draws in this condition and in the matched medium condition of",
        r"  Table~\ref{tab:hosted-reasoning-matched-levels}, which is the population",
        r"  every \pmd{} reading in this appendix uses.",
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
    effects = _level_effects()
    # The paragraph below names a direction per deployment, so each one is
    # asserted here; "flat" means inside the rounding the table prints.
    require(set(effects) == {row["model"] for row in record["rows"]}
            and all(len(values) == 3 for values in effects.values())
            and all(value < 0 for value in effects["gpt-5.6-sol"])
            and all(value > 0 for value in effects["grok-4.3"])
            and all(value > 0 for value in effects["FW-Kimi-K3"])
            and all(abs(value) < 0.02 for value in effects["claude-opus-4-8"])
            and abs(effects["gpt-5.4"][0]) < 0.02
            and all(value < -0.02 for value in effects["gpt-5.4"][1:])
            and all(min(values) * max(values) >= 0 or min(abs(v) for v in values) < 0.02
                    for values in effects.values()),
            "Per-level PCMD directions differ from the narrative that reads them")
    require(all(level["reportable"] for cell in pmd_by_deployment().values()
                for level in cell["by_level"].values()),
            "A level cell lost its paired PCMD support and the caption still claims it")
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
        r"\begin{table}[H]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{llrrcrrc}", r"    \toprule",
        r"    & & \multicolumn{3}{c}{Medium reasoning} & \multicolumn{3}{c}{Reasoning disabled} \\",
        r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"    Model & Level & \texttt{pass@8} (\%) & \# modes & \pmd{}"
        r" & \texttt{pass@8} (\%) & \# modes & \pmd{} \\",
        r"    \midrule"]
    # The across-level averages were a second table with the same seven columns
    # and the same model rows, so every deployment's name was printed twice to
    # separate its levels from their mean. They are an "All" row here.
    overall_pmd = pmd_by_deployment()
    for index, row in enumerate(record["rows"]):
        if index:
            lines.append(r"    \addlinespace")
        levels = pmd_levels(row["model"])
        for level in LEVELS:
            values = [model_label(record, row) if level == 1 else "", str(level)]
            for condition, field in (("on", "reasoning_on"), ("off", "reasoning_off")):
                values.extend(pair_values(row["levels"][str(level)][condition]))
                values.append(pmd_tex(levels.get(str(level)), field))
            lines.append("    " + " & ".join(values) + r" \\")
        cell = overall_pmd.get(row["model"])
        values = ["", r"\textit{All}"]
        for condition, field in (("on", "reasoning_on"), ("off", "reasoning_off")):
            values.extend(pair_values(row["overall"][condition]))
            values.append("---" if cell is None
                          else f"{cell[field]:.3f}".replace("0.", ".", 1))
        lines.append(r"    \cmidrule(l){2-8}")
        lines.append("    " + " & ".join(values) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}",
        r"  \caption{\textbf{Matched medium and disabled reasoning, by level.}",
        r"  Each row averages 160 prompts across five domains. All eight responses",
        r"  and zero-success prompts are retained. \# modes denotes mean",
        r"  \texttt{distinct@8}; estimates use the frozen formatting normalizer.",
        r"  \pmd{} is paired within the level: it averages over the prompts of that",
        r"  level which return two verified draws in both conditions, between",
        f"  {_paired_span('by_level')[0]:,} and {_paired_span('by_level')[1]:,}"
        r" of the 160 per cell, and every level cell here clears the",
        r"  thirty-prompt support bar.",
        r"  The \texttt{pass@8} and \# modes columns use all 160 prompts, so the two",
        r"  axes are read on the same draws but not on the same denominators, and the",
        r"  paired count behind every level cell is retained in the accompanying",
        r"  machine-readable record alongside the per-domain counts.",
        r"  The \textit{All} row averages the same 480 prompts and 3,840 responses",
        r"  per condition with equal weight on every domain and level; its \pmd{}",
        r"  is the paired reading over the",
        f"  {_paired_span('total')[0]:,}--{_paired_span('total')[1]:,} prompts per"
        r" deployment that return two verified draws in both conditions.",
        r"  That column separates concentration from lost accuracy, since a prompt",
        r"  one condition stopped answering twice leaves the estimate rather than",
        r"  entering it as a smaller number; the per-domain readings behind it are",
        r"  retained in the accompanying JSON, together with the prompts that only",
        r"  one of the two conditions answers twice and so enter neither average.}",
        r"  \label{tab:hosted-reasoning-matched-levels}", r"\end{table}", "",
        r"\paragraph{Incomplete deployments.}",
        # Ordered so the measurement is read before the exclusion. The earlier
        # wording opened on "is excluded", and a reader could reasonably finish
        # the paragraph believing a frontier deployment had been discarded over
        # one missing response, when its effect is reported two sentences later.
        r"Five of seven registered deployments passed complete-cohort admission,",
        r"and neither partial cohort receives a score. They fall short in different",
        r"ways, and that decides what can still be read from each.", "",
        r"DeepSeek V4 Pro's draws are intact, so its measurement survives its",
        r"exclusion and is reported here: read on the "
        + f"{_deepseek()['paired_prompts']:,}" + r" prompts both of its",
        r"conditions define, disabling reasoning moves \pmd{} from",
        r"$" + f"{_deepseek()['reasoning_on']:.3f}".replace("0.", ".", 1) + r"$ to $"
        + f"{_deepseek()['reasoning_off']:.3f}".replace("0.", ".", 1) + r"$, a change of $"
        + f"{_deepseek()['effect']:+.3f}".replace("0.", ".", 1) + r"$. What that figure",
        r"cannot do is join the table, for two reasons and neither is the recovery",
        r"shortfall: both of its arms are graded strictly, because the frozen",
        r"normalizer was only ever run on admitted cohorts, and one PantryPlan prompt",
        r"holds seven draws rather than eight. A caveat cannot put a strictly graded",
        r"number on the same scale as a normalized one. The shortfall itself is",
        r"slight --- 3,839 of 3,840 sample receipts were recovered, leaving one",
        r"unresolved request, which is not counted as a failure --- and it is not",
        r"what keeps the deployment out of the table. The figure is reported so the",
        r"evidence is not lost, not so it can be compared.", "",
        r"Claude Opus 5 admits no such reading. A native response returned a thinking",
        r"block and 40 thinking tokens despite an explicit disabled-thinking request,",
        r"so its disabled condition did not disable reasoning. There is no",
        r"reasoning-off measurement to preserve, rather than one withheld.", "",
        r"\paragraph{Scope of the comparison.}",
        r"Across all five complete deployments, disabling reasoning lowers both",
        r"overall \texttt{pass@8} and observed \distk{}. Reduced success also",
        r"reduces opportunities to observe distinct successful modes, so those two",
        r"columns cannot separate concentration from lost accuracy. \pmd{} can, and",
        r"the answer is not uniform: on the prompts both conditions define, breadth",
        r"falls for GPT-5.6 Sol and GPT-5.4, is unchanged for Claude Opus 4.8, and",
        r"rises for Kimi K3 and Grok 4.3, which more than doubles from $.154$ to",
        r"$.364$ while its \distk{} falls. Disabling deliberation concentrates some",
        r"deployments and spreads others, so the common fall in \distk{} tracks the",
        r"accuracy drop rather than a shared collapse.",
        r"Table~\ref{tab:hosted-reasoning-matched-levels} reads the same contrast at",
        r"each level, and the direction belongs to the deployment rather than to the",
        r"level: GPT-5.6 Sol falls at every level, Grok 4.3 and Kimi K3 rise at",
        r"every level, GPT-5.4 is flat at Level 1 and falls at Levels 2 and 3, and",
        r"Claude Opus 4.8 stays within $"
        + three(max(abs(value) for value in effects["claude-opus-4-8"]))
        + r"$ of zero throughout.",
        r"No deployment reverses its own direction from one level to the next, and",
        r"no level cell loses the measurement: every one clears the support bar on",
        r"prompts both conditions answer twice. Medium and disabled outputs",
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
