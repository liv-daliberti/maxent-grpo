#!/usr/bin/env python3
"""Summarize an eight-draw, inference-only hosted ModeBench evaluation.

Reads the immutable prompt snapshot and successful API responses. Transport errors
are counted separately; a returned incomplete response is still a scored draw.
Only prompts with all eight distinct sample indices enter outcome statistics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

DRAWS = 8
DOMAIN_ORDER = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
DOMAIN_LABELS = {"graph_coloring": "Graph coloring", "countdown": "Countdown",
                 "python_factors": "Python factors", "mathir": "MathIR", "pantry_plan": "Pantry"}
METRICS = (
    "pass1", "pass8", "distinct8", "correct_pair_collision", "one_mode_rate",
    "effective_modes_correct", "correct_repeat_fraction", "uniform_expected_distinct8",
    "distinct8_gap_to_uniform", "uniform_correct_pair_collision",
    "correct_pair_collision_excess_uniform", "uniform_one_mode_rate",
)
COLUMNS = (
    "prompts", "correct_draws", "prompts_with_correct", "distinct_correct_modes",
    "colliding_correct_pairs", "correct_pairs", "prompts_with_two_correct",
    "one_mode_prompts", "effective_modes_sum", "repeated_correct_draws",
    "known_support_prompts", "uniform_distinct_sum", "known_support_distinct_sum",
    "known_support_correct_pairs", "uniform_colliding_pairs_sum",
    "known_support_colliding_pairs", "known_support_two_correct_prompts",
    "uniform_one_mode_sum",
)


def canonical_domain(value: str) -> str:
    return "pantry_plan" if value == "pantry" else value


def prompt_id(row: dict[str, Any]) -> tuple[int, str, int]:
    return int(row["level"]), canonical_domain(row["domain"]), int(row["row_index"])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON at {path}:{lineno}") from exc
    return result


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def support_count(row: dict[str, Any]) -> int | None:
    # Countdown metadata enumerates a binary-expression library, while the
    # hosted verifier also accepts distinct unary-negation AST modes.
    if canonical_domain(row.get("domain", "")) == "countdown":
        return None
    metadata = row.get("metadata", {})
    if metadata.get("support_is_open"):
        return None
    count = metadata.get("answer_mode_count")
    if isinstance(count, (int, float)) and not isinstance(count, bool):
        if math.isfinite(count) and count > 0 and int(count) == count:
            return int(count)
    return None


def prompt_statistics(row: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Canonical mode occupancy among correct draws; invalid text creates no mode."""
    if len(samples) != DRAWS or {s["sample_index"] for s in samples} != set(range(DRAWS)):
        raise ValueError("Prompt statistics require all eight unique sample indices")
    keys = []
    for sample in samples:
        if not isinstance(sample.get("verified"), bool):
            raise ValueError("Every response needs a boolean verified outcome")
        if sample["verified"]:
            if sample.get("canonical_key") is None:
                raise ValueError("Verified responses must have a validated canonical key")
            keys.append(json.dumps(sample["canonical_key"], sort_keys=True, separators=(",", ":"), allow_nan=False))
    counts = Counter(keys)
    correct = len(keys)
    distinct = len(counts)
    pairs = correct * (correct - 1) // 2
    collisions = sum(n * (n - 1) // 2 for n in counts.values())
    effective = math.exp(-sum((n / correct) * math.log(n / correct) for n in counts.values())) if correct else None
    modes = support_count(row)
    if modes is not None and distinct > modes:
        raise ValueError(f"Observed modes exceed certified support for {prompt_id(row)}")
    uniform_distinct = modes * (-math.expm1(correct * math.log1p(-1 / modes))) if modes and modes > 1 else (float(correct > 0) if modes else None)
    level, domain, index = prompt_id(row)
    return {
        "level": level, "domain": domain, "row_index": index,
        "correct_draws": correct, "pass1": correct / DRAWS, "pass8": int(correct > 0),
        "distinct8": distinct, "correct_pairs": pairs, "colliding_correct_pairs": collisions,
        "correct_pair_collision": collisions / pairs if pairs else None,
        "one_mode": int(distinct == 1) if correct >= 2 else None,
        "effective_modes_correct": effective, "repeated_correct_draws": correct - distinct,
        "certified_support_count": modes,
        "uniform_expected_distinct_given_correct": uniform_distinct,
        "uniform_expected_colliding_correct_pairs": pairs / modes if modes else None,
        "uniform_one_mode_probability_given_correct": modes ** (1 - correct) if modes and correct >= 2 else None,
        "response_status_counts": dict(Counter(s.get("response_status", "unknown") for s in samples)),
    }


def as_vector(stat: dict[str, Any]) -> list[float]:
    known = stat["certified_support_count"] is not None
    return [
        1, stat["correct_draws"], stat["pass8"], stat["distinct8"],
        stat["colliding_correct_pairs"], stat["correct_pairs"], int(stat["correct_draws"] >= 2),
        stat["one_mode"] or 0, stat["effective_modes_correct"] or 0, stat["repeated_correct_draws"],
        int(known), stat["uniform_expected_distinct_given_correct"] or 0,
        stat["distinct8"] if known else 0, stat["correct_pairs"] if known else 0,
        stat["uniform_expected_colliding_correct_pairs"] or 0,
        stat["colliding_correct_pairs"] if known else 0,
        int(known and stat["correct_draws"] >= 2), stat["uniform_one_mode_probability_given_correct"] or 0,
    ]


def metric_values(sums: np.ndarray) -> np.ndarray:
    """Compute ratio estimands after whole-prompt resampling, preserving denominators."""
    s = np.asarray(sums, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack((s[..., 1] / (DRAWS * s[..., 0]), s[..., 2] / s[..., 0],
            s[..., 3] / s[..., 0], s[..., 4] / s[..., 5], s[..., 7] / s[..., 6],
            s[..., 8] / s[..., 2], s[..., 9] / s[..., 1], s[..., 11] / s[..., 10],
            (s[..., 11] - s[..., 12]) / s[..., 10], s[..., 14] / s[..., 13],
            (s[..., 15] - s[..., 14]) / s[..., 13], s[..., 17] / s[..., 16]), axis=-1)


def bootstrap_cell(stats: list[dict[str, Any]], replicates: int, rng: np.random.Generator) -> tuple[dict[str, Any], np.ndarray]:
    if not stats:
        return {"complete_prompts": 0, "counts": {key: 0 for key in COLUMNS},
                "metrics": describe_metrics(np.full(len(METRICS), np.nan), np.full((replicates, len(METRICS)), np.nan))}, np.full((replicates, len(METRICS)), np.nan)
    vectors = np.asarray([as_vector(s) for s in stats], dtype=float)
    sums = vectors.sum(axis=0)
    boot = np.empty((replicates, len(METRICS)), dtype=float)
    for start in range(0, replicates, 256):
        end = min(start + 256, replicates)
        indices = rng.integers(len(stats), size=(end - start, len(stats)))
        boot[start:end] = metric_values(vectors[indices].sum(axis=1))
    return {"complete_prompts": len(stats), "counts": dict(zip(COLUMNS, sums.tolist())),
            "metrics": describe_metrics(metric_values(sums), boot)}, boot


def describe_metrics(point: np.ndarray, boot: np.ndarray) -> dict[str, Any]:
    output = {}
    for index, name in enumerate(METRICS):
        finite = boot[:, index][np.isfinite(boot[:, index])]
        value = float(point[index]) if np.isfinite(point[index]) else None
        ci = np.quantile(finite, [.025, .975]).tolist() if len(finite) and value is not None else None
        output[name] = {"estimate": value, "ci95": ci, "defined_bootstrap_replicates": len(finite)}
    return output


def safe_macro(values: np.ndarray) -> np.ndarray:
    """An equal-cell macro is undefined unless every included cell is defined."""
    return np.mean(values, axis=0)


def summarize(directory: Path, replicates: int = 2000, seed: int = 20260911,
              sample_records: list[dict[str, Any]] | None = None,
              primary_samples_path: Path | None = None) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    rows = read_jsonl(directory / "rows.jsonl")
    if not rows:
        raise ValueError("rows.jsonl must contain the frozen expected prompt inventory")
    expected = {prompt_id(row): row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate prompt IDs in rows.jsonl")
    primary_path = primary_samples_path or directory / "samples.jsonl"
    samples = read_jsonl(primary_path) if sample_records is None else sample_records
    if primary_path.resolve() != (directory / "samples.jsonl").resolve():
        validate_audited_primary(directory, samples)
    errors = read_jsonl(directory / "errors.jsonl")
    grouped: dict[tuple[int, str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for sample in samples:
        key = prompt_id(sample)
        if key not in expected:
            raise ValueError(f"Response references an unexpected prompt {key}")
        index = sample["sample_index"]
        if not isinstance(index, int) or isinstance(index, bool) or index not in range(DRAWS):
            raise ValueError(f"Invalid sample index for prompt {key}")
        if index in grouped[key]:
            raise ValueError(f"Duplicate response for {key}, sample {index}")
        grouped[key][index] = sample
    cell_stats = defaultdict(list)
    cell_expected: Counter = Counter()
    incomplete = []
    for key, row in expected.items():
        cell_expected[key[:2]] += 1
        group = grouped.get(key, {})
        if len(group) == DRAWS:
            cell_stats[key[:2]].append(prompt_statistics(row, list(group.values())))
        else:
            incomplete.append({"level": key[0], "domain": key[1], "row_index": key[2],
                               "received_samples": len(group), "missing_sample_indices": sorted(set(range(DRAWS)) - group.keys())})
    rng = np.random.default_rng(seed)
    cells, cell_boots = {}, {}
    for cell in sorted(cell_expected):
        identifier = f"level{cell[0]}/{cell[1]}"
        result, boot = bootstrap_cell(cell_stats[cell], replicates, rng)
        result.update(level=cell[0], domain=cell[1], expected_prompts=cell_expected[cell],
                      missing_prompts=cell_expected[cell] - len(cell_stats[cell]))
        cells[identifier], cell_boots[cell] = result, boot
    levels = {}
    for level in sorted({key[0] for key in cell_expected}):
        keys = sorted(cell for cell in cell_expected if cell[0] == level)
        points = np.asarray([[cells[f"level{k[0]}/{k[1]}"]["metrics"][m]["estimate"] for m in METRICS] for k in keys], dtype=float)
        level_boot = safe_macro(np.asarray([cell_boots[k] for k in keys]))
        levels[str(level)] = {
            "level": level, "domains": [k[1] for k in keys], "expected_prompts": sum(cell_expected[k] for k in keys),
            "complete_prompts": sum(len(cell_stats[k]) for k in keys),
            "metrics": describe_metrics(safe_macro(points), level_boot),
            "aggregation": "Equal-domain arithmetic mean; a metric is undefined if any domain has no eligible observations.",
        }
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    inventory_path = directory / "datasets.json"
    inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else {}
    usage_totals: Counter = Counter()
    for sample in samples:
        for name, value in flatten_usage(sample.get("usage") or {}).items():
            usage_totals[name] += value
    latencies = [s["latency_seconds"] for s in samples if isinstance(s.get("latency_seconds"), (int, float))]
    prompt_stats = [stat for key in sorted(cell_stats) for stat in cell_stats[key]]
    return {
        "schema": "frontier_modebench_summary_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "primary_samples_path": str(primary_path.resolve()),
        "primary_samples_sha256": digest(primary_path),
        "primary_grading_audit": load_primary_audit(directory, primary_path, samples),
        "status": "complete" if not incomplete else "incomplete", "directory": str(directory.resolve()),
        "expected_prompts": len(expected), "expected_responses": DRAWS * len(expected),
        "received_responses": len(samples), "missing_responses": DRAWS * len(expected) - len(samples),
        "response_records_sha256": hashlib.sha256(json.dumps(samples, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "complete_prompts": len(prompt_stats), "incomplete_prompts": incomplete,
        "excluded_responses_in_incomplete_groups": sum(i["received_samples"] for i in incomplete),
        "response_status_counts": dict(Counter(s.get("response_status", "unknown") for s in samples)),
        "api_error_attempts": len(errors),
        "api_error_types": dict(Counter(str(e.get("error_type", e.get("type", "unknown"))) for e in errors)),
        "models_returned": dict(Counter(str(s.get("model", "unknown")) for s in samples)),
        "returned_sampling": {name: dict(Counter(json.dumps(sample.get(name), sort_keys=True) for sample in samples))
                              for name in ("temperature", "top_p", "reasoning", "service_tier")},
        "usage_totals": dict(usage_totals),
        "deployment_header_examples": deployment_headers(directory, samples),
        "latency_seconds": {"mean": float(np.mean(latencies)), "median": float(np.median(latencies)),
                            "p95": float(np.quantile(latencies, .95))} if latencies else {},
        "run_configuration": {key: manifest[key] for key in ("model", "deployment", "reasoning_effort", "max_output_tokens", "samples_per_prompt", "temperature", "top_p", "instructions", "created_at_utc", "api", "base_url", "endpoint", "sample_count", "tools", "training", "conversation_state") if key in manifest},
        "dataset_provenance_notes": inventory.get("provenance_notes", []),
        "input_sha256": {name: digest(directory / name) for name in ("rows.jsonl", "datasets.json", "manifest.json", "samples.jsonl", "errors.jsonl")},
        "bootstrap": {"replicates": replicates, "seed": seed, "interval": "pointwise 95% percentile",
                      "unit": "Whole prompts, resampled independently within each domain and level; all eight draws stay together.",
                      "scope": "Prompt population uncertainty under the empirical prompt distribution; not a separate resampling of generations and not simultaneous intervals."},
        "metric_definitions": metric_definitions(), "levels": levels, "cells": cells, "prompts": prompt_stats,
        "limitations": limitations(),
    }




def validate_audited_primary(directory: Path, samples: list[dict[str, Any]]) -> None:
    raw = {(*prompt_id(sample), sample["sample_index"]): sample for sample in read_jsonl(directory / "samples.jsonl")}
    audited_keys = [(*prompt_id(sample), sample["sample_index"]) for sample in samples]
    if len(audited_keys) != len(raw) or set(audited_keys) != set(raw):
        raise ValueError("Audited primary sidecar is stale or incomplete relative to saved API responses")
    for sample in samples:
        key = (*prompt_id(sample), sample["sample_index"])
        if key not in raw:
            raise ValueError(f"Audited sample absent from original API receipts: {key}")
        for name, original in raw[key].items():
            if name not in ("verified", "canonical_key", "graded_text") and sample.get(name) != original:
                raise ValueError(f"Audited primary changed non-grading field {name}: {key}")


def load_primary_audit(directory: Path, primary_path: Path, samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    if primary_path.resolve() == (directory / "samples.jsonl").resolve():
        return None
    audit_path = directory / "primary_python_regrade_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    expected_sidecar_hash = audit.get("derived_primary", {}).get("sha256")
    if expected_sidecar_hash and digest(primary_path) != expected_sidecar_hash:
        raise ValueError("Audited primary sidecar differs from its grading audit digest")
    return {"path": str(audit_path.resolve()) if audit_path.exists() else None, "sha256": digest(audit_path),
            "corrected_records": sum(bool(sample.get("primary_regrade_corrected")) for sample in samples),
            "status": audit.get("status"), "source_sample_count": len(samples),
            "full_sampling_complete": audit.get("full_sampling_complete"),
            "correction_counts": audit.get("correction_counts", {}),
            "description": "Strict executable grading after a serial frozen-worker infrastructure audit; original API responses and grading receipts are preserved. These corrections are not formatting rescues."}


def load_cache_reconciliation(directory: Path) -> dict[str, Any] | None:
    path = directory / "normalization_python_reconciliation.json"
    if not path.exists():
        return None
    audit = json.loads(path.read_text())
    return {"path": str(path.resolve()), "sha256": digest(path), **{name: audit[name] for name in
            ("status", "graded_python_records", "changed_records", "correction_breakdown")}}


def warm_python_verifier(samples: list[dict[str, Any]], rows: dict, grader) -> dict[str, Any]:
    from oat_drgrpo.python_modebench_process import _SHARED_VERIFIER
    _SHARED_VERIFIER._start()
    time.sleep(1.25)
    known_good = next((sample for sample in samples if sample["domain"] == "python_factors" and sample["verified"]), None)
    result = {"startup_grace_seconds": 1.25, "unchanged_verifier_timeout_seconds": _SHARED_VERIFIER.timeout_seconds,
              "known_good_confirmed": False, "attempts": 0}
    if known_good is None:
        return result
    for attempt in range(3):
        grade = grader(known_good["level"], known_good["domain"], rows[prompt_id(known_good)], known_good["text"])
        result["attempts"] = attempt + 1
        if grade["verified"] and grade["canonical_key"] == known_good["canonical_key"]:
            result["known_good_confirmed"] = True
            return result
        _SHARED_VERIFIER._start()
        time.sleep(1.25)
    raise ValueError("Frozen Python worker failed its bounded known-good warmup; refusing secondary grading")


def deployment_headers(directory: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    paths = [directory / "api_preflight.json"]
    if samples:
        paths += [directory / sample["raw_receipt"] for sample in (samples[0], samples[-1]) if sample.get("raw_receipt")]
    for path in dict.fromkeys(paths):
        if path.exists():
            receipt = json.loads(path.read_text())
            headers = receipt.get("headers", {})
            examples.append({"source": str(path.relative_to(directory)),
                             "served_model": headers.get("x-ms-served-model"),
                             "region": headers.get("x-ms-region"), "sha256": digest(path)})
    return examples


def flatten_usage(value: dict[str, Any], prefix: str = "") -> dict[str, float]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten_usage(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = item
    return result


def metric_definitions() -> dict[str, str]:
    return {
        "pass1": "Average verified fraction of the eight independent draws per prompt (correct/8), not only the first draw.",
        "pass8": "Fraction of prompts with at least one verified answer in eight draws.",
        "distinct8": "Mean number of unique validated canonical modes among eight draws, including zero for unsolved prompts.",
        "correct_pair_collision": "Within-prompt unordered correct pairs sharing a canonical mode, summed across prompts, divided by all within-prompt correct pairs. Undefined without a prompt with >=2 correct draws; pair weighted within each cell.",
        "one_mode_rate": "Fraction of prompts with >=2 correct draws whose correct draws all share one canonical mode.",
        "effective_modes_correct": "Mean exp(empirical Shannon entropy) across prompts with >=1 correct draw. At most eight observed modes; this sample-limited quantity is not total model support.",
        "correct_repeat_fraction": "Sum(correct draws minus distinct correct modes) divided by all correct draws: correct draws repeating an already observed mode.",
        "uniform_expected_distinct8": "Mean M*(1-(1-1/M)^c), conditional on each prompt's observed number c of correct draws and its certified mode count M.",
        "distinct8_gap_to_uniform": "Mean uniform expected distinct minus observed distinct, on prompts with known certified support; positive means fewer observed modes than uniform sampling predicts.",
        "uniform_correct_pair_collision": "Uniform expected collision 1/M, weighted by each prompt's observed number of correct pairs, on prompts with known certified support.",
        "correct_pair_collision_excess_uniform": "Observed minus uniform expected correct-pair collision on the same known-support prompts and pair weights.",
        "uniform_one_mode_rate": "Mean M^(1-c) conditional on observed c>=2 correct draws, on prompts with known certified support.",
    }


def limitations() -> list[str]:
    return [
        "This is an inference-only snapshot of one deployed model. It can show concentration of observed correct modes, but cannot establish a training-induced collapse, compare before and after training, or isolate an effect of model size.",
        "Eight draws per prompt give only a sample-limited view: unseen valid modes can retain positive probability. Neither repeated answers nor a low distinct8 proves that a mode has zero support.",
        "Countdown metadata counts an enumerated binary-expression library, but its verifier also accepts additional canonical modes containing unary negation. Its total verifier support is unknown here, so no certified-support uniform reference is reported for Countdown.",
        "The uniform baseline conditions on the observed number of correct draws and the certified canonical-mode count. It is a reference distribution, not a claim that a useful model must sample all correct modes uniformly.",
        "Correctness and output-format compliance affect observed diversity. Conditional-correct metrics help separate correctness from concentration, but condition on a selected set of successful draws.",
        "Level 1 Pantry selects a six-bit support mask whose quantities are supplied by the benchmark's deterministic trusted projection; levels 2 and 3 emit explicit quantities. Interpret their cross-level correctness differences with this interface distinction in mind.",
        "Inherited Level 2–3 prompts explicitly prescribe MathIR's move-variable, remove-constant, divide order; suggest a small-divisor conditional chain for Python; and favor particular ingredient families for Pantry. Concentration can reflect following these solution preferences. There is no prompt-ablation control, and no prompt asks for uniform coverage or a different valid mode on each call.",
        "MathIR grades symbolic-program compliance, which can reject a numerically valid shortcut. For a=1 in a*x+b=c, subtracting b may solve the numerical equation, while the symbolic checker retains a*x and requires division by a to reach a final x node. Such failures do not establish inability to solve the numerical equation.",
        "Canonical modes are operational output equivalence classes: graph color assignments, Countdown expression trees, Python returned-divisor vectors, MathIR symbolic state paths, and Pantry ingredient sets. They are not direct observations of hidden reasoning or a shared measure of cognitive strategy diversity.",
        "Level 3 datasets were calibrated against particular small-model reference results; their level labels do not guarantee increasing difficulty for this hosted model. Returned decoding settings, reasoning effort, and output budget also differ from the small-model protocol.",
        "Requests are stateless with separate response identifiers, but provider RNG independence cannot be audited directly. Hidden reasoning text and internal probability distributions are not exposed, so these results do not measure their contents or entropy.",
        "Hosted natural-language generation does not use the local training sampler's strict token-level output constraint. Results characterize this hosted prompting and reasoning configuration.",
        "Prompts differ across levels and domain support sizes differ. Level summaries weight each domain equally; raw distinct counts alone are not directly comparable measures of total support.",
        "Bootstrap intervals resample entire prompts within each cell. They are descriptive pointwise intervals, not a test of model training effects or correction for many comparisons.",
        "API error attempts are excluded from model-answer scoring and may include retries that later succeeded. Returned incomplete responses remain scored draws; only fully sampled eight-draw prompt groups enter metrics.",
    ]


def format_metric(record: dict[str, Any], name: str, percent: bool = False, ci: bool = False) -> str:
    stat = record["metrics"][name]
    value = stat["estimate"]
    if value is None:
        return "—"
    scale = 100 if percent else 1
    digits = 1 if percent else 2
    suffix = "%" if percent else ""
    output = f"{value * scale:.{digits}f}{suffix}"
    if ci and stat["ci95"] is not None:
        low, high = stat["ci95"]
        output += f" [{low * scale:.{digits}f}, {high * scale:.{digits}f}]"
    return output


def render_report(summary: dict[str, Any]) -> str:
    s = summary
    lines = ["# Hosted ModeBench evaluation", "",
        f"Status: **{s['status']}**. Received **{s['received_responses']:,}/{s['expected_responses']:,}** responses; "
        f"**{s['complete_prompts']:,}/{s['expected_prompts']:,}** prompts have all eight draws and enter the metrics.", "",
        f"Missing responses: {s['missing_responses']:,}. Responses excluded because their prompt group is unfinished: "
        f"{s['excluded_responses_in_incomplete_groups']:,}. API error attempts, scored separately: {s['api_error_attempts']:,}.", "",
        f"Returned model identifiers: `{json.dumps(s['models_returned'], sort_keys=True)}`. "
        f"Response statuses: `{json.dumps(s['response_status_counts'], sort_keys=True)}`.", ""]
    if s.get("primary_grading_audit"):
        audit = s["primary_grading_audit"]
        lines += [f"Primary scores use audited strict grades: **{audit['corrected_records']:,}** original grading records were corrected by the frozen serial-worker audit. "
                  "Original API responses and grading receipts are preserved. These operational corrections are separate from formatting normalization.", ""]
    if s["run_configuration"]:
        lines += ["Run configuration:", "", "```json", json.dumps(s["run_configuration"], indent=2, sort_keys=True), "```", ""]
    if s["status"] != "complete":
        lines += ["**Partial results:** only complete prompt groups enter this report. Their completion order may be selective, so these are not full-test estimates.", ""]
    lines += ["## Equal-domain level averages (strict primary grading)", "", "Entries include pointwise 95% prompt-bootstrap intervals. Collision rates are pair weighted inside each domain, then averaged equally across domains.", "",
              "| Level | Complete prompts | Pass@1 | Pass@8 | Distinct@8 | Correct-pair collision | One-mode rate (≥2 correct) | Effective modes (≥1 correct) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, record in s["levels"].items():
        columns = [key, f"{record['complete_prompts']}/{record['expected_prompts']}"]
        columns += [format_metric(record, metric, percent, True) for metric, percent in (
            ("pass1", True), ("pass8", True), ("distinct8", False), ("correct_pair_collision", True),
            ("one_mode_rate", True), ("effective_modes_correct", False))]
        lines.append("| " + " | ".join(columns) + " |")
    lines += ["", "## Domain and level results", "",
              "| Level | Domain | Complete prompts | Pass@1 | Pass@8 | Distinct@8 | Correct-pair collision | One-mode rate | Effective modes |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for record in s["cells"].values():
        columns = [str(record["level"]), DOMAIN_LABELS.get(record["domain"], record["domain"]), f"{record['complete_prompts']}/{record['expected_prompts']}"]
        columns += [format_metric(record, metric, percent) for metric, percent in (
            ("pass1", True), ("pass8", True), ("distinct8", False), ("correct_pair_collision", True),
            ("one_mode_rate", True), ("effective_modes_correct", False))]
        lines.append("| " + " | ".join(columns) + " |")
    lines += ["", "Per-cell intervals and exact pair counts are included in `summary.json`.", "",
              "## Finite-sample uniform reference", "",
              "This comparison fixes each prompt's observed number of correct draws. Positive mode-count gaps or collision excess indicate concentration relative to uniform sampling over its certified correct modes.", "",
              "| Level | Domain | Uniform expected distinct | Expected − observed distinct (95% CI) | Uniform pair collision | Observed − uniform collision (95% CI) | Correct pairs | Prompts with ≥2 correct |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for record in s["cells"].values():
        columns = [str(record["level"]), DOMAIN_LABELS.get(record["domain"], record["domain"]),
                   format_metric(record, "uniform_expected_distinct8"), format_metric(record, "distinct8_gap_to_uniform", ci=True),
                   format_metric(record, "uniform_correct_pair_collision", percent=True),
                   format_metric(record, "correct_pair_collision_excess_uniform", percent=True, ci=True),
                   str(int(record["counts"]["correct_pairs"])), str(int(record["counts"]["prompts_with_two_correct"]))]
        lines.append("| " + " | ".join(columns) + " |")
    lines += ["", "![Accuracy and observed mode diversity](modebench_frontier.png)", ""]
    if "normalized_secondary" in s:
        secondary = s["normalized_secondary"]
        lines += ["## Formatting normalization (post hoc secondary analysis)", "",
                  "The primary results above retain the frozen strict grader. This secondary check deterministically normalizes supported LaTeX and presentation syntax, then reuses the original executable verifier. It repairs no mathematical values or program logic. Strict successes and their canonical keys are retained exactly.", "",
                  f"Additional verified responses after normalization: **{secondary['additional_verified_responses']:,}**. "
                  f"Strict verified responses: {secondary['strict_verified_responses']:,}; secondary verified responses: {secondary['normalized_verified_responses']:,}.", "",
                  "| Level / domain | Strict pass@1 | Normalized pass@1 | Normalized pass@8 | Normalized distinct@8 | Normalized correct-pair collision | Normalized effective modes |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        records = [(f"L{k} macro", rec, s["levels"][k]) for k, rec in secondary["levels"].items()]
        records += [(f"L{rec['level']} / {DOMAIN_LABELS.get(rec['domain'], rec['domain'])}", rec, s["cells"][k]) for k, rec in secondary["cells"].items()]
        for label, record, strict in records:
            columns = [label, format_metric(strict, "pass1", True), format_metric(record, "pass1", True)]
            columns += [format_metric(record, name, percentage) for name, percentage in (
                ("pass8", True), ("distinct8", False), ("correct_pair_collision", True), ("effective_modes_correct", False))]
            lines.append("| " + " | ".join(columns) + " |")
        if secondary.get("python_cache_reconciliation"):
            audit = secondary["python_cache_reconciliation"]
            lines += ["", f"Secondary Python grading audit: {audit['graded_python_records']:,} cached records checked serially; {audit['changed_records']:,} operational grading corrections. The previous cache and every comparison are preserved in the reconciliation audit.", ""]
        lines += ["", "All secondary confidence intervals, support-calibrated comparisons, and per-prompt occupancies are in `summary.json`. "
                  "This formatting policy was introduced after seeing the first fifteen responses, so it is a diagnostic sensitivity analysis, not a preregistered replacement endpoint.", "",
                  f"Applied transformation counts: `{json.dumps(secondary['transformation_counts'], sort_keys=True)}`.", "",
                  "![Secondary results after formatting normalization](modebench_frontier_normalized.png)", ""]
    lines += ["## Interpretation limits", "", "See [the interface interpretation notes](interpretation_notes.md) for frozen prompt excerpts, verifier semantics, and concrete saved-response examples.", ""]
    lines += [f"- {note}" for note in s["limitations"]]
    lines += ["", "## Metric definitions", ""]
    lines += [f"- **{name}:** {definition}" for name, definition in s["metric_definitions"].items()]
    lines += ["", "## Usage and provenance", "", f"Reported token usage: `{json.dumps(s['usage_totals'], sort_keys=True)}`.", "",
              "Usage totals cover saved responses; unsuccessful API attempts may have incurred unreported usage. Monetary cost is not estimated without a verified deployment-specific rate.", "",
              f"Latency in seconds: `{json.dumps(s['latency_seconds'], sort_keys=True)}`.", "",
              f"Sampling settings returned by the API: `{json.dumps(s['returned_sampling'], sort_keys=True)}`.", "",
              f"Deployment header examples (preflight and first/last saved responses): `{json.dumps(s['deployment_header_examples'], sort_keys=True)}`.", "",
              f"Bootstrap: {s['bootstrap']['replicates']:,} replicates, seed {s['bootstrap']['seed']}. {s['bootstrap']['unit']}", "",
              "Input SHA-256 digests are saved in `summary.json`. The prompt snapshot contains only the frozen evaluation split.", ""]
    if s["dataset_provenance_notes"]:
        notes = s["dataset_provenance_notes"]
        lines += ["Dataset provenance:", "", "```json", json.dumps(notes, indent=2, sort_keys=True), "```", ""]
    return "\n".join(lines)


def make_plot(summary: dict[str, Any], directory: Path, stem: str = "modebench_frontier") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    domains = [domain for domain in DOMAIN_ORDER if any(c["domain"] == domain for c in summary["cells"].values())]
    domains += sorted({c["domain"] for c in summary["cells"].values()} - set(domains))
    levels = sorted(int(level) for level in summary["levels"])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, len(domains), figsize=(3.05 * len(domains), 9.0), squeeze=False, sharex=True)
    plot_rows = (("pass1", None, "Verified draws (%)", 100),
                 ("distinct8", "uniform_expected_distinct8", "Distinct correct modes / 8 draws", 1),
                 ("correct_pair_collision", "uniform_correct_pair_collision", "Correct-pair collision (%)", 100))
    for col, domain in enumerate(domains):
        for row, (metric, baseline, ylabel, scale) in enumerate(plot_rows):
            ax = axes[row, col]
            for series, color, marker, linestyle, label in ((metric, "#185a9d", "o", "-", "Observed"),
                (baseline, "#cc6d18", "s", "--", "Uniform, conditional on correct draws")):
                if series is None:
                    continue
                values, xs, errors = [], [], []
                for level in levels:
                    record = summary["cells"].get(f"level{level}/{domain}")
                    if record is None or record["metrics"][series]["estimate"] is None:
                        continue
                    stat = record["metrics"][series]
                    value = stat["estimate"] * scale
                    low, high = stat["ci95"] or [stat["estimate"], stat["estimate"]]
                    xs.append(level); values.append(value)
                    errors.append([max(0, value - low * scale), max(0, high * scale - value)])
                if values:
                    ax.errorbar(xs, values, yerr=np.asarray(errors).T, color=color, marker=marker,
                                linestyle=linestyle, linewidth=1.5, markersize=5, capsize=3, label=label)
            ax.set_xticks(levels, [f"L{level}" for level in levels])
            ax.grid(axis="y", alpha=.22)
            ax.set_xlim(min(levels) - .3, max(levels) + .3)
            ax.set_ylim(0, 102 if scale == 100 else 8.2)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(DOMAIN_LABELS.get(domain, domain), fontweight="bold")
            if row == 2:
                ax.set_xlabel("Benchmark level")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=2, frameon=False)
    fig.suptitle(f"Hosted ModeBench · {summary['received_responses']:,}/{summary['expected_responses']:,} responses · {summary['status']}", fontsize=15, y=.98)
    fig.text(.5, .012, "Eight draws per prompt; 95% whole-prompt bootstrap intervals. Observed concentration does not establish training-induced collapse.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .945), h_pad=2.2)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)



def add_normalized_analysis(summary: dict[str, Any], directory: Path, replicates: int, seed: int) -> None:
    """Regrade only strict failures; cache by exact receipt and normalizer source."""
    frozen_root = directory.resolve() / "code"
    frozen_grader = None
    frozen_contract = frozen_root / "ops/frontier_modebench_contract.py"
    if frozen_contract.exists():
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        for relative, expected_hash in manifest.get("code_sha256", {}).items():
            if digest(frozen_root / relative) != expected_hash:
                raise ValueError(f"Frozen grading source changed: {relative}")
        for module_name, module in list(sys.modules.items()):
            if module_name in ("frontier_modebench_contract", "oat_drgrpo") or module_name.startswith("oat_drgrpo."):
                source = getattr(module, "__file__", None)
                if source and not Path(source).resolve().is_relative_to(frozen_root):
                    raise ValueError(f"Refusing an already imported unfrozen grader module: {module_name}")
        sys.path.insert(0, str(frozen_root / "src"))
        sys.path.insert(0, str(frozen_root / "ops"))
        import frontier_modebench_contract as contract
        frozen_grader = contract.grade_response
    secondary_root = directory.resolve() / "secondary_code/ops"
    if (secondary_root / "frontier_modebench_normalization.py").exists():
        loaded_normalizer = sys.modules.get("frontier_modebench_normalization")
        if loaded_normalizer is not None and not Path(loaded_normalizer.__file__).resolve().is_relative_to(secondary_root):
            raise ValueError("Refusing an already imported unfrozen normalization module")
        sys.path.insert(0, str(secondary_root))
    import frontier_modebench_normalization as normalization
    code_hash = digest(Path(normalization.__file__))
    initial_audit_path = directory / "secondary_initial15_audit.json"
    if initial_audit_path.exists():
        audit = json.loads(initial_audit_path.read_text())
        expected_normalizer_hash = audit.get("normalizer", {}).get("sha256")
        if expected_normalizer_hash and code_hash != expected_normalizer_hash:
            raise ValueError("Normalization source differs from the frozen initial rule audit")
    rows = {prompt_id(r): r for r in read_jsonl(directory / "rows.jsonl")}
    samples = read_jsonl(Path(summary["primary_samples_path"]))[:summary["received_responses"]]
    snapshot_hash = hashlib.sha256(json.dumps(samples, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if snapshot_hash != summary["response_records_sha256"]:
        raise ValueError("Strict response snapshot changed before normalization")
    cache_path = directory / "normalized_samples.jsonl"
    cache = {}
    for record in read_jsonl(cache_path):
        if record.get("normalization_source_sha256") == code_hash:
            cache[record["strict_receipt_sha256"]] = record
    raw_lookup = {(*prompt_id(sample), sample["sample_index"]): sample for sample in read_jsonl(directory / "samples.jsonl")}
    def cache_receipt(sample):
        raw = raw_lookup.get((*prompt_id(sample), sample["sample_index"]), sample)
        return {**raw, **{name: sample[name] for name in ("verified", "canonical_key", "graded_text") if name in sample}}
    worker_warmup = None
    if frozen_grader is not None and any(s["domain"] == "python_factors" for s in samples):
        worker_warmup = warm_python_verifier(samples, rows, frozen_grader)
    normalized = []
    transformations: Counter = Counter()
    strict_correct = additional = 0
    with cache_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            receipt_hash = hashlib.sha256(json.dumps(cache_receipt(sample), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            cached = cache.get(receipt_hash)
            if cached is None:
                if sample["verified"]:
                    grade = {"verified": True, "canonical_key": sample["canonical_key"],
                             "graded_text": sample.get("graded_text", sample["text"]),
                             "original_text": sample["text"], "normalized_text": sample["text"],
                             "transformations": [], "posthoc": True, "strict_success_retained": True}
                else:
                    kwargs = {"strict_grade": sample}
                    if frozen_grader is not None:
                        kwargs["grader"] = frozen_grader
                    grade = normalization.normalize_and_grade(rows[prompt_id(sample)], sample["text"], **kwargs)
                cached = {"normalization_source_sha256": code_hash, "strict_receipt_sha256": receipt_hash,
                          "level": sample["level"], "domain": sample["domain"], "row_index": sample["row_index"],
                          "sample_index": sample["sample_index"], "normalization": grade}
                handle.write(json.dumps(cached, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
            grade = cached["normalization"]
            if sample["verified"] and (not grade["verified"] or grade["canonical_key"] != sample["canonical_key"]):
                raise ValueError("Secondary analysis must retain every strict success and its canonical key")
            normalized.append({**sample, "verified": grade["verified"], "canonical_key": grade["canonical_key"],
                               "graded_text": grade.get("graded_text", grade.get("normalized_text", sample["text"]))})
            strict_correct += int(sample["verified"])
            additional += int(grade["verified"] and not sample["verified"])
            transformations.update(grade.get("transformations", []))
    secondary = summarize(directory, replicates, seed, sample_records=normalized)
    summary["normalized_secondary"] = {
        "analysis": "Post hoc, formatting-only sensitivity analysis; strict successes retained exactly.",
        "normalization_source_sha256": code_hash, "normalization_source_path": str(Path(normalization.__file__).resolve()), "cache_sha256": digest(cache_path),
        "normalization_rules": getattr(normalization, "RULE_DESCRIPTIONS", {}),
        "initial_rule_audit_sha256": digest(initial_audit_path),
        "python_worker_warmup": worker_warmup,
        "python_cache_reconciliation": load_cache_reconciliation(directory),
        "frozen_grader_contract_sha256": digest(frozen_contract),
        "frozen_grader_contract_path": str(frozen_contract) if frozen_contract.exists() else None,
        "strict_verified_responses": strict_correct, "normalized_verified_responses": strict_correct + additional,
        "additional_verified_responses": additional, "transformation_counts": dict(transformations),
        "levels": secondary["levels"], "cells": secondary["cells"], "prompts": secondary["prompts"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--primary-samples", type=Path, help="Optional audited strict-grade sidecar, relative to input-dir or absolute; original API receipts remain immutable.")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--with-normalization", action="store_true", help="Add a cached, post hoc formatting-only diagnostic with the original verifier.")
    args = parser.parse_args(argv)
    primary_path = args.primary_samples
    if primary_path is not None and not primary_path.is_absolute():
        primary_path = args.input_dir / primary_path
    result = summarize(args.input_dir, args.bootstrap_replicates, args.seed, primary_samples_path=primary_path)
    if args.with_normalization:
        add_normalized_analysis(result, args.input_dir, args.bootstrap_replicates, args.seed)
    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "report.md").write_text(render_report(result))
    if not args.no_plot:
        make_plot(result, output)
        if "normalized_secondary" in result:
            plot_summary = {**result, **result["normalized_secondary"]}
            plot_summary["status"] = result["status"] + " · formatting normalized (post hoc)"
            make_plot(plot_summary, output, "modebench_frontier_normalized")
    print(json.dumps({key: result[key] for key in ("status", "received_responses", "expected_responses", "complete_prompts", "missing_responses")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
