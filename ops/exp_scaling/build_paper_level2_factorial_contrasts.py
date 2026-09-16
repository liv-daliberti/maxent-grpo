#!/usr/bin/env python3
"""Derive E119 terminal factorial contrasts from a frozen endpoint audit.

Run with ``--date YYYY-MM-DD`` after build_paper_latest_results.py. Every
contrast within a domain uses the same all-four-arm seed intersection. The
JSON includes all five simple/interaction contrasts; the compact TeX body
reports the two estimator effects and the interaction, complementing the
separate replay-effect table. Neither live run files nor prior results change.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
DOMAIN_LABELS = {
    "graph_coloring": "Graph", "countdown": "Countdown",
    "python_factors": "Python", "mathir": "MathIR", "pantry_plan": "PantryPlan",
}
ARMS = ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
SEEDS = (43, 44, 45, 46, 47)
METRICS = ("pass8", "distinct8", "breadth8", "mean8")
CONTRASTS = {
    "maxrl_minus_drgrpo": {"maxrl": 1, "drgrpo": -1},
    "replay_maxrl_minus_replay_drgrpo": {"replay_maxrl": 1, "replay_drgrpo": -1},
    "replay_drgrpo_minus_drgrpo": {"replay_drgrpo": 1, "drgrpo": -1},
    "replay_maxrl_minus_maxrl": {"replay_maxrl": 1, "maxrl": -1},
    "factorial_interaction": {
        "replay_maxrl": 1, "maxrl": -1, "replay_drgrpo": -1, "drgrpo": 1,
    },
}
TABLE_CONTRASTS = {
    "maxrl_minus_drgrpo": "MaxRL $-$ Dr.GRPO",
    "replay_maxrl_minus_replay_drgrpo": "Re:Max $-$ Re:Dr",
    "factorial_interaction": "Replay $\\times$ MaxRL interaction",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def summarize(values: dict[int, float]) -> dict[str, Any]:
    seeds = sorted(values)
    result: dict[str, Any] = {
        "n": len(seeds), "per_seed": {str(seed): values[seed] for seed in seeds},
    }
    if seeds:
        result["mean"] = statistics.fmean(values.values())
    if seeds == list(SEEDS):
        half = 2.7764451051977987 * statistics.stdev(values.values()) / math.sqrt(5)
        result["student_t_95"] = [result["mean"] - half, result["mean"] + half]
    return result


def build(audit: dict[str, Any]) -> dict[str, Any]:
    rows = audit["campaigns"]["e119"]["rows"]
    registered = {(domain, arm, seed) for domain in DOMAINS for arm in ARMS for seed in SEEDS}
    seen = set()
    admitted: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in rows:
        key = (row["domain"], row["arm"], int(row["seed"]))
        model = row.get("scale", row.get("model_key", "qwen05b"))
        if key not in registered or key in seen or model != "qwen05b":
            raise ValueError(f"unexpected or duplicate E119 cell: {key}")
        seen.add(key)
        if row["endpoint_status"] != "admitted":
            continue
        integrity = row["integrity_audit"]
        if (integrity["status"] != "admitted" or integrity["step"] != 3072
                or integrity["observed_draws"] != [0, 1, 2, 3]
                or integrity.get("conflicting_retry_selected") is not False):
            raise ValueError(f"invalid admitted endpoint audit: {key}")
        endpoint = row["endpoint"]
        values = {metric: float(endpoint[metric]) for metric in METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite admitted endpoint: {key}")
        if not math.isclose(values["breadth8"], values["distinct8"] - values["pass8"], abs_tol=1e-12):
            raise ValueError(f"inconsistent extra-mode decomposition: {key}")
        admitted[key] = values
    if seen != registered:
        raise ValueError(f"E119 audit is missing registered cells: {sorted(registered - seen)}")

    blocks = []
    for domain in DOMAINS:
        arm_seeds = {arm: sorted(seed for d, a, seed in admitted if d == domain and a == arm)
                     for arm in ARMS}
        common = sorted(set.intersection(*(set(seeds) for seeds in arm_seeds.values())))
        contrasts = {}
        for name, weights in CONTRASTS.items():
            summaries = {}
            for metric in METRICS:
                values = {seed: sum(weight * admitted[domain, arm, seed][metric]
                                    for arm, weight in weights.items()) for seed in common}
                summaries[metric] = summarize(values)
            contrasts[name] = {"coefficients": weights, "summaries": summaries}
        blocks.append({
            "model_key": "qwen05b", "domain": domain,
            "terminal_seeds_by_arm": arm_seeds,
            "paired_seeds": common, "n": len(common),
            "complete_five_seed_block": common == list(SEEDS),
            "arm_summaries": {
                arm: {metric: summarize({seed: admitted[domain, arm, seed][metric] for seed in common})
                      for metric in METRICS} for arm in ARMS
            },
            "contrasts": contrasts,
        })
    return {
        "registered_cells": len(registered), "admitted_terminal_endpoints": len(admitted),
        "registered_blocks": len(DOMAINS),
        "complete_five_seed_blocks": sum(block["complete_five_seed_block"] for block in blocks),
        "blocks": blocks,
    }


def signed(value: float) -> str:
    return f"{value:+.3f}".replace("+0.", "+.").replace("-0.", "-.")


def render_table(result: dict[str, Any]) -> str:
    lines = ["% Domain & paired n & contrast & delta pass@8 & delta distinct@8 & delta extra modes"]
    for block in result["blocks"]:
        for name, label in TABLE_CONTRASTS.items():
            cells = [DOMAIN_LABELS[block["domain"]], str(block["n"]), label]
            for metric in ("pass8", "distinct8", "breadth8"):
                summary = block["contrasts"][name]["summaries"][metric]
                if not summary["n"]:
                    cells.append(r"\textemdash")
                elif "student_t_95" in summary:
                    lower, upper = summary["student_t_95"]
                    cells.append(f"${signed(summary['mean'])}\\;[{signed(lower)},{signed(upper)}]$")
                else:
                    cells.append(f"${signed(summary['mean'])}$")
            lines.append(" & ".join(cells) + r" \\")
    lines.append(r"    \bottomrule")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="analysis date, YYYY-MM-DD")
    parser.add_argument("--audit", type=Path, help="frozen latest_endpoints.json; defaults to the dated audit")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/results")
    args = parser.parse_args()
    analysis_date = date.fromisoformat(args.date).isoformat()
    stamp = analysis_date.replace("-", "")
    audit_path = args.audit or ROOT / f"paper/audits/results_refresh_{stamp}/latest_endpoints.json"
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("schema") != "paper-latest-endpoint-audit-v1" or "e119" not in audit.get("campaigns", {}):
        raise ValueError("the source must be a frozen endpoint audit containing the E119 campaign")
    result = {
        "schema": "paper-level2-factorial-contrasts-v1", "analysis_date": analysis_date,
        "target_step": 3072, "model": "Qwen2.5-0.5B-Instruct", "level": 2,
        "collection_started_at_utc": audit["collected_at_utc"],
        "collection_finished_at_utc": audit["finished_at_utc"],
        "source_audit": {"path": relative(audit_path), "sha256": hashlib.sha256(audit_bytes).hexdigest()},
        "builder": {"path": relative(Path(__file__)), "sha256": digest(Path(__file__))},
        "ledger": audit["campaigns"]["e119"]["ledger"],
        "ledger_sha256": audit["campaigns"]["e119"]["ledger_sha256"],
        "scope": (
            "Exact step-3072 endpoints from the frozen source-admissible four-draw audit. "
            "All contrasts and arm means use the all-four-arm seed intersection within each domain. "
            "Five registered seeds receive unadjusted descriptive paired Student-t 95% intervals; "
            "partial blocks have exact seed sets and no interval. No completion, endpoint, or "
            "missing cell is imputed; no cross-domain pooling."
        ),
        "metric_definitions": {
            "pass8": "Probability of at least one correct sample among eight (co-primary).",
            "distinct8": "Distinct verified semantic modes among eight samples (co-primary).",
            "breadth8": "distinct8 minus pass8; extra modes beyond the first, still coupled to correctness.",
            "mean8": "Mean per-sample correctness among eight samples (secondary).",
        },
        "interaction_definition": "(Re:Max - MaxRL) - (Re:Dr - Dr.GRPO)",
        **build(audit),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"level2_factorial_contrasts_{stamp}.json"
    table = args.output_dir / f"level2_factorial_contrasts_{stamp}_table_body.tex"
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    table.write_text(render_table(result), encoding="utf-8")
    print(json.dumps({"json": relative(output), "table": relative(table),
                      "complete_five_seed_blocks": result["complete_five_seed_blocks"],
                      "paired_seeds": {block["domain"]: block["paired_seeds"] for block in result["blocks"]}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
