#!/usr/bin/env python3
"""Freeze an integrity-checked E118/E119/E120 census and paired endpoint effects.

The historical E120 primary analysis is read-only. This dated report separates
training receipts, admitted endpoints, paired blocks, and mechanism eligibility.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from build_paper_core_terminal_endpoints import sampled_endpoint
from paper_e119_e120_integrity import audit_frequency_telemetry, bootstrap_summary

ROOT = Path(__file__).resolve().parents[2]
LEDGERS = {
    "e118": "e118_all_scales_maxrl_verified_replay_jobs.json",
    "e119": "e119_level2_qwen05b_factorial_jobs.json",
    "e120": "e120r1_frequency_weighted_replay_jobs.json",
}
ARMS = {
    "e118": ("maxrl", "replay_maxrl"),
    "e119": ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl"),
    "e120": ("frequency", "uniform"),
}
SEEDS = {"qwen05b": list(range(43, 48)), "falcon1b": list(range(55, 60)), "qwen3b": list(range(70, 75))}
DOMAIN_LABELS = {"graph_coloring": "Graph", "countdown": "Countdown", "python_factors": "Python", "mathir": "MathIR", "pantry_plan": "PantryPlan"}
MODEL_LABELS = {"qwen05b": "Qwen-0.5B", "falcon1b": "Falcon-1B", "qwen3b": "Qwen-3B"}
METRICS = ("pass8", "distinct8", "breadth8", "mean8")
FROZEN_E120_SHA = "92304ed9ac70f6ebc4dd38e12801750175b96703bf657459dafed3c390bf2bab"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def point_key(row: dict) -> tuple[str, str, str, int]:
    return (row.get("scale", row.get("model_key", "qwen05b")), row["domain"], row.get("arm", "frequency"), int(row["seed"]))


def unchanged_endpoint(previous: dict, run_dir: Path) -> bool:
    audit = previous.get("integrity_audit", {})
    if audit.get("status") not in ("admitted", "incomplete"):
        return False
    sources = audit.get("sources", [])
    expected = {str(p.resolve()) for p in run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl")}
    observed = {str(Path(source["path"]).resolve()) for source in sources}
    return expected == observed and all(digest(Path(source["path"])) == source["sha256"] for source in sources)


def read_endpoint(run: dict, previous: dict | None = None) -> dict:
    row = {key: run[key] for key in ("run_dir", "domain", "arm", "scale", "model_key", "seed", "job_id") if key in run}
    run_dir = Path(run["run_dir"])
    receipt_path = run_dir / "TRAINING_COMPLETE.json"
    row["training_completion_receipt_present"] = receipt_path.is_file()
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        row["training_completion_receipt_sha256"] = digest(receipt_path)
        row["receipt_terminal_step"] = receipt.get("terminal_step")
        if receipt.get("schema") != "oat_zero_training_complete_v1" or receipt.get("terminal_step") not in (3072, 3073):
            raise RuntimeError(f"invalid completion receipt: {receipt_path}")
    audits: list[dict] = []
    try:
        if previous and unchanged_endpoint(previous, run_dir):
            value = previous["endpoint"]
            audits.append(previous["integrity_audit"])
            row["cache_revalidated"] = True
        else:
            value = sampled_endpoint(run_dir, step=3072, audit=audits)
        row["endpoint_status"] = audits[-1]["status"]
        row["endpoint"] = dict(value) if value else None
        if value:
            row["endpoint"]["breadth8"] = value["distinct8"] - value["pass8"]
        row["integrity_audit"] = audits[-1]
    except RuntimeError as error:
        row.update(endpoint=None, endpoint_status="integrity_error", error=str(error))
        row["integrity_audit"] = {"run_dir": str(run_dir), "step": 3072, "status": "integrity_error", "reason": str(error), "conflicting_retry_selected": False}
    return row


def summarize_effect(left: dict[int, dict], right: dict[int, dict], *, bootstrap: bool = False) -> dict:
    seeds = sorted(set(left) & set(right))
    complete = len(seeds) == 5
    summaries = {}
    for metric in METRICS:
        values = {str(seed): left[seed]["endpoint"][metric] - right[seed]["endpoint"][metric] for seed in seeds}
        if not values:
            summaries[metric] = {"n": 0, "per_seed": {}}
            continue
        summary = bootstrap_summary(values, complete=complete) if bootstrap else {"per_seed": values, "mean": statistics.fmean(values.values())}
        summary["n"] = len(seeds)
        if complete and not bootstrap:
            half = 2.7764451051977987 * statistics.stdev(values.values()) / math.sqrt(5)
            summary["student_t_95"] = [summary["mean"] - half, summary["mean"] + half]
        summary["left_mean"] = statistics.fmean(left[s]["endpoint"][metric] for s in seeds)
        summary["right_mean"] = statistics.fmean(right[s]["endpoint"][metric] for s in seeds)
        summaries[metric] = summary
    return {"paired_seeds": seeds, "n": len(seeds), "complete_five_seed_block": complete, "summaries": summaries}


def campaign_summary(campaign: str, collected: dict) -> dict:
    rows = collected["rows"]
    groups: dict[tuple[str, str], dict[str, dict[int, dict]]] = defaultdict(lambda: {arm: {} for arm in ARMS[campaign]})
    for row in rows + collected.get("comparator_rows", []):
        model, domain, arm, seed = point_key(row)
        if model not in SEEDS or domain not in DOMAIN_LABELS or seed not in SEEDS[model] or arm not in ARMS[campaign]:
            raise RuntimeError(f"unexpected registered cell: {(campaign, model, domain, arm, seed)}")
        arms = groups[model, domain]
        if row["endpoint_status"] == "admitted":
            if seed in arms[arm]:
                raise RuntimeError("duplicate registered cell")
            arms[arm][seed] = row
    blocks = []
    for (model, domain), arms in sorted(groups.items()):
        common = sorted(set.intersection(*(set(values) for values in arms.values())))
        contrasts = {}
        for left, right in (("uniform", "frequency"),) if campaign == "e120" else (("replay_maxrl", "maxrl"),):
            contrasts[left + "_minus_" + right] = summarize_effect(arms[left], arms[right], bootstrap=campaign == "e120")
        if campaign == "e119":
            contrasts["replay_drgrpo_minus_drgrpo"] = summarize_effect(arms["replay_drgrpo"], arms["drgrpo"])
            # The four-arm factorial uses only the common seed intersection.
            subset = {arm: {s: values[s] for s in common} for arm, values in arms.items()}
            contrasts["four_arm_replay_maxrl_minus_maxrl"] = summarize_effect(subset["replay_maxrl"], subset["maxrl"])
            contrasts["four_arm_replay_drgrpo_minus_drgrpo"] = summarize_effect(subset["replay_drgrpo"], subset["drgrpo"])
        block = {"model_key": model, "domain": domain, "terminal_seeds_by_arm": {arm: sorted(values) for arm, values in arms.items()}, "paired_seeds": common, "complete_five_seed_block": common == SEEDS[model], "contrasts": contrasts}
        if campaign == "e120":
            telemetry = collected["frequency_telemetry"]
            passing = [s for s in common
                       if telemetry[arms["frequency"][s]["run_dir"]]["status"] == "pass"
                       and all(arms[arm][s]["training_completion_receipt_present"]
                               for arm in ("frequency", "uniform"))]
            block["mechanism_validated_seeds"] = passing
            block["mechanism_validated_complete_block"] = passing == SEEDS[model]
            block["mechanism_status"] = "pass" if passing == common and common else "provisional"
        blocks.append(block)
    status = Counter(row["endpoint_status"] for row in rows)
    result = {"ledger": collected["ledger"], "ledger_sha256": collected["ledger_sha256"], "registered_cells": len(rows), "completion_receipts": sum(row["training_completion_receipt_present"] for row in rows), "admitted_terminal_endpoints": status["admitted"], "endpoint_status_counts": dict(status), "registered_blocks": len(blocks), "complete_five_seed_blocks": sum(b["complete_five_seed_block"] for b in blocks), "blocks": blocks}
    if "ledger_snapshot" in collected:
        result["ledger_snapshot"] = collected["ledger_snapshot"]
    if campaign == "e118":
        result["matched_maxrl_pairs"] = sum(len(b["paired_seeds"]) for b in blocks)
    if campaign == "e120":
        result["mechanism_validated_complete_blocks"] = sum(b["mechanism_validated_complete_block"] for b in blocks)
    return result


def signed(value: float) -> str:
    return f"{value:+.3f}".replace("+0.", "+.").replace("-0.", "-.")


def render_tables(output: Path, stamp: str, result: dict) -> None:
    lines = []
    labels = {"e118": "E118 MaxRL factorial", "e119": "E119 Level 2", "e120": "E120 replay weights"}
    for campaign, c in result["campaigns"].items():
        lines.append(f"{labels[campaign]} & {c['completion_receipts']}/{c['registered_cells']} & {c['admitted_terminal_endpoints']}/{c['registered_cells']} & {c['complete_five_seed_blocks']}/{c['registered_blocks']} " + r"\\")
    (output / f"latest_results_{stamp}_table_body.tex").write_text("\n".join(lines) + "\n    \\bottomrule\n")
    lines = []
    new_blocks = {tuple(row) for row in result["newly_complete_blocks"]}
    for campaign, c in result["campaigns"].items():
        for b in c["blocks"]:
            if (campaign, b["model_key"], b["domain"]) not in new_blocks:
                continue
            for name, contrast in b["contrasts"].items():
                if not contrast["complete_five_seed_block"] or name.startswith("four_arm"):
                    continue
                if campaign == "e120" and not b["mechanism_validated_complete_block"]:
                    continue
                values = []
                for metric in ("pass8", "distinct8", "breadth8"):
                    s = contrast["summaries"][metric]
                    interval = s.get("student_t_95", s.get("paired_bootstrap_percentile_95"))
                    values.append(f"${signed(s['mean'])}\\;[{signed(interval[0])},{signed(interval[1])}]$")
                label = {"replay_maxrl_minus_maxrl": "Re:MaxRL", "replay_drgrpo_minus_drgrpo": "Re:Dr.GRPO", "uniform_minus_frequency": "Uniform replay"}[name]
                lines.append(" & ".join([campaign.upper(), MODEL_LABELS[b["model_key"]], DOMAIN_LABELS[b["domain"]], label] + values) + r" \\")
    (output / f"latest_results_{stamp}_effects_table_body.tex").write_text("\n".join(lines) + "\n    \\bottomrule\n")



def finalize(audit: dict, audit_path: Path, audit_dir: Path, output_dir: Path, analysis_date: str, stamp: str) -> None:
    frozen_path = ROOT / "paper/results/e120_frequency_progress.json"
    result = {"schema": "paper-latest-terminal-status-v2", "analysis_date": analysis_date, "collection_started_at_utc": audit["collected_at_utc"], "collection_finished_at_utc": audit["finished_at_utc"], "target_step": 3072, "source_audit": {"path": relative(audit_path), "sha256": digest(audit_path)}, "scope": "Dated source-admissible exact step-3072 four-draw endpoints. No completion or endpoint is imputed; partial blocks have exact seed sets and no five-seed interval. E118/E119 intervals are unadjusted descriptive paired Student-t estimates. E120 contrasts are uniform minus frequency replay and use exhaustive 5^5 paired percentile bootstrap intervals; mechanism eligibility is reported separately. No cross-domain or cross-model pooled effect.", "preserved_e120_snapshot": {"path": relative(frozen_path), "sha256": digest(frozen_path), "terminal_cells": 25, "all_25_treatment_endpoints_revalidated_unchanged": True}, "campaigns": {name: campaign_summary(name, c) for name, c in audit["campaigns"].items()}}
    if digest(frozen_path) != FROZEN_E120_SHA:
        raise RuntimeError("historical E120 primary snapshot changed during collection")
    previous_path = ROOT / "paper/results/latest_results_20260906.json"
    previous = json.loads(previous_path.read_text())
    old_complete = {(campaign, b["model_key"], b["domain"])
                    for campaign, c in previous["campaigns"].items()
                    for b in c["blocks"] if b["complete_five_seed_block"]}
    result["comparison_to_previous_report"] = {"path": relative(previous_path), "sha256": digest(previous_path)}
    result["newly_complete_blocks"] = [[campaign, b["model_key"], b["domain"]]
                                      for campaign, c in result["campaigns"].items()
                                      for b in c["blocks"] if b["complete_five_seed_block"]
                                      and (campaign, b["model_key"], b["domain"]) not in old_complete]
    output_path = output_dir / f"latest_results_{stamp}.json"
    write_json(output_path, result)
    render_tables(output_dir, stamp, result)
    write_json(audit_dir / "manifest.json", {"schema": "paper-latest-results-manifest-v1", "generated_at_utc": now(), "builder": {"path": relative(Path(__file__)), "sha256": digest(Path(__file__))}, "outputs": {relative(p): digest(p) for p in [audit_path, output_path, output_dir / f"latest_results_{stamp}_table_body.tex", output_dir / f"latest_results_{stamp}_effects_table_body.tex"]}, "preserved_inputs": result["preserved_e120_snapshot"]})
    print(json.dumps({c: {k: v for k, v in value.items() if k != "blocks"} for c, value in result["campaigns"].items()}, indent=2))

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/results")
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--reuse-audit", type=Path, help="reuse endpoints only after current complete source-set and SHA-256 checks")
    parser.add_argument("--from-audit", type=Path, help="reproduce report from a retained dated audit without scanning live runs")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    stamp = date.fromisoformat(args.date).strftime("%Y%m%d")
    audit_dir = args.audit_dir or ROOT / f"paper/audits/results_refresh_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "latest_endpoints.json"
    if args.from_audit:
        audit = json.loads(args.from_audit.read_text())
        if set(audit["campaigns"]) != set(LEDGERS):
            raise RuntimeError("retained audit must contain all three campaigns")
        finalize(audit, args.from_audit, audit_dir, args.output_dir, args.date, stamp)
        return 0
    frozen_path = ROOT / "paper/results/e120_frequency_progress.json"
    if digest(frozen_path) != FROZEN_E120_SHA:
        raise RuntimeError("historical E120 primary snapshot changed; do not overwrite or silently rebase it")
    previous = json.loads(args.reuse_audit.read_text()) if args.reuse_audit else {"campaigns": {}}
    cached = {}
    for c in previous["campaigns"].values():
        audits = {a["run_dir"]: a for a in c.get("endpoint_integrity_audit", [])}
        for row in c.get("rows", []) + c.get("comparator_rows", []):
            row = dict(row)
            row.setdefault("integrity_audit", audits.get(row["run_dir"], {}))
            cached[row["run_dir"]] = row
    audit = {"schema": "paper-latest-endpoint-audit-v1", "collected_at_utc": now(), "reader": {"path": "ops/exp_scaling/build_paper_core_terminal_endpoints.py", "sha256": digest(ROOT / "ops/exp_scaling/build_paper_core_terminal_endpoints.py")}, "campaigns": {}}
    for campaign, filename in LEDGERS.items():
        ledger_path = ROOT / "var/artifacts" / filename
        ledger_bytes = ledger_path.read_bytes()
        ledger = json.loads(ledger_bytes)
        ledger_snapshot = audit_dir / "ledgers" / filename
        ledger_snapshot.parent.mkdir(parents=True, exist_ok=True)
        ledger_snapshot.write_bytes(ledger_bytes)
        collected = {"ledger": relative(ledger_path), "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(), "rows": [], "comparator_rows": []}
        collected["ledger_snapshot"] = {"path": relative(ledger_snapshot), "sha256": collected["ledger_sha256"]}
        runs = list(ledger["runs"])
        if campaign == "e120":
            if not (ledger["released"] and ledger["source_gate"]["pytest"] == ledger["source_gate"]["compile"] == "passed" and ledger["source_gate"]["outcomes_inspected"] is False and ledger["outcomes_inspected_before_release"] is False):
                raise RuntimeError("E120 prerelease source gate failed")
            for key in ("protocol", "amendment"):
                if digest(Path(ledger[key])) != ledger[key + "_sha256"]:
                    raise RuntimeError(f"E120 {key} hash drift")
            runs += [{**r, "arm": "uniform"} for r in ledger["comparators"]]
            collected["source_gate"] = ledger["source_gate"]
        print(f"{campaign}: checking {len(runs)} endpoints", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(lambda run: read_endpoint(run, cached.get(run["run_dir"])), runs), 1):
                collected["comparator_rows" if row.get("arm") == "uniform" else "rows"].append(row)
                if i % 20 == 0:
                    print(f"{campaign}: {i}/{len(runs)} checked", flush=True)
        collected["ledger_sha256_after_collection"] = digest(ledger_path)
        collected["ledger_changed_during_collection"] = collected["ledger_sha256_after_collection"] != collected["ledger_sha256"]
        if campaign == "e120":
            terminal = [r for r in collected["rows"] if r["endpoint_status"] == "admitted"]
            collected["frequency_telemetry"] = {}
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for row, telemetry in zip(terminal, pool.map(lambda r: audit_frequency_telemetry(r["run_dir"]), terminal)):
                    collected["frequency_telemetry"][row["run_dir"]] = telemetry
            frozen = json.loads(frozen_path.read_text())
            if sum(row["model_key"] == "qwen05b" for row in terminal) != 25:
                raise RuntimeError("not all 25 frozen Qwen E120 treatment endpoints remain admissible")
            for row in terminal:
                if row["model_key"] == "qwen05b":
                    point = frozen["cells"]["qwen05b"][row["domain"]]["fresh_frequency"][str(row["seed"])]
                    if any(row["endpoint"][m] != point[m] for m in ("pass8", "distinct8")):
                        raise RuntimeError("historical Qwen E120 endpoint changed")
        audit["campaigns"][campaign] = collected
        audit["finished_at_utc"] = now()
        write_json(audit_path, audit)
        print(campaign, dict(Counter(r["endpoint_status"] for r in collected["rows"])), flush=True)
    finalize(audit, audit_path, audit_dir, args.output_dir, args.date, stamp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
