#!/usr/bin/env python3
"""Extend the preserved collision sample cache to amendment 2's frozen census.

This program reconstructs existing P,D,M measurements only. It never computes
collision, conditional eligibility, effect sizes, or inferential summaries.
Unchanged checkpoints are reused only after exact frozen admission equality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "paper/audits/conditional_concentration_20260911"
BINDING = AUDIT / "amendment_2_current_census_binding.json"
OUTPUT = AUDIT / "verified_samples_completed_cohort.jsonl.gz"
RECEIPT = AUDIT / "collection_receipt_completed_cohort.json"
CERTIFICATE = AUDIT / "stream_source_audit_completed_cohort.json"
sys.path.insert(0, str(ROOT / "ops"))
from exp_scaling import load_paper_collision_samples as loader

FIELDS = ("level", "scale", "domain", "method", "seed")
STEPS = ("0", "3072")


def dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_path(record):
    path = ROOT / record["path"]
    if sha(path) != record["sha256"]:
        raise ValueError(f"frozen input changed: {path}")
    return path


def cell_key(cell):
    return tuple(cell[field] for field in FIELDS)


def checkpoint_signature(checkpoint):
    return hashlib.sha256(dump(checkpoint).encode()).hexdigest()


def change_plan(old, new):
    old_index = {cell_key(c): c for c in old["cells"]}
    new_index = {cell_key(c): c for c in new["cells"]}
    if set(old_index) != set(new_index) or len(new_index) != len(new["cells"]):
        raise ValueError("scientific registration changed outside the frozen census extension")
    plan = {"added": [], "removed": [], "changed_complete": [], "reused": [], "terminal_admission_changes": []}
    for key in sorted(new_index):
        a, b = old_index[key], new_index[key]
        if a["terminal_admitted"] != b["terminal_admitted"]:
            plan["terminal_admission_changes"].append({"cell": list(key), "old": a["terminal_admitted"], "new": b["terminal_admitted"]})
        for step in STEPS:
            x, y = a["complete_checkpoints"].get(step), b["complete_checkpoints"].get(step)
            record = {"cell": list(key), "step": int(step)}
            if x is None and y is not None:
                plan["added"].append(record)
            elif x is not None and y is None:
                plan["removed"].append(record)
            elif x is not None and y is not None:
                if x == y:
                    plan["reused"].append({**record, "frozen_checkpoint_sha256": checkpoint_signature(x)})
                else:
                    plan["changed_complete"].append(record)
    return plan


def compact_checkpoint(checkpoint):
    """Reconstruct only original per-draw pass8, distinct8, and mean8."""
    prompts, stream_checks = {}, []
    for draw in checkpoint["draws"]:
        parent = draw["metadata"]["seed"]
        if type(parent) is not int:
            raise ValueError("neutral draw requires an integer parent seed")
        p_values, d_values, m_values = [], [], []
        for prompt in draw["prompts"]:
            keys = prompt["verified_keys"]
            if len(keys) != 8:
                raise ValueError("intact K8 groups required")
            request = prompt.get("request_seeds_by_option")
            options = prompt.get("option_ids")
            if request not in (None, [], [parent]):
                raise ValueError("unsupported neutral request-seed branch")
            if options is not None and (len(options) != 8 or any(option is not None for option in options)):
                raise ValueError("non-neutral answer options outside amendment 2")
            correct = [key for key in keys if key is not None]
            p_values.append(float(bool(correct)))
            d_values.append(len(set(correct)))
            m_values.append(len(correct) / 8)
            record = prompts.setdefault(prompt["prompt_id"], {"prompt_index": prompt["prompt_index"],
                "draws": [], "request_seeds_by_draw": [], "option_ids_by_draw": []})
            record["draws"].append(keys)
            record["request_seeds_by_draw"].append(request)
            record["option_ids_by_draw"].append(options)
        for values, field in ((p_values, "any_correct_at_k"), (d_values, "distinct_correct_modes_at_k"), (m_values, "mean_at_k")):
            if abs(statistics.mean(values) - draw["observed_metrics"][field]) > 1e-12:
                raise ValueError(f"original K8 metric does not reconstruct: {field}")
        stream_checks.append({"draw_index": draw["draw_index"], "parent_seed": parent,
                              "nominal_child_seeds": list(range(parent, parent + 8)),
                              "neutral_option_and_request_metadata_verified": True})
    if len(prompts) != 128 or any(len(p["draws"]) != 4 for p in prompts.values()):
        raise ValueError("compact checkpoint lacks its complete prompt/draw grid")
    compact = {"prompts": prompts, "sampling_certificate": checkpoint["sampling_certificate"],
               "origins": [{"draw_index": draw["draw_index"], "origins": draw["origins"],
                            "raw_payload_sha256": draw["raw_payload_sha256"]} for draw in checkpoint["draws"]],
               "primary_metrics_reconstructed": True}
    return compact, stream_checks


def new_manifest(snapshot, source_record):
    cohorts = [{k: deepcopy(p[k]) for k in ("level", "scale", "domain", "paired_cohorts")} for p in snapshot["panels"]]
    totals = Counter()
    for panel in cohorts:
        for objective, seeds in panel["paired_cohorts"].items():
            if len(set(seeds)) != len(seeds):
                raise ValueError("duplicate cohort seeds")
            totals[panel["level"], objective] += len(seeds)
    if (totals["level1", "drgrpo"], totals["level1", "maxrl"]) != (74, 75):
        raise ValueError("completed census differs from amendment 2's 74/75 pairs")
    cohort_keys = {(p["level"], p["scale"], p["domain"], method, seed)
                   for p in cohorts for objective, seeds in p["paired_cohorts"].items()
                   for method in (objective, "replay_" + objective) for seed in seeds}
    return {"schema": "paper-collision-source-manifest-v1", "source_snapshot": source_record,
            "cohorts": cohorts, "cohort_counts": [{"level": level, "objective": objective, "n": n}
                                                   for (level, objective), n in sorted(totals.items())]}, cohort_keys


def extend(workers=4):
    if any(path.exists() for path in (OUTPUT, RECEIPT, CERTIFICATE)):
        raise ValueError("refusing to overwrite a completed extension artifact")
    binding = json.loads(BINDING.read_text())
    amendment = ROOT / binding["amendment_path"]
    if sha(amendment) != binding["amendment_sha256"] or binding["status"] != "frozen_after_initial_results_before_new_source_effects":
        raise ValueError("amendment 2 binding differs")
    old_path, new_path = bound_path(binding["original_snapshot"]), bound_path(binding["completed_snapshot"])
    old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    old_cache = bound_path(binding["original_cache"])
    bound_path(binding["preserved_initial_results"])
    old_receipt = json.loads(bound_path(binding["original_collection_receipt"]).read_text())
    if old_receipt["sha256"] != sha(old_cache):
        raise ValueError("original cache receipt differs")
    with gzip.open(old_cache, "rt") as handle:
        old_header = json.loads(next(handle))
        cached_cells = [json.loads(line) for line in handle]
    if old_header["primary"]["source_snapshot"]["sha256"] != sha(old_path):
        raise ValueError("initial sample cache is bound to another snapshot")
    cached = {cell_key(cell): cell for cell in cached_cells}
    if len(cached) != len(cached_cells) or len(cached) != 475:
        raise ValueError("expected 475 unique cached cells")
    plan = change_plan(old, new)
    if len(plan["added"]) != 13 or len(plan["removed"]) != 1 or plan["changed_complete"]:
        raise ValueError("current source change set differs from amendment 2")
    new_index = {cell_key(cell): cell for cell in new["cells"]}
    primary, cohort_keys = new_manifest(new, {"path": str(new_path.relative_to(ROOT)), "sha256": sha(new_path)})
    reload_steps = {}
    for record in plan["added"] + plan["changed_complete"]:
        reload_steps.setdefault(tuple(record["cell"]), []).append(record["step"])
    loaded = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [(key, pool.submit(loader._load_cell, new_index[key], key in cohort_keys, tuple(steps)))
                   for key, steps in sorted(reload_steps.items())]
        for number, (key, future) in enumerate(futures, 1):
            loaded[key] = future.result()
            print(f"Validated new source cell {number}/{len(futures)}: {key}", flush=True)
    removed = {(tuple(r["cell"]), str(r["step"])) for r in plan["removed"]}
    added = {(tuple(r["cell"]), str(r["step"])) for r in plan["added"] + plan["changed_complete"]}
    reused = {(tuple(r["cell"]), str(r["step"])) for r in plan["reused"]}
    result_cells, new_source_checks, stream_rows, failures = [], [], [], []
    for old_cell in cached_cells:
        key = cell_key(old_cell)
        result = deepcopy(old_cell)
        if old_cell["method"] == "grpo":
            result_cells.append(result)
            continue
        source = new_index[key]
        for field in (*FIELDS, "run_dir", "ledger", "registered_job_id", "terminal_admitted", "terminal_matches_census", "terminal_reference", "approved_exclusion"):
            result[field] = deepcopy(source.get(field))
        result["in_terminal_paired_cohort"] = key in cohort_keys
        actions = {}
        for step in STEPS:
            if (key, step) in added:
                raw = loaded[key]
                result["sample_issues"] = [issue for issue in result["sample_issues"] if issue.get("step") != int(step)]
                result["sample_issues"].extend(deepcopy(raw["sample_issues"]))
                cp = raw["checkpoints"].get(step)
                result["checkpoints"][step] = None
                if cp is not None:
                    try:
                        result["checkpoints"][step], checks = compact_checkpoint(cp)
                        stream_rows.append({"cell": list(key), "step": int(step), "draws": checks,
                                            "source_bound_fingerprint": cp["sampling_certificate"]["source_bound_fingerprint"]})
                    except (ValueError, KeyError, TypeError) as exc:
                        result["sample_issues"].append({"kind": "completed_census_compact_integrity_failure", "step": int(step), "reason": str(exc)})
                if result["checkpoints"][step] is None:
                    failures.append({"cell": list(key), "step": int(step), "issues": raw["sample_issues"] + result["sample_issues"]})
                result["source_checks"].extend(deepcopy(raw["source_checks"]))
                new_source_checks.extend(deepcopy(raw["source_checks"]))
                actions[step] = "new_frozen_source_ingested"
            elif (key, step) in removed:
                result["checkpoints"][step] = None
                result["sample_issues"].append({"kind": "withdrawn_by_completed_census_conflict", "step": int(step),
                    "reason": "Newly authorized source conflicts with the prior initial checkpoint; no attempt selected.",
                    "frozen_issues": [issue for issue in source["issues"] if issue.get("step") == int(step)]})
                actions[step] = "withdrawn_new_source_conflict"
            elif (key, step) in reused:
                actions[step] = "reused_exact_frozen_checkpoint_identity"
            else:
                if result["checkpoints"].get(step) is not None:
                    raise ValueError("cached checkpoint cannot be reused outside exact frozen equality")
                actions[step] = "remains_unavailable"
        result["before_after_available"] = all(result["checkpoints"].get(step) is not None for step in STEPS)
        result["frozen_reference_metrics"] = {step: {"pass8": source["complete_checkpoints"][step]["mean_metrics"]["any_correct_at_k"],
            "distinct8": source["complete_checkpoints"][step]["mean_metrics"]["distinct_correct_modes_at_k"]}
            for step in STEPS if step in source["complete_checkpoints"]}
        result["cohort_extension_source_actions"] = actions
        result_cells.append(result)
    code_paths = [Path(__file__).resolve(), Path(loader.__file__).resolve()]
    code_hashes = {str(path.relative_to(ROOT)): sha(path) for path in code_paths}
    for path in code_paths:
        dest = AUDIT / "extension_source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.read_bytes() != path.read_bytes():
            raise ValueError("extension source snapshot already differs")
        dest.write_bytes(path.read_bytes())
    header = deepcopy(old_header)
    header.update(created_at_utc=datetime.now(timezone.utc).isoformat(), primary=primary,
        cohort_extension={"binding_path": str(BINDING.relative_to(ROOT)), "binding_sha256": sha(BINDING), "binding": binding},
        extension_code_sha256=code_hashes, parent_cache={"path": str(old_cache.relative_to(ROOT)), "sha256": sha(old_cache)})
    temp = OUTPUT.with_suffix(OUTPUT.suffix + ".partial")
    if temp.exists():
        raise ValueError("partial extension already exists")
    with gzip.open(temp, "wt", encoding="utf-8") as handle:
        handle.write(dump(header) + "\n")
        for cell in result_cells:
            handle.write(dump(cell) + "\n")
    temp.replace(OUTPUT)
    counts, availability = Counter(), Counter()
    for cell in result_cells:
        counts[cell["method"]] += 1
        for step, cp in cell["checkpoints"].items():
            if cp is not None:
                availability[f"{cell['level']}/{cell['method']}/{step}"] += 1
    cert = {"schema": "collision-completed-census-stream-source-extension-v1", "no_collision_effects_computed": True,
        "amendment_binding": {"path": str(BINDING.relative_to(ROOT)), "sha256": sha(BINDING)},
        "completed_snapshot": primary["source_snapshot"], "newly_loaded_checkpoint_metadata": stream_rows,
        "new_source_prefix_checks": new_source_checks,
        "retained_source_certificates": {str(p.relative_to(ROOT)): sha(p) for p in (AUDIT/'stream_source_audit.json', AUDIT/'stream_source_audit_grpo_extension.json')},
        "scope": "The same neutral n8 parent-plus-output-index mapping is used. Empty request lists and exact [parent] lists are accepted; option IDs must be null. Parent seeds are shared across prompts. Archived dependency identity and iid sampling are assumptions, not established by this certificate. Unchanged checkpoints inherit the source mapping audit through exact frozen metadata/origin equality.",
        "new_sample_failures": failures}
    CERTIFICATE.write_text(json.dumps(cert, indent=2, sort_keys=True) + "\n")
    receipt = {"status": "collected", "path": str(OUTPUT.relative_to(ROOT)), "sha256": sha(OUTPUT),
        "cells_by_method": dict(counts), "available_checkpoints": dict(availability),
        "sample_issue_count": sum(len(cell["sample_issues"]) for cell in result_cells),
        "cohort_extension": binding, "binding_path": str(BINDING.relative_to(ROOT)), "binding_sha256": sha(BINDING),
        "primary_cohort_counts": primary["cohort_counts"], "source_change_plan": plan,
        "new_source_prefix_files": len(new_source_checks), "new_source_prefix_bytes": sum(row["read_bytes"] for row in new_source_checks),
        "new_sample_failures": failures, "source_stream_certificate": {"path": str(CERTIFICATE.relative_to(ROOT)), "sha256": sha(CERTIFICATE)},
        "extension_code_sha256": code_hashes, "effects_computed": False,
        "preserved_original_artifact_checks": {name: {"path": binding[name]["path"], "sha256": sha(bound_path(binding[name]))}
            for name in ("original_snapshot", "original_cache", "original_collection_receipt", "preserved_initial_results")}}
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "path", "sha256", "cells_by_method", "sample_issue_count", "new_source_prefix_bytes", "new_sample_failures")}, indent=2), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    extend(args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
