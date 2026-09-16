#!/usr/bin/env python3
"""Freeze primary-method training histories against the dated main-paper census.

Collection is explicit; normal figure builds read the frozen JSON only. Each
objective uses its own terminal paired cohort, fixed across every checkpoint.
Independent and unfinished arms are retained as separate descriptive histories.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ops"))
from exp_scaling.build_paper_core_terminal_endpoints import approved_run_exclusion
from exp_scaling.plot_paper_aligned_domain_strips import registered_evaluation_sources
from exp_scaling.snapshot_evaluation_coverage import read_cell

OUTPUT = ROOT / "paper/results/training_curve_snapshot_20260911.json"
AUDIT = ROOT / "paper/audits/training_curves_20260911"
ENDPOINT_AUDIT = ROOT / "paper/audits/results_refresh_20260911/latest_endpoints.json"
CORE = ROOT / "paper/results/core_terminal_endpoints.json"
FIGURE = ROOT / "paper/figures/e118_all_scale_factorial_progress.json"
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
SCALES = ("qwen05b", "falcon1b", "qwen3b")
METHODS = ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
PAIRS = {"drgrpo": ("drgrpo", "replay_drgrpo"), "maxrl": ("maxrl", "replay_maxrl")}
METRICS = {"pass8": "any_correct_at_k", "distinct8": "distinct_correct_modes_at_k"}
STEPS = list(range(0, 3073, 192))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def key(cell):
    return tuple(cell[field] for field in ("level", "scale", "domain", "method", "seed"))


def validate_checkpoint(checkpoint):
    """Do not let an otherwise complete draw set hide an evaluator mismatch."""
    if checkpoint.get("draw_count") != 4:
        return "expected four evaluation draws"
    draws = checkpoint.get("draws", [])
    if [row.get("draw_index") for row in draws] != [0, 1, 2, 3]:
        return "draw indices must be exactly 0, 1, 2, 3"
    for row in draws:
        meta = row["metadata"]
        if (meta.get("evaluation_kind"), meta.get("sample_count"), meta.get("temperature"),
                meta.get("prompt_count")) != ("fixed_seed_sampled_k_neutral", 8, 1, 128):
            return "expected fixed-seed sampled K=8, temperature 1, 128 prompts"
    return None


def collect_cell(task):
    run = task["run"]
    exclusion = approved_run_exclusion(Path(run["run_dir"]))
    fields = {name: task[name] for name in ("level", "scale", "domain", "method", "seed", "ledger")}
    if exclusion:
        return {**fields, "run_dir": run["run_dir"], "registered_job_id": run["job_id"],
                "complete_steps": [], "complete_checkpoints": {}, "source_files": [],
                "invalid_or_conflicted_steps": [], "incomplete_checkpoints": {},
                "issues": [], "approved_exclusion": exclusion,
                "missing_registered_steps": STEPS, "source_binding": None}
    if "authorized_sources" in task:
        authorized = {str(Path(row["path"]).resolve()) for row in task["authorized_sources"]}
        observed = list(Path(run["run_dir"]).glob("debug_job*/eval_mode_coverage_draws.jsonl"))
        excluded = [path for path in observed if str(path.resolve()) not in authorized]
        binding = {"policy": "source paths admitted by the frozen terminal census; registered continuation histories retained together",
                   "registered_job_id": run["job_id"], "terminal_census_sources": task["authorized_sources"],
                   "excluded_sources": [{"path": str(path), "status": "source_not_in_frozen_terminal_census",
                                          "outcome_value_selected": False} for path in excluded]}
        excluded_ids = [path.parent.name.removeprefix("debug_job") for path in excluded]
    else:
        _selected, binding = registered_evaluation_sources(run)
        excluded_ids = binding["superseded_job_ids"]
    if task.get("frozen_coverage") is not None:
        cell = deepcopy(task["frozen_coverage"])
        if {row["path"] for row in cell["source_files"]} - authorized:
            raise RuntimeError("reused frozen coverage contains a source outside the terminal census")
        cell["reused_coverage_source"] = task["frozen_coverage_source"]
    else:
        cell = read_cell(run["run_dir"], excluded_job_ids=excluded_ids)
    cell.update(fields, registered_job_id=run["job_id"], source_binding=binding,
                approved_exclusion=None)
    for step_text, checkpoint in list(cell["complete_checkpoints"].items()):
        step = int(step_text)
        reason = validate_checkpoint(checkpoint)
        if step not in STEPS:
            reason = "checkpoint is outside the registered 192-step grid"
        if reason:
            cell["issues"].append({"kind": "invalid_checkpoint_contract", "step": step, "reason": reason})
            cell["invalid_or_conflicted_steps"].append(step)
            cell["incomplete_checkpoints"][step_text] = {
                "draw_indices": [x["draw_index"] for x in checkpoint["draws"]],
                "conflicted_or_invalid": True, "reason": reason}
            del cell["complete_checkpoints"][step_text]
    cell["complete_steps"] = sorted(map(int, cell["complete_checkpoints"]))
    cell["invalid_or_conflicted_steps"] = sorted(set(cell["invalid_or_conflicted_steps"]))
    cell["missing_registered_steps"] = [step for step in STEPS if step not in cell["complete_steps"]]
    return cell


def source_tasks(audit_directory, endpoint_audit=ENDPOINT_AUDIT):
    sources, frozen_inputs = {}, []
    for path in (endpoint_audit, CORE, FIGURE):
        frozen_path = audit_directory / path.name
        if frozen_path.exists():
            raw = frozen_path.read_bytes()
            if path != FIGURE and raw != path.read_bytes():
                raise RuntimeError(f"published census changed after collection: {path}")
        else:
            raw = path.read_bytes()
            frozen_path.write_bytes(raw)
        source_path = frozen_path if path == FIGURE else path
        sources[str(source_path.relative_to(ROOT))] = digest(raw)
        frozen_inputs.append(raw)
    audit, core, figure = map(json.loads, frozen_inputs)
    tasks, admissions = [], {}
    core_audit = {row["run_dir"]: row for row in core["endpoint_audit"]}
    for model, source in core["sources"].items():
        path = Path(source["path"])
        frozen_path = audit_directory / "ledgers" / path.name
        raw = frozen_path.read_bytes() if frozen_path.exists() else path.read_bytes()
        if digest(raw) != source["sha256"]:
            raise RuntimeError(f"core ledger drifted from published endpoint source: {path}")
        scale = core["models"][model]["scale"]
        ledger = json.loads(raw)
        ledger_name = str(frozen_path.relative_to(ROOT))
        sources[ledger_name] = digest(raw)
        if not frozen_path.exists():
            frozen_path.write_bytes(raw)
        check_grid(ledger)
        for run in ledger["runs"]:
            method = {"control": "drgrpo", "replay": "replay_drgrpo"}[run["arm"]]
            task = dict(level="level1", scale=scale, domain=run["domain"], method=method,
                        seed=int(run["seed"]), ledger=ledger_name, run=run,
                        authorized_sources=core_audit[run["run_dir"]].get("sources", []))
            tasks.append(task)
            block = core["models"][model]["domains"][run["domain"]]["methods"][run["arm"]]
            admissions[key(task)] = block["per_seed"].get(str(run["seed"]))
    for campaign, level in (("e118", "level1"), ("e119", "level2")):
        source = audit["campaigns"][campaign]
        path = ROOT / source["ledger_snapshot"]["path"]
        raw = path.read_bytes()
        if digest(raw) != source["ledger_snapshot"]["sha256"]:
            raise RuntimeError(f"frozen audit ledger hash differs: {path}")
        ledger = json.loads(raw)
        frozen_path = audit_directory / "ledgers" / path.name
        ledger_name = str(frozen_path.relative_to(ROOT))
        sources[ledger_name] = digest(raw)
        if frozen_path.exists() and frozen_path.read_bytes() != raw:
            raise RuntimeError(f"retained ledger copy differs from endpoint audit: {frozen_path}")
        if not frozen_path.exists():
            frozen_path.write_bytes(raw)
        check_grid(ledger)
        rows = {(r.get("scale", "qwen05b"), r["domain"], r["arm"], int(r["seed"])): r
                for r in source["rows"]}
        for run in ledger["runs"]:
            scale = run.get("scale", "qwen05b")
            task = dict(level=level, scale=scale, domain=run["domain"], method=run["arm"],
                        seed=int(run["seed"]), ledger=ledger_name, run=run)
            row = rows[scale, run["domain"], run["arm"], int(run["seed"])]
            if row["run_dir"] != run["run_dir"] or int(row["job_id"]) != int(run["job_id"]):
                raise RuntimeError("terminal census and frozen ledger disagree on registered source")
            task["authorized_sources"] = row["integrity_audit"].get("sources", [])
            tasks.append(task)
            admissions[key(task)] = row["endpoint"] if row["endpoint_status"] == "admitted" else None
    if len({key(task) for task in tasks}) != 400 or len(tasks) != 400:
        raise RuntimeError("expected exactly 400 registered primary-method cells")
    prior_path = ROOT / "paper/results/modebench_level_comparison_snapshot.json"
    prior = json.loads(prior_path.read_text())
    for level, reference in prior["input_snapshots"].items():
        path = ROOT / reference["path"]
        raw = path.read_bytes()
        if digest(raw) != reference["sha256"]:
            raise RuntimeError("existing frozen level coverage hash differs")
        records = {row["run_dir"]: row for row in json.loads(raw)["cells"]}
        sources[str(path.relative_to(ROOT))] = digest(raw)
        for task in tasks:
            if task["level"] == level and task["run"]["run_dir"] in records:
                task["frozen_coverage"] = records[task["run"]["run_dir"]]
                task["frozen_coverage_source"] = {"path": str(path.relative_to(ROOT)), "sha256": digest(raw)}
    return tasks, admissions, sources, figure


def check_grid(ledger):
    if (ledger["checkpoint_interval_steps"], ledger["target_steps"], ledger["train_rows"],
            ledger["passes"]) != (192, 3072, 384, 8):
        raise RuntimeError("registered checkpoint geometry differs")


def series(index, prefix, method, seeds, policy):
    points = []
    for step in STEPS:
        per_seed = {}
        for seed in seeds:
            cell = index[(*prefix, method, seed)]
            checkpoint = cell["complete_checkpoints"].get(str(step))
            if checkpoint:
                per_seed[str(seed)] = {metric: checkpoint["mean_metrics"][field]
                                       for metric, field in METRICS.items()}
        complete = bool(seeds) and len(per_seed) == len(seeds)
        points.append({"step": step, "training_pass": step / 384, "complete": complete,
                       "observed_n": len(per_seed), "missing_seeds": [s for s in seeds if str(s) not in per_seed],
                       "per_seed": per_seed,
                       "mean": {metric: statistics.fmean(row[metric] for row in per_seed.values())
                                if complete else None for metric in METRICS}})
    return {"cohort_seeds": seeds, "cohort_n": len(seeds), "cohort_policy": policy, "points": points}


def compose(cells, admissions, sources, figure, collection):
    index = {key(cell): cell for cell in cells}
    if len(index) != len(cells):
        raise RuntimeError("duplicate scientific cell")
    for cell in cells:
        # Some evaluators wrote additional 96-step observations. The ledgers
        # register a 192-step grid; omission of an extra point is not a conflict.
        omitted = sorted({issue["step"] for issue in cell["issues"]
                          if issue.get("reason") == "checkpoint is outside the registered 192-step grid"})
        cell["omitted_unregistered_steps"] = omitted
        cell["invalid_or_conflicted_steps"] = [step for step in cell["invalid_or_conflicted_steps"] if step not in omitted]
        for issue in cell["issues"]:
            if issue.get("step") in omitted and issue.get("kind") == "invalid_checkpoint_contract":
                issue["kind"] = "unregistered_checkpoint_omitted"
        endpoint = admissions[key(cell)]
        cell["terminal_admitted"] = endpoint is not None
        cell["terminal_reference"] = {metric: endpoint[metric] for metric in METRICS} if endpoint else None
        cell["terminal_matches_census"] = None
        if endpoint is not None:
            checkpoint = cell["complete_checkpoints"].get("3072")
            if checkpoint is None:
                raise RuntimeError(f"published terminal point absent from valid frozen trajectory: {key(cell)}")
            if any(checkpoint["mean_metrics"][field] != endpoint[metric] for metric, field in METRICS.items()):
                raise RuntimeError(f"frozen trajectory terminal differs from main-paper census: {key(cell)}")
            cell["terminal_matches_census"] = True
    panels = []
    for level, scales in (("level1", SCALES), ("level2", ("qwen05b",))):
        for scale in scales:
            for domain in DOMAINS:
                prefix = (level, scale, domain)
                paired = {}
                for objective, methods in PAIRS.items():
                    arm_seeds = [{k[-1] for k, endpoint in admissions.items()
                                  if k[:-2] == prefix and k[-2] == method and endpoint is not None}
                                 for method in methods]
                    paired[objective] = sorted(set.intersection(*arm_seeds))
                    if level == "level1":
                        for method in methods:
                            expected = figure["cells"][scale][domain]["method_seeds"][method]
                            if paired[objective] != expected:
                                raise RuntimeError(f"training cohort differs from Figure 5: {prefix}, {method}")
                methods, supplementary = {}, {}
                for method in METHODS:
                    objective = "maxrl" if "maxrl" in method else "drgrpo"
                    seeds = paired[objective]
                    methods[method] = series(index, prefix, method, seeds,
                                            "paired terminal seeds within objective; fixed across checkpoints")
                    extra = sorted(k[-1] for k, cell in index.items()
                                   if k[:-2] == prefix and k[-2] == method and k[-1] not in seeds
                                   and cell["complete_steps"])
                    supplementary[method] = series(index, prefix, method, extra,
                                                  "partial/unpaired histories; no paired effect")
                panels.append({"level": level, "scale": scale, "domain": domain,
                               "paired_cohorts": paired, "methods": methods,
                               "supplementary_methods": supplementary})
    return {"schema": "training-curve-frozen-snapshot-v1", "collection": collection,
            "registered_steps": STEPS, "target_step": 3072, "train_rows": 384, "metrics": METRICS,
            "levels": ["level1", "level2"], "scales": list(SCALES), "domains": list(DOMAINS),
            "methods": list(METHODS), "source_sha256": sources,
            "selection_policy": "Main-paper terminal paired seeds within each objective, fixed across every training step; "
                                "independently observed arms and partial histories retained separately. No endpoint substitution.",
            "checkpoint_policy": "Exactly four fixed-seed sampled K=8 draws at temperature 1 on 128 prompts; "
                                 "registered 192-step grid only; identical repeats deduplicated; conflicts invalidate the entire checkpoint; "
                                 "registered continuation histories retained from census-authorized source paths; later sources excluded. "
                                 "No imputation or connection across gaps.",
            "aggregation_policy": "Four draws averaged within seed. Mean and seed range only when every fixed cohort seed is observed. "
                                  "Partial/unpaired histories have no paired effect or inferential interval.",
            "panels": panels, "cells": sorted(cells, key=key)}



def validate_snapshot(snapshot, main_figure=None):
    """Validate frozen evidence and reconstructions without opening live run logs."""
    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    require(snapshot.get("schema") == "training-curve-frozen-snapshot-v1", "training snapshot schema differs")
    require(snapshot["registered_steps"] == STEPS and snapshot["metrics"] == METRICS,
            "registered checkpoint or metric contract differs")
    cells = snapshot["cells"]
    index = {key(cell): cell for cell in cells}
    require(len(index) == len(cells) == 400, "expected 400 unique training cells")
    expected_panels = {(level, scale, domain) for level, scales in (("level1", SCALES), ("level2", ("qwen05b",)))
                       for scale in scales for domain in DOMAINS}
    require({k[:3] for k in index} == expected_panels, "training scope misses a model/domain/level")
    for prefix in expected_panels:
        expected_seeds = {"qwen05b": set(range(43,48)), "falcon1b": set(range(55,60)),
                          "qwen3b": set(range(70,75))}[prefix[1]]
        for method in METHODS:
            require({k[-1] for k in index if k[:-2] == prefix and k[-2] == method} == expected_seeds,
                    "registered primary-method seeds differ")
    for cell in cells:
        checkpoints = cell["complete_checkpoints"]
        require(cell["complete_steps"] == sorted(map(int, checkpoints)), "complete-step index differs")
        require(not set(cell["complete_steps"]) - set(STEPS), "unregistered checkpoint admitted")
        require(cell["missing_registered_steps"] == [step for step in STEPS if step not in cell["complete_steps"]],
                "missing checkpoints not reported exactly")
        require(not set(cell["complete_steps"]) & set(cell["invalid_or_conflicted_steps"]),
                "invalid/conflicted checkpoint admitted")
        for step, checkpoint in checkpoints.items():
            require(validate_checkpoint(checkpoint) is None, "invalid fixed evaluation contract")
            require(checkpoint["step"] == int(step), "checkpoint step differs from key")
            for field in METRICS.values():
                values = [draw["metrics"][field] for draw in checkpoint["draws"]]
                require(all(isinstance(v, (int,float)) and not isinstance(v, bool) and math.isfinite(v) for v in values),
                        "nonfinite or invalid primary metric")
                require(checkpoint["mean_metrics"][field] == statistics.fmean(values), "draw mean differs")
            for draw in checkpoint["draws"]:
                require(bool(draw["origins"]), "missing exact draw-line provenance")
                for origin in draw["origins"]:
                    source = next((row for row in cell["source_files"] if row["path"] == origin["path"]), None)
                    require(source is not None and 1 <= origin["line"] <= source["line_count"],
                            "draw provenance lies outside the frozen source prefix")
                    require(len(source["sha256_read_prefix"]) == 64 and source["read_bytes"] == source["size_before"],
                            "source prefix is not fully identified")
        if cell["terminal_admitted"]:
            require(cell["terminal_matches_census"] is True and "3072" in checkpoints,
                    "admitted endpoint is absent from the frozen trajectory")
            require(all(checkpoints["3072"]["mean_metrics"][field] == cell["terminal_reference"][metric]
                        for metric, field in METRICS.items()), "terminal differs from frozen census")
    panels = snapshot["panels"]
    require(len(panels) == 20 and {(p["level"], p["scale"], p["domain"]) for p in panels} == expected_panels,
            "training panel scope differs")
    for panel in panels:
        prefix = (panel["level"], panel["scale"], panel["domain"])
        require(set(panel["methods"]) == set(METHODS), "a primary method is missing")
        for objective, pair in PAIRS.items():
            seeds = sorted(set.intersection(*({k[-1] for k, cell in index.items()
                                               if k[:-2] == prefix and k[-2] == method and cell["terminal_admitted"]}
                                              for method in pair)))
            require(panel["paired_cohorts"][objective] == seeds, "paired cohort differs from terminal admissions")
            for method in pair:
                record = panel["methods"][method]
                require(record["cohort_seeds"] == seeds, "primary curve uses a different paired cohort")
                require(record == series(index, prefix, method, seeds, record["cohort_policy"]),
                        "primary curve means/gaps or seed values do not reconstruct")
                if main_figure is not None and prefix[0] == "level1":
                    main_cell = main_figure["cells"][prefix[1]][prefix[2]]
                    require(seeds == main_cell["method_seeds"][method],
                            "training curve cohort differs from main Figure 5")
                    for metric in METRICS:
                        require([record["points"][-1]["per_seed"][str(seed)][metric] for seed in seeds]
                                == main_cell["methods"][method][metric],
                                "training terminal observations differ from main Figure 5")
                extra = sorted(k[-1] for k, cell in index.items()
                               if k[:-2] == prefix and k[-2] == method and k[-1] not in seeds and cell["complete_steps"])
                record = panel["supplementary_methods"][method]
                require(record["cohort_seeds"] == extra, "an independent training history was lost")
                require(record == series(index, prefix, method, extra, record["cohort_policy"]),
                        "independent curve values/gaps do not reconstruct")
    return {"registered_cells": len(cells), "panels": len(panels),
            "terminal_admitted_cells": sum(cell["terminal_admitted"] for cell in cells)}


def summarize(snapshot):
    cells = snapshot["cells"]
    return {"schema": snapshot["schema"], "registered_cells": len(cells),
            "terminal_admitted_cells": sum(cell["terminal_admitted"] for cell in cells),
            "terminal_matches_census": all(cell["terminal_matches_census"] is True
                                            for cell in cells if cell["terminal_admitted"]),
            "complete_checkpoints": sum(len(cell["complete_steps"]) for cell in cells),
            "invalid_or_conflicted_checkpoints": sum(len(cell["invalid_or_conflicted_steps"]) for cell in cells),
            "excluded_runs": [key(cell) for cell in cells if cell["approved_exclusion"]],
            "panel_cohorts": [{field: panel[field] for field in ("level", "scale", "domain", "paired_cohorts")}
                              for panel in snapshot["panels"]],
            "missing_checkpoint_cells": [{**{field: cell[field] for field in ("level", "scale", "domain", "method", "seed")},
                                          "missing_registered_steps": cell["missing_registered_steps"]}
                                         for cell in cells if cell["missing_registered_steps"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit-directory", type=Path, default=AUDIT)
    parser.add_argument("--endpoint-audit", type=Path, default=ENDPOINT_AUDIT,
                        help="Frozen terminal census to use for admissions and source authorization")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-coverage", action="store_true", help="Reuse the exact frozen cell coverage")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    audit = args.audit_directory.resolve()
    (audit / "ledgers").mkdir(parents=True, exist_ok=True)
    tasks, admissions, sources, figure = source_tasks(audit, args.endpoint_audit.resolve())
    reader_paths = [Path(__file__), ROOT / "ops/exp_scaling/snapshot_evaluation_coverage.py",
                    ROOT / "ops/exp_scaling/plot_paper_aligned_domain_strips.py",
                    ROOT / "ops/exp_scaling/build_paper_core_terminal_endpoints.py"]
    for path in reader_paths:
        sources[str(path.relative_to(ROOT))] = digest(path.read_bytes())
    coverage_path = audit / "coverage_snapshot.json"
    if args.reuse_coverage:
        coverage = json.loads(coverage_path.read_text())
        cells, collection = coverage["cells"], coverage["collection"]
    else:
        cells, collection = [], {"started_at_utc": now()}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(collect_cell, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                cells.append(future.result())
                if count % 20 == 0:
                    print(f"Frozen {count}/{len(tasks)} primary training cells", flush=True)
        collection["finished_at_utc"] = now()
        write_json(coverage_path, {"schema": "primary-training-coverage-v1", "collection": collection,
                                   "source_sha256": sources, "cells": sorted(cells, key=key)})
    sources[str(coverage_path.relative_to(ROOT))] = digest(coverage_path.read_bytes())
    snapshot = compose(cells, admissions, sources, figure, collection)
    validate_snapshot(snapshot, figure)
    write_json(args.output, snapshot)
    summary = summarize(snapshot)
    summary["snapshot_sha256"] = digest(args.output.read_bytes())
    write_json(audit / "selection_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("missing_checkpoint_cells", "panel_cohorts")}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
