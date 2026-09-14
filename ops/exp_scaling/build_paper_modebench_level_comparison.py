#!/usr/bin/env python3
"""Select an outcome-independent, matched interim comparison across levels."""
from __future__ import annotations

import math
import statistics

LEVELS = ("level1", "level2")
METHODS = ("drgrpo", "replay_drgrpo", "maxrl", "replay_maxrl")
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
SEEDS = (43, 44, 45, 46, 47)
METRICS = {"pass8": "any_correct_at_k", "distinct8": "distinct_correct_modes_at_k"}
TARGET_STEP = 3072
TRAIN_ROWS = 384


def admitted_checkpoints(evaluations: list[dict]) -> dict[tuple, dict]:
    """Deduplicate identical sampled draws; refuse any conflicting retry."""
    draws: dict[tuple, dict[int, dict]] = {}
    for row in evaluations:
        if row.get("evaluation_kind") != "fixed_seed_sampled_k_neutral":
            continue
        if row.get("sample_count") != 8:
            continue
        level, domain, method = (row[name] for name in ("level", "domain", "method"))
        seed, step, draw = (row[name] for name in ("seed", "step", "draw_index"))
        if level not in LEVELS or domain not in DOMAINS or method not in METHODS or seed not in SEEDS:
            raise RuntimeError("unregistered level/domain/method/seed in frozen comparison")
        if not isinstance(step, int) or not 0 <= step <= TARGET_STEP:
            raise RuntimeError("invalid frozen comparison checkpoint")
        if not isinstance(draw, int) or isinstance(draw, bool):
            raise RuntimeError("invalid sampled draw index")
        metrics = row.get("metrics", {})
        for field in METRICS.values():
            value = metrics.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"non-finite sampled comparison metric: {field}")
        key = (level, domain, method, seed, step)
        current = draws.setdefault(key, {})
        if draw in current:
            if current[draw]["metrics"] != metrics:
                raise RuntimeError(f"conflicting duplicate comparison draw: {key}, draw={draw}")
            continue
        current[draw] = row
    return {
        key: {
            "draws": [rows[draw] for draw in range(4)],
            "means": {
                metric: statistics.fmean(float(rows[draw]["metrics"][field]) for draw in range(4))
                for metric, field in METRICS.items()
            },
        }
        for key, rows in draws.items() if set(rows) == {0, 1, 2, 3}
    }


def build_interim_comparison(evaluations: list[dict], availability: list[dict] | None = None) -> dict:
    """Use every seed with a common observed checkpoint for all eight series.

    A checkpoint is selected solely from availability, before accessing the
    values used in its mean. Domain means receive equal weight even when the
    numbers of eligible seeds differ. A missing domain fails closed.
    """
    admitted = admitted_checkpoints(evaluations)
    expected_steps = None
    if availability is not None:
        available = {}
        for cell in availability:
            key = (cell["level"], cell["domain"], cell["method"], cell["seed"])
            if key in available:
                raise RuntimeError("duplicate frozen availability cell")
            steps = cell["complete_steps"]
            if any(type(step) is not int or not 0 <= step <= TARGET_STEP for step in steps):
                raise RuntimeError("invalid frozen availability checkpoint")
            available[key] = set(steps)
        required = {(level, domain, method, seed) for level in LEVELS for domain in DOMAINS
                    for method in METHODS for seed in SEEDS}
        if set(available) != required:
            raise RuntimeError("frozen availability must enumerate all 200 registered cells")
        expected_steps = {}
        for domain in DOMAINS:
            for seed in SEEDS:
                common = set.intersection(*(available[level, domain, method, seed]
                                            for level in LEVELS for method in METHODS))
                if common:
                    expected_steps[domain, seed] = max(common)
    selected = []
    coverage = {}
    for domain in DOMAINS:
        domain_cells = []
        for seed in SEEDS:
            step_sets = [
                {step for lev, dom, arm, cell_seed, step in admitted
                 if (lev, dom, arm, cell_seed) == (level, domain, method, seed) and step >= 0}
                for level in LEVELS for method in METHODS
            ]
            common_steps = set.intersection(*step_sets)
            if not common_steps:
                continue
            step = max(common_steps)
            cell = {
                "domain": domain, "seed": seed, "step": step,
                "training_pass": step / TRAIN_ROWS,
                "series": {
                    level: {method: admitted[level, domain, method, seed, step]
                            for method in METHODS}
                    for level in LEVELS
                },
            }
            domain_cells.append(cell)
            selected.append(cell)
        if not domain_cells:
            raise RuntimeError(f"cannot emit a five-domain comparison: no common observed checkpoint for {domain}")
        steps = {str(cell["seed"]): cell["step"] for cell in domain_cells}
        coverage[domain] = {
            "n": len(domain_cells), "seeds": [cell["seed"] for cell in domain_cells],
            "steps_by_seed": steps,
            "step_range": [min(steps.values()), max(steps.values())],
            "training_pass_range": [min(steps.values()) / TRAIN_ROWS, max(steps.values()) / TRAIN_ROWS],
            "initial_checkpoint_only": max(steps.values()) == 0,
            "initial_checkpoint_seeds": [int(seed) for seed, step in steps.items() if step == 0],
        }
    if expected_steps is not None:
        actual_steps = {(cell["domain"], cell["seed"]): cell["step"] for cell in selected}
        if actual_steps != expected_steps:
            raise RuntimeError("selected draws do not match latest common frozen availability")
    domain_means = {
        domain: {
            level: {
                method: {
                    metric: statistics.fmean(cell["series"][level][method]["means"][metric]
                                            for cell in selected if cell["domain"] == domain)
                    for metric in METRICS
                } for method in METHODS
            } for level in LEVELS
        } for domain in DOMAINS
    }
    means = {
        level: {
            method: {
                metric: statistics.fmean(domain_means[domain][level][method][metric] for domain in DOMAINS)
                for metric in METRICS
            } for method in METHODS
        } for level in LEVELS
    }
    return {
        "status": "descriptive interim; mixed progress across cells, matched progress within every cell",
        "model": "Qwen2.5-0.5B-Instruct",
        "selection_rule": "For each registered domain and seed, select the latest observed checkpoint, including an actually measured initial step0 when necessary, with all four fixed-seed sampled-K=8 draws for all eight level-by-method combinations; include every qualifying seed.",
        "aggregation_rule": "Mean draws within a cell; mean all eligible seeds within each domain; equal mean of all five domain means. The same selected cells and steps are used for all eight series.",
        "uncertainty": "Descriptive means only; no confidence intervals or pooled seed points.",
        "initial_checkpoint_policy": "Admit an observed step0 only when all eight series have complete valid sampled draws; never substitute admission estimates or impute missing observations.",
        "initial_only_domains": [domain for domain in DOMAINS if coverage[domain]["initial_checkpoint_only"]],
        "target_step": TARGET_STEP, "train_rows": TRAIN_ROWS,
        "domains": list(DOMAINS), "levels": list(LEVELS), "methods": list(METHODS),
        "eligible_domain_seed_cells": len(selected),
        "expected_domain_seed_cells": len(DOMAINS) * len(SEEDS),
        "selected_cells": selected, "coverage_by_domain": coverage,
        "domain_means": domain_means, "means": means,
    }


def build_terminal_progress(evaluations: list[dict]) -> dict:
    """Keep exact Level2 pass8 coverage separate from the interim averages."""
    admitted = admitted_checkpoints(evaluations)
    by_domain = {domain: {method: {} for method in METHODS} for domain in DOMAINS}
    for (level, domain, method, seed, step), checkpoint in admitted.items():
        if level == 'level2' and step == TARGET_STEP:
            by_domain[domain][method][seed] = checkpoint['means']
    progress = {}
    for domain, arms in by_domain.items():
        common_four = sorted(set.intersection(*(set(arms[method]) for method in METHODS)))
        contrasts = {}
        for control, replay in (('drgrpo', 'replay_drgrpo'), ('maxrl', 'replay_maxrl')):
            common = sorted(set(arms[control]) & set(arms[replay]))
            contrasts[f'{replay}_minus_{control}'] = {
                'matched_seeds': common, 'n': len(common),
                'mean_effects': {
                    metric: statistics.fmean(arms[replay][seed][metric] - arms[control][seed][metric]
                                             for seed in common)
                    for metric in METRICS
                } if common else {},
            }
        progress[domain] = {
            'terminal_seeds_by_arm': {method: sorted(values) for method, values in arms.items()},
            'four_arm_matched_seeds': common_four, 'four_arm_n': len(common_four),
            'complete_block': len(common_four) == len(SEEDS), 'contrasts': contrasts,
        }
    return progress


def build_terminal_comparison(evaluations: list[dict], admission: list[dict]) -> dict:
    """Compare only complete terminal domain blocks; retain partial arm evidence.

    Main-figure domain membership depends on all five registered seeds being
    admitted in all eight series. Partial domains remain separate, with exact
    per-arm sample counts; they never enter the across-domain mean.
    """
    admitted = admitted_checkpoints(evaluations)
    if any(key[-1] != TARGET_STEP for key in admitted):
        raise RuntimeError("terminal comparison cannot include earlier checkpoints")
    required = {(level, domain, method, seed) for level in LEVELS
                for domain in DOMAINS for method in METHODS for seed in SEEDS}
    declared = {}
    for cell in admission:
        key = tuple(cell[field] for field in ("level", "domain", "method", "seed"))
        if key in declared:
            raise RuntimeError("duplicate terminal admission cell")
        declared[key] = cell["admitted"]
    if set(declared) != required:
        raise RuntimeError("terminal admission must enumerate all 200 registered cells")
    expected = {key for key, value in declared.items() if value}
    if {key[:-1] for key in admitted} != expected:
        raise RuntimeError("terminal draws disagree with frozen endpoint admission")
    by_domain = {}
    for domain in DOMAINS:
        series = {}
        seed_sets = []
        for level in LEVELS:
            series[level] = {}
            for method in METHODS:
                values = {seed: admitted[level, domain, method, seed, TARGET_STEP]["means"]
                          for seed in SEEDS if (level, domain, method, seed, TARGET_STEP) in admitted}
                seed_sets.append(set(values))
                series[level][method] = {
                    "n": len(values), "seeds": sorted(values),
                    "per_seed": {str(seed): value for seed, value in values.items()},
                    "means": {metric: statistics.fmean(value[metric] for value in values.values())
                              for metric in METRICS} if values else {},
                }
        common = sorted(set.intersection(*seed_sets))
        by_domain[domain] = {
            "series": series, "eight_series_matched_seeds": common,
            "n": len(common), "complete_block": common == list(SEEDS),
        }
    complete = [domain for domain in DOMAINS if by_domain[domain]["complete_block"]]
    if not complete:
        raise RuntimeError("no complete terminal Level1/Level2 domain block")
    means = {
        level: {method: {metric: statistics.fmean(
            by_domain[domain]["series"][level][method]["means"][metric]
            for domain in complete) for metric in METRICS} for method in METHODS}
        for level in LEVELS
    }
    return {
        "model": "Qwen2.5-0.5B-Instruct", "target_step": TARGET_STEP, "training_pass": 8,
        "selection_rule": "Include a domain in the main mean only when every registered seed has an admitted pass-8 endpoint in all four methods at both levels; select by availability before reading effects.",
        "aggregation_rule": "Mean four sampled draws within each seed, mean the five registered seeds within each complete domain, then weight complete domains equally for all eight series.",
        "partial_domain_policy": "Retain available terminal arm means and exact seed n separately; never substitute earlier checkpoints or include partial blocks in the across-domain mean.",
        "uncertainty": "Descriptive seed means; no confidence intervals.",
        "complete_domains": complete,
        "partial_domains": [domain for domain in DOMAINS if domain not in complete],
        "seeds_per_complete_domain": len(SEEDS),
        "admitted_terminal_cells_by_level": {level: sum(key[0] == level for key in expected)
                                             for level in LEVELS},
        "expected_terminal_cells_per_level": len(DOMAINS) * len(METHODS) * len(SEEDS),
        "domain_results": by_domain, "means": means,
    }
