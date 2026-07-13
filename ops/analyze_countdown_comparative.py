#!/usr/bin/env python3
"""Regression analysis for the ModeBench comparative (paper protocol).

Consumes the outputs of ops/run_countdown_comparative_eval.sh: for each
evaluation seed a coverage-eval directory
    {output_root}/{stamp_prefix}_e{eseed}/checkpoints/{arm}_s{seed}/{split}/
containing attempts.json and prompt_metrics.json, plus a
{stamp_prefix}_greedy directory for the deterministic pass.

Builds prompt-level outcomes (pass@K, mean@K, distinct@K, coverage@K, greedy
pass@1) and, for each treatment arm versus the baseline, estimates the linear
probability model

    Y = alpha + gamma * 1{arm=treatment} + train-seed FE + eval-seed FE + eps

with standard errors clustered by evaluation prompt (CR1). Reports gamma, its
standard error, a 95% CI, and a normal-approximation p-value per outcome.

Runs with the paper310 python (numpy required, nothing else).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ALIAS_RE = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)$")


def _load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def _discover_eval_dirs(output_root: Path, stamp_prefix: str):
    sampled = {}
    greedy = None
    for entry in sorted(output_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == f"{stamp_prefix}_greedy":
            greedy = entry
        else:
            match = re.fullmatch(re.escape(stamp_prefix) + r"_e(\d+)", entry.name)
            if match:
                sampled[int(match.group(1))] = entry
    return sampled, greedy


def _iter_alias_dirs(eval_dir: Path):
    checkpoints = eval_dir / "checkpoints"
    if not checkpoints.is_dir():
        return
    for alias_dir in sorted(checkpoints.iterdir()):
        match = ALIAS_RE.match(alias_dir.name)
        if match is None:
            continue
        yield match.group("arm"), int(match.group("seed")), alias_dir


def _prompt_rows(alias_dir: Path, k: int):
    """Yield per-prompt outcome dicts for every split under an alias dir."""
    for split_dir in sorted(p for p in alias_dir.iterdir() if p.is_dir()):
        attempts_path = split_dir / "attempts.json"
        metrics_path = split_dir / "prompt_metrics.json"
        if not attempts_path.is_file():
            continue
        attempts = _load_json(attempts_path)
        mode_counts = {}
        if metrics_path.is_file():
            for row in _load_json(metrics_path):
                mode_counts[int(row["dataset_index"])] = float(
                    row.get("answer_mode_count") or 0.0
                )
        by_prompt = defaultdict(list)
        for row in attempts:
            if int(row["sample_index"]) <= k:
                by_prompt[int(row["dataset_index"])].append(row)
        for prompt_idx, rows in sorted(by_prompt.items()):
            correct = [bool(r["correct"]) for r in rows]
            keys = {
                r["answer_key"]
                for r in rows
                if bool(r["correct"]) and r.get("answer_key")
            }
            total_modes = mode_counts.get(prompt_idx, 0.0)
            yield {
                "split": split_dir.name,
                "prompt": prompt_idx,
                "pass_at_k": float(any(correct)),
                "mean_at_k": float(np.mean(correct)) if correct else 0.0,
                "distinct_at_k": float(len(keys)),
                "coverage_at_k": (
                    float(len(keys)) / total_modes if total_modes > 0 else 0.0
                ),
            }


def build_observations(output_root: Path, stamp_prefix: str, k: int):
    sampled_dirs, greedy_dir = _discover_eval_dirs(output_root, stamp_prefix)
    if not sampled_dirs:
        raise SystemExit(
            f"no {stamp_prefix}_e* eval directories under {output_root}"
        )
    observations = []
    for eseed, eval_dir in sorted(sampled_dirs.items()):
        for arm, seed, alias_dir in _iter_alias_dirs(eval_dir):
            for row in _prompt_rows(alias_dir, k):
                observations.append(
                    {"arm": arm, "train_seed": seed, "eval_seed": eseed, **row}
                )
    greedy_rows = []
    if greedy_dir is not None:
        for arm, seed, alias_dir in _iter_alias_dirs(greedy_dir):
            for row in _prompt_rows(alias_dir, 1):
                greedy_rows.append(
                    {
                        "arm": arm,
                        "train_seed": seed,
                        "prompt": row["prompt"],
                        "split": row["split"],
                        "pass_at_1_greedy": row["pass_at_k"],
                    }
                )
    return observations, greedy_rows


def _fixed_effect_columns(values):
    levels = sorted(set(values))
    columns = []
    for level in levels[1:]:
        columns.append(np.asarray([1.0 if v == level else 0.0 for v in values]))
    return columns


def clustered_lpm(y, treat, clusters, fe_lists):
    """OLS of y on [1, treat, FE dummies] with CR1 cluster-robust SEs.

    Returns (gamma_hat, se, n_obs, n_clusters).
    """
    columns = [np.ones(len(y)), np.asarray(treat, dtype=float)]
    for values in fe_lists:
        columns.extend(_fixed_effect_columns(values))
    x = np.column_stack(columns)
    y = np.asarray(y, dtype=float)
    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta

    cluster_ids = defaultdict(list)
    for i, c in enumerate(clusters):
        cluster_ids[c].append(i)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for idx in cluster_ids.values():
        xg = x[idx]
        eg = resid[idx]
        score = xg.T @ eg
        meat += np.outer(score, score)
    n, p = x.shape
    g = len(cluster_ids)
    if g < 2 or n <= p:
        return float(beta[1]), float("nan"), n, g
    correction = (g / (g - 1)) * ((n - 1) / (n - p))
    cov = correction * (xtx_inv @ meat @ xtx_inv)
    return float(beta[1]), float(math.sqrt(max(cov[1, 1], 0.0))), n, g


def _p_value(gamma, se):
    if not (se and math.isfinite(se) and se > 0):
        return float("nan")
    z = abs(gamma / se)
    return math.erfc(z / math.sqrt(2.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("var/artifacts"))
    parser.add_argument("--stamp-prefix", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--baseline", default="grpo")
    parser.add_argument("--split", default="multi_answer")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    observations, greedy_rows = build_observations(
        args.output_root, args.stamp_prefix, args.k
    )
    observations = [o for o in observations if o["split"] == args.split]
    greedy_rows = [o for o in greedy_rows if o["split"] == args.split]
    arms = sorted({o["arm"] for o in observations})
    if args.baseline not in arms:
        raise SystemExit(f"baseline arm {args.baseline!r} not found in {arms}")

    # Design-completeness report: an arm silently missing a train seed or an
    # eval seed (crashed job, stale alias from an earlier eval run) would
    # unbalance the regression without changing the headline estimate's look.
    grid = defaultdict(lambda: (set(), set()))
    for o in observations:
        seeds, eseeds = grid[o["arm"]]
        seeds.add(o["train_seed"])
        eseeds.add(o["eval_seed"])
    all_seed_sets = {frozenset(v[0]) for v in grid.values()}
    all_eseed_sets = {frozenset(v[1]) for v in grid.values()}
    print("design grid (arm: train_seeds / eval_seeds):")
    for arm in arms:
        seeds, eseeds = grid[arm]
        print(f"  {arm}: {sorted(seeds)} / {sorted(eseeds)}")
    if len(all_seed_sets) > 1 or len(all_eseed_sets) > 1:
        print(
            "WARNING: unbalanced design — arms differ in train/eval seed "
            "coverage; interpret coefficients with caution.\n"
        )

    outcomes = {
        f"pass@{args.k}": "pass_at_k",
        f"mean@{args.k}": "mean_at_k",
        f"coverage@{args.k}": "coverage_at_k",
        f"distinct@{args.k}": "distinct_at_k",
    }
    results = []
    for arm in arms:
        if arm == args.baseline:
            continue
        subset = [o for o in observations if o["arm"] in (args.baseline, arm)]
        for label, field_name in outcomes.items():
            gamma, se, n, g = clustered_lpm(
                [o[field_name] for o in subset],
                [1.0 if o["arm"] == arm else 0.0 for o in subset],
                [o["prompt"] for o in subset],
                [
                    [o["train_seed"] for o in subset],
                    [o["eval_seed"] for o in subset],
                ],
            )
            results.append(
                {
                    "treatment": arm,
                    "outcome": label,
                    "gamma": gamma,
                    "se": se,
                    "ci_low": gamma - 1.96 * se if math.isfinite(se) else None,
                    "ci_high": gamma + 1.96 * se if math.isfinite(se) else None,
                    "p": _p_value(gamma, se),
                    "n_obs": n,
                    "n_prompt_clusters": g,
                }
            )
        greedy_subset = [
            o for o in greedy_rows if o["arm"] in (args.baseline, arm)
        ]
        if greedy_subset:
            gamma, se, n, g = clustered_lpm(
                [o["pass_at_1_greedy"] for o in greedy_subset],
                [1.0 if o["arm"] == arm else 0.0 for o in greedy_subset],
                [o["prompt"] for o in greedy_subset],
                [[o["train_seed"] for o in greedy_subset]],
            )
            results.append(
                {
                    "treatment": arm,
                    "outcome": "pass@1 (greedy)",
                    "gamma": gamma,
                    "se": se,
                    "ci_low": gamma - 1.96 * se if math.isfinite(se) else None,
                    "ci_high": gamma + 1.96 * se if math.isfinite(se) else None,
                    "p": _p_value(gamma, se),
                    "n_obs": n,
                    "n_prompt_clusters": g,
                }
            )

    out_path = args.out or (
        args.output_root / f"{args.stamp_prefix}_comparative_regression.json"
    )
    out_path.write_text(json.dumps(results, indent=2))

    header = (
        f"{'treatment':<16} {'outcome':<18} {'gamma':>8} {'se':>8} "
        f"{'95% CI':>20} {'p':>8} {'N':>7} {'clusters':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        ci = (
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
            if r["ci_low"] is not None
            else "[--, --]"
        )
        print(
            f"{r['treatment']:<16} {r['outcome']:<18} {r['gamma']:+8.4f} "
            f"{r['se']:8.4f} {ci:>20} {r['p']:8.4f} {r['n_obs']:7d} "
            f"{r['n_prompt_clusters']:8d}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
