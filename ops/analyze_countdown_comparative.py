#!/usr/bin/env python3
"""Regression analysis for the ModeBench comparative (paper protocol).

Consumes the outputs of ops/run_countdown_comparative_eval.sh: for each
evaluation seed a coverage-eval directory
    {output_root}/{stamp_prefix}_e{eseed}/checkpoints/{arm}_s{seed}/{split}/
containing attempts.json and prompt_metrics.json, plus a
{stamp_prefix}_greedy directory for the deterministic pass.

Builds prompt-level outcomes (pass@K, mean@K, distinct@K, coverage@K, greedy
pass@1) and, for each treatment arm versus the baseline, estimates the linear
probability model of the paper's Equation (3)

    Y = alpha + gamma * 1{arm=treatment}
        + train-seed FE + eval-seed FE [+ task-domain FE] + eps

with standard errors clustered by evaluation prompt (CR1), plus a robustness
pass clustering by training run (domain x arm x train seed), as pre-specified
in the paper's regression protocol. Reports gamma, both standard errors, a 95%
CI, and normal-approximation p-values per outcome. The pre-specified primary
arm (default xdr_tau0p5) is reported with its raw p-value; every other arm is
exploratory and additionally gets a Holm-adjusted p-value, adjusted within
outcome across the exploratory arms.

Single-domain usage (unchanged):
    --stamp-prefix cdcomp4_stable
Pooled two-domain usage (task-domain fixed effects, prompt clusters qualified
by domain so prompt indices cannot collide across tasks):
    --stamp-prefix countdown=cdcomp4_stable \
    --stamp-prefix graph_coloring=gccomp1_stable

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
        if not metrics_path.is_file():
            # Silently zeroing coverage for one arm would bias the comparison;
            # fail loudly instead.
            raise SystemExit(
                f"missing {metrics_path}; cannot compute coverage for "
                f"{alias_dir.name} without the prompt mode counts"
            )
        mode_counts = {}
        for row in _load_json(metrics_path):
            mode_counts[int(row["dataset_index"])] = float(
                row.get("answer_mode_count") or 0.0
            )
        by_prompt = defaultdict(list)
        for row in attempts:
            if int(row["sample_index"]) <= k:
                by_prompt[int(row["dataset_index"])].append(row)
        unparseable_correct = 0
        for prompt_idx, rows in sorted(by_prompt.items()):
            correct = [bool(r["correct"]) for r in rows]
            keys = set()
            for i, r in enumerate(rows):
                if not bool(r["correct"]):
                    continue
                if r.get("answer_key"):
                    keys.add(r["answer_key"])
                else:
                    # A verified-correct completion whose clustering key could
                    # not be extracted still represents a found mode; count it
                    # as its own singleton rather than dropping it (dropping
                    # would penalize arms with messier surface formatting on
                    # exactly the headline coverage metrics).
                    keys.add(("__unparsed__", prompt_idx, i))
                    unparseable_correct += 1
            total_modes = mode_counts.get(prompt_idx, 0.0)
            yield {
                "split": split_dir.name,
                "prompt": prompt_idx,
                "pass_at_k": float(any(correct)),
                "mean_at_k": float(np.mean(correct)) if correct else 0.0,
                "distinct_at_k": float(min(len(keys), total_modes) if total_modes > 0 else len(keys)),
                "coverage_at_k": (
                    min(float(len(keys)) / total_modes, 1.0)
                    if total_modes > 0
                    else 0.0
                ),
            }
        if unparseable_correct:
            print(
                f"note: {alias_dir.name}/{split_dir.name}: "
                f"{unparseable_correct} correct completions lacked a "
                "clustering key; counted as singleton modes"
            )


def build_observations(output_root: Path, stamp_prefix: str, k: int, domain: str):
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
                    {
                        "arm": arm,
                        "train_seed": seed,
                        "eval_seed": eseed,
                        "domain": domain,
                        **row,
                    }
                )
    greedy_rows = []
    if greedy_dir is not None:
        for arm, seed, alias_dir in _iter_alias_dirs(greedy_dir):
            for row in _prompt_rows(alias_dir, 1):
                greedy_rows.append(
                    {
                        "arm": arm,
                        "train_seed": seed,
                        "domain": domain,
                        "prompt": row["prompt"],
                        "split": row["split"],
                        "pass_at_1_greedy": row["pass_at_k"],
                    }
                )
    return observations, greedy_rows


def parse_stamp_specs(specs: list[str]) -> list[tuple[str, str]]:
    """Parse --stamp-prefix values into (domain_label, stamp) pairs.

    A bare stamp keeps the historical single-domain behavior (domain label =
    stamp). A 'domain=stamp' spec labels the task domain for pooled runs.
    """
    pairs = []
    for spec in specs:
        if "=" in spec:
            domain, stamp = spec.split("=", 1)
        else:
            domain, stamp = spec, spec
        if not domain or not stamp:
            raise SystemExit(f"malformed --stamp-prefix {spec!r}")
        pairs.append((domain, stamp))
    if len({domain for domain, _ in pairs}) != len(pairs):
        raise SystemExit("duplicate domain labels in --stamp-prefix")
    return pairs


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment (monotone, capped at 1)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (math.inf if math.isnan(p_values[i]) else p_values[i]))
    adjusted = [float("nan")] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = p_values[idx]
        if math.isnan(p):
            adjusted[idx] = float("nan")
            continue
        running_max = max(running_max, min(1.0, (m - rank) * p))
        adjusted[idx] = running_max
    return adjusted


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


def _estimate(subset, y_field, arm, multi_domain, fe_builder):
    """Point estimate with prompt-clustered and run-clustered CR1 SEs.

    Prompt clusters are domain-qualified so integer prompt indices from
    different task datasets can never merge; the run-cluster robustness pass
    treats each trained checkpoint (domain x arm x train seed) as one cluster,
    as pre-specified in the paper's regression protocol.
    """
    y = [o[y_field] for o in subset]
    treat = [1.0 if o["arm"] == arm else 0.0 for o in subset]
    fe_lists = fe_builder(subset)
    if multi_domain:
        fe_lists = fe_lists + [[o["domain"] for o in subset]]
    prompt_clusters = [f"{o['domain']}:{o['prompt']}" for o in subset]
    run_clusters = [f"{o['domain']}:{o['arm']}:{o['train_seed']}" for o in subset]
    gamma, se, n, g_prompt = clustered_lpm(y, treat, prompt_clusters, fe_lists)
    gamma_run, se_run, _n, g_run = clustered_lpm(y, treat, run_clusters, fe_lists)
    assert abs(gamma - gamma_run) < 1e-9, "point estimate must not depend on clustering"
    return {
        "gamma": gamma,
        "se": se,
        "ci_low": gamma - 1.96 * se if math.isfinite(se) else None,
        "ci_high": gamma + 1.96 * se if math.isfinite(se) else None,
        "p": _p_value(gamma, se),
        "se_run_cluster": se_run,
        "p_run_cluster": _p_value(gamma, se_run),
        "n_obs": n,
        "n_prompt_clusters": g_prompt,
        "n_run_clusters": g_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("var/artifacts"))
    parser.add_argument(
        "--stamp-prefix",
        action="append",
        required=True,
        help="stamp, or domain=stamp; repeat the flag to pool task domains",
    )
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--baseline", default="grpo")
    parser.add_argument(
        "--primary-arm",
        default="xdr_tau0p5",
        help="pre-specified primary treatment; all other arms get "
        "Holm-adjusted p-values (within outcome)",
    )
    parser.add_argument("--split", default="multi_answer")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    stamp_pairs = parse_stamp_specs(args.stamp_prefix)
    multi_domain = len(stamp_pairs) > 1
    observations, greedy_rows = [], []
    for domain, stamp in stamp_pairs:
        domain_obs, domain_greedy = build_observations(
            args.output_root, stamp, args.k, domain
        )
        observations.extend(domain_obs)
        greedy_rows.extend(domain_greedy)
    observations = [o for o in observations if o["split"] == args.split]
    greedy_rows = [o for o in greedy_rows if o["split"] == args.split]
    arms = sorted({o["arm"] for o in observations})
    if args.baseline not in arms:
        raise SystemExit(f"baseline arm {args.baseline!r} not found in {arms}")
    if args.primary_arm not in arms:
        print(
            f"note: primary arm {args.primary_arm!r} absent from this eval; "
            "all arms treated as exploratory (Holm-adjusted)"
        )

    # Design-completeness report: an arm silently missing a train seed, an
    # eval seed, or a whole domain (crashed job, stale alias from an earlier
    # eval run) would unbalance the regression without changing the headline
    # estimate's look.
    grid = defaultdict(lambda: (set(), set(), set()))
    for o in observations:
        seeds, eseeds, domains = grid[o["arm"]]
        seeds.add(o["train_seed"])
        eseeds.add(o["eval_seed"])
        domains.add(o["domain"])
    all_seed_sets = {frozenset(v[0]) for v in grid.values()}
    all_eseed_sets = {frozenset(v[1]) for v in grid.values()}
    all_domain_sets = {frozenset(v[2]) for v in grid.values()}
    print("design grid (arm: train_seeds / eval_seeds / domains):")
    for arm in arms:
        seeds, eseeds, domains = grid[arm]
        print(f"  {arm}: {sorted(seeds)} / {sorted(eseeds)} / {sorted(domains)}")
    if len(all_seed_sets) > 1 or len(all_eseed_sets) > 1 or len(all_domain_sets) > 1:
        print(
            "WARNING: unbalanced design — arms differ in train-seed, "
            "eval-seed, or domain coverage; interpret coefficients with "
            "caution.\n"
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
            estimate = _estimate(
                subset,
                field_name,
                arm,
                multi_domain,
                lambda rows: [
                    [o["train_seed"] for o in rows],
                    [o["eval_seed"] for o in rows],
                ],
            )
            results.append({"treatment": arm, "outcome": label, **estimate})
        greedy_subset = [
            o for o in greedy_rows if o["arm"] in (args.baseline, arm)
        ]
        if greedy_subset:
            estimate = _estimate(
                greedy_subset,
                "pass_at_1_greedy",
                arm,
                multi_domain,
                lambda rows: [[o["train_seed"] for o in rows]],
            )
            results.append(
                {"treatment": arm, "outcome": "pass@1 (greedy)", **estimate}
            )

    # Multiplicity: the pre-specified primary arm keeps its raw p-value; the
    # exploratory arms are Holm-adjusted within each outcome family.
    for r in results:
        r["role"] = "primary" if r["treatment"] == args.primary_arm else "exploratory"
        r["p_holm"] = None
    for outcome in {r["outcome"] for r in results}:
        family = [
            r
            for r in results
            if r["outcome"] == outcome and r["role"] == "exploratory"
        ]
        adjusted = holm_adjust([r["p"] for r in family])
        for r, p_adj in zip(family, adjusted):
            r["p_holm"] = p_adj

    default_stem = "_".join(stamp for _domain, stamp in stamp_pairs)
    out_path = args.out or (
        args.output_root / f"{default_stem}_comparative_regression.json"
    )
    out_path.write_text(json.dumps(results, indent=2))

    domains_note = ", ".join(f"{d}={s}" for d, s in stamp_pairs)
    print(
        f"\nmodel: LPM with train-seed + eval-seed"
        f"{' + domain' if multi_domain else ''} fixed effects; domains: {domains_note}"
    )
    header = (
        f"{'treatment':<16} {'outcome':<16} {'gamma':>8} {'se':>7} "
        f"{'95% CI':>20} {'p':>7} {'pHolm':>7} {'seRun':>7} {'pRun':>7} "
        f"{'N':>6} {'cl':>5} {'runs':>5}"
    )
    print(header)
    print("-" * len(header))

    def _fmt_p(value):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "     --"
        return f"{value:7.4f}"

    for r in results:
        ci = (
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
            if r["ci_low"] is not None
            else "[--, --]"
        )
        marker = "*" if r["role"] == "primary" else " "
        print(
            f"{r['treatment']:<15}{marker} {r['outcome']:<16} {r['gamma']:+8.4f} "
            f"{r['se']:7.4f} {ci:>20} {_fmt_p(r['p'])} {_fmt_p(r['p_holm'])} "
            f"{r['se_run_cluster']:7.4f} {_fmt_p(r['p_run_cluster'])} "
            f"{r['n_obs']:6d} {r['n_prompt_clusters']:5d} {r['n_run_clusters']:5d}"
        )
    print(
        "\n* = pre-specified primary arm (raw p); exploratory arms report "
        "Holm-adjusted p within outcome.\n"
        "seRun/pRun: robustness pass clustering by training run "
        "(domain x arm x train seed).\n"
        f"wrote {out_path}"
    )


if __name__ == "__main__":
    main()
