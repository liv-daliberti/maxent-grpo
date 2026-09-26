#!/usr/bin/env python3
"""Audit and aggregate E72 B1b ordinary verified-response rehearsal.

B1b is the published matched Dr.GRPO control with a live verified-likelihood
replay gradient, but no rarity credit and no uniform mode-balancing gradient.
The frozen protocol reports terminal pass 12, five paired seeds per domain,
without pooling domains or carrying forward incomplete checkpoints.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

SEEDS = (43, 44, 45, 46, 47)
DOMAINS = (
    ("graph_coloring", "Graph coloring"),
    ("countdown", "Countdown"),
    ("python_factors", "Python factors"),
    ("mathir", "MathIR action menu"),
    ("pantry_plan", "PantryPlan"),
)
TERMINAL_STEP = 4608
MARGIN = 0.15
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 72_011_043
INTERVAL_PERCENTILES = (2.5, 97.5)

METRIC_KEYS = {
    "greedy": "eval/multi_answer/accuracy",
    "mean8": "eval/multi_answer/sampled_mean_at_8",
    "pass8": "eval/multi_answer/sampled_any_correct_at_8",
    "distinct8": "eval/multi_answer/sampled_distinct_correct_at_8",
}
EXPECTED_OVERRIDES = {
    "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.0",
    "OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE": "0",
    "OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE": "0",
    "OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE": "verified_likelihood_per_rollout",
}

REPLAY_GRADIENT_KEYS = (
    "train/canonical_replay_applied_score_gradient_l2",
    "train/canonical_replay_applied_score_gradient_sum",
)
BALANCE_GRADIENT_KEYS = (
    "train/canonical_replay_balance_score_gradient_l2",
    "train/canonical_replay_balance_score_gradient_sum",
)
MASS_GRADIENT_KEYS = (
    "train/canonical_replay_mass_score_gradient_l2",
    "train/canonical_replay_mass_score_gradient_sum",
)
ZERO_DISCOVERY_KEYS = (
    "train/online_canonical_entropy_advantage_rms",
    "train/online_canonical_combined_advantage_rms",
)
ACTIVE_DISCOVERY_KEYS = (
    "train/semantic_shannon_separate_advantage_active",
    "train/semantic_shannon_success_conditioned_signed_advantage_active",
)
COMPUTE_ONLY_KEYS = (
    "train/canonical_replay_compute_only",
    "train/canonical_replay_compute_only_configured",
)
VERIFIED_LIKELIHOOD_KEY = "train/canonical_replay_verified_likelihood_active"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _max_abs(record: dict[str, Any], keys: tuple[str, ...], current: float) -> tuple[float, int]:
    seen = 0
    for key in keys:
        value = record.get(key)
        if finite(value):
            current = max(current, abs(float(value)))
            seen += 1
    return current, seen


def read_run(run_dir: Path) -> dict[str, Any]:
    """Read one B1b cell and enforce the observed arm-integrity contract."""

    terminal_rows: list[dict[str, float]] = []
    counts = {
        "metric_files": 0,
        "records": 0,
        "replay_gradient_records": 0,
        "balance_gradient_records": 0,
        "mass_gradient_records": 0,
        "compute_only_records": 0,
        "verified_likelihood_records": 0,
        "discovery_records": 0,
    }
    replay_gradient_max = 0.0
    balance_gradient_max = 0.0
    mass_gradient_max = 0.0
    discovery_max = 0.0
    compute_only_values: set[float] = set()
    verified_likelihood_values: set[float] = set()
    discovery_switch_values: dict[str, set[float]] = {
        key: set() for key in ACTIVE_DISCOVERY_KEYS
    }

    for path_text in sorted(
        glob.glob(str(run_dir / "debug_job*" / "train_metrics.jsonl"))
    ):
        counts["metric_files"] += 1
        with Path(path_text).open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                counts["records"] += 1
                if int(record.get("misc/global_step", -1)) == TERMINAL_STEP:
                    values = {
                        metric: float(record[key])
                        for metric, key in METRIC_KEYS.items()
                        if finite(record.get(key))
                    }
                    if len(values) == len(METRIC_KEYS):
                        terminal_rows.append(values)

                replay_gradient_max, seen = _max_abs(
                    record, REPLAY_GRADIENT_KEYS, replay_gradient_max
                )
                counts["replay_gradient_records"] += seen
                balance_gradient_max, seen = _max_abs(
                    record, BALANCE_GRADIENT_KEYS, balance_gradient_max
                )
                counts["balance_gradient_records"] += seen
                mass_gradient_max, seen = _max_abs(
                    record, MASS_GRADIENT_KEYS, mass_gradient_max
                )
                counts["mass_gradient_records"] += seen

                for key in ZERO_DISCOVERY_KEYS:
                    value = record.get(key)
                    if finite(value):
                        counts["discovery_records"] += 1
                        discovery_max = max(discovery_max, abs(float(value)))
                for key in ACTIVE_DISCOVERY_KEYS:
                    value = record.get(key)
                    if finite(value):
                        discovery_switch_values[key].add(float(value))
                for key in COMPUTE_ONLY_KEYS:
                    value = record.get(key)
                    if finite(value):
                        counts["compute_only_records"] += 1
                        compute_only_values.add(float(value))
                value = record.get(VERIFIED_LIKELIHOOD_KEY)
                if finite(value):
                    counts["verified_likelihood_records"] += 1
                    verified_likelihood_values.add(float(value))

    violations: list[str] = []
    if not (run_dir / "TRAINING_COMPLETE.json").is_file():
        violations.append("missing TRAINING_COMPLETE.json")
    if not terminal_rows:
        violations.append("missing pass-12 evaluation")
        terminal = None
    else:
        for metric in METRIC_KEYS:
            values = {round(row[metric], 12) for row in terminal_rows}
            if len(values) != 1:
                violations.append(f"conflicting pass-12 {metric} values")
        terminal = terminal_rows[-1]
    if counts["replay_gradient_records"] == 0 or replay_gradient_max <= 0:
        violations.append("nonzero verified-likelihood replay gradient not observed")
    if not compute_only_values or compute_only_values != {0.0}:
        violations.append(
            f"replay compute-only values are {sorted(compute_only_values)}"
        )
    if (
        not verified_likelihood_values
        or 1.0 not in verified_likelihood_values
        or verified_likelihood_values - {0.0, 1.0}
    ):
        violations.append(
            "verified-likelihood activation values are "
            f"{sorted(verified_likelihood_values)}"
        )
    if balance_gradient_max != 0.0:
        violations.append("uniform-balance score gradient was nonzero")
    if mass_gradient_max != 0.0:
        violations.append("verified-mass score gradient was nonzero")
    if discovery_max != 0.0:
        violations.append("rarity/discovery advantage was nonzero")
    if any(values - {0.0} for values in discovery_switch_values.values()):
        violations.append("a disabled discovery switch became active")

    return {
        "run_dir": str(run_dir),
        "terminal": terminal is not None and not violations,
        "metrics": terminal,
        "integrity": {
            **counts,
            "replay_gradient_max": replay_gradient_max,
            "balance_gradient_max": balance_gradient_max,
            "mass_gradient_max": mass_gradient_max,
            "discovery_advantage_max": discovery_max,
            "compute_only_values": sorted(compute_only_values),
            "verified_likelihood_values": sorted(verified_likelihood_values),
            "discovery_switch_values": {
                key: sorted(values) for key, values in discovery_switch_values.items()
            },
            "violations": violations,
        },
    }


def load_cells(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    ledger_path = root / "var" / "artifacts" / "e72_b1b_ablation_jobs.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("arm") != "b1b":
        raise ValueError("B1b ledger has the wrong arm")
    if ledger.get("variant") != "verified_first_replay_rehearsal_only":
        raise ValueError("B1b ledger has the wrong runtime variant")
    if ledger.get("overrides") != EXPECTED_OVERRIDES:
        raise ValueError("B1b effective overrides differ from the frozen protocol")
    expected = {(domain, seed) for domain, _ in DOMAINS for seed in SEEDS}
    registered = {
        (str(run["domain"]), int(run["seed"])) for run in ledger.get("runs", [])
    }
    if registered != expected or len(ledger.get("runs", [])) != len(expected):
        raise ValueError("B1b launch ledger is not the frozen 5x5 grid")
    return {
        (str(run["domain"]), int(run["seed"])): read_run(Path(run["save_path"]))
        for run in ledger["runs"]
    }


def load_references(root: Path) -> dict[tuple[str, str, int], dict[str, float]]:
    manifest_path = root / "var" / "artifacts" / "e72_frontier_source_runs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    references = {
        (str(run["domain"]), str(run["arm"]), int(run["seed"])): {
            metric: float(run["published_terminal"][metric]) for metric in METRIC_KEYS
        }
        for run in manifest["runs"]
        if run.get("arm") in {"drgrpo", "xgrpo"} and run.get("published_terminal")
    }
    expected = {
        (domain, arm, seed)
        for domain, _ in DOMAINS
        for arm in ("drgrpo", "xgrpo")
        for seed in SEEDS
    }
    if set(references) != expected:
        raise ValueError("reference manifest does not contain the frozen 5x5 paired arms")

    # B1b's frozen protocol also names B3a and the exploratory B1a cohort as
    # fixed comparators. Their outcomes were known before B1b was launched;
    # keeping that provenance explicit makes the B1b comparison prospective
    # without relabelling B1a's original 43--47 sample as confirmatory.
    b3a = json.loads(
        (root / "var" / "artifacts" / "e72_b3a_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if b3a.get("interim_mode") or set(b3a.get("reportable_domains", [])) != {
        domain for domain, _ in DOMAINS
    }:
        raise ValueError("B3a comparator is not the terminal five-domain artifact")
    for domain_row in b3a["domains"]:
        domain = str(domain_row["domain"])
        by_seed = {int(row["seed"]): row for row in domain_row["per_seed"]}
        if set(by_seed) != set(SEEDS):
            raise ValueError(f"B3a {domain} does not contain seeds 43--47")
        for seed in SEEDS:
            references[(domain, "b3a", seed)] = {
                metric: float(by_seed[seed]["b3a"][metric]) for metric in METRIC_KEYS
            }

    b1a = json.loads(
        (root / "var" / "artifacts" / "e72_b1a_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if b1a.get("schema") != "e72_b1a_discovery_sample_v1":
        raise ValueError("B1a comparator is not the frozen discovery-sample artifact")
    by_domain = {str(row["domain"]): row for row in b1a["domains"]}
    if set(by_domain) != {domain for domain, _ in DOMAINS}:
        raise ValueError("B1a comparator is not the frozen five-domain artifact")
    for domain, _ in DOMAINS:
        row = by_domain[domain]
        if tuple(row["seeds"]) != SEEDS or len(row["per_seed_b1a"]) != len(SEEDS):
            raise ValueError(f"B1a {domain} does not contain ordered seeds 43--47")
        for seed, value in zip(SEEDS, row["per_seed_b1a"]):
            # The B1a artifact predates the four-metric terminal aggregator and
            # freezes only the preregistered primary quantity.
            references[(domain, "b1a", seed)] = {"distinct8": float(value)}
    return references


def percentile_interval(
    deltas: list[float], rng: np.random.Generator
) -> list[float]:
    values = np.asarray(deltas, dtype=float)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, INTERVAL_PERCENTILES)
    return [float(low), float(high)]


def verdict(interval: list[float]) -> str:
    low, high = interval
    if low > -MARGIN and high < MARGIN:
        return "E"
    if high < 0.0:
        return "W"
    if low > MARGIN:
        return "B"
    return "U"


def registered_interpretations(domains: list[dict[str, Any]]) -> list[str]:
    if not all(row.get("reportable") for row in domains):
        return ["WITHHELD"]
    p1 = sum(row["comparisons"]["xgrpo"]["verdict"] == "E" for row in domains) >= 3
    p2 = (
        sum(
            row["comparisons"]["drgrpo"]["verdict"] == "B"
            and row["comparisons"]["xgrpo"]["verdict"] == "W"
            for row in domains
        )
        >= 3
    )
    p3 = (
        sum(row["comparisons"]["drgrpo"]["verdict"] == "E" for row in domains)
        >= 3
    )
    p4 = any(
        row["comparisons"]["xgrpo"]["mean"] > 0.0 for row in domains
    )
    fired = [
        code
        for code, condition in (("P1", p1), ("P2", p2), ("P3", p3), ("P4", p4))
        if condition
    ]
    return fired or ["PER_DOMAIN_ONLY"]


def summarize(
    cells: dict[tuple[str, int], dict[str, Any]],
    references: dict[tuple[str, str, int], dict[str, float]],
) -> dict[str, Any]:
    domains: list[dict[str, Any]] = []
    for domain_index, (domain, title) in enumerate(DOMAINS):
        violations = []
        terminal_seeds = []
        for seed in SEEDS:
            row = cells[(domain, seed)]
            if row["terminal"]:
                terminal_seeds.append(seed)
            if row["integrity"]["violations"]:
                violations.append(
                    {"seed": seed, "violations": row["integrity"]["violations"]}
                )
        reportable = len(terminal_seeds) == len(SEEDS) and not violations
        if not reportable:
            domains.append(
                {
                    "domain": domain,
                    "title": title,
                    "reportable": False,
                    "terminal_seeds": terminal_seeds,
                    "violations": violations,
                }
            )
            continue

        per_seed = []
        for seed in SEEDS:
            b1b = cells[(domain, seed)]
            per_seed.append(
                {
                    "seed": seed,
                    "b1b": b1b["metrics"],
                    "b3a": references[(domain, "b3a", seed)],
                    "b1a": references[(domain, "b1a", seed)],
                    "drgrpo": references[(domain, "drgrpo", seed)],
                    "xgrpo": references[(domain, "xgrpo", seed)],
                    "integrity": b1b["integrity"],
                }
            )
        means = {
            arm: {
                metric: statistics.fmean(row[arm][metric] for row in per_seed)
                for metric in METRIC_KEYS
                if metric in per_seed[0][arm]
            }
            for arm in ("drgrpo", "b3a", "b1b", "b1a", "xgrpo")
        }
        comparisons = {}
        for opponent_index, opponent in enumerate(
            ("drgrpo", "b3a", "b1a", "xgrpo")
        ):
            deltas = [
                row["b1b"]["distinct8"] - row[opponent]["distinct8"]
                for row in per_seed
            ]
            interval = percentile_interval(
                deltas,
                np.random.default_rng(
                    BOOTSTRAP_SEED + 10 * domain_index + opponent_index
                ),
            )
            comparisons[opponent] = {
                "mean": statistics.fmean(deltas),
                "interval": interval,
                "per_seed": deltas,
                "wins": sum(delta > 0.0 for delta in deltas),
                "verdict": verdict(interval),
            }
        domains.append(
            {
                "domain": domain,
                "title": title,
                "reportable": True,
                "terminal_seeds": list(SEEDS),
                "means": means,
                "comparisons": comparisons,
                "per_seed": per_seed,
                "violations": [],
            }
        )

    all_reportable = all(row["reportable"] for row in domains)
    interpretations = registered_interpretations(domains)
    return {
        "schema": "e72_b1b_summary_v1",
        "terminal_step": TERMINAL_STEP,
        "metric": "distinct@8",
        "seeds": list(SEEDS),
        "bootstrap": {
            "method": "paired seed-level percentile",
            "resamples": BOOTSTRAP_RESAMPLES,
            "rng": "numpy.PCG64",
            "seed": BOOTSTRAP_SEED,
            "interval_percentiles": list(INTERVAL_PERCENTILES),
            "equivalence_margin": MARGIN,
        },
        "all_25_cells_terminal_and_valid": all_reportable,
        "reportable_domains": [
            row["domain"] for row in domains if row["reportable"]
        ],
        "registered_interpretations": interpretations,
        "domains": domains,
    }


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_report(summary: dict[str, Any], path: Path) -> None:
    state = "terminal registered analysis" if summary["all_25_cells_terminal_and_valid"] else "WITHHELD"
    lines = [
        f"# E72 B1b ordinary verified rehearsal — {state}",
        "",
        "Primary metric: distinct@8 at pass 12; five paired seeds per domain.",
        "",
        "| domain | Dr.GRPO | discovery only (B3a) | rehearsal (B1b) | replay + balance (B1a) | xGRPO | rehearsal-xGRPO [95% CI] | verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in summary["domains"]:
        if not row["reportable"]:
            lines.append(
                f"| {row['title']} | _withheld ({len(row['terminal_seeds'])}/5 terminal)_ | | | | | | |"
            )
            continue
        means = row["means"]
        xg = row["comparisons"]["xgrpo"]
        lines.append(
            f"| {row['title']} | {means['drgrpo']['distinct8']:.3f} | "
            f"{means['b3a']['distinct8']:.3f} | {means['b1b']['distinct8']:.3f} | "
            f"{means['b1a']['distinct8']:.3f} | {means['xgrpo']['distinct8']:.3f} | "
            f"{xg['mean']:+.3f} "
            f"[{xg['interval'][0]:+.3f}, {xg['interval'][1]:+.3f}] | {xg['verdict']} |"
        )
    if summary["all_25_cells_terminal_and_valid"]:
        lines += [
            "",
            "## Paired B1b contrasts",
            "",
            "| domain | versus Dr.GRPO | versus B3a | versus B1a | versus xGRPO |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in summary["domains"]:
            cells = []
            for opponent in ("drgrpo", "b3a", "b1a", "xgrpo"):
                comparison = row["comparisons"][opponent]
                cells.append(
                    f"{comparison['mean']:+.3f} "
                    f"[{comparison['interval'][0]:+.3f}, "
                    f"{comparison['interval'][1]:+.3f}] "
                    f"({comparison['verdict']})"
                )
            lines.append(f"| {row['title']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Registered interpretations: **"
        + ", ".join(summary["registered_interpretations"])
        + "**.",
        "",
        "P1 requires equivalence to xGRPO in at least three domains; P2 requires "
        "a material gain over Dr.GRPO but a loss to xGRPO in at least three; P3 "
        "requires equivalence to Dr.GRPO in at least three. P4 records any "
        "positive B1b-xGRPO point estimate as promised, but does not license "
        "superiority unless the corresponding verdict is B.",
        "",
        "E = interval wholly within (-0.15, +0.15); W = interval excludes zero "
        "below; B = interval wholly above +0.15; otherwise U.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "var" / "artifacts" / "e72_b1b_summary.json",
    )
    args = parser.parse_args()
    summary = summarize(load_cells(root), load_references(root))
    atomic_json(summary, args.output)
    report = root / "paper" / "results" / "e72_b1b_terminal.md"
    write_report(summary, report)
    print(
        f"[e72-b1b] {len(summary['reportable_domains'])}/5 domains reportable; "
        f"{summary['registered_interpretations']}"
    )
    print(f"[e72-b1b] wrote {args.output} and {report}")
    return 0 if summary["all_25_cells_terminal_and_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
