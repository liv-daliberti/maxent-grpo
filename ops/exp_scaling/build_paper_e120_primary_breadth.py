#!/usr/bin/env python3
"""Recover E120's registered primary estimands from the frozen paper snapshot.

This builder reads no live experiment directories and leaves the source snapshot
unchanged. Percentile-bootstrap implementation details are analysis choices;
the preregistration specifies paired bootstrap intervals, not these details.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper/results/e120_frequency_progress.json"
PROTOCOL = ROOT / "paper/preregistration/e120_frequency_weighted_replay_ablation_20260902.md"
OUTPUT = ROOT / "paper/results/e120_primary_breadth.json"
TABLE = ROOT / "paper/results/e120_primary_breadth_table_body.tex"
SEED_TABLE = ROOT / "paper/results/e120_primary_breadth_seeds_table_body.tex"
FROZEN_DATE = "2026-09-04T15:40:26.896956+00:00"
FROZEN_SHA256 = "92304ed9ac70f6ebc4dd38e12801750175b96703bf657459dafed3c390bf2bab"
SEEDS = (43, 44, 45, 46, 47)
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
LABELS = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
    "five_domain_mean": "Five-domain mean",
}
METRICS = ("breadth8", "pass8", "distinct8")


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number, got {value!r}")
    return float(value)


def validate_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            validate_finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_finite_tree(item, f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        finite_number(value, label)


def require_seed_keys(value: dict[str, Any], label: str) -> None:
    if not isinstance(value, dict) or set(value) != {str(seed) for seed in SEEDS}:
        raise ValueError(f"{label}: expected exactly seeds {SEEDS}")


def close(actual: Any, expected: float, label: str) -> None:
    number = finite_number(actual, label)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: stored contrast disagrees with arm endpoints")


def paired_bootstrap(per_seed: dict[str, float]) -> dict[str, Any]:
    require_seed_keys(per_seed, "bootstrap per_seed")
    values = [finite_number(per_seed[str(seed)], f"seed {seed}") for seed in SEEDS]
    draws = sorted(statistics.fmean(draw) for draw in itertools.product(values, repeat=5))

    def percentile(q: float) -> float:
        position = q * (len(draws) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        return draws[lower] + (position - lower) * (draws[upper] - draws[lower])

    return {
        "mean": statistics.fmean(values),
        "per_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
        "paired_bootstrap_percentile_95": [percentile(0.025), percentile(0.975)],
    }


def analyze(source: dict[str, Any]) -> dict[str, Any]:
    """Validate paired source values before computing any reported estimate."""
    if source.get("schema") != "e120-frequency-progress-v1":
        raise ValueError("unexpected E120 source schema")
    if source.get("contrast") != "fresh-frequency replay minus uniform key-balanced replay":
        raise ValueError("unexpected source contrast direction")
    if source.get("generated_at") != FROZEN_DATE:
        raise ValueError("source is not the registered paper's frozen September 4 snapshot")
    datetime.fromisoformat(source["generated_at"])
    if not source.get("selection_rule", "").startswith("exact step-3072 four-draw endpoints only;"):
        raise ValueError("source endpoint selection rule drifted")
    cells = source["cells"]["qwen05b"]
    if set(cells) != set(DOMAINS):
        raise ValueError("expected all five registered Qwen-0.5B domains")
    rows: dict[str, Any] = {}
    for domain in DOMAINS:
        cell = cells[domain]
        validate_finite_tree(cell, domain)
        if cell.get("complete_block") is not True or cell.get("n") != 5:
            raise ValueError(f"{domain}: incomplete five-seed block")
        for field in ("registered_seeds", "terminal_seeds"):
            if cell.get(field) != list(SEEDS):
                raise ValueError(f"{domain}.{field}: expected exactly seeds {SEEDS}")
        for arm in ("uniform_key_replay", "fresh_frequency"):
            require_seed_keys(cell[arm], f"{domain}.{arm}")
            for seed in SEEDS:
                point = cell[arm][str(seed)]
                p = finite_number(point["pass8"], f"{domain}.{arm}.{seed}.pass8")
                d = finite_number(point["distinct8"], f"{domain}.{arm}.{seed}.distinct8")
                if not 0 <= p <= 1 or not p <= d <= 8:
                    raise ValueError(f"{domain}.{arm}.{seed}: impossible pass/distinct endpoint")
        effects: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
        for metric in ("pass8", "distinct8"):
            stored = cell["fresh_frequency_minus_uniform"][metric]
            require_seed_keys(stored["per_seed"], f"{domain}.{metric}.stored_delta")
            for seed in SEEDS:
                key = str(seed)
                delta = cell["uniform_key_replay"][key][metric] - cell["fresh_frequency"][key][metric]
                close(stored["per_seed"][key], -delta, f"{domain}.{metric}.{seed}")
                effects[metric][key] = delta
            close(stored["mean"], -statistics.fmean(effects[metric].values()), f"{domain}.{metric}.mean")
        for seed in SEEDS:
            key = str(seed)
            effects["breadth8"][key] = effects["distinct8"][key] - effects["pass8"][key]
        rows[domain] = {
            "label": LABELS[domain],
            "n": 5,
            "uniform_key_replay": cell["uniform_key_replay"],
            "fresh_frequency": cell["fresh_frequency"],
            "uniform_minus_frequency": {metric: paired_bootstrap(effects[metric]) for metric in METRICS},
        }
    # A seed is the resampling unit across domains: average its five effects
    # first, then bootstrap these five averages with replacement.
    pooled = {
        metric: {
            str(seed): statistics.fmean(
                rows[domain]["uniform_minus_frequency"][metric]["per_seed"][str(seed)]
                for domain in DOMAINS
            )
            for seed in SEEDS
        }
        for metric in METRICS
    }
    rows["five_domain_mean"] = {
        "label": LABELS["five_domain_mean"],
        "n": 5,
        "domains_per_seed": 5,
        "uniform_minus_frequency": {metric: paired_bootstrap(pooled[metric]) for metric in METRICS},
    }
    return rows


def signed(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:+.3f}".replace("+0.", "+.").replace("-0.", "-.")


def render_tables(rows: dict[str, Any]) -> tuple[str, str]:
    main, seeds = [], []
    for domain in (*DOMAINS, "five_domain_mean"):
        row = rows[domain]
        effects = row["uniform_minus_frequency"]
        cells = [row["label"]]
        for metric in ("breadth8", "pass8"):
            summary = effects[metric]
            interval = ",".join(signed(value) for value in summary["paired_bootstrap_percentile_95"])
            cells.extend((f"${signed(summary['mean'])}$", f"$[{interval}]$"))
        main.append(" & ".join(cells) + r" \\")
        cells = [row["label"]]
        for seed in SEEDS:
            pair = "/".join(signed(effects[metric]["per_seed"][str(seed)]) for metric in ("breadth8", "pass8"))
            cells.append(f"${pair}$")
        seeds.append(" & ".join(cells) + r" \\")
    return tuple("\n".join(lines) + "\n    \\bottomrule\n" for lines in (main, seeds))


def build() -> dict[str, Any]:
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_SHA256:
        raise ValueError("source bytes differ from the frozen September 4 paper snapshot")
    source = json.loads(raw)
    rows = analyze(source)
    return {
        "schema": "e120-primary-breadth-frozen-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_generated_at": source["generated_at"],
        "preregistration": str(PROTOCOL.relative_to(ROOT)),
        "preregistration_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "builder": str(Path(__file__).resolve().relative_to(ROOT)),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Complete Qwen2.5-0.5B primary block only; frozen step-3072 four-draw endpoint estimates; no live files read.",
        "contrast": "uniform key-balanced replay minus fresh-frequency replay",
        "metrics": {
            "breadth8": "Primary: (distinct@8 - pass@8)_uniform - (distinct@8 - pass@8)_frequency",
            "pass8": "Co-primary safety: pass@8_uniform - pass@8_frequency",
            "distinct8": "Supporting: raw distinct@8_uniform - raw distinct@8_frequency",
        },
        "seeds": list(SEEDS),
        "domain_order": list(DOMAINS),
        "bootstrap": {
            "method": "paired-seed percentile bootstrap",
            "confidence_level": 0.95,
            "ordered_resamples": 5**5,
            "enumeration": "All 5^5 ordered draws of five paired seeds with replacement; no Monte Carlo randomness.",
            "quantiles": "Linear interpolation at positions q*(3125-1), q in {0.025, 0.975}.",
            "five_domain_mean": "Equal-weight domain mean within each seed, followed by resampling five entire paired-seed vectors; n=5, not n=25.",
            "registration_boundary": "The protocol prespecifies paired bootstrap 95% intervals. Exact enumeration, percentile intervals, linear interpolation, and preserving the shared seed across domains are explicit implementation choices, not claimed as preregistered details.",
            "multiplicity": "Unadjusted descriptive estimation intervals.",
        },
        "rows": rows,
    }


def main() -> None:
    payload = build()
    table, seed_table = render_tables(payload["rows"])
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    TABLE.write_text(table, encoding="utf-8")
    SEED_TABLE.write_text(seed_table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()
