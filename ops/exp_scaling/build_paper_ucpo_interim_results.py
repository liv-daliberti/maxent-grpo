#!/usr/bin/env python3
"""Build the paper-facing balanced interim UCPO comparison.

The final preregistered UCPO estimand uses five paired seeds.  While that cohort
is running, this builder reports only the largest seed set whose UCPO cells are
terminal in every registered domain.  The same seeds are then used for the
already-completed matched Dr.GRPO and x-Mode Dr.GRPO arms.  This prevents an
unbalanced completion pattern from changing the domain comparison.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
UCPO_LEDGER = ROOT / "var/artifacts/e97_ucpo_05b_jobs.json"
CORE_LEDGER = ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json"
OUTPUT = ROOT / "paper/results/ucpo_interim_05b.json"
TABLE_BODY = ROOT / "paper/results/ucpo_interim_05b_table_body.tex"
EXPECTED_DOMAINS = ("graph_coloring", "python_factors", "pantry_plan")
EXPECTED_DRAWS = 4
METRIC_FIELDS = {
    "pass8": "any_correct_at_k",
    "distinct8": "distinct_correct_modes_at_k",
}
DOMAIN_TITLES = {
    "graph_coloring": "Graph coloring",
    "python_factors": "Python factors",
    "pantry_plan": "PantryPlan",
}


def _finite(value: Any, *, where: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"{where}: expected a finite number, got {value!r}")
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completion_marker(run_dir: Path, *, target: int) -> Path | None:
    marker = run_dir / "TRAINING_COMPLETE.json"
    if not marker.is_file():
        return None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    terminal_step = payload.get("terminal_step")
    if not isinstance(terminal_step, int) or terminal_step < target:
        raise RuntimeError(f"{marker}: terminal step {terminal_step!r} is below {target}")
    return marker


def _terminal_metrics(
    run_dir: Path,
    *,
    target: int,
) -> tuple[dict[str, float], list[Path]]:
    """Return four-draw terminal means and the exact sampled-evaluation files."""

    records: dict[int, dict[str, Any]] = {}
    paths = sorted(run_dir.glob("debug_job*/eval_mode_coverage_draws.jsonl"))
    if not paths:
        raise RuntimeError(f"{run_dir}: no sampled-evaluation log")
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if (
                    row.get("evaluation_kind") == "fixed_seed_sampled_k_neutral"
                    and row.get("step") == target
                    and isinstance(row.get("draw_index"), int)
                    and row["draw_index"] in range(EXPECTED_DRAWS)
                    and isinstance(row.get("metrics"), dict)
                ):
                    records[row["draw_index"]] = row["metrics"]
    if tuple(sorted(records)) != tuple(range(EXPECTED_DRAWS)):
        raise RuntimeError(
            f"{run_dir}: terminal draw indexes {sorted(records)}, expected 0--3"
        )
    return (
        {
            output_name: statistics.fmean(
                _finite(
                    records[draw].get(source_name),
                    where=f"{run_dir}:{target}:draw{draw}:{source_name}",
                )
                for draw in range(EXPECTED_DRAWS)
            )
            for output_name, source_name in METRIC_FIELDS.items()
        },
        paths,
    )


def _plain(value: float) -> str:
    return f"{value:.3f}".replace("0.", ".", 1) if 0 <= value < 1 else f"{value:.3f}"


def _signed(value: float) -> str:
    return f"{value:+.3f}".replace("+0.", "+.").replace("-0.", "-.")


def main() -> None:
    ucpo_ledger = json.loads(UCPO_LEDGER.read_text(encoding="utf-8"))
    core_ledger = json.loads(CORE_LEDGER.read_text(encoding="utf-8"))
    domains = tuple(str(domain) for domain in ucpo_ledger["domains"])
    if set(domains) != set(EXPECTED_DOMAINS):
        raise RuntimeError(f"unexpected UCPO domains: {domains}")
    target = int(ucpo_ledger["target_steps"])
    expected_seeds = tuple(int(seed) for seed in ucpo_ledger["seeds"])

    ucpo_runs: dict[tuple[str, int], dict[str, Any]] = {}
    terminal_by_domain: dict[str, set[int]] = defaultdict(set)
    completion_markers: dict[tuple[str, int], Path] = {}
    for run in ucpo_ledger["runs"]:
        domain = str(run["domain"])
        seed = int(run["seed"])
        key = (domain, seed)
        if key in ucpo_runs:
            raise RuntimeError(f"duplicate UCPO cell {key}")
        ucpo_runs[key] = run
        marker = _completion_marker(Path(run["run_dir"]), target=target)
        if marker is not None:
            terminal_by_domain[domain].add(seed)
            completion_markers[key] = marker

    balanced_seeds = tuple(
        seed
        for seed in expected_seeds
        if all(seed in terminal_by_domain[domain] for domain in EXPECTED_DOMAINS)
    )
    if not balanced_seeds:
        raise RuntimeError("UCPO has no seed terminal in all three registered domains")

    core_runs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for run in core_ledger["runs"]:
        key = (str(run["domain"]), str(run["arm"]), int(run["seed"]))
        if key in core_runs:
            raise RuntimeError(f"duplicate completed-core cell {key}")
        core_runs[key] = run

    input_paths: set[Path] = {UCPO_LEDGER, CORE_LEDGER}
    domain_output: dict[str, Any] = {}
    for domain in EXPECTED_DOMAINS:
        per_seed: dict[str, dict[str, dict[str, float]]] = {}
        for seed in balanced_seeds:
            arm_runs = {
                "control": Path(core_runs[(domain, "control", seed)]["run_dir"]),
                "ucpo": Path(ucpo_runs[(domain, seed)]["run_dir"]),
                "xmode": Path(core_runs[(domain, "replay", seed)]["run_dir"]),
            }
            seed_values: dict[str, dict[str, float]] = {}
            for arm, run_dir in arm_runs.items():
                metrics, source_paths = _terminal_metrics(run_dir, target=target)
                seed_values[arm] = metrics
                input_paths.update(source_paths)
            input_paths.add(completion_markers[(domain, seed)])
            per_seed[str(seed)] = seed_values

        arm_means = {
            arm: {
                metric: statistics.fmean(
                    per_seed[str(seed)][arm][metric] for seed in balanced_seeds
                )
                for metric in METRIC_FIELDS
            }
            for arm in ("control", "ucpo", "xmode")
        }
        for arm in arm_means:
            arm_means[arm]["adjusted_breadth8"] = (
                arm_means[arm]["distinct8"] - arm_means[arm]["pass8"]
            )
        contrasts = {}
        for name, left, right in (
            ("ucpo_minus_control", "ucpo", "control"),
            ("xmode_minus_ucpo", "xmode", "ucpo"),
        ):
            contrasts[name] = {
                metric: arm_means[left][metric] - arm_means[right][metric]
                for metric in ("pass8", "distinct8", "adjusted_breadth8")
            }
            contrasts[name]["per_seed"] = {
                str(seed): {
                    "pass8": per_seed[str(seed)][left]["pass8"]
                    - per_seed[str(seed)][right]["pass8"],
                    "distinct8": per_seed[str(seed)][left]["distinct8"]
                    - per_seed[str(seed)][right]["distinct8"],
                    "adjusted_breadth8": (
                        per_seed[str(seed)][left]["distinct8"]
                        - per_seed[str(seed)][left]["pass8"]
                    )
                    - (
                        per_seed[str(seed)][right]["distinct8"]
                        - per_seed[str(seed)][right]["pass8"]
                    ),
                }
                for seed in balanced_seeds
            }
        domain_output[domain] = {
            "title": DOMAIN_TITLES[domain],
            "arms": arm_means,
            "contrasts": contrasts,
            "per_seed": per_seed,
        }

    output = {
        "schema": "ucpo_interim_paper_results_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "interim; the preregistered five-seed UCPO estimand is incomplete",
        "selection_rule": (
            "largest registered seed set with terminal UCPO cells in every domain; "
            "the same seeds are used for completed Dr.GRPO and x-Mode Dr.GRPO"
        ),
        "design": {
            "model": str(ucpo_ledger["model"]),
            "domains": list(EXPECTED_DOMAINS),
            "expected_seeds": list(expected_seeds),
            "balanced_terminal_seeds": list(balanced_seeds),
            "balanced_n": len(balanced_seeds),
            "ucpo_terminal_cells": sum(len(seeds) for seeds in terminal_by_domain.values()),
            "ucpo_registered_cells": len(ucpo_ledger["runs"]),
            "terminal_step": target,
            "evaluation_draws": EXPECTED_DRAWS,
        },
        "domains": domain_output,
        "input_sha256": {
            str(path.resolve()): _sha256(path) for path in sorted(input_paths)
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    rows = []
    for domain in EXPECTED_DOMAINS:
        entry = domain_output[domain]
        control = entry["arms"]["control"]
        ucpo = entry["arms"]["ucpo"]
        xmode = entry["arms"]["xmode"]
        contrast = entry["contrasts"]["xmode_minus_ucpo"]
        rows.append(
            "    "
            + " & ".join(
                [
                    entry["title"],
                    str(len(balanced_seeds)),
                    _plain(control["pass8"]),
                    _plain(control["distinct8"]),
                    _plain(ucpo["pass8"]),
                    _plain(ucpo["distinct8"]),
                    _plain(xmode["pass8"]),
                    _plain(xmode["distinct8"]),
                    _signed(contrast["distinct8"]),
                    _signed(contrast["adjusted_breadth8"]),
                ]
            )
            + r" \\"
        )
    TABLE_BODY.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        f"wrote {OUTPUT} and {TABLE_BODY}: balanced seeds {list(balanced_seeds)}, "
        f"UCPO terminal cells {output['design']['ucpo_terminal_cells']}/"
        f"{output['design']['ucpo_registered_cells']}"
    )


if __name__ == "__main__":
    main()
