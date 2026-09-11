#!/usr/bin/env python3
"""Aggregate E72 decoding-frontier cells into the three registered estimands.

Estimands (see docs/e72_baseline_and_decoding_control_suite.md, section 2):

E1  decoding frontier   breadth attainable at or above a given accuracy, where
                        the accuracy/breadth pair is traced by sweeping
                        temperature at the frozen terminal checkpoint
E2  temperature-repair  best breadth an arm reaches at ANY temperature, divided
    index               by the reference arm's temperature-one breadth
E3  coverage-budget     distinct@k and pass@k against sample budget k

Nothing is reported until the reproduction gate passes: every cell measured at
temperature 1.0, top_p 1.0, K=8 must reproduce the published terminal value for
that domain/arm/seed. A gate failure means the eval-only path is not the path
that produced the paper's numbers, which would make every other cell
uninterpretable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

CELL_TAG = re.compile(
    r"^T(?P<t>[0-9p]+)_p(?P<p>[0-9p]+)_K(?P<k>\d+)_d(?P<d>\d+)$"
)

# Absolute tolerance floors for the reproduction gate. Bounded rates and the
# unbounded mode count are held to different floors because a 0.02 discrepancy
# means something different for a probability than for a count.
GATE_FLOOR_RATE = 0.02
GATE_FLOOR_COUNT = 0.05
GATE_SE_MULTIPLE = 3.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def unformat_number(text: str) -> float:
    return float(text.replace("p", "."))


def metric_keys(k: int, split: str = "multi_answer") -> dict[str, str]:
    return {
        "greedy": f"eval/{split}/accuracy",
        "mean": f"eval/{split}/sampled_mean_at_{k}",
        "pass": f"eval/{split}/sampled_any_correct_at_{k}",
        "distinct": f"eval/{split}/sampled_distinct_correct_at_{k}",
    }


def read_cell(cell_dir: Path) -> dict[str, Any] | None:
    """Return one measured cell, or None when it never completed."""
    markers = sorted(cell_dir.glob("*/EVAL_ONLY_COMPLETE.json"))
    if not markers:
        return None
    marker = json.loads(markers[-1].read_text())
    attempt = markers[-1].parent
    metrics_path = attempt / "train_metrics.jsonl"
    if not metrics_path.is_file():
        return None

    k = int(marker["eval_mode_coverage_k"])
    keys = metric_keys(k)
    record: dict[str, Any] | None = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if keys["distinct"] in payload:
                record = payload
                break
    if record is None:
        return None

    return {
        "attempt": str(attempt),
        "k": k,
        "draws": int(marker["eval_mode_coverage_draws"]),
        "temperature": float(marker["eval_mode_coverage_temperature"]),
        "top_p": float(marker["eval_mode_coverage_top_p"]),
        "coverage_seed": int(marker["eval_mode_coverage_seed"]),
        "checkpoint": marker["pretrain"],
        "greedy": record.get(keys["greedy"]),
        "mean": record.get(keys["mean"]),
        "pass": record.get(keys["pass"]),
        "distinct": record.get(keys["distinct"]),
        "mean_draw_se": record.get(f"{keys['mean']}_draw_se"),
        "pass_draw_se": record.get(f"{keys['pass']}_draw_se"),
        "distinct_draw_se": record.get(f"{keys['distinct']}_draw_se"),
    }


def collect(root: Path, stages: Iterable[str]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    frontier_root = root / "var" / "data" / "e72_frontier"
    for stage in stages:
        stage_root = frontier_root / stage
        if not stage_root.is_dir():
            continue
        for cell_dir in sorted(stage_root.glob("*/*/*")):
            match = CELL_TAG.match(cell_dir.name)
            if match is None:
                continue
            domain = cell_dir.parent.parent.name
            arm_seed = cell_dir.parent.name
            arm, _, seed_text = arm_seed.rpartition("_s")
            measured = read_cell(cell_dir)
            if measured is None:
                continue
            measured.update(
                {
                    "stage": stage,
                    "domain": domain,
                    "arm": arm,
                    "seed": int(seed_text),
                    "cell": cell_dir.name,
                    "cell_dir": str(cell_dir),
                }
            )
            cells.append(measured)
    return cells


def base_pass0_reference(
    manifest: dict[str, Any], root: Path
) -> dict[tuple[str, str], dict[str, Any]]:
    """Pass-0 reference for each (domain, base arm), from the frozen curves.

    A base cell measures the pre-RL policy, so its published counterpart is the
    cohort's step-0 row rather than a terminal one. Rows are restricted to the
    seeds that ran on the same GPU model, because pass-0 values differ between
    GPU models and averaging across them would blur the reference the cell is
    checked against.
    """
    nodes = {
        (run["domain"], int(run["seed"]), run["curve_arm_label"]): run["source_node"]
        for run in manifest["runs"]
        if run.get("curve_arm_label")
    }
    reference: dict[tuple[str, str], dict[str, Any]] = {}
    for run in manifest["runs"]:
        arm = str(run["arm"])
        if not arm.startswith("base_"):
            continue
        node = str(run["source_node"])
        curve = json.loads((root / "var" / "artifacts" / run["curve_artifact"]).read_text())
        rows = [
            row
            for row in curve
            if row.get("split") == "multi_answer"
            and int(row["step"]) == 0
            and nodes.get((run["domain"], int(row["seed"]), str(row["arm"]))) == node
        ]
        if not rows:
            continue
        def mean(key: str) -> float | None:
            values = [row[key] for row in rows if row.get(key) is not None]
            return sum(values) / len(values) if values else None

        reference[(run["domain"], arm)] = {
            "greedy": mean("greedy"),
            "mean8": mean("mean8"),
            "pass8": mean("pass8"),
            "distinct8": mean("distinct8"),
            "mean8_draw_se": mean("mean8_draw_se"),
            "pass8_draw_se": mean("pass8_draw_se"),
            "distinct8_draw_se": mean("distinct8_draw_se"),
            "contributing_rows": len(rows),
        }
    return reference


def reproduction_gate(
    cells: list[dict[str, Any]],
    manifest: dict[str, Any],
    base_reference: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    base_reference = base_reference or {}
    published = {
        (run["domain"], run["arm"], int(run["seed"])): run["published_terminal"]
        for run in manifest["runs"]
        if run.get("published_terminal")
    }
    checks: list[dict[str, Any]] = []
    for cell in cells:
        if not (
            cell["k"] == 8
            and math.isclose(cell["temperature"], 1.0)
            and math.isclose(cell["top_p"], 1.0)
        ):
            continue
        if str(cell["arm"]).startswith("base_"):
            # A base cell reproduces the frozen pass-0 row, not a terminal one.
            reference = base_reference.get((cell["domain"], cell["arm"]))
        else:
            reference = published.get((cell["domain"], cell["arm"], cell["seed"]))
        if reference is None:
            checks.append(
                {
                    "cell_dir": cell["cell_dir"],
                    "status": "no_published_reference",
                    "passed": False,
                }
            )
            continue
        for metric, published_key, se_key, floor in (
            ("mean", "mean8", "mean8_draw_se", GATE_FLOOR_RATE),
            ("pass", "pass8", "pass8_draw_se", GATE_FLOOR_RATE),
            ("distinct", "distinct8", "distinct8_draw_se", GATE_FLOOR_COUNT),
            ("greedy", "greedy", None, GATE_FLOOR_RATE),
        ):
            measured = cell.get(metric)
            expected = reference.get(published_key)
            if measured is None or expected is None:
                checks.append(
                    {
                        "domain": cell["domain"],
                        "arm": cell["arm"],
                        "seed": cell["seed"],
                        "metric": metric,
                        "status": "missing_value",
                        "passed": False,
                    }
                )
                continue
            se = float(reference.get(se_key) or 0.0) if se_key else 0.0
            tolerance = max(GATE_SE_MULTIPLE * se, floor)
            delta = float(measured) - float(expected)
            checks.append(
                {
                    "domain": cell["domain"],
                    "arm": cell["arm"],
                    "seed": cell["seed"],
                    "metric": metric,
                    "measured": float(measured),
                    "published": float(expected),
                    "delta": delta,
                    "tolerance": tolerance,
                    "passed": abs(delta) <= tolerance,
                }
            )
    failures = [check for check in checks if not check["passed"]]
    return {
        "checked": len(checks),
        "failed": len(failures),
        "failures": failures[:40],
        "passed": bool(checks) and not failures,
        "tolerance_rule": (
            f"abs(delta) <= max({GATE_SE_MULTIPLE} * published draw SE, "
            f"{GATE_FLOOR_RATE} for rates / {GATE_FLOOR_COUNT} for mode counts)"
        ),
    }


def seed_mean(values: list[float]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def frontier_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """E1 and E2 over the untruncated (top_p = 1) temperature sweep."""
    grouped: dict[tuple[str, str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        if not math.isclose(cell["top_p"], 1.0):
            continue
        grouped[(cell["domain"], cell["arm"], cell["temperature"], cell["k"])].append(
            cell
        )

    points: list[dict[str, Any]] = []
    for (domain, arm, temperature, k), rows in sorted(grouped.items()):
        points.append(
            {
                "domain": domain,
                "arm": arm,
                "temperature": temperature,
                "k": k,
                "seeds": sorted(row["seed"] for row in rows),
                "n_seeds": len(rows),
                "mean_at_k": seed_mean([row["mean"] for row in rows]),
                "pass_at_k": seed_mean([row["pass"] for row in rows]),
                "distinct_at_k": seed_mean([row["distinct"] for row in rows]),
                "greedy": seed_mean([row["greedy"] for row in rows]),
                "per_seed": {
                    row["seed"]: {
                        "mean_at_k": row["mean"],
                        "pass_at_k": row["pass"],
                        "distinct_at_k": row["distinct"],
                    }
                    for row in sorted(rows, key=lambda item: item["seed"])
                },
            }
        )

    # E2: best breadth at any temperature, against the reference arm at T=1.
    repair: list[dict[str, Any]] = []
    by_domain_arm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        if point["k"] != 8:
            continue
        by_domain_arm[(point["domain"], point["arm"])].append(point)
    for (domain, arm), arm_points in sorted(by_domain_arm.items()):
        usable = [p for p in arm_points if p["distinct_at_k"] is not None]
        if not usable:
            continue
        best = max(usable, key=lambda item: item["distinct_at_k"])
        at_one = next(
            (p for p in usable if math.isclose(p["temperature"], 1.0)), None
        )
        reference = next(
            (
                p
                for p in by_domain_arm.get((domain, "xgrpo"), [])
                if math.isclose(p["temperature"], 1.0)
                and p["distinct_at_k"] is not None
            ),
            None,
        )
        repair.append(
            {
                "domain": domain,
                "arm": arm,
                "best_temperature": best["temperature"],
                "best_distinct_at_8": best["distinct_at_k"],
                "accuracy_at_best": best["mean_at_k"],
                "distinct_at_8_at_T1": at_one["distinct_at_k"] if at_one else None,
                "accuracy_at_T1": at_one["mean_at_k"] if at_one else None,
                "reference_arm": "xgrpo",
                "reference_distinct_at_8_at_T1": (
                    reference["distinct_at_k"] if reference else None
                ),
                "repair_index_rho": (
                    best["distinct_at_k"] / reference["distinct_at_k"]
                    if reference and reference["distinct_at_k"]
                    else None
                ),
            }
        )
    return {"points": points, "temperature_repair": repair}


def write_report(payload: dict[str, Any], path: Path) -> None:
    """Human-readable frontier report: the estimands, not the raw cells."""
    gate = payload["reproduction_gate"]
    lines = [
        "# E72 decoding frontier (live)",
        "",
        f"Cells measured: {payload['cells_measured']} "
        f"(stage A target {payload['stage_a_expected']}).",
        f"Reproduction gate: {'PASS' if gate['passed'] else 'FAIL'} "
        f"({gate['checked'] - gate['failed']}/{gate['checked']} checks).",
        f"Tolerance: {gate['tolerance_rule']}.",
        "",
        "## E2 temperature-repair index",
        "",
        "`rho` is the best breadth an arm reaches at ANY temperature, divided by",
        "xGRPO's temperature-one breadth. `rho < 1` means no decoding setting",
        "recovers the treatment's breadth. The accuracy column is what that best",
        "breadth cost.",
        "",
        "| domain | arm | best T | distinct@8 at best T | accuracy there | distinct@8 at T=1 | rho |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["frontier"]["temperature_repair"]:
        def fmt(value: float | None, digits: int = 3) -> str:
            return "n/a" if value is None else f"{value:.{digits}f}"

        lines.append(
            f"| {row['domain']} | {row['arm']} | {row['best_temperature']:g} | "
            f"{fmt(row['best_distinct_at_8'])} | {fmt(row['accuracy_at_best'])} | "
            f"{fmt(row['distinct_at_8_at_T1'])} | {fmt(row['repair_index_rho'], 2)} |"
        )

    lines += [
        "",
        "## E1 frontier points (five-seed means)",
        "",
        "| domain | arm | T | mean@8 | pass@8 | distinct@8 | seeds |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for point in payload["frontier"]["points"]:
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        lines.append(
            f"| {point['domain']} | {point['arm']} | {point['temperature']:g} | "
            f"{fmt(point['mean_at_k'])} | {fmt(point['pass_at_k'])} | "
            f"{fmt(point['distinct_at_k'])} | {point['n_seeds']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="a,b,c")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "var" / "artifacts" / "e72_frontier_source_runs.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "var" / "artifacts" / "e72_decoding_frontier_summary.json",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=root / "var" / "artifacts" / "e72_decoding_frontier_cells.jsonl",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    stages = [part for part in args.stages.split(",") if part]
    cells = collect(root, stages)
    base_reference = base_pass0_reference(manifest, root)
    gate = reproduction_gate(cells, manifest, base_reference)
    summary = frontier_summary([cell for cell in cells if cell["k"] == 8])

    expected_stage_a = len(manifest["runs"]) * 6
    payload = {
        "schema": "e72_decoding_frontier_summary_v1",
        "stages": stages,
        "cells_measured": len(cells),
        "stage_a_expected": expected_stage_a,
        "reproduction_gate": gate,
        "frontier": summary,
    }

    with args.raw.open("w", encoding="utf-8") as sink:
        for cell in sorted(
            cells,
            key=lambda item: (
                item["domain"],
                item["arm"],
                item["seed"],
                item["k"],
                item["temperature"],
                item["top_p"],
            ),
        ):
            sink.write(json.dumps(cell, sort_keys=True) + "\n")

    handle, temporary = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    with os.fdopen(handle, "w", encoding="utf-8") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n")
    os.replace(temporary, args.output)
    report_path = (
        repo_root() / "paper" / "results" / "e72_decoding_frontier_live.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_report(payload, report_path)

    print(
        f"[e72-frontier] cells={len(cells)} "
        f"gate_checked={gate['checked']} gate_failed={gate['failed']} "
        f"gate_passed={gate['passed']}"
    )
    for failure in gate["failures"][:10]:
        print(f"[e72-frontier] gate failure: {failure}")
    print(f"[e72-frontier] wrote {args.output} and {args.raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
