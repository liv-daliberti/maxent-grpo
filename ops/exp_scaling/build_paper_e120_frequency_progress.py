#!/usr/bin/env python3
"""Build partial-safe E120 fresh-frequency versus uniform-replay results."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e120r1_frequency_weighted_replay_jobs.json"
OUTPUT = ROOT / "paper/results/e120_frequency_progress.json"
TABLE = ROOT / "paper/results/e120_frequency_progress_table_body.tex"
TARGET_STEP = 3072
DRAWS = {0, 1, 2, 3}
MODELS = ("qwen05b", "falcon1b", "qwen3b")
DOMAINS = ("graph_coloring", "countdown", "python_factors", "mathir", "pantry_plan")
DOMAIN_LABELS = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail_lines(path: Path, max_lines: int = 96, chunk_bytes: int = 1_048_576) -> list[str]:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        chunks: list[bytes] = []
        newlines = 0
        while position > 0 and newlines <= max_lines:
            take = min(chunk_bytes, position)
            position -= take
            handle.seek(position)
            chunk = handle.read(take)
            chunks.append(chunk)
            newlines += chunk.count(b"\n")
    return b"".join(reversed(chunks)).decode("utf-8", errors="replace").splitlines()[-max_lines:]


def endpoint(run_dir: str) -> dict[str, float] | None:
    draws: dict[int, dict[str, Any]] = {}
    for path in sorted(Path(run_dir).glob("debug_job*/eval_mode_coverage_draws.jsonl")):
        for line in tail_lines(path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row.get("step", -1)) == TARGET_STEP and row.get("draw_index") is not None:
                draws[int(row["draw_index"])] = row
    if set(draws) != DRAWS:
        return None
    return {
        "pass8": statistics.fmean(
            float(row["metrics"]["any_correct_at_k"]) for row in draws.values()
        ),
        "distinct8": statistics.fmean(
            float(row["metrics"]["distinct_correct_modes_at_k"]) for row in draws.values()
        ),
    }


def signed(value: float) -> str:
    return f"{value:+.3f}".replace("+0.", "+.").replace("-0.", "-.")


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if ledger.get("schema") != "e120r1_frequency_weighted_replay_jobs_v1":
        raise RuntimeError("E120-R1 ledger schema drifted")
    comparators = {
        (row["model_key"], row["domain"], int(row["seed"])): endpoint(row["run_dir"])
        for row in ledger["comparators"]
    }
    if any(value is None for value in comparators.values()):
        raise RuntimeError("a registered uniform-replay comparator is not terminal")
    fresh = {
        (row["model_key"], row["domain"], int(row["seed"])): endpoint(row["run_dir"])
        for row in ledger["runs"]
    }

    cells: dict[str, dict[str, Any]] = {}
    terminal_total = 0
    complete_blocks = 0
    for model in MODELS:
        cells[model] = {}
        for domain in DOMAINS:
            registered = sorted(
                seed for m, d, seed in fresh if m == model and d == domain
            )
            if not registered:
                continue
            terminal = [seed for seed in registered if fresh[(model, domain, seed)] is not None]
            terminal_total += len(terminal)
            effects = {
                metric: [
                    fresh[(model, domain, seed)][metric] - comparators[(model, domain, seed)][metric]
                    for seed in terminal
                ]
                for metric in ("pass8", "distinct8")
            }
            contrast: dict[str, Any] = {}
            if terminal:
                for metric, values in effects.items():
                    summary: dict[str, Any] = {
                        "per_seed": {
                            str(seed): value for seed, value in zip(terminal, values)
                        },
                        "mean": statistics.fmean(values),
                    }
                    if len(terminal) == len(registered) == 5:
                        half = 2.776445105 * statistics.stdev(values) / math.sqrt(5)
                        summary["student_t_95"] = [
                            summary["mean"] - half,
                            summary["mean"] + half,
                        ]
                    contrast[metric] = summary
            record = {
                "registered_seeds": registered,
                "terminal_seeds": terminal,
                "n": len(terminal),
                "complete_block": len(terminal) == len(registered) == 5,
                "fresh_frequency": {
                    str(seed): fresh[(model, domain, seed)] for seed in terminal
                },
                "uniform_key_replay": {
                    str(seed): comparators[(model, domain, seed)] for seed in terminal
                },
                "fresh_frequency_minus_uniform": contrast if terminal else {},
            }
            complete_blocks += int(record["complete_block"])
            cells[model][domain] = record

    payload = {
        "schema": "e120-frequency-progress-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "exact step-3072 four-draw endpoints only; complete five-seed blocks "
            "receive unadjusted descriptive Student-t intervals, while partial seed "
            "prefixes receive no interval or confirmatory interpretation"
        ),
        "contrast": "fresh-frequency replay minus uniform key-balanced replay",
        "source": str(LEDGER.relative_to(ROOT)),
        "source_sha256": digest(LEDGER),
        "terminal_cells": terminal_total,
        "registered_cells": len(ledger["runs"]),
        "complete_blocks": complete_blocks,
        "cells": cells,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    qwen = cells["qwen05b"]
    lines = []
    for domain in DOMAINS:
        row = qwen[domain]
        seeds = ",".join(str(seed) for seed in row["terminal_seeds"]) or "--"
        if row["n"]:
            effects = row["fresh_frequency_minus_uniform"]
            pass_effect = signed(effects["pass8"]["mean"])
            distinct_effect = signed(effects["distinct8"]["mean"])
        else:
            pass_effect = distinct_effect = "--"
        lines.append(
            f"{DOMAIN_LABELS[domain]} & {row['n']} & {seeds} & "
            f"{pass_effect} & {distinct_effect} \\\\"
        )
    lines.append(r"    \bottomrule")
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(TABLE)
    print(f"terminal={terminal_total}/{len(ledger['runs'])} complete_blocks={complete_blocks}")


if __name__ == "__main__":
    main()
