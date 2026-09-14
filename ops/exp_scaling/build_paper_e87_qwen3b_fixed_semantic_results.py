#!/usr/bin/env python3
"""Freeze the completed seed-70 Qwen2.5-3B fixed-semantic block for the paper.

This is intentionally a descriptive single-seed result. It validates and
reports the matched Dr.GRPO, Re:Dr.GRPO, and fixed Semantic MaxEnt plus
Re:Dr.GRPO endpoints on all five static domains, but never constructs an
interval or promotes the observation to a five-seed estimand.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from build_paper_e78_terminal_results import _curve


ROOT = Path(__file__).resolve().parents[2]
PARENT_LEDGER = (
    ROOT / "var/artifacts/e80r1_qwen3b_aligned_verified_replay_jobs.json"
)
TREATMENT_LEDGER = (
    ROOT / "var/artifacts/e87_qwen3b_semantic_maxent_seed70_jobs.json"
)
OUTPUT = ROOT / "paper/results/e87_qwen3b_fixed_semantic_seed70.json"
TABLE_BODY = (
    ROOT / "paper/results/e87_qwen3b_fixed_semantic_seed70_table_body.tex"
)
DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
DOMAIN_LABELS = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
}
SEED = 70
METHOD_ARMS = {
    "drgrpo": "control",
    "replay_grpo": "replay",
    "replay_semantic_maxent": "semantic",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _decimal(value: float) -> str:
    rendered = f"{value:.3f}"
    if rendered.startswith("0."):
        return rendered[1:]
    if rendered.startswith("-0."):
        return "-" + rendered[2:]
    return rendered


def main() -> None:
    parent = _load(PARENT_LEDGER)
    treatment = _load(TREATMENT_LEDGER)
    if parent.get("schema") != "e80r1_qwen3b_aligned_verified_replay_jobs_v1":
        raise RuntimeError("Qwen2.5-3B parent ledger contract drifted")
    if treatment.get("schema") != "e87_qwen3b_semantic_maxent_seed70_jobs_v1":
        raise RuntimeError("Qwen2.5-3B fixed-semantic ledger contract drifted")
    if treatment.get("seeds") != [SEED]:
        raise RuntimeError("fixed-semantic result is not the registered seed-70 block")
    if tuple(treatment.get("domains", ())) != DOMAINS:
        raise RuntimeError("fixed-semantic domain order drifted")
    for field in ("train_rows", "target_steps", "checkpoint_interval_steps", "passes"):
        if int(parent[field]) != int(treatment[field]):
            raise RuntimeError(f"parent/treatment {field} mismatch")
    if float(treatment.get("semantic_coefficient", -1)) != 0.1:
        raise RuntimeError("fixed semantic coefficient drifted")
    if float(treatment.get("replay_weight", -1)) != 0.1:
        raise RuntimeError("fixed replay weight drifted")

    runs: dict[tuple[str, str], Path] = {}
    for run in parent["runs"]:
        if int(run["seed"]) == SEED and run.get("arm") in {"control", "replay"}:
            runs[(str(run["domain"]), str(run["arm"]))] = Path(run["run_dir"])
    for run in treatment["runs"]:
        if int(run["seed"]) != SEED or run.get("arm") != "semantic":
            raise RuntimeError("unexpected fixed-semantic run identity")
        runs[(str(run["domain"]), "semantic")] = Path(run["run_dir"])
    expected = {
        (domain, arm) for domain in DOMAINS for arm in METHOD_ARMS.values()
    }
    if set(runs) != expected:
        raise RuntimeError(f"incomplete seed-70 method grid: {sorted(set(runs) ^ expected)}")

    target = int(treatment["target_steps"])
    interval = int(treatment["checkpoint_interval_steps"])
    domains: dict[str, Any] = {}
    rows: list[str] = []
    for domain in DOMAINS:
        methods = {
            method: _curve(runs[(domain, arm)], interval=interval, target=target)[target]
            for method, arm in METHOD_ARMS.items()
        }
        replay = methods["replay_grpo"]
        fixed = methods["replay_semantic_maxent"]
        delta = {
            metric: fixed[metric] - replay[metric]
            for metric in ("pass8", "mean8", "distinct8")
        }
        domains[domain] = {"methods": methods, "fixed_minus_replay": delta}
        rows.append(
            f"{DOMAIN_LABELS[domain]} & "
            f"{_decimal(replay['pass8'])} & {_decimal(replay['distinct8'])} & "
            f"{_decimal(fixed['pass8'])} & {_decimal(fixed['distinct8'])} & "
            f"{_decimal(delta['pass8'])} & {_decimal(delta['distinct8'])} \\\\"
        )

    output = {
        "schema": "paper-e87-qwen3b-fixed-semantic-seed70-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "terminal five-domain single-seed descriptive evidence",
        "model": "Qwen2.5-3B-Instruct",
        "seed": SEED,
        "n": 1,
        "training_pass": 8.0,
        "target_steps": target,
        "evaluation_draws": 4,
        "semantic_coefficient": float(treatment["semantic_coefficient"]),
        "replay_weight": float(treatment["replay_weight"]),
        "methods": list(METHOD_ARMS),
        "uncertainty": "none; one training seed, no interval or significance claim",
        "domains": domains,
        "sources": {
            "parent_ledger": {
                "path": str(PARENT_LEDGER.relative_to(ROOT)),
                "sha256": _sha256(PARENT_LEDGER),
            },
            "treatment_ledger": {
                "path": str(TREATMENT_LEDGER.relative_to(ROOT)),
                "sha256": _sha256(TREATMENT_LEDGER),
            },
        },
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    rows.append(r"\bottomrule")
    TABLE_BODY.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"wrote {TABLE_BODY}")


if __name__ == "__main__":
    main()
