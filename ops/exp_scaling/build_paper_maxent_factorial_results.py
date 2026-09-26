#!/usr/bin/env python3
"""Build the paper-facing fixed semantic-MaxEnt factorial summary.

The completed Qwen2.5-0.5B design is a five-domain, five-seed 2x2 factorial:
matched Dr.GRPO, verified replay, semantic MaxEnt without replay, and verified
replay plus semantic MaxEnt.  PantryPlan's original semantic runs are excluded
because their canonical-action keys never reached the estimator; the registered
E85 repair cells replace them.  This script fails closed unless all 100 arm x
domain x seed terminal evaluations, including four K=8 draws, are present.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from build_paper_e78_terminal_results import _curve, _paired_summary, _signed


ROOT = Path(__file__).resolve().parents[2]
E78_LEDGER = ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json"
E81_LEDGER = (
    ROOT / "var/artifacts/e81_semantic_maxent_verified_replay_05b_jobs.json"
)
E83_LEDGER = ROOT / "var/artifacts/e83_semantic_maxent_without_replay_05b_jobs.json"
E85_LEDGER = ROOT / "var/artifacts/e85_pantry_semantic_repair_jobs.json"
OUTPUT = ROOT / "paper/results/maxent_factorial_05b.json"
EFFECT_TABLE = ROOT / "paper/results/maxent_factorial_05b_table_body.tex"
UNCERTAINTY_TABLE = (
    ROOT / "paper/results/maxent_factorial_05b_uncertainty_table_body.tex"
)

EXPECTED_DOMAINS = (
    "graph_coloring",
    "countdown",
    "python_factors",
    "mathir",
    "pantry_plan",
)
EXPECTED_SEEDS = (43, 44, 45, 46, 47)
EXPECTED_ARMS = ("control", "replay", "semantic_only", "semantic")
DOMAIN_TITLES = {
    "graph_coloring": "Graph coloring",
    "countdown": "Countdown",
    "python_factors": "Python factors",
    "mathir": "MathIR",
    "pantry_plan": "PantryPlan",
}
CONTRASTS = {
    "maxent_without_replay": ("semantic_only", "control"),
    "maxent_with_replay": ("semantic", "replay"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _register(
    run_dirs: dict[str, dict[str, dict[int, Path]]],
    *,
    arm: str,
    domain: str,
    seed: int,
    run_dir: str,
) -> None:
    slot = run_dirs[domain][arm]
    if seed in slot:
        raise RuntimeError(f"duplicate run for {domain}/{arm}/seed {seed}")
    slot[seed] = Path(run_dir)


def _effect_metrics(
    left: dict[str, float], right: dict[str, float]
) -> dict[str, float]:
    delta_pass = left["pass8"] - right["pass8"]
    delta_distinct = left["distinct8"] - right["distinct8"]
    return {
        "pass8": delta_pass,
        "distinct8": delta_distinct,
        "adjusted_breadth": delta_distinct - delta_pass,
    }


def _ci_cell(summary: dict[str, Any]) -> str:
    low, high = summary["student_t_95"]
    return rf"${_signed(summary['mean'])}\;[{_signed(low)},{_signed(high)}]$"


def main() -> None:
    ledgers = {
        "e78": _load(E78_LEDGER),
        "e81": _load(E81_LEDGER),
        "e83": _load(E83_LEDGER),
        "e85": _load(E85_LEDGER),
    }
    reference = ledgers["e78"]
    interval = int(reference["checkpoint_interval_steps"])
    target = int(reference["target_steps"])
    for name, payload in ledgers.items():
        if int(payload["checkpoint_interval_steps"]) != interval:
            raise RuntimeError(f"{name}: checkpoint interval mismatch")
        if int(payload["target_steps"]) != target:
            raise RuntimeError(f"{name}: target-step mismatch")

    run_dirs: dict[str, dict[str, dict[int, Path]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for run in reference["runs"]:
        _register(
            run_dirs,
            arm=str(run["arm"]),
            domain=str(run["domain"]),
            seed=int(run["seed"]),
            run_dir=str(run["run_dir"]),
        )
    for ledger_name, arm in (("e81", "semantic"), ("e83", "semantic_only")):
        for run in ledgers[ledger_name]["runs"]:
            if str(run["domain"]) == "pantry_plan":
                continue
            _register(
                run_dirs,
                arm=arm,
                domain=str(run["domain"]),
                seed=int(run["seed"]),
                run_dir=str(run["run_dir"]),
            )
    repair_parent_to_arm = {"e81": "semantic", "e83": "semantic_only"}
    for run in ledgers["e85"]["runs"]:
        parent = str(run["parent"])
        if parent not in repair_parent_to_arm:
            continue
        if str(run["domain"]) != "pantry_plan":
            raise RuntimeError(f"E85 repaired unexpected domain: {run['domain']}")
        _register(
            run_dirs,
            arm=repair_parent_to_arm[parent],
            domain="pantry_plan",
            seed=int(run["seed"]),
            run_dir=str(run["run_dir"]),
        )

    observed_domains = tuple(run_dirs)
    if set(observed_domains) != set(EXPECTED_DOMAINS):
        raise RuntimeError(f"unexpected domains: {observed_domains}")
    for domain in EXPECTED_DOMAINS:
        if set(run_dirs[domain]) != set(EXPECTED_ARMS):
            raise RuntimeError(
                f"{domain}: expected arms {EXPECTED_ARMS}, got {tuple(run_dirs[domain])}"
            )
        for arm in EXPECTED_ARMS:
            if tuple(sorted(run_dirs[domain][arm])) != EXPECTED_SEEDS:
                raise RuntimeError(
                    f"{domain}/{arm}: expected seeds {EXPECTED_SEEDS}, "
                    f"got {sorted(run_dirs[domain][arm])}"
                )

    curves: dict[str, dict[str, dict[int, dict[int, dict[str, float]]]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for domain in EXPECTED_DOMAINS:
        for arm in EXPECTED_ARMS:
            for seed, run_dir in run_dirs[domain][arm].items():
                curves[domain][arm][seed] = _curve(
                    run_dir, interval=interval, target=target
                )

    output: dict[str, Any] = {
        "schema": "maxent_factorial_05b_paper_results_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "model": "Qwen2.5-0.5B-Instruct",
            "domains": list(EXPECTED_DOMAINS),
            "arms": list(EXPECTED_ARMS),
            "paired_seeds": list(EXPECTED_SEEDS),
            "evaluation_draws": 4,
            "checkpoint_interval_steps": interval,
            "target_steps": target,
            "passes": int(reference["passes"]),
            "pantry_repair_parents": ["e81", "e83"],
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in (
                ("e78", E78_LEDGER),
                ("e81", E81_LEDGER),
                ("e83", E83_LEDGER),
                ("e85", E85_LEDGER),
            )
        },
        "uncertainty": (
            "two-sided 95% Student-t intervals over five paired training-seed "
            "effects (df=4); no pooling across domains"
        ),
        "domains": {},
    }

    effect_rows: list[str] = []
    uncertainty_rows: list[str] = []
    for domain in EXPECTED_DOMAINS:
        record: dict[str, Any] = {"arms": {}, "contrasts": {}}
        for arm in EXPECTED_ARMS:
            record["arms"][arm] = {
                "terminal_per_seed": {
                    str(seed): curves[domain][arm][seed][target]
                    for seed in EXPECTED_SEEDS
                }
            }

        per_contrast: dict[str, dict[int, dict[str, float]]] = {}
        for contrast, (left_arm, right_arm) in CONTRASTS.items():
            per_seed = {
                seed: _effect_metrics(
                    curves[domain][left_arm][seed][target],
                    curves[domain][right_arm][seed][target],
                )
                for seed in EXPECTED_SEEDS
            }
            per_contrast[contrast] = per_seed
            record["contrasts"][contrast] = {
                metric: _paired_summary(
                    {seed: values[metric] for seed, values in per_seed.items()}
                )
                for metric in ("pass8", "distinct8", "adjusted_breadth")
            }

        interaction = {
            seed: {
                metric: (
                    per_contrast["maxent_with_replay"][seed][metric]
                    - per_contrast["maxent_without_replay"][seed][metric]
                )
                for metric in ("pass8", "distinct8", "adjusted_breadth")
            }
            for seed in EXPECTED_SEEDS
        }
        record["contrasts"]["factorial_interaction"] = {
            metric: _paired_summary(
                {seed: values[metric] for seed, values in interaction.items()}
            )
            for metric in ("pass8", "distinct8", "adjusted_breadth")
        }
        output["domains"][domain] = record

        no_replay = record["contrasts"]["maxent_without_replay"]
        with_replay = record["contrasts"]["maxent_with_replay"]
        interaction_summary = record["contrasts"]["factorial_interaction"]
        effect_rows.append(
            "    "
            + DOMAIN_TITLES[domain]
            + " & "
            + " & ".join(
                _signed(summary["mean"])
                for summary in (
                    no_replay["pass8"],
                    no_replay["distinct8"],
                    no_replay["adjusted_breadth"],
                    with_replay["pass8"],
                    with_replay["distinct8"],
                    with_replay["adjusted_breadth"],
                    interaction_summary["distinct8"],
                )
            )
            + r" \\"
        )
        uncertainty_rows.append(
            "    "
            + DOMAIN_TITLES[domain]
            + " & "
            + " & ".join(
                _ci_cell(summary)
                for summary in (
                    no_replay["distinct8"],
                    with_replay["distinct8"],
                    interaction_summary["distinct8"],
                )
            )
            + r" \\"
        )

    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    EFFECT_TABLE.write_text(
        "\n".join([*effect_rows, r"    \bottomrule"]) + "\n", encoding="utf-8"
    )
    UNCERTAINTY_TABLE.write_text(
        "\n".join([*uncertainty_rows, r"    \bottomrule"]) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    print(f"wrote {EFFECT_TABLE.relative_to(ROOT)}")
    print(f"wrote {UNCERTAINTY_TABLE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
