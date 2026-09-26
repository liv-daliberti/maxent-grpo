#!/usr/bin/env python3
"""Apply the frozen E76 Stage A or Stage B validation selector."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

PASS_KEY = "eval/multi_answer/sampled_any_correct_at_8"
DISTINCT_KEY = "eval/multi_answer/sampled_distinct_correct_at_8"
LEDGERS = {
    "a": "var/artifacts/e76_tuned_scale_stage_a_jobs.json",
    "b": "var/artifacts/e76_tuned_scale_stage_b_jobs.json",
}
OUTPUTS = {
    "a": "var/artifacts/e76_tuned_scale_stage_a_selection.json",
    "b": "var/artifacts/e76_tuned_scale_stage_b_selection.json",
}
PASS_FLOOR = 0.02


def repo_root() -> Path:
    override = os.environ.get("E76_REPO_ROOT")
    return Path(override).resolve() if override else Path(__file__).resolve().parents[2]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as sink:
            json.dump(payload, sink, indent=2, sort_keys=True)
            sink.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_evaluations(run: dict[str, Any]) -> dict[int, dict[str, float]]:
    run_dir = Path(run["run_dir"])
    paths = sorted(
        (Path(path) for path in glob.glob(str(run_dir / "debug_job*" / "train_metrics.jsonl"))),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    if not paths:
        raise ValueError(f"{run['run_stamp']}: no train_metrics.jsonl")
    values: dict[int, list[tuple[float, float]]] = defaultdict(list)
    deepest = 0
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("misc/global_step") is not None:
                deepest = max(deepest, int(row["misc/global_step"]))
            if PASS_KEY not in row or DISTINCT_KEY not in row:
                continue
            step = int(row.get("misc/global_step", -1))
            pair = (float(row[PASS_KEY]), float(row[DISTINCT_KEY]))
            if not all(math.isfinite(value) for value in pair):
                raise ValueError(f"{run['run_stamp']}: non-finite validation metric at {step}")
            values[step].append(pair)
    target = int(run["target_steps"])
    if deepest < target:
        raise ValueError(f"{run['run_stamp']}: incomplete ({deepest} < {target})")
    resolved: dict[int, dict[str, float]] = {}
    for step, pairs in values.items():
        first = pairs[0]
        if any(abs(pair[0] - first[0]) > 1e-7 or abs(pair[1] - first[1]) > 1e-7 for pair in pairs[1:]):
            raise ValueError(f"{run['run_stamp']}: conflicting validation metrics at step {step}")
        resolved[step] = {"pass_at_8": first[0], "distinct_at_8": first[1]}
    if target not in resolved:
        raise ValueError(f"{run['run_stamp']}: no terminal validation at step {target}")
    return resolved


def cell_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return (
        run["model"], run["domain"], run["arm"], float(run["lr"]),
        float(run["beta"]), run.get("dose"), int(run["seed"]),
    )


def load_complete_runs(root: Path, stage: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ledger_path = root / LEDGERS[stage]
    if not ledger_path.is_file():
        raise SystemExit(f"missing stage ledger: {ledger_path}")
    ledger = json.loads(ledger_path.read_text())
    runs = ledger.get("runs", [])
    expected = 48 if stage == "a" else 28
    if len(runs) != expected:
        raise SystemExit(f"stage {stage}: expected {expected} runs, found {len(runs)}")
    keys = [cell_key(run) for run in runs]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"stage {stage}: duplicate cells in ledger")
    failures: list[str] = []
    for run in runs:
        try:
            run["validation_metrics"] = run_evaluations(run)
        except ValueError as exc:
            failures.append(str(exc))
    if failures:
        raise SystemExit(
            f"stage {stage} cannot advance; {len(failures)} cell(s) failed closed:\n  - "
            + "\n  - ".join(failures)
        )
    return ledger, runs


def choose_domain_checkpoint(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {run["arm"]: run for run in runs}
    if set(by_arm) != {"grpo", "xmode"}:
        raise ValueError(f"Stage A configuration lacks both arms: {sorted(by_arm)}")
    initial: dict[str, dict[str, float]] = {}
    for arm, run in by_arm.items():
        metrics = run["validation_metrics"]
        if 0 not in metrics:
            raise ValueError(f"{run['run_stamp']}: no step-zero validation")
        initial[arm] = metrics[0]
    target = min(int(run["target_steps"]) for run in runs)
    checkpoints = list(range(320, target + 1, 320))
    candidates: list[dict[str, Any]] = []
    for step in checkpoints:
        if any(step not in run["validation_metrics"] for run in runs):
            missing = [run["run_stamp"] for run in runs if step not in run["validation_metrics"]]
            raise ValueError(f"missing full-pass validation at step {step}: {missing}")
        arms: dict[str, Any] = {}
        for arm, run in by_arm.items():
            metric = run["validation_metrics"][step]
            arms[arm] = {
                **metric,
                "initial_pass_at_8": initial[arm]["pass_at_8"],
                "initial_distinct_at_8": initial[arm]["distinct_at_8"],
                "utility": (
                    metric["distinct_at_8"] - initial[arm]["distinct_at_8"]
                    + metric["pass_at_8"] - initial[arm]["pass_at_8"]
                ),
            }
        candidates.append(
            {
                "step": step,
                "pass": step // 320,
                "feasible": all(
                    entry["pass_at_8"] >= entry["initial_pass_at_8"] - PASS_FLOOR
                    for entry in arms.values()
                ),
                "mean_utility": sum(entry["utility"] for entry in arms.values()) / 2,
                "mean_pass_at_8": sum(entry["pass_at_8"] for entry in arms.values()) / 2,
                "mean_distinct_at_8": sum(entry["distinct_at_8"] for entry in arms.values()) / 2,
                "arms": arms,
            }
        )
    feasible = [candidate for candidate in candidates if candidate["feasible"]]
    if feasible:
        chosen = max(
            feasible,
            key=lambda row: (row["mean_utility"], row["mean_pass_at_8"], -row["step"]),
        )
        fallback = False
    else:
        chosen = max(
            candidates,
            key=lambda row: (row["mean_pass_at_8"], row["mean_distinct_at_8"], -row["step"]),
        )
        fallback = True
    return {**chosen, "fallback_no_feasible_checkpoint": fallback, "candidates": candidates}


def select_stage_a(root: Path) -> dict[str, Any]:
    ledger, runs = load_complete_runs(root, "a")
    grouped: dict[tuple[str, float, float, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["model"], float(run["lr"]), float(run["beta"]), run["domain"])].append(run)
    models: dict[str, Any] = {}
    for model in sorted({run["model"] for run in runs}):
        configurations: list[dict[str, Any]] = []
        combos = sorted(
            {(float(run["lr"]), float(run["beta"])) for run in runs if run["model"] == model}
        )
        for lr, beta in combos:
            domains = {
                domain: choose_domain_checkpoint(grouped[(model, lr, beta, domain)])
                for domain in sorted({run["domain"] for run in runs if run["model"] == model})
            }
            utilities = [choice["mean_utility"] for choice in domains.values()]
            configurations.append(
                {
                    "learning_rate": lr,
                    "beta": beta,
                    "mean_domain_utility": sum(utilities) / len(utilities),
                    "worst_domain_utility": min(utilities),
                    "domains": domains,
                }
            )
        chosen = max(
            configurations,
            key=lambda row: (
                row["mean_domain_utility"], row["worst_domain_utility"],
                -row["beta"], -row["learning_rate"],
            ),
        )
        models[model] = {
            "learning_rate": chosen["learning_rate"],
            "beta": chosen["beta"],
            "mean_domain_utility": chosen["mean_domain_utility"],
            "worst_domain_utility": chosen["worst_domain_utility"],
            "domains": {
                domain: {
                    "stopping_step": choice["step"],
                    "stopping_pass": choice["pass"],
                    "fallback_no_feasible_checkpoint": choice["fallback_no_feasible_checkpoint"],
                    "mean_utility": choice["mean_utility"],
                    "mean_pass_at_8": choice["mean_pass_at_8"],
                    "mean_distinct_at_8": choice["mean_distinct_at_8"],
                    "arms": choice["arms"],
                }
                for domain, choice in chosen["domains"].items()
            },
            "configuration_screen": configurations,
        }
    return {
        "schema": "e76_tuned_scale_stage_a_selection_v1",
        "complete": True,
        "source_ledger": LEDGERS["a"],
        "source_snapshot": ledger.get("snapshot_root"),
        "pass_floor_absolute": PASS_FLOOR,
        "models": models,
    }


def terminal(run: dict[str, Any]) -> dict[str, float]:
    return run["validation_metrics"][int(run["target_steps"])]


def select_stage_b(root: Path) -> dict[str, Any]:
    ledger, runs = load_complete_runs(root, "b")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["model"], run["domain"])].append(run)
    models: dict[str, Any] = {}
    for model in sorted({run["model"] for run in runs}):
        domain_output: dict[str, Any] = {}
        for domain in sorted({run["domain"] for run in runs if run["model"] == model}):
            cells = grouped[(model, domain)]
            controls = [run for run in cells if run["arm"] == "grpo"]
            if len(controls) != 1:
                raise SystemExit(f"{model}/{domain}: expected one Dr.GRPO control")
            control_metric = terminal(controls[0])
            entry: dict[str, Any] = {
                "target_steps": int(controls[0]["target_steps"]),
                "stopping_pass": int(controls[0]["target_passes"]),
                "grpo": control_metric,
            }
            for arm in ("xmode", "rehearsal"):
                arm_runs = [run for run in cells if run["arm"] == arm]
                if len(arm_runs) != 3:
                    raise SystemExit(f"{model}/{domain}/{arm}: expected three doses")
                candidates = []
                for run in arm_runs:
                    metric = terminal(run)
                    candidates.append(
                        {
                            "dose": float(run["dose"]),
                            **metric,
                            "feasible": metric["pass_at_8"] >= control_metric["pass_at_8"] - PASS_FLOOR,
                            "run_stamp": run["run_stamp"],
                        }
                    )
                feasible = [candidate for candidate in candidates if candidate["feasible"]]
                if feasible:
                    chosen = max(
                        feasible,
                        key=lambda row: (row["distinct_at_8"], row["pass_at_8"], -row["dose"]),
                    )
                    fallback = False
                else:
                    chosen = max(
                        candidates,
                        key=lambda row: (row["pass_at_8"], row["distinct_at_8"], -row["dose"]),
                    )
                    fallback = True
                entry[arm] = {
                    **chosen,
                    "fallback_no_feasible_dose": fallback,
                    "candidates": candidates,
                }
            domain_output[domain] = entry
        models[model] = {"domains": domain_output}
    return {
        "schema": "e76_tuned_scale_stage_b_selection_v1",
        "complete": True,
        "source_ledger": LEDGERS["b"],
        "source_snapshot": ledger.get("snapshot_root"),
        "pass_floor_absolute": PASS_FLOOR,
        "models": models,
    }


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("a", "b"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or root / OUTPUTS[args.stage]
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen selection: {output}")
    try:
        payload = select_stage_a(root) if args.stage == "a" else select_stage_b(root)
    except ValueError as exc:
        raise SystemExit(f"stage {args.stage} selector failed closed: {exc}") from exc
    atomic_json(output, payload)
    print(f"[e76 selector] stage={args.stage} wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
