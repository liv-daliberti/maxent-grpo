#!/usr/bin/env python3
"""Regenerate the E78 Python output-collapse and token-entropy diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "var/artifacts/e78_verified_replay_only_05b_jobs.json"
CORE = ROOT / "paper/results/core_terminal_endpoints.json"
OUTPUT = ROOT / "paper/results/e78_python_collapse_diagnostics.json"
DOMAIN = "python_factors"
TARGET_STEP = 3072
STEPS_PER_PASS = 384
EXPECTED_PROMPTS = 128
EXPECTED_DRAWS = 4
EXPECTED_K = 8
METRIC_KEYS = {
    "mean8": "mean_at_k",
    "pass8": "any_correct_at_k",
    "distinct8": "distinct_correct_modes_at_k",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(run_dir: Path, name: str) -> Path:
    paths = list(run_dir.glob(f"**/{name}"))
    if not paths:
        raise FileNotFoundError(f"{run_dir}: missing {name}")
    return max(paths, key=lambda path: path.stat().st_size)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def checkpoint_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    greedy_rows = [row for row in rows if float(row["temperature"]) == 0.0]
    draw_rows = [row for row in rows if float(row["temperature"]) == 1.0]
    if len(greedy_rows) != 1 or len(draw_rows) != EXPECTED_DRAWS:
        raise ValueError(
            f"expected one greedy and {EXPECTED_DRAWS} sampled rows; "
            f"got {len(greedy_rows)} and {len(draw_rows)}"
        )
    if any(int(row["sample_count"]) != EXPECTED_K for row in draw_rows):
        raise ValueError("sampled row does not use registered K=8")

    greedy = {
        int(prompt["prompt_index"]): prompt["responses"][0]
        for prompt in greedy_rows[0]["prompts"]
    }
    draws = [
        {int(prompt["prompt_index"]): prompt for prompt in row["prompts"]}
        for row in draw_rows
    ]
    if len(greedy) != EXPECTED_PROMPTS:
        raise ValueError(f"expected {EXPECTED_PROMPTS} prompts, got {len(greedy)}")

    greedy_matches = 0
    within_draw_unique: list[int] = []
    across_draw_unique: list[int] = []
    for prompt_index, greedy_response in greedy.items():
        all_responses: list[str] = []
        for draw in draws:
            responses = draw[prompt_index]["responses"]
            if len(responses) != EXPECTED_K:
                raise ValueError("prompt does not contain K=8 sampled responses")
            all_responses.extend(responses)
            within_draw_unique.append(len(set(responses)))
        across_draw_unique.append(len(set(all_responses)))
        greedy_matches += int(all(response == greedy_response for response in all_responses))

    draw_std = {
        name: statistics.pstdev(float(row["metrics"][key]) for row in draw_rows)
        for name, key in METRIC_KEYS.items()
    }
    return {
        "greedy_match_prompt_count": greedy_matches,
        "prompt_count": EXPECTED_PROMPTS,
        "mean_unique_raw_completions_within_k8": statistics.mean(within_draw_unique),
        "mean_unique_raw_completions_across_4xk8": statistics.mean(across_draw_unique),
        "endpoint_draw_population_std": draw_std,
    }


def raw_trajectory(path: Path) -> dict[str, Any]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        by_step.setdefault(int(row["step"]), []).append(row)
    checkpoints = {
        step: checkpoint_stats(rows) for step, rows in sorted(by_step.items())
    }
    steps = sorted(checkpoints)
    persistent_step = None
    for index, step in enumerate(steps):
        current = checkpoints[step]
        if current["greedy_match_prompt_count"] != EXPECTED_PROMPTS:
            continue
        if all(
            checkpoints[later]["greedy_match_prompt_count"] == EXPECTED_PROMPTS
            for later in steps[index:]
        ):
            persistent_step = step
            break
    return {
        "source": str(path.relative_to(ROOT)),
        "checkpoint_steps": steps,
        "first_persistent_raw_point_mass_step": persistent_step,
        "first_persistent_raw_point_mass_pass": (
            persistent_step / STEPS_PER_PASS if persistent_step is not None else None
        ),
        "pass0": checkpoints.get(0),
        "terminal": checkpoints.get(TARGET_STEP),
    }


def token_entropy(path: Path) -> dict[str, Any]:
    by_step: dict[int, float] = {}
    for row in load_jsonl(path):
        step = int(row.get("misc/global_step", row.get("trainer/global_step", 0)))
        value = row.get("train/entropy")
        if value is not None and 1 <= step <= TARGET_STEP:
            by_step[step] = float(value)
    if set(by_step) != set(range(1, TARGET_STEP + 1)):
        missing = sorted(set(range(1, TARGET_STEP + 1)) - set(by_step))
        raise ValueError(f"entropy trace is incomplete; first missing steps: {missing[:10]}")
    pass_means = []
    for pass_index in range(8):
        lo = pass_index * STEPS_PER_PASS
        hi = (pass_index + 1) * STEPS_PER_PASS
        pass_means.append(statistics.mean(by_step[step] for step in range(lo + 1, hi + 1)))
    return {
        "source": str(path.relative_to(ROOT)),
        "units": "masked-mean policy token entropy in nats over sampled response tokens",
        "pass_means": pass_means,
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "seed_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    relevant = [
        run for run in ledger["runs"]
        if run["domain"] == DOMAIN and run["arm"] in {"control", "replay"}
    ]
    if len(relevant) != 10:
        raise ValueError(f"expected 10 E78 Python runs, got {len(relevant)}")

    arms: dict[str, dict[str, Any]] = {"control": {}, "replay": {}}
    for run in relevant:
        run_dir = Path(run["run_dir"])
        sidecar = artifact(run_dir, "eval_mode_coverage_draws.jsonl")
        metrics = artifact(run_dir, "train_metrics.jsonl")
        arms[run["arm"]][str(run["seed"])] = {
            "raw_output": raw_trajectory(sidecar),
            "token_entropy": token_entropy(metrics),
        }

    core = json.loads(CORE.read_text(encoding="utf-8"))
    equality: dict[str, Any] = {}
    for domain in ("python_factors", "pantry_plan"):
        equal: list[dict[str, Any]] = []
        unequal: list[dict[str, Any]] = []
        for model, model_record in core["models"].items():
            per_seed = model_record["domains"][domain]["methods"]["control"]["per_seed"]
            for seed, values in per_seed.items():
                record = {"model": model, "seed": int(seed), **values}
                target = equal if values["mean8"] == values["pass8"] == values["distinct8"] else unequal
                target.append(record)
        equality[domain] = {
            "equal_count": len(equal),
            "total_count": len(equal) + len(unequal),
            "equal_cells": equal,
            "unequal_cells": unequal,
            "interpretation": "screening signature only; metric equality is not sufficient to prove raw-output collapse",
        }

    aggregate: dict[str, Any] = {}
    for arm, seeds in arms.items():
        entropy_passes = []
        for pass_index in range(8):
            entropy_passes.append(summarize([
                seed_record["token_entropy"]["pass_means"][pass_index]
                for seed_record in seeds.values()
            ]))
        aggregate[arm] = {
            "token_entropy_by_pass": entropy_passes,
            "pass0_mean_unique_raw_completions_within_k8": summarize([
                seed_record["raw_output"]["pass0"]["mean_unique_raw_completions_within_k8"]
                for seed_record in seeds.values()
            ]),
            "terminal_mean_unique_raw_completions_within_k8": summarize([
                seed_record["raw_output"]["terminal"]["mean_unique_raw_completions_within_k8"]
                for seed_record in seeds.values()
            ]),
            "terminal_mean_unique_raw_completions_across_4xk8": summarize([
                seed_record["raw_output"]["terminal"]["mean_unique_raw_completions_across_4xk8"]
                for seed_record in seeds.values()
            ]),
        }

    control_steps = [
        seed_record["raw_output"]["first_persistent_raw_point_mass_step"]
        for seed_record in arms["control"].values()
    ]
    if any(step is None for step in control_steps):
        raise ValueError("not every control seed reaches persistent raw point-mass collapse")

    payload = {
        "schema": "e78-python-collapse-diagnostics-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question": "does metric equality reflect raw output collapse in the E78 Python control, and how does masked-mean token entropy evolve?",
        "post_hoc_descriptive": True,
        "sources": {
            "ledger": str(LEDGER.relative_to(ROOT)),
            "ledger_sha256": sha256(LEDGER),
            "terminal_endpoints": str(CORE.relative_to(ROOT)),
            "terminal_endpoints_sha256": sha256(CORE),
        },
        "registered_evaluation": {
            "prompt_count": EXPECTED_PROMPTS,
            "sampled_draws": EXPECTED_DRAWS,
            "samples_per_draw": EXPECTED_K,
            "sample_temperature": 1.0,
            "sample_top_p": 1.0,
            "greedy_temperature": 0.0,
        },
        "criterion": (
            "raw point mass means that, for every prompt, all 32 temperature-1 sampled strings "
            "are byte-identical to the separately logged temperature-0 greedy string; aggregate "
            "endpoint equality or zero draw SD is not used as the criterion"
        ),
        "terminal_endpoint_equality": equality,
        "control_persistent_raw_point_mass_step_range": [min(control_steps), max(control_steps)],
        "control_persistent_raw_point_mass_pass_range": [
            min(control_steps) / STEPS_PER_PASS,
            max(control_steps) / STEPS_PER_PASS,
        ],
        "arms": arms,
        "aggregate": aggregate,
        "limitations": [
            "The raw point-mass timing audit is specific to the five E78 Qwen2.5-0.5B Python control seeds.",
            "Masked-mean token entropy is response-length dependent and is not canonical-key entropy.",
            "No decoding temperature or top-p sweep of the current arms is included.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(json.dumps({
        "control_collapse_steps": control_steps,
        "control_terminal_unique_k8": aggregate["control"]["terminal_mean_unique_raw_completions_within_k8"]["mean"],
        "replay_terminal_unique_k8": aggregate["replay"]["terminal_mean_unique_raw_completions_within_k8"]["mean"],
        "control_entropy_pass_means": [x["mean"] for x in aggregate["control"]["token_entropy_by_pass"]],
        "replay_entropy_pass_means": [x["mean"] for x in aggregate["replay"]["token_entropy_by_pass"]],
        "endpoint_equality_counts": {k: [v["equal_count"], v["total_count"]] for k, v in equality.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
