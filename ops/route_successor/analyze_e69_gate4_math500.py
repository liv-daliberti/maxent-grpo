#!/usr/bin/env python3
"""Fail-closed analysis for E69's one-time held-out MATH-500 transfer."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from ops.route_successor.audit_e69_gate3_confirmatory import (
    crossed_bootstrap_interval,
)


ROOT = Path(__file__).resolve().parents[2]
IDENTITY = ROOT / "var/artifacts/e69_gate4_math500_identity.json"
GATE3_AUDIT = ROOT / "var/artifacts/e69_gate3_confirmatory_audit_latest.json"
OUT_JSON = ROOT / "var/artifacts/e69_gate4_math500_analysis.json"
OUT_CSV = ROOT / "paper/results/e69_gate4_math500_seed_results.csv"
OUT_MD = ROOT / "paper/results/e69_gate4_math500_transfer.md"
CONTROL = "grpo"
SUCCESSOR = "verified_first_global_replay_canonical"
SEEDS = (43, 44, 45)
METRICS = ("greedy", "mean8", "pass8")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 690402
HARD_FAILURE = re.compile(
    r"CUDA out of memory|torch\.OutOfMemoryError|ChildFailedError|"
    r"RayActorError|segmentation fault|Traceback \(most recent call last\)",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _close(left: Any, right: float, tolerance: float = 1e-12) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isfinite(float(left))
        and math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    )


def _prompt_contributions(
    result: dict[str, Any],
    *,
    label: str,
) -> tuple[dict[str, list[float]], list[str], list[str]]:
    violations: list[str] = []
    prompt_hashes: list[str] = []
    contributions = {metric: [] for metric in METRICS}
    prompts = result.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 500:
        return contributions, prompt_hashes, [f"{label}: expected 500 prompts"]
    worker_errors = 0
    for expected_index, row in enumerate(prompts):
        if not isinstance(row, dict) or row.get("index") != expected_index:
            violations.append(f"{label}: prompt index {expected_index} mismatch")
            continue
        problem_hash = row.get("problem_sha256")
        if not isinstance(problem_hash, str) or len(problem_hash) != 64:
            violations.append(f"{label}: prompt {expected_index} hash invalid")
            continue
        prompt_hashes.append(problem_hash)
        greedy = row.get("greedy")
        sampled = row.get("sampled")
        if not isinstance(greedy, dict) or not isinstance(sampled, list):
            violations.append(f"{label}: prompt {expected_index} result missing")
            continue
        if len(sampled) != 8:
            violations.append(f"{label}: prompt {expected_index} sampled width")
            continue
        rewards = [greedy.get("reward")] + [
            sample.get("reward") if isinstance(sample, dict) else None
            for sample in sampled
        ]
        if not all(_close(reward, 0.0) or _close(reward, 1.0) for reward in rewards):
            violations.append(f"{label}: prompt {expected_index} reward invalid")
            continue
        worker_errors += sum(
            bool(
                sample.get("verifier_info", {}).get("verifier_worker_error")
            )
            for sample in [greedy, *sampled]
            if isinstance(sample, dict)
        )
        sampled_rewards = [float(sample["reward"]) for sample in sampled]
        contributions["greedy"].append(float(greedy["reward"]))
        contributions["mean8"].append(sum(sampled_rewards) / 8.0)
        contributions["pass8"].append(
            float(any(reward > 0 for reward in sampled_rewards))
        )
    if worker_errors:
        violations.append(f"{label}: {worker_errors} verifier worker errors")
    return contributions, prompt_hashes, violations


def classify_final(
    executable_deltas: dict[str, dict[str, float]],
    math_deltas: dict[str, float],
    positive_support: dict[str, bool],
    mechanism_by_domain: dict[str, bool],
) -> dict[str, Any]:
    """Apply the frozen five-area final classification."""

    executable_quality = {
        domain: (
            row["greedy"] >= -0.02 - 1e-12
            and row["pass8"] >= -0.02 - 1e-12
        )
        for domain, row in executable_deltas.items()
    }
    heldout_math_quality = all(
        math_deltas[metric] >= -0.02 - 1e-12 for metric in METRICS
    )
    checks = {
        "four_executable_areas_task_quality_noninferior": all(
            executable_quality.values()
        ),
        "heldout_math500_task_quality_noninferior": heldout_math_quality,
        "three_executable_domains_positive_support": (
            sum(bool(value) for value in positive_support.values()) >= 3
        ),
        "mechanism_present_in_three_executable_domains": (
            sum(bool(value) for value in mechanism_by_domain.values()) >= 3
        ),
    }
    support_mechanism = (
        checks["three_executable_domains_positive_support"]
        and checks["mechanism_present_in_three_executable_domains"]
    )
    if all(checks.values()):
        status = "successful_exploration"
    elif support_mechanism:
        status = "mechanism_result"
    else:
        status = "null_or_negative"
    return {
        "status": status,
        "checks": checks,
        "executable_task_quality_noninferior": executable_quality,
        "heldout_math500_task_quality_noninferior": heldout_math_quality,
    }


def _write_csv(seed_metrics: dict[str, dict[int, dict[str, float]]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", *METRICS])
        for arm, seeds in seed_metrics.items():
            for seed, metrics in seeds.items():
                writer.writerow([arm, seed, *(metrics[name] for name in METRICS)])


def _write_markdown(payload: dict[str, Any]) -> None:
    lines = [
        "# E69 held-out MATH-500 transfer",
        "",
        f"Audit status: **{payload['status']}**.",
        "",
        "This is the preregistered one-time held-out transfer. MATH-500 is not "
        "a ModeBench domain, and no result was used for tuning or checkpoint "
        "selection.",
    ]
    if payload.get("seed_metrics"):
        lines.extend(
            [
                "",
                "| Arm | Seed | Greedy | Mean@8 | Pass@8 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for arm, seeds in payload["seed_metrics"].items():
            for seed, metrics in seeds.items():
                lines.append(
                    f"| {arm} | {seed} | {metrics['greedy']:.4f} | "
                    f"{metrics['mean8']:.4f} | {metrics['pass8']:.4f} |"
                )
    if payload.get("paired_deltas"):
        lines.extend(
            [
                "",
                "| Metric | Three-seed mean delta | Descriptive 95% interval | "
                "Seed 43 / 44 / 45 |",
                "|---|---:|---:|---|",
            ]
        )
        for metric, row in payload["paired_deltas"].items():
            seed_text = " / ".join(
                f"{row['seeds'][str(seed)]:+.4f}" for seed in SEEDS
            )
            lines.append(
                f"| {metric} | {row['mean']:+.4f} | "
                f"[{row['lower']:+.4f}, {row['upper']:+.4f}] | "
                f"{seed_text} |"
            )
    if payload.get("final_classification"):
        lines.extend(["", "## Frozen final classification", ""])
        lines.append(
            f"Result: **{payload['final_classification']['status']}**."
        )
        for name, passed in payload["final_classification"]["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    if payload.get("pending"):
        lines.extend(["", "## Pending", ""])
        lines.extend(f"- {value}" for value in payload["pending"])
    if payload.get("violations"):
        lines.extend(["", "## Integrity violations", ""])
        lines.extend(f"- {value}" for value in payload["violations"])
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not IDENTITY.is_file():
        raise SystemExit(f"E69 Gate 4 identity is absent: {IDENTITY}")
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    gate3 = json.loads(GATE3_AUDIT.read_text(encoding="utf-8"))
    violations: list[str] = []
    pending: list[str] = []
    if identity.get("schema") != "e69_gate4_math500_one_time_transfer_v1":
        violations.append("Gate 4 identity schema mismatch")
    if (
        gate3.get("schema") != "e69_gate3_confirmatory_audit_v1"
        or gate3.get("status") != "complete"
        or gate3.get("summary", {}).get("integrity_violations") != 0
    ):
        violations.append("Gate 3 audit is not clean and complete")

    identity_hash = _sha256(IDENTITY)
    seed_metrics: dict[str, dict[int, dict[str, float]]] = {
        CONTROL: {},
        SUCCESSOR: {},
    }
    prompt_values: dict[str, dict[int, dict[str, list[float]]]] = {
        CONTROL: {},
        SUCCESSOR: {},
    }
    prompt_hashes: dict[str, dict[int, list[str]]] = {
        CONTROL: {},
        SUCCESSOR: {},
    }
    results: list[dict[str, Any]] = []
    for evaluation in identity.get("evaluations", []):
        arm = str(evaluation["arm"])
        seed = int(evaluation["seed"])
        alias = str(evaluation["alias"])
        job_id = int(evaluation["job_id"])
        output = Path(evaluation["output"])
        label = f"{arm}/seed{seed}/job{job_id}"
        if not output.is_file():
            pending.append(f"{label}: result absent")
            for suffix in ("out", "err"):
                log = ROOT / f"var/artifacts/logs/e69_math500-{job_id}.{suffix}"
                if log.is_file():
                    match = HARD_FAILURE.search(
                        log.read_text(encoding="utf-8", errors="replace")
                    )
                    if match:
                        violations.append(f"{label}: {match.group(0)}")
            continue
        result = json.loads(output.read_text(encoding="utf-8"))
        if (
            result.get("schema") != "e69_gate4_math500_checkpoint_result_v1"
            or result.get("identity_sha256") != identity_hash
            or result.get("alias") != alias
            or result.get("arm") != arm
            or int(result.get("seed", -1)) != seed
            or result.get("checkpoint_tree_sha256")
            != evaluation["checkpoint_tree_sha256"]
            or result.get("data") != identity["data"]
            or result.get("requests") != identity["requests"]
        ):
            violations.append(f"{label}: result identity mismatch")
        contributions, hashes, row_violations = _prompt_contributions(
            result,
            label=label,
        )
        violations.extend(row_violations)
        recomputed = {
            metric: sum(values) / len(values) if values else float("nan")
            for metric, values in contributions.items()
        }
        for metric in METRICS:
            if not _close(result.get("metrics", {}).get(metric), recomputed[metric]):
                violations.append(f"{label}: stored {metric} does not recompute")
        seed_metrics[arm][seed] = recomputed
        prompt_values[arm][seed] = contributions
        prompt_hashes[arm][seed] = hashes
        results.append(
            {
                "alias": alias,
                "arm": arm,
                "seed": seed,
                "job_id": job_id,
                "output": str(output.resolve()),
                "output_sha256": _sha256(output),
                "metrics": recomputed,
                "diagnostics": result.get("metrics", {}),
            }
        )

    complete = (
        len(results) == 6
        and all(set(seed_metrics[arm]) == set(SEEDS) for arm in (CONTROL, SUCCESSOR))
        and not pending
    )
    paired_deltas: dict[str, dict[str, Any]] = {}
    final_classification = None
    if complete and not violations:
        for seed in SEEDS:
            if prompt_hashes[CONTROL][seed] != prompt_hashes[SUCCESSOR][seed]:
                violations.append(f"seed {seed}: paired prompt hashes differ")
        reference_hashes = prompt_hashes[CONTROL][43]
        for arm in (CONTROL, SUCCESSOR):
            for seed in SEEDS:
                if prompt_hashes[arm][seed] != reference_hashes:
                    violations.append(f"{arm}/seed{seed}: MATH-500 order differs")
        if not violations:
            for metric in METRICS:
                per_seed_prompt = {
                    seed: [
                        treatment - control
                        for treatment, control in zip(
                            prompt_values[SUCCESSOR][seed][metric],
                            prompt_values[CONTROL][seed][metric],
                        )
                    ]
                    for seed in SEEDS
                }
                seed_delta = {
                    seed: (
                        seed_metrics[SUCCESSOR][seed][metric]
                        - seed_metrics[CONTROL][seed][metric]
                    )
                    for seed in SEEDS
                }
                lower, upper = crossed_bootstrap_interval(
                    per_seed_prompt,
                    replicates=BOOTSTRAP_REPLICATES,
                    seed=BOOTSTRAP_SEED,
                )
                paired_deltas[metric] = {
                    "mean": sum(seed_delta.values()) / len(SEEDS),
                    "lower": lower,
                    "upper": upper,
                    "seeds": {str(seed): value for seed, value in seed_delta.items()},
                }
            executable = {
                domain: gate3["aggregate_deltas"][domain]["6"]
                for domain in (
                    "graph_coloring",
                    "countdown",
                    "python_factor",
                    "mathir",
                )
            }
            final_classification = classify_final(
                executable,
                {
                    metric: paired_deltas[metric]["mean"]
                    for metric in METRICS
                },
                gate3["classification"]["positive_support"],
                gate3["classification"]["mechanism_by_domain"],
            )

    status = (
        "fail_integrity"
        if violations
        else "complete"
        if complete
        else "in_progress"
    )
    payload = {
        "schema": "e69_gate4_math500_analysis_v1",
        "status": status,
        "identity": str(IDENTITY.resolve()),
        "identity_sha256": identity_hash,
        "math500_unsealed": True,
        "summary": {
            "expected_results": 6,
            "observed_results": len(results),
            "pending_items": len(set(pending)),
            "integrity_violations": len(set(violations)),
        },
        "results": results,
        "seed_metrics": {
            arm: {
                str(seed): metrics for seed, metrics in seeds.items()
            }
            for arm, seeds in seed_metrics.items()
        },
        "paired_deltas": paired_deltas,
        "final_classification": final_classification,
        "pending": sorted(set(pending)),
        "violations": sorted(set(violations)),
    }
    _atomic_json(OUT_JSON, payload)
    _write_csv(seed_metrics)
    _write_markdown(payload)
    print(
        f"[e69-gate4-analysis] status={status} "
        f"results={len(results)}/6 "
        f"violations={len(set(violations))}"
    )


if __name__ == "__main__":
    main()
