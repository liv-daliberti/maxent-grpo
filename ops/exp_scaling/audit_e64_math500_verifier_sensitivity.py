#!/usr/bin/env python3
"""Audit MATH-500 verifier stability from the campaign's saved raw traces.

The frozen E64 process auditor intentionally treats any printed traceback as
fatal.  ``math_verify`` can also print caught, per-example diagnostics and
continue.  This supplemental auditor never changes the primary scores.  It
uses the saved response-level traces to test whether those diagnostics caused
observable grading nondeterminism and requires complete raw traces at every
registered checkpoint before declaring the sensitivity audit terminal.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
E64_AUDIT = ROOT / "var/artifacts/e64_math500_realism_matched_audit_latest.json"
OUT = ROOT / "var/artifacts/e64_math500_verifier_sensitivity_latest.json"
FIXED_STEPS = tuple(training_pass * 384 for training_pass in (0, 2, 4, 6, 8, 10, 12))
KINDS = (
    "deterministic_greedy_trace_neutral",
    "fixed_seed_sampled_k_neutral",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _close(left: Any, right: Any) -> bool:
    return (
        _finite(left)
        and _finite(right)
        and math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _scan_trace(
    trace: dict[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], list[str], list[tuple[tuple[str, str, str], float]]]:
    violations: list[str] = []
    prompts = trace.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 500:
        return (
            {},
            [f"{label}: trace does not contain exactly 500 prompts"],
            [],
        )
    kind = str(trace.get("evaluation_kind", ""))
    expected_k = 1 if kind == KINDS[0] else 8 if kind == KINDS[1] else None
    if expected_k is None:
        violations.append(f"{label}: unexpected evaluation kind {kind!r}")
    observed_rewards = 0.0
    observed_any = 0
    response_rewards: list[tuple[tuple[str, str, str], float]] = []
    response_surface: list[Any] = []
    reward_surface: list[Any] = []
    for prompt_index, prompt in enumerate(prompts):
        responses = prompt.get("responses")
        rewards = prompt.get("rewards")
        answer_keys = prompt.get("answer_keys")
        if (
            not isinstance(responses, list)
            or not isinstance(rewards, list)
            or not isinstance(answer_keys, list)
            or len(responses) != len(rewards)
            or len(responses) != len(answer_keys)
            or (expected_k is not None and len(responses) != expected_k)
        ):
            violations.append(
                f"{label}: prompt {prompt_index} has malformed K-way rows"
            )
            continue
        prompt_text = str(prompt.get("prompt", ""))
        reference = str(prompt.get("reference", ""))
        numeric_rewards: list[float] = []
        for response, reward in zip(responses, rewards, strict=True):
            if not _finite(reward):
                violations.append(
                    f"{label}: prompt {prompt_index} has nonfinite reward"
                )
                continue
            numeric_reward = float(reward)
            if numeric_reward not in (0.0, 1.0):
                violations.append(
                    f"{label}: prompt {prompt_index} reward is not binary"
                )
            numeric_rewards.append(numeric_reward)
            response_rewards.append(
                ((prompt_text, reference, str(response)), numeric_reward)
            )
        observed_rewards += sum(numeric_rewards)
        observed_any += int(any(value > 0 for value in numeric_rewards))
        response_surface.append((prompt_text, reference, responses))
        reward_surface.append(rewards)
    denominator = 500 * expected_k if expected_k is not None else 0
    recomputed_mean = observed_rewards / denominator if denominator else None
    recomputed_any = observed_any / 500.0
    metrics = trace.get("metrics", {})
    if not _close(metrics.get("mean_at_k"), recomputed_mean):
        violations.append(f"{label}: saved mean@K does not match raw rewards")
    if not _close(metrics.get("any_correct_at_k"), recomputed_any):
        violations.append(f"{label}: saved pass@K does not match raw rewards")
    return (
        {
            "step": int(trace.get("step", -1)),
            "evaluation_kind": kind,
            "response_sha256": _digest(response_surface),
            "reward_sha256": _digest(reward_surface),
            "recomputed_mean_at_k": recomputed_mean,
            "recomputed_any_correct_at_k": recomputed_any,
        },
        violations,
        response_rewards,
    )


def main() -> None:
    e64 = _load_json(E64_AUDIT)
    violations: list[str] = []
    runs: list[dict[str, Any]] = []
    identical_response_rewards: dict[
        tuple[str, str, str],
        set[float],
    ] = defaultdict(set)
    step_zero_by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in e64.get("runs", []):
        job_id = int(run["job_id"])
        label = f"{run['arm']}/s{run['seed']}/j{job_id}"
        run_dir_value = run.get("run_dir")
        run_dir = ROOT / str(run_dir_value) if run_dir_value else None
        trace_path = (
            run_dir / "eval_mode_coverage_draws.jsonl"
            if run_dir is not None
            else None
        )
        trace_summaries: list[dict[str, Any]] = []
        if trace_path is None or not trace_path.is_file():
            violations.append(f"{label}: saved raw evaluation trace is missing")
        else:
            with trace_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        trace = json.loads(line)
                    except json.JSONDecodeError:
                        violations.append(
                            f"{label}: invalid trace JSON line {line_number}"
                        )
                        continue
                    step = int(trace.get("step", -1))
                    if step not in FIXED_STEPS:
                        continue
                    summary, trace_violations, response_rewards = _scan_trace(
                        trace,
                        label=f"{label}/step{step}/line{line_number}",
                    )
                    violations.extend(trace_violations)
                    if summary:
                        trace_summaries.append(summary)
                        if step == 0:
                            step_zero_by_kind[
                                summary["evaluation_kind"]
                            ].append({"job_id": job_id, **summary})
                    for key, reward in response_rewards:
                        identical_response_rewards[key].add(reward)
        available = {
            (int(trace["step"]), str(trace["evaluation_kind"]))
            for trace in trace_summaries
        }
        latest_step = int(run.get("latest_step", -1))
        expected_so_far = {
            (step, kind)
            for step in FIXED_STEPS
            if step <= latest_step
            for kind in KINDS
        }
        missing = sorted(expected_so_far - available)
        if missing:
            violations.append(
                f"{label}: missing registered raw traces {missing}"
            )
        log_timeout_count = 0
        log_traceback_count = 0
        for suffix in ("out", "err"):
            log = ROOT / f"var/artifacts/logs/xdr_train-{job_id}.{suffix}"
            if not log.is_file():
                continue
            text = log.read_text(encoding="utf-8", errors="replace")
            log_timeout_count += text.count("Timeout during comparison")
            log_traceback_count += text.count(
                "Traceback (most recent call last)"
            )
        runs.append(
            {
                "arm": run["arm"],
                "seed": int(run["seed"]),
                "job_id": job_id,
                "latest_step": latest_step,
                "registered_trace_count": len(trace_summaries),
                "expected_registered_trace_count_so_far": len(
                    expected_so_far
                ),
                "verifier_timeout_diagnostics": log_timeout_count,
                "caught_traceback_diagnostics": log_traceback_count,
                "traces": trace_summaries,
            }
        )
    conflicts = sum(
        len(rewards) > 1 for rewards in identical_response_rewards.values()
    )
    if conflicts:
        violations.append(
            f"{conflicts} identical prompt/reference/response tuples received "
            "inconsistent rewards"
        )
    step_zero_reproducible: dict[str, Any] = {}
    for kind in KINDS:
        rows = step_zero_by_kind.get(kind, [])
        response_hashes = {row["response_sha256"] for row in rows}
        reward_hashes = {row["reward_sha256"] for row in rows}
        reproducible = (
            len(rows) == 6
            and len(response_hashes) == 1
            and len(reward_hashes) == 1
        )
        if not reproducible:
            violations.append(
                f"step-0 {kind} response/reward redundancy is not reproducible"
            )
        step_zero_reproducible[kind] = {
            "run_count": len(rows),
            "unique_response_hashes": len(response_hashes),
            "unique_reward_hashes": len(reward_hashes),
            "reproducible": reproducible,
        }
    all_terminal = (
        int(e64.get("summary", {}).get("terminal_runs", 0)) == 6
        and all(
            run["registered_trace_count"] == len(FIXED_STEPS) * len(KINDS)
            for run in runs
        )
    )
    status = "fail" if violations else "pass" if all_terminal else "in_progress"
    payload = {
        "schema": "e64_math500_verifier_sensitivity_v1",
        "status": status,
        "evidential_role": (
            "supplemental response-level verifier stability audit; does not "
            "replace or alter frozen primary scores"
        ),
        "fixed_steps": list(FIXED_STEPS),
        "summary": {
            "expected_runs": 6,
            "runs_with_raw_traces": sum(
                run["registered_trace_count"] > 0 for run in runs
            ),
            "terminal_runs": int(
                e64.get("summary", {}).get("terminal_runs", 0)
            ),
            "verifier_timeout_diagnostics": sum(
                run["verifier_timeout_diagnostics"] for run in runs
            ),
            "caught_traceback_diagnostics": sum(
                run["caught_traceback_diagnostics"] for run in runs
            ),
            "unique_response_tuples_checked": len(
                identical_response_rewards
            ),
            "identical_response_reward_conflicts": conflicts,
            "step_zero_reproducible": step_zero_reproducible,
        },
        "runs": runs,
        "violations": sorted(set(violations)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUT.name}.", dir=OUT.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, OUT)
    print(
        "[e64-verifier-sensitivity] "
        f"status={status} timeouts="
        f"{payload['summary']['verifier_timeout_diagnostics']} "
        f"conflicts={conflicts} violations={len(payload['violations'])}"
    )


if __name__ == "__main__":
    main()
