#!/usr/bin/env python3
"""Verify retained CodeContests capability receipts and write a development report.

Uses only the Python standard library. Does not import a model, execute candidate
code, modify the frozen execution bundle, or infer a training treatment effect.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROBLEMS = ("359_B", "988_A", "1399_D")
SCHEMA = "codecontests-capability-report-20260921-v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_probability(n: int, successes: int, k: int) -> float:
    if not 0 <= successes <= n or not 1 <= k <= n:
        raise ValueError("invalid without-replacement sample counts")
    return 1.0 if n - successes < k else 1 - math.comb(n - successes, k) / math.comb(n, k)


def _request_key(row: dict[str, Any]) -> tuple[int, int]:
    return row["row_index"], row["sample_index"]


def _indexed(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    indexed = {_request_key(row): row for row in rows}
    if len(indexed) != 192 or len(rows) != 192 or set(indexed) != {(r, s) for r in range(3) for s in range(64)}:
        raise ValueError("receipt requires exactly 192 unique frozen request identities")
    return indexed


def load_verified(receipt_path: Path) -> tuple[dict[str, Any], dict, dict]:
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema_version") != "constructive-code-coder-7b-capability-pilot-20260921-v1":
        raise ValueError("wrong capability receipt schema")
    if receipt.get("status") not in ("pass", "capability_fail", "audit_fail"):
        raise ValueError("capability stage has no terminal outcome")
    boundary = receipt["information_boundary"]
    if tuple(boundary["loaded_problem_ids"]) != PROBLEMS or boundary["evaluation_rows_loaded"] is not False:
        raise ValueError("development information boundary mismatch")
    artifacts = {}
    for name in ("responses", "attempts"):
        descriptor = receipt["artifacts"][name]
        path = Path(descriptor["path"])
        if not path.is_absolute():
            path = receipt_path.parent / path
        if sha256(path) != descriptor["sha256"]:
            raise ValueError(f"{name} sidecar SHA-256 mismatch")
        artifacts[name] = _indexed([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
    attempts = _indexed(receipt["attempts"])
    if attempts != artifacts["attempts"]:
        raise ValueError("incremental attempt sidecar differs from final receipt")
    for key, response in artifacts["responses"].items():
        attempt = attempts[key]
        if response["source_problem_id"] != PROBLEMS[key[0]] or attempt["source_problem_id"] != PROBLEMS[key[0]]:
            raise ValueError("request problem identity mismatch")
        expected_seed = 77101 + 10000 * key[0] + key[1]
        if response["request_seed"] != expected_seed or attempt["request_seed"] != expected_seed:
            raise ValueError("request seed mismatch")
        for text_field, hash_field in (("emitted_text", "emitted_text_sha256"), ("code", "executed_source_sha256")):
            digest = hashlib.sha256(response[text_field].encode()).hexdigest()
            if digest != response[hash_field] or digest != attempt[hash_field]:
                raise ValueError("raw response/executed source hash mismatch")
        if response["token_count"] != len(response["token_ids"]) or response["token_count"] != attempt["token_count"]:
            raise ValueError("response token count mismatch")
        if attempt["accepted"]:
            repeated = attempt.get("stability_recheck")
            if attempt["hard_violations"] or not attempt.get("stability_recheck_required") or not repeated or not repeated["accepted"] or repeated["canonical_key"] != attempt["canonical_key"] or repeated["hard_violations"]:
                raise ValueError("accepted result lacks a stable, clean independent replay")
    return receipt, artifacts["responses"], attempts


def _failure_stage(attempt: dict[str, Any]) -> str:
    if attempt["accepted"]:
        return "accepted"
    if attempt["hard_violations"]:
        return "audit_violation"
    if not attempt["terminal_worker_record"]:
        return "worker_exception"
    execution = attempt.get("execution") or {}
    failure = execution.get("first_failure") or {}
    return str(failure.get("stage") or "unclassified_rejection")


def summarize(receipt: dict[str, Any], responses: dict, attempts: dict, scheduler: Any = None) -> dict[str, Any]:
    tasks, curves, examples = [], [], []
    for row_index, problem in enumerate(PROBLEMS):
        rows = [attempts[(row_index, sample)] for sample in range(64)]
        accepted = [row for row in rows if row["accepted"]]
        counts = Counter(row["canonical_key"] for row in accepted)
        m = len(accepted)
        pcmd = 1 - sum(v * (v - 1) for v in counts.values()) / (m * (m - 1)) if m >= 30 else None
        task = {
            "source_problem_id": problem, "samples": 64, "accepted": m,
            "distinct_modes": len(counts), "canonical_key_counts": dict(sorted(counts.items())),
            "pcmd": pcmd, "pcmd_eligible": m >= 30,
            "prefix_16_accepted": sum(row["accepted"] for row in rows[:16]),
            "failure_stages": dict(Counter(_failure_stage(row) for row in rows)),
            "finish_reasons": dict(Counter(responses[(row_index, sample)]["finish_reason"] for sample in range(64))),
            **{f"pass_at_{k}": draw_probability(64, m, k) for k in (1, 8, 32, 64)},
        }
        recorded = receipt["prompt_results"][row_index]
        for field, value in (("accepted", m), ("distinct_valid_modes", len(counts)), ("pcmd", pcmd), *[(f"pass_at_{k}", task[f"pass_at_{k}"]) for k in (1, 8, 32)]):
            if recorded.get(field) != value and not (isinstance(recorded.get(field), (float, int)) and isinstance(value, (float, int)) and math.isclose(recorded[field], value, abs_tol=1e-12)):
                raise ValueError(f"recomputed {problem}/{field} differs from evaluator")
        tasks.append(task)
        curves.append({"source_problem_id": problem, "points": [{"k": k, "pass_at_k": draw_probability(64, m, k), "expected_distinct_correct_modes_at_k": sum(draw_probability(64, count, k) for count in counts.values())} for k in range(1, 65)]})
        seen = set()
        pair = []
        for attempt in accepted:
            mode = attempt["canonical_key"]
            if mode in seen:
                continue
            seen.add(mode)
            response = responses[_request_key(attempt)]
            pair.append({"sample_index": attempt["sample_index"], "canonical_key": mode, "executed_source_sha256": response["executed_source_sha256"], "code": response["code"]})
            if len(pair) == 2:
                break
        if len(pair) == 2:
            examples.append({"source_problem_id": problem, "selection": "earliest accepted sample for each of the first two distinct canonical modes", "programs": pair})
    accepted_total = sum(task["accepted"] for task in tasks)
    multimode = sum(task["distinct_modes"] >= 2 for task in tasks)
    hard = [value for row in attempts.values() for value in row["hard_violations"]]
    audit = not hard and all(row["terminal_worker_record"] for row in attempts.values())
    capability = accepted_total / 192 >= 0.10 and multimode >= 2
    expected_status = "pass" if audit and capability else "audit_fail" if not audit else "capability_fail"
    if expected_status != receipt["status"]:
        raise ValueError("independently recomputed gate status differs from receipt")
    eligible = [task["pcmd"] for task in tasks if task["pcmd_eligible"]]
    scheduled_hours = None
    if isinstance(scheduler, dict):
        explicit = scheduler.get("allocated_gpu_hours")
        count, seconds = scheduler.get("allocated_gpu_count"), scheduler.get("elapsed_seconds")
        if isinstance(explicit, (int, float)) and math.isfinite(explicit) and explicit >= 0:
            scheduled_hours = explicit
        elif isinstance(count, (int, float)) and isinstance(seconds, (int, float)) and count >= 0 and seconds >= 0:
            scheduled_hours = count * seconds / 3600
    return {
        "schema_version": SCHEMA, "scope": "development capability only; no training or MaxRL versus Re:Max comparison",
        "status": expected_status, "audit_passed": audit, "capability_passed": capability,
        "accepted": accepted_total, "samples": 192, "aggregate_accuracy": accepted_total / 192,
        "multimode_tasks": multimode, "pcmd_eligible_tasks": len(eligible), "pcmd_total_tasks": 3,
        "macro_pcmd_over_eligible_tasks": sum(eligible) / len(eligible) if eligible else None,
        "macro_pass_at_k": {str(k): sum(task[f"pass_at_{k}"] for task in tasks) / 3 for k in (1, 8, 32, 64)},
        "tasks": tasks, "sampling_curves": curves, "program_pairs": examples,
        "failure_stages": dict(Counter(_failure_stage(row) for row in attempts.values())),
        "hard_violation_count": len(hard), "hard_violations": hard,
        "stability_rechecks": sum(bool(row.get("stability_recheck_required")) for row in attempts.values()),
        "model": receipt["model"], "model_tree_sha256": receipt["model_tree_sha256"],
        "gpu": receipt["gpu"], "timing": receipt["timing"],
        "scheduler": scheduler, "scheduler_allocated_gpu_hours": scheduled_hours,
        "source_hash": receipt["source_hash"], "execution_hash": receipt["execution_hash"],
        "prompt_overlay": receipt["prompt_overlay"], "artifacts": receipt["artifacts"],
    }


def _pct(value: float | None) -> str:
    return "not eligible" if value is None else f"{100 * value:.1f}%"


def report(summary: dict[str, Any]) -> str:
    outcome = {"pass": "The capability gate passed.", "capability_fail": "The capability gate did not pass.", "audit_fail": "The audit failed; this run cannot establish capability."}[summary["status"]]
    lines = ["# CodeContests+ development capability pilot", "", outcome,
        f"Accepted {summary['accepted']}/{summary['samples']} candidates ({_pct(summary['aggregate_accuracy'])}); {summary['multimode_tasks']}/3 tasks produced at least two verified modes. These are three development problems, with 64 samples each, from Qwen2.5-Coder-7B-Instruct. No model training or Re:Max–MaxRL comparison was performed.", "",
        "| Problem | Accepted / 64 | pass@1 | pass@8 | pass@32 | Distinct valid modes | PCMD |", "|---|---:|---:|---:|---:|---:|---:|"]
    for task in summary["tasks"]:
        lines.append(f"| {task['source_problem_id']} | {task['accepted']} | {_pct(task['pass_at_1'])} | {_pct(task['pass_at_8'])} | {_pct(task['pass_at_32'])} | {task['distinct_modes']} | {_pct(task['pcmd'])} |")
    lines += ["", f"PCMD is eligible on {summary['pcmd_eligible_tasks']}/3 tasks (at least 30 accepted samples required); missing values are not zero. Pass@k uses the without-replacement combinatorial estimator from all 64 draws. Distinct counts and the retained expected-distinct@k curves count verified behavior, not algorithm families. Curves describe subsets of the observed draws; they do not establish extrapolated scaling beyond 64 samples.", "",
        f"Each initially accepted program was replayed independently on the entire frozen suite; {summary['stability_rechecks']} such rechecks are retained. Only identical accepted behavior keys count as stable successes. Hard audit violations: {summary['hard_violation_count']}.", "",
        "| Terminal stage | Candidates |", "|---|---:|"]
    for stage, count in sorted(summary["failure_stages"].items()):
        lines.append(f"| {stage} | {count} |")
    gpu, timing = summary["gpu"], summary["timing"]
    lines += ["", "## Measured execution", "", f"Hardware: {', '.join(gpu['names'])}; {gpu['visible_count']} visible GPU(s). Generation took {timing['generation_wall_seconds']:.1f} s ({timing['generation_samples_per_second']:.2f} samples/s; {timing['generation_tokens_per_second']:.1f} output tokens/s). Verification took {timing['execution_wall_seconds']:.1f} s. Model loading took {timing['model_load_seconds']:.1f} s."]
    if summary["scheduler_allocated_gpu_hours"] is not None:
        lines += ["", f"Scheduler allocation: {summary['scheduler_allocated_gpu_hours']:.4f} GPU-hours."]
    else:
        lines += ["", "Final scheduler GPU-hour accounting is not available in the supplied structured receipt."]
    lines += [f"Evaluator elapsed-time estimate: {timing['estimated_allocated_gpu_hours_during_evaluator']:.4f} allocated GPU-hours; this excludes scheduler startup and teardown. These measurements do not estimate full training cost.", "",
        "## Interpretation and provenance", "", "This is a capability diagnostic on previously selected development tasks. A pass supports considering a separately specified paired engineering smoke. It establishes neither a treatment benefit nor held-out generalization and does not authorize a full study.", "",
        "The questions originate in human programming contests; CodeContests+ supplies generated tests and custom multiple-answer checkers. [CodeContests+ (Wang et al., 2025)](https://arxiv.org/abs/2506.05817). Repeated-sampling evaluation is motivated by [Large Language Monkeys (Brown et al., 2024)](https://arxiv.org/abs/2407.21787); this pilot does not reproduce its scale.", "",
        "Problem 359_B had its missing public equation image transcribed before sampling. Only the image placeholder was replaced, with original and effective statement hashes retained. Its result is consequently not a pure model-size comparison with the July pilot. [Original Codeforces equation image](https://espresso.codeforces.com/b54693338584d5268d5ec3ab8c4f8e90b87dea39.png). The other two statements are unchanged."]
    if summary["program_pairs"]:
        lines += ["", "## Examples of distinct verified behavior", "", "These are the earliest two accepted modes per eligible task, selected mechanically. Different keys show distinct canonical output behavior on the fixed suite; the source excerpts do not establish different algorithms. Complete source is retained in summary.json and the raw response sidecar."]
        for pair in summary["program_pairs"]:
            lines += ["", f"### {pair['source_problem_id']}"]
            for program in pair["programs"]:
                code = program["code"]
                excerpt = "\n".join(code.splitlines()[:24])[:1200]
                if excerpt != code:
                    excerpt += "\n# ... excerpt truncated; full source retained in summary.json"
                lines += ["", f"Sample {program['sample_index']}, canonical key `{program['canonical_key']}`:", "", "````python", excerpt, "````"]
    if summary["hard_violations"]:
        lines += ["", "## Retained audit violations", ""]
        lines.extend(f"- {value}" for value in summary["hard_violations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt_path = args.receipt or args.rootdir / "capability.json"
    receipt, responses, attempts = load_verified(receipt_path)
    scheduler_path = args.rootdir / "scheduler_receipt.json"
    scheduler = json.loads(scheduler_path.read_text()) if scheduler_path.exists() else None
    summary = summarize(receipt, responses, attempts, scheduler)
    summary["capability_receipt"] = {"path": str(receipt_path.resolve()), "sha256": sha256(receipt_path)}
    args.rootdir.mkdir(parents=True, exist_ok=True)
    (args.rootdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.rootdir / "report.md").write_text(report(summary))
    print(json.dumps({"status": summary["status"], "accepted": summary["accepted"], "report": str(args.rootdir / "report.md")}))


if __name__ == "__main__":
    main()
