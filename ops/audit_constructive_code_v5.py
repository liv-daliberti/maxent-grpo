#!/usr/bin/env python3
"""Fail-closed audit for ConstructiveCode v5 paired smoke or Stage B."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
SRC = Path(os.environ.get("OAT_ZERO_SOURCE_ROOT", ROOT / "src")).resolve()
for path in (OPS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_constructive_code_v5 as train  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row {path}:{line_number}")
            rows.append(row)
    return rows


def resolve_repo_path(raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("identity contains an invalid artifact path")
    path = (ROOT / raw).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("identity artifact path leaves repository") from error
    return path


def finite_metric_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        not isinstance(value, float) or math.isfinite(value)
        for row in rows
        for value in row.values()
    )


def scheduler_state(job_id: int) -> str:
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-j",
            str(job_id),
            "--format=JobIDRaw,State,ExitCode",
            "-n",
            "-P",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )
    rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    root_rows = [row for row in rows if row[0] == str(job_id)]
    if len(root_rows) != 1 or len(root_rows[0]) < 3:
        raise ValueError(f"scheduler did not return one root row for {job_id}")
    return f"{root_rows[0][1]}|{root_rows[0][2]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=train.MODES, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--gate-identity", type=Path, required=True)
    parser.add_argument("--viability-receipt", type=Path, required=True)
    parser.add_argument("--paired-audit", type=Path, default=None)
    parser.add_argument("--slate-root", type=Path, required=True)
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-workers", type=int, default=16)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--execution-hash", required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def identity_records(identity: Mapping[str, Any], mode: str) -> dict[str, Mapping[str, Any]]:
    key = "runs" if mode == train.PAIRED_MODE else "cells"
    records = identity.get(key)
    if not isinstance(records, Mapping):
        raise ValueError(f"ConstructiveCode identity lacks {key}")
    expected = (
        set(train.ARMS)
        if mode == train.PAIRED_MODE
        else {
            f"{arm}/s{seed}" for arm in train.ARMS for seed in (43, 44, 45, 46, 47)
        }
    )
    if set(records) != expected or not all(
        isinstance(value, Mapping) for value in records.values()
    ):
        raise ValueError("ConstructiveCode identity does not contain exact run set")
    return {str(key): value for key, value in records.items()}


def validate_run(
    *,
    label: str,
    record: Mapping[str, Any],
    mode: str,
    identity: Path,
    protocol: Path,
    gate_audit: Path,
    gate_identity: Path,
    viability: Path,
    paired_audit: Path | None,
) -> dict[str, Any]:
    receipt_path = resolve_repo_path(record.get("receipt"))
    metrics_path = resolve_repo_path(record.get("metrics"))
    candidates_path = resolve_repo_path(record.get("candidates"))
    evaluations_path = (
        None
        if mode == train.PAIRED_MODE
        else resolve_repo_path(record.get("evaluations"))
    )
    for path in (receipt_path, metrics_path, candidates_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if mode == train.STAGE_B_MODE and (
        evaluations_path is None or not evaluations_path.is_file()
    ):
        raise FileNotFoundError(evaluations_path)
    receipt = read_json(receipt_path)
    metrics = read_jsonl(metrics_path)
    candidates = read_jsonl(candidates_path)
    evaluations = [] if evaluations_path is None else read_jsonl(evaluations_path)
    arm = label if mode == train.PAIRED_MODE else label.split("/s", 1)[0]
    seed = 78101 if mode == train.PAIRED_MODE else int(label.rsplit("s", 1)[1])
    expected_updates = 4 if mode == train.PAIRED_MODE else 48
    expected_candidates = expected_updates * train.SAMPLES
    if (
        receipt.get("schema") != "constructive-code-v5-run-receipt-v1"
        or receipt.get("status") != "complete"
        or receipt.get("mode") != mode
        or receipt.get("arm") != arm
        or receipt.get("seed") != seed
        or receipt.get("job_id") != record.get("job_id")
        or receipt.get("source_hash") != record.get("source_hash")
        or receipt.get("execution_hash") != record.get("execution_hash")
        or receipt.get("protocol_sha256") != train.sha256_file(protocol)
        or receipt.get("identity_sha256") != train.sha256_file(identity)
        or receipt.get("gate_audit_sha256") != train.sha256_file(gate_audit)
        or receipt.get("gate_identity_sha256") != train.sha256_file(gate_identity)
        or receipt.get("viability_receipt_sha256") != train.sha256_file(viability)
        or receipt.get("paired_audit_sha256")
        != (None if paired_audit is None else train.sha256_file(paired_audit))
        or receipt.get("metrics_sha256") != train.sha256_file(metrics_path)
        or receipt.get("candidate_ledger_sha256")
        != train.sha256_file(candidates_path)
        or receipt.get("evaluation_ledger_sha256")
        != (
            None
            if evaluations_path is None
            else train.sha256_file(evaluations_path)
        )
    ):
        raise ValueError(f"ConstructiveCode receipt identity drift: {label}")
    counts = receipt.get("counts", {})
    expected_evaluations = 0 if mode == train.PAIRED_MODE else 49
    expected_eval_programs = 0 if mode == train.PAIRED_MODE else 49 * 132
    if (
        counts.get("updates") != expected_updates
        or counts.get("training_requests") != expected_candidates
        or counts.get("training_terminal_executions") != expected_candidates
        or counts.get("policy_score_rows") != expected_candidates
        or counts.get("policy_response_token_slots")
        != expected_candidates * train.RESPONSE_TOKENS
        or counts.get("replay_score_rows_per_pass")
        != expected_updates * train.REPLAY_CAPACITY
        or counts.get("replay_score_passes") != 2
        or counts.get("evaluation_coordinates") != expected_evaluations
        or counts.get("evaluation_programs") != expected_eval_programs
    ):
        raise ValueError(f"ConstructiveCode fixed traversal drift: {label}")
    training_metrics = [
        row
        for row in metrics
        if row.get("schema") == "constructive-code-v5-training-metric-v1"
    ]
    evaluation_metrics = [
        row
        for row in metrics
        if row.get("schema")
        == "constructive-code-v5-stage-b-evaluation-metric-v1"
    ]
    if (
        len(training_metrics) != expected_updates
        or [row.get("trainer/global_step") for row in training_metrics]
        != list(range(1, expected_updates + 1))
        or len(evaluation_metrics) != expected_evaluations
        or [row.get("trainer/global_step") for row in evaluation_metrics]
        != list(range(expected_evaluations))
        or not finite_metric_rows(metrics)
    ):
        raise ValueError(f"ConstructiveCode metric schedule drift: {label}")
    if len(candidates) != expected_candidates:
        raise ValueError(f"ConstructiveCode candidate count drift: {label}")
    by_update = Counter(int(row.get("update", -1)) for row in candidates)
    if by_update != Counter({update: 16 for update in range(1, expected_updates + 1)}):
        raise ValueError(f"ConstructiveCode candidate grouping drift: {label}")
    for row in candidates:
        code = row.get("code")
        if (
            row.get("schema") != "constructive-code-v5-training-candidate-v1"
            or row.get("mode") != mode
            or row.get("arm") != arm
            or row.get("seed") != seed
            or not isinstance(code, str)
            or hashlib_sha256_text(code) != row.get("executed_source_sha256")
            or row.get("terminal_worker_record") is not True
            or not isinstance(row.get("replay"), Mapping)
            or train.hard_replay_violations(row["replay"])
        ):
            raise ValueError(f"ConstructiveCode candidate ledger drift: {label}")
    if mode == train.STAGE_B_MODE:
        if (
            len(evaluations) != 49
            or [row.get("update") for row in evaluations] != list(range(49))
            or any(row.get("programs") != 132 for row in evaluations)
            or any(
                not isinstance(row.get("tasks"), list)
                or len(row["tasks"]) != 4
                or any(
                    not isinstance(task.get("draws"), list)
                    or len(task["draws"]) != 4
                    or any(not isinstance(draw, list) or len(draw) != 8 for draw in task["draws"])
                    for task in row["tasks"]
                )
                for row in evaluations
            )
        ):
            raise ValueError(f"ConstructiveCode evaluation ledger drift: {label}")
    if arm == train.CONTROL:
        if any(
            float(row.get("applied_exploration_advantage_rms", math.nan)) != 0.0
            or float(row.get("canonical_replay_applied_score_gradient_l2", math.nan))
            != 0.0
            or float(row.get("canonical_replay_compute_only", math.nan)) != 1.0
            for row in training_metrics
        ):
            raise ValueError(f"ConstructiveCode control applied derivative: {label}")
    else:
        if any(
            (
                float(row.get("raw_exploration_advantage_rms", 0.0)) > 0.0
                and float(row.get("applied_exploration_advantage_rms", 0.0))
                <= 0.0
            )
            or (
                float(row.get("canonical_replay_raw_score_gradient_l2", 0.0))
                > 0.0
                and float(
                    row.get("canonical_replay_applied_score_gradient_l2", 0.0)
                )
                <= 0.0
            )
            or float(row.get("canonical_replay_compute_only", math.nan)) != 0.0
            for row in training_metrics
        ):
            raise ValueError(f"ConstructiveCode treatment skipped derivative: {label}")
    return {
        "label": label,
        "arm": arm,
        "seed": seed,
        "job_id": int(record["job_id"]),
        "receipt_path": receipt_path,
        "metrics_path": metrics_path,
        "candidates_path": candidates_path,
        "evaluation_path": evaluations_path,
        "receipt": receipt,
        "metrics": training_metrics,
        "candidates": candidates,
        "scheduler_state": scheduler_state(int(record["job_id"])),
    }


def hashlib_sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def second_replay(
    *,
    runs: Sequence[Mapping[str, Any]],
    tasks: Mapping[str, Any],
    base: Any,
    launcher: Path,
    runtime_root: Path,
    scratch_root: Path,
    workers: int,
) -> dict[str, Any]:
    rows = []
    for run in runs:
        for candidate in run["candidates"]:
            rows.append((str(run["label"]), candidate))
    replay_records = []
    mismatches = []
    grouped: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for label, row in rows:
        grouped[str(row["source_problem_id"])].append((label, row))
    global_slot = 0
    for problem_id in sorted(grouped):
        source_rows = grouped[problem_id]
        candidates = []
        for label, row in source_rows:
            candidates.append(
                {
                    "slot": global_slot,
                    "request_namespace": int(row["request_namespace"]),
                    "request_stream_offset": int(row["request_stream_offset"]),
                    "emitted_text_sha256": str(row["emitted_text_sha256"]),
                    "executed_source_sha256": str(row["executed_source_sha256"]),
                    "fence_stripped": bool(row["fence_stripped"]),
                    "token_count": int(row["token_count"]),
                    "finish_reason": str(row["finish_reason"]),
                    "code": str(row["code"]),
                }
            )
            global_slot += 1
        replayed = train.execute_candidates(
            candidates=candidates,
            task=tasks[problem_id],
            base=base,
            launcher=launcher,
            runtime_root=runtime_root,
            scratch_root=scratch_root,
            workers=workers,
        )
        for (label, original), observed in zip(source_rows, replayed):
            match = (
                observed["accepted"] == original["accepted"]
                and observed["canonical_key"] == original["canonical_key"]
                and observed["executed_source_sha256"]
                == original["executed_source_sha256"]
            )
            record = {
                "label": label,
                "update": int(original["update"]),
                "slot": int(original["slot"]),
                "source_problem_id": problem_id,
                "executed_source_sha256": original["executed_source_sha256"],
                "original_accepted": bool(original["accepted"]),
                "replayed_accepted": bool(observed["accepted"]),
                "original_key": original["canonical_key"],
                "replayed_key": observed["canonical_key"],
                "match": match,
            }
            replay_records.append(record)
            if not match:
                mismatches.append(record)
    return {
        "count": len(replay_records),
        "mismatch_count": len(mismatches),
        "records_sha256": train.canonical_sha256(replay_records),
        "mismatches": mismatches,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"fresh ConstructiveCode audit required: {args.output}")
    if args.execution_workers != 16:
        raise ValueError("ConstructiveCode audit worker count drift")
    for path in (
        args.identity,
        args.submission,
        args.protocol,
        args.gate_audit,
        args.gate_identity,
        args.viability_receipt,
        args.slate_root / "manifest.json",
        args.image,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.mode == train.STAGE_B_MODE and (
        args.paired_audit is None or not args.paired_audit.is_file()
    ):
        raise ValueError("ConstructiveCode Stage B audit requires paired audit")
    if args.mode == train.PAIRED_MODE and args.paired_audit is not None:
        raise ValueError("ConstructiveCode paired audit cannot consume itself")
    identity = read_json(args.identity)
    submission = read_json(args.submission)
    records = identity_records(identity, args.mode)
    expected_schema = (
        "constructive-code-v5-paired-smoke-identity-v1"
        if args.mode == train.PAIRED_MODE
        else "constructive-code-v5-stage-b-identity-v1"
    )
    if (
        identity.get("schema") != expected_schema
        or identity.get("source_hash") != args.source_hash
        or identity.get("execution_hash") != args.execution_hash
        or identity.get("protocol_sha256") != train.sha256_file(args.protocol)
        or identity.get("gate_audit_sha256") != train.sha256_file(args.gate_audit)
        or identity.get("gate_identity_sha256") != train.sha256_file(
            args.gate_identity
        )
        or identity.get("viability_receipt_sha256") != train.sha256_file(
            args.viability_receipt
        )
    ):
        raise ValueError("ConstructiveCode audit identity header drift")
    expected_submission_schema = (
        "constructive-code-v5-paired-smoke-submission-v1"
        if args.mode == train.PAIRED_MODE
        else "constructive-code-v5-stage-b-submission-v1"
    )
    if (
        submission.get("schema") != expected_submission_schema
        or submission.get("identity_sha256") != train.sha256_file(args.identity)
        or submission.get("held_job_audit") != "pass"
        or submission.get("released") is not True
        or not isinstance(submission.get("scheduler_records"), Mapping)
        or set(submission["scheduler_records"]) != set(records)
    ):
        raise ValueError("ConstructiveCode held-job submission contract drift")
    if args.mode == train.STAGE_B_MODE:
        assert args.paired_audit is not None
        if identity.get("paired_audit_sha256") != train.sha256_file(
            args.paired_audit
        ):
            raise ValueError("ConstructiveCode Stage B paired-audit binding drift")
        manifest = resolve_repo_path(identity.get("comparative_jobs_manifest"))
        if (
            not manifest.is_file()
            or identity.get("comparative_jobs_manifest_sha256")
            != train.sha256_file(manifest)
        ):
            raise ValueError("ConstructiveCode comparative manifest hash drift")
        manifest_lines = manifest.read_text(encoding="utf-8").splitlines()
        expected_lines = ["arm\tseed\tjob_id\trun_stamp"]
        for arm in train.ARMS:
            for seed in (43, 44, 45, 46, 47):
                label = f"{arm}/s{seed}"
                expected_lines.append(
                    f"{arm}\t{seed}\t{records[label]['job_id']}\t"
                    f"cce70_clean_stage_b_05b_12pass_{arm}_s{seed}"
                )
        if manifest_lines != expected_lines:
            raise ValueError("ConstructiveCode comparative manifest row drift")

    runs = [
        validate_run(
            label=label,
            record=record,
            mode=args.mode,
            identity=args.identity,
            protocol=args.protocol,
            gate_audit=args.gate_audit,
            gate_identity=args.gate_identity,
            viability=args.viability_receipt,
            paired_audit=args.paired_audit,
        )
        for label, record in sorted(records.items())
    ]
    if any(run["scheduler_state"] != "COMPLETED|0:0" for run in runs):
        raise ValueError("ConstructiveCode run is not scheduler-terminal success")
    initial_hashes = {run["receipt"]["initial_model_tree_sha256"] for run in runs}
    if len(initial_hashes) != 1:
        raise ValueError("ConstructiveCode arms did not share one checkpoint")
    if args.mode == train.PAIRED_MODE:
        if any(
            run["receipt"]["counts"]["verified_training_candidates"] < 1
            or run["receipt"]["counts"]["multimode_updates"] < 1
            for run in runs
        ):
            raise ValueError("ConstructiveCode paired smoke lacks success or multimode")
    else:
        by_seed = defaultdict(dict)
        for run in runs:
            by_seed[int(run["seed"])][str(run["arm"])] = run
        for seed, arms in by_seed.items():
            if set(arms) != set(train.ARMS):
                raise ValueError(f"ConstructiveCode seed {seed} lacks paired arms")
            control = arms[train.CONTROL]["receipt"]["counts"]
            treatment = arms[train.TREATMENT]["receipt"]["counts"]
            traversal_keys = (
                "updates",
                "training_requests",
                "training_terminal_executions",
                "policy_score_rows",
                "policy_response_token_slots",
                "replay_score_rows_per_pass",
                "replay_score_passes",
                "evaluation_coordinates",
                "evaluation_programs",
            )
            if any(control[key] != treatment[key] for key in traversal_keys):
                raise ValueError(f"ConstructiveCode compute traversal differs: seed {seed}")

    gate = read_json(args.gate_audit)
    args.runtime_root.parent.mkdir(parents=True, exist_ok=True)
    args.build_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    _v5, _replay_v5, base, _materialize = train._replay_modules()
    launcher_sha = base.build_launcher(base.SANDBOX_SOURCE, args.launcher)
    runtime_identity = base.prepare_runtime(args.image, args.runtime_root)
    problem_ids = (
        train.DEVELOPMENT_PROBLEMS
        if args.mode == train.PAIRED_MODE
        else train.TRAIN_PROBLEMS
    )
    expected_split = {
        key: "development" if args.mode == train.PAIRED_MODE else "train"
        for key in problem_ids
    }
    tasks, _public, checker_builds, source_manifest = train.load_frozen_tasks(
        problem_ids=problem_ids,
        expected_split=expected_split,
        slate_root=args.slate_root,
        v1_root=args.v1_root,
        build_root=args.build_root,
        gate=gate,
    )
    replay_audit = second_replay(
        runs=runs,
        tasks=tasks,
        base=base,
        launcher=args.launcher,
        runtime_root=args.runtime_root,
        scratch_root=args.scratch_root,
        workers=args.execution_workers,
    )
    expected_second_replays = 128 if args.mode == train.PAIRED_MODE else 7680
    if (
        replay_audit["count"] != expected_second_replays
        or replay_audit["mismatch_count"] != 0
    ):
        raise ValueError("ConstructiveCode second official-checker replay failed")
    payload = {
        "schema": (
            "constructive-code-v5-paired-smoke-audit-v1"
            if args.mode == train.PAIRED_MODE
            else "constructive-code-v5-stage-b-audit-v1"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "decision": (
            "eligible_for_ten_constructive_code_stage_b_jobs"
            if args.mode == train.PAIRED_MODE
            else "constructive_code_terminal_eligible"
        ),
        "mode": args.mode,
        "job_id": int(args.job_id),
        "source_hash": args.source_hash,
        "execution_hash": args.execution_hash,
        "identity_sha256": train.sha256_file(args.identity),
        "submission_sha256": train.sha256_file(args.submission),
        "protocol_sha256": train.sha256_file(args.protocol),
        "gate_audit_sha256": train.sha256_file(args.gate_audit),
        "gate_identity_sha256": train.sha256_file(args.gate_identity),
        "viability_receipt_sha256": train.sha256_file(args.viability_receipt),
        "paired_audit_sha256": (
            None
            if args.paired_audit is None
            else train.sha256_file(args.paired_audit)
        ),
        "source_manifest_sha256": train.canonical_sha256(source_manifest),
        "runtime": {
            "launcher_binary_sha256": launcher_sha,
            "runtime_identity": asdict(runtime_identity),
            "checker_builds": checker_builds,
        },
        "checks": {
            "exact_run_manifest": True,
            "shared_initial_checkpoint": True,
            "terminal_scheduler_success": True,
            "held_job_submission_contract": True,
            "fixed_policy_traversal": True,
            "fixed_replay_traversal": True,
            "control_derivatives_zero": True,
            "treatment_derivatives_applied_when_eligible": True,
            "finite_metrics": True,
            "second_official_checker_replay": True,
            "compute_match_by_seed": True,
        },
        "runs": {
            run["label"]: {
                "job_id": run["job_id"],
                "scheduler_state": run["scheduler_state"],
                "receipt_sha256": train.sha256_file(run["receipt_path"]),
                "metrics_sha256": train.sha256_file(run["metrics_path"]),
                "candidates_sha256": train.sha256_file(run["candidates_path"]),
                "verified_training_candidates": run["receipt"]["counts"][
                    "verified_training_candidates"
                ],
                "multimode_updates": run["receipt"]["counts"][
                    "multimode_updates"
                ],
            }
            for run in runs
        },
        "second_replay": replay_audit,
        "violations": [],
    }
    train.atomic_json(args.output, payload)
    print(
        "[constructive-v5-audit] "
        f"mode={args.mode} status=pass runs={len(runs)} "
        f"second_replays={replay_audit['count']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
