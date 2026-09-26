#!/usr/bin/env python3
"""Fail-closed terminal audit and development-only analysis for E117 Stage 1."""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_e117_same_plumbing_component_preflight as mechanism  # noqa: E402
import build_e117_stage1_development_results as builder  # noqa: E402
import launch_e117_stage1_development as stage  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_mechanism_contract() -> None:
    mechanism.e117.SEED = 201
    mechanism.e117.TRAIN_ROWS = stage.TRAIN_ROWS
    mechanism.e117.PASSES = stage.PASSES
    mechanism.e117.TARGET_STEPS = stage.TARGET_STEPS
    mechanism.e117.CHECKPOINT_INTERVAL = stage.EVAL_INTERVAL
    mechanism.e117.EVAL_DRAWS = stage.EVAL_DRAWS


def accounting(job_ids: list[int]) -> dict[int, dict[str, Any]]:
    result = subprocess.run(
        [
            "sacct", "-X", "-n", "-P", "-j", ",".join(str(value) for value in job_ids),
            "--format=JobIDRaw,State,ExitCode,Restarts,NodeList,ElapsedRaw,AllocTRES,Start,End",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not query Stage 1 accounting: {result.stderr.strip()}")
    rows: dict[int, dict[str, Any]] = {}
    wanted = set(job_ids)
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 9 or not fields[0].isdigit():
            continue
        job_id = int(fields[0])
        if job_id not in wanted:
            continue
        rows[job_id] = {
            "state": fields[1].split("+", 1)[0],
            "exit_code": fields[2],
            "restarts": int(fields[3] or 0),
            "node": fields[4],
            "gpu_seconds": int(fields[5] or 0),
            "alloc_tres": fields[6],
            "start": fields[7],
            "end": fields[8],
        }
    return rows


def rows(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open(encoding="utf-8", errors="strict") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict) or not mechanism.finite_json(row):
                raise RuntimeError(f"{path}:{line_number}: invalid JSON object")
            result.append(row)
    return result


def coverage_receipts(debug: Path) -> dict[int, list[dict[str, Any]]]:
    path = debug / "eval_mode_coverage_draws.jsonl"
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows(path):
        step = row.get("step")
        if type(step) is not int or step not in stage.CHECKPOINTS:
            raise RuntimeError(f"{path}: invalid/extra evaluation step {step!r}")
        by_step[step].append(row)
    if set(by_step) != set(stage.CHECKPOINTS):
        raise RuntimeError(f"{path}: evaluation checkpoint coverage drifted")
    for step, selected in by_step.items():
        sampled = [row for row in selected if row.get("evaluation_kind") == builder.SAMPLED_KIND]
        greedy = [row for row in selected if row.get("evaluation_kind") == "deterministic_greedy_trace_neutral"]
        if len(sampled) != stage.EVAL_DRAWS or len(greedy) != 1:
            raise RuntimeError(f"{path}: checkpoint {step} does not have 16 sampled plus one greedy row")
        if [row.get("draw_index") for row in sampled] != list(range(stage.EVAL_DRAWS)):
            raise RuntimeError(f"{path}: checkpoint {step} draw order drifted")
        if greedy[0].get("draw_index") is not None or len(selected) != stage.EVAL_DRAWS + 1:
            raise RuntimeError(f"{path}: checkpoint {step} evaluation kinds drifted")
        for draw, row in enumerate(sampled):
            expected = {
                "benchmark": "multi_answer", "sample_count": 8,
                "schema_version": 1, "seed": stage.EVAL_SEED_BASE + draw,
                "temperature": 1.0,
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"{path}: checkpoint {step}/draw {draw} request metadata drifted")
            prompts = row.get("prompts")
            if not isinstance(prompts, list) or len(prompts) != 128:
                raise RuntimeError(f"{path}: checkpoint {step}/draw {draw} prompt count drifted")
    return dict(by_step)


def validate_provenance(ledger: dict[str, Any]) -> list[str]:
    failures = []
    checks = (
        ("protocol", "protocol_sha256"),
        ("scheduler_amendment", "scheduler_amendment_sha256"),
        ("audit_scheduler_amendment", "audit_scheduler_amendment_sha256"),
        ("launcher", "launcher_sha256"),
        ("storage_amendment", "storage_amendment_sha256"),
        ("superseded_storage_unsafe_ledger", "superseded_storage_unsafe_ledger_sha256"),
        ("superseded_storage_unsafe_audit_job", "superseded_storage_unsafe_audit_job_sha256"),
        ("terminal_auditor", "terminal_auditor_sha256"),
        ("table_builder", "table_builder_sha256"),
        ("effective_contract_manifest", "effective_contract_manifest_sha256"),
        ("r2_ledger", "r2_ledger_sha256"),
        ("r2_audit", "r2_audit_sha256"),
    )
    for path_key, digest_key in checks:
        path = Path(str(ledger.get(path_key, "")))
        if not path.is_file() or sha256(path) != ledger.get(digest_key):
            failures.append(f"frozen provenance drifted: {path_key}")
    snapshot = Path(str(ledger.get("snapshot_root", "")))
    identity = snapshot / "SNAPSHOT_IDENTITY.json"
    if not identity.is_file() or sha256(identity) != ledger.get("snapshot_identity_sha256"):
        failures.append("Stage 1 snapshot identity drifted")
    try:
        stage.validate_v11(stage.root())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        failures.append(f"v11 binding failed: {error}")
    try:
        r2 = json.loads(Path(str(ledger["r2_audit"])).read_text(encoding="utf-8"))
        if (
            r2.get("passed") is not True
            or r2.get("failures") != []
            or r2.get("stage1_execution_readiness", {}).get("ready") is not True
            or ledger.get("r2_official_audit_job_id") != stage.OFFICIAL_R2_AUDIT_JOB
        ):
            failures.append("official R2 authorization no longer validates")
    except (KeyError, OSError, json.JSONDecodeError) as error:
        failures.append(f"official R2 authorization unreadable: {error}")
    return failures


def validate_design(ledger: dict[str, Any], runs: list[dict[str, Any]]) -> list[str]:
    failures = []
    expected = {
        "schema": "e117_stage1_development_jobs_v1",
        "released": True,
        "development_only": True,
        "confirmation_reserve_read": False,
        "pointmaze": "excluded",
        "sentinels": [f"{scale}/{domain}" for scale, domain in stage.SENTINELS],
        "arms": list(stage.ARMS),
        "training_seeds": list(stage.SEEDS),
        "train_rows": stage.TRAIN_ROWS,
        "passes": stage.PASSES,
        "target_steps": stage.TARGET_STEPS,
        "checkpoint_interval_steps": stage.EVAL_INTERVAL,
        "checkpoints": list(stage.CHECKPOINTS),
        "evaluation_draws": list(range(stage.EVAL_DRAWS)),
        "evaluation_request_seeds": [stage.EVAL_SEED_BASE + draw for draw in range(stage.EVAL_DRAWS)],
        "storage_policy": {
            "resume_interval_steps": stage.RESUME_INTERVAL,
            "maximum_resume_checkpoints": stage.MAX_RESUME_CHECKPOINTS,
            "prune_resume_on_success": True,
            "terminal_model_export": False,
        },
        "concurrency_policy": {
            "kind": "one completion-serialized lane per physical node",
            "dependency": "afterok",
            "nodes": ["node202", "node203"],
            "maximum_active_jobs": 2,
            "maximum_active_jobs_per_node": 1,
            "measured_worst_case_atomic_checkpoint_gib": 104,
        },
    }
    for key, value in expected.items():
        if ledger.get(key) != value:
            failures.append(f"registered design drifted: {key}")
    identities = {
        (str(run.get("scale")), str(run.get("domain")), int(run.get("seed", -1)), str(run.get("arm")))
        for run in runs
    }
    expected_identities = {
        (scale, domain, seed, arm)
        for scale, domain in stage.SENTINELS for seed in stage.SEEDS for arm in stage.ARMS
    }
    if identities != expected_identities or len(runs) != 36:
        failures.append("Stage 1 ledger cell identity drifted")
    return failures


def validate_launch_blocks(
    ledger: dict[str, Any], runs: list[dict[str, Any]], status: dict[int, dict[str, Any]]
) -> list[str]:
    failures = []
    varying = {
        "SAVE_PATH", "RUN_STAMP", "OAT_ZERO_SEMANTIC_SHANNON_COEF",
        "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY",
    }
    by_block: dict[tuple[str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    run_by_block: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        block = (str(run["scale"]), str(run["domain"]), int(run["seed"]))
        arm = str(run["arm"])
        record = str(run["held_scheduler_record"])
        try:
            environment_text = mechanism.exported_environment_text(record)
            environment = mechanism.exported_environment(record)
        except ValueError as error:
            failures.append(f"{block}/{arm}: scheduler export invalid: {error}")
            continue
        if hashlib.sha256(environment_text.encode()).hexdigest() != run.get("scientific_environment_sha256"):
            failures.append(f"{block}/{arm}: scheduler export hash drifted")
        by_block[block][arm] = environment
        run_by_block[block].append(run)
        job_id = int(run["job_id"])
        actual = status.get(job_id, {})
        if actual.get("node") != run.get("node") or run.get("node") != stage.NODES[block]:
            failures.append(f"{block}/{arm}: actual/registered node drifted")
        if "gres/gpu:a5000=1" not in str(actual.get("alloc_tres", "")):
            failures.append(f"{block}/{arm}: A5000 allocation was not retained")
        expected_fixed = {
            "OAT_ZERO_SOURCE_ROOT": str(Path(ledger["snapshot_root"]) / "src"),
            "OAT_ZERO_OPS_SNAPSHOT_ROOT": str(Path(ledger["snapshot_root"]) / "ops"),
            "OAT_ZERO_SEED": str(run["seed"]),
            "OAT_ZERO_MAX_TRAIN": str(stage.TRAIN_ROWS),
            "OAT_ZERO_NUM_PROMPT_EPOCH": str(stage.PASSES),
            "OAT_ZERO_MAX_PROMPT_EPOCHS": str(stage.PASSES),
            "OAT_ZERO_EVAL_PROMPT_INTERVAL": str(stage.EVAL_INTERVAL),
            "OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS": str(stage.EVAL_DRAWS),
            "OAT_ZERO_EVAL_MODE_COVERAGE_SEED": str(stage.EVAL_SEED_BASE),
            "OAT_ZERO_RESUME_STEPS": str(stage.RESUME_INTERVAL),
            "OAT_ZERO_EXPORT_STEPS": "-1",
            "OAT_ZERO_MAX_RESUME_NUM": str(stage.MAX_RESUME_CHECKPOINTS),
            "OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_ADMISSION_COMPUTE_ONLY": "1" if arm == "c" else "0",
            "OAT_ZERO_SEMANTIC_SHANNON_COEF": "0.1" if arm == "f" else "0.0",
        }
        for key, value in expected_fixed.items():
            if environment.get(key) != value:
                failures.append(f"{block}/{arm}: export drifted: {key}")
        eval_path = environment.get("OAT_ZERO_EVAL_DATA", "")
        expected_eval = str(stage.root() / "var/data/e117_evaluation_reserve_v1/development" / block[1] / "eval")
        if eval_path != expected_eval or "confirmation" in eval_path:
            failures.append(f"{block}/{arm}: development reserve path drifted")
    for block, environments in by_block.items():
        if set(environments) != set(stage.ARMS):
            failures.append(f"{block}: C/P/F export block incomplete")
            continue
        common = {
            arm: {key: value for key, value in env.items() if key not in varying}
            for arm, env in environments.items()
        }
        if len({json.dumps(value, sort_keys=True) for value in common.values()}) != 1:
            failures.append(f"{block}: C/P/F scientific exports differ")
        ordered = sorted(run_by_block[block], key=lambda row: int(row["start_position"]))
        expected_order = list(stage.START_ORDERS[block[2]])
        if [str(row["arm"]) for row in ordered] != expected_order:
            failures.append(f"{block}: registered start order drifted")
        starts = [str(status.get(int(row["job_id"]), {}).get("start", "")) for row in ordered]
        if any(not value or value == "Unknown" for value in starts) or any(
            right < left for left, right in zip(starts, starts[1:])
        ):
            failures.append(f"{block}: actual start order was not retained: {starts}")
    for node in ("node202", "node203"):
        lane = sorted(
            (run for run in runs if str(run.get("node")) == node),
            key=lambda row: int(row.get("node_lane_position", -1)),
        )
        if len(lane) != 18 or [row.get("node_lane_position") for row in lane] != list(range(18)):
            failures.append(f"{node}: completion lane identity drifted")
            continue
        for index, row in enumerate(lane):
            dependency = row.get("start_dependency_job_id")
            expected_dependency = None if index == 0 else int(lane[index - 1]["job_id"])
            if dependency != expected_dependency or row.get("start_dependency_type") != "afterok":
                failures.append(f"{node}: completion dependency chain drifted at lane position {index}")
            held_dependency = stage.field(str(row["held_scheduler_record"]), "Dependency")
            if expected_dependency is None:
                if held_dependency not in ("", "(null)"):
                    failures.append(f"{node}: first lane job had a dependency")
            elif f"afterok:{expected_dependency}" not in held_dependency:
                failures.append(f"{node}: held afterok dependency drifted at lane position {index}")
            if index == 0:
                continue
            previous = status.get(int(lane[index - 1]["job_id"]), {})
            current = status.get(int(row["job_id"]), {})
            previous_end = str(previous.get("end", ""))
            current_start = str(current.get("start", ""))
            if (
                not previous_end or previous_end == "Unknown"
                or not current_start or current_start == "Unknown"
                or current_start < previous_end
            ):
                failures.append(f"{node}: jobs overlapped or lacked terminal times at lane position {index}")
    return failures


def metric_sum(train: dict[int, dict[str, Any]], key: str) -> float | str:
    values = []
    for row in train.values():
        if key not in row:
            return "unavailable"
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return "unavailable"
        values.append(float(value))
    return sum(values)


def product_sum(train: dict[int, dict[str, Any]], left: str, right: str) -> float | str:
    total = 0.0
    for row in train.values():
        values = (row.get(left), row.get(right))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            return "unavailable"
        total += float(values[0]) * float(values[1])
    return total


def compute_accounting(
    run: dict[str, Any], train: dict[int, dict[str, Any]], status: dict[str, Any]
) -> dict[str, Any]:
    return {
        "optimizer_updates": len(train),
        "gpu": {"class": "a5000", "count": 1, "seconds": status.get("gpu_seconds", "unavailable")},
        "node": status.get("node", "unavailable"),
        "restarts": status.get("restarts", "unavailable"),
        "paths": {
            "neutral": {
                "generated_rows": metric_sum(train, "actor/num_data"),
                "charged_prompt_tokens": "unavailable",
                "charged_response_token_budget": product_sum(train, "actor/num_data", "actor/sampling_max_tokens"),
                "realized_prompt_tokens": "unavailable",
                "realized_response_tokens": product_sum(train, "actor/num_data", "actor/response_tok_len"),
                "response_token_total_derivation": "num_data * mean response_tok_len",
            },
            "fixed_control_request": {
                "generated_rows": metric_sum(train, "actor/counterfactual_fixed_control_rows_generated"),
                "charged_prompt_tokens": "unavailable",
                "charged_response_token_budget": metric_sum(train, "actor/counterfactual_fixed_control_charged_response_token_budget"),
                "realized_prompt_tokens": metric_sum(train, "actor/counterfactual_fixed_control_realized_prompt_tokens"),
                "realized_response_tokens": metric_sum(train, "actor/counterfactual_fixed_control_realized_response_tokens"),
            },
            "proposal": {
                "generated_rows": metric_sum(train, "actor/counterfactual_proposal_rows_generated"),
                "charged_prompt_tokens": "unavailable",
                "charged_response_token_budget": "unavailable",
                "realized_prompt_tokens": "unavailable",
                "realized_response_tokens": "unavailable",
            },
            "replay": {
                "generated_rows": "unavailable",
                "charged_prompt_tokens": "unavailable",
                "charged_response_token_budget": metric_sum(train, "train/canonical_replay_charged_response_token_budget"),
                "realized_prompt_tokens": metric_sum(train, "train/canonical_replay_realized_prompt_tokens"),
                "realized_response_tokens": metric_sum(train, "train/canonical_replay_realized_response_tokens"),
            },
        },
        "unavailable_values_imputed": False,
        "cost_enters_scientific_gate": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    configure_mechanism_contract()
    ledger_path = args.ledger.resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    runs = list(ledger.get("runs", []))
    job_ids = [int(run["job_id"]) for run in runs]
    status = accounting(job_ids)
    noncompleted = {
        str(job_id): status.get(job_id, {"state": "UNKNOWN"})
        for job_id in job_ids
        if status.get(job_id, {}).get("state") != "COMPLETED"
        or status.get(job_id, {}).get("exit_code") != "0:0"
    }
    if noncompleted:
        payload = {
            "schema": "e117_stage1_development_audit_v1",
            "terminal": True,
            "passed": False,
            "development_analysis_run": False,
            "confirmation_reserve_read": False,
            "noncompleted_jobs": noncompleted,
            "failures": ["all 36 Stage 1 jobs did not complete successfully"],
        }
        stage.atomic_json(args.output.resolve(), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    failures = validate_design(ledger, runs)
    failures.extend(validate_provenance(ledger))
    failures.extend(validate_launch_blocks(ledger, runs, status))
    artifacts: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    compute = []
    for run in runs:
        identity = (str(run["scale"]), str(run["domain"]), int(run["seed"]), str(run["arm"]))
        run_dir = Path(str(run["run_dir"]))
        complete = run_dir / "TRAINING_COMPLETE.json"
        debug = run_dir / f"debug_job{int(run['job_id'])}"
        try:
            if not complete.is_file() or not debug.is_dir():
                raise RuntimeError("training completion receipt/debug directory absent")
            train = mechanism.training_rows(debug)
            evaluations = coverage_receipts(debug)
            summary, arm_failures = mechanism.audit_arm_rows(identity[3], train)
            failures.extend(f"{identity}: {value}" for value in arm_failures)
            artifacts[identity] = {"train": train, "evaluations": evaluations, "mechanism": summary}
            compute.append(
                {"scale": identity[0], "domain": identity[1], "training_seed": identity[2],
                 "arm": identity[3], "job_id": int(run["job_id"]),
                 "accounting": compute_accounting(run, train, status[int(run["job_id"])])}
            )
        except (IndexError, KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"{identity}: {error}")
    identity_keys = (
        "actor/sampling_request_seed",
        "actor/counterfactual_fixed_control_groups_generated",
        "actor/counterfactual_fixed_control_rows_generated",
        "actor/counterfactual_fixed_control_charged_response_token_budget",
        "actor/counterfactual_fixed_control_request_seed_min",
        "actor/counterfactual_fixed_control_request_seed_max",
    )
    for scale, domain in stage.SENTINELS:
        for seed in stage.SEEDS:
            block = (scale, domain, seed)
            if any((*block, arm) not in artifacts for arm in stage.ARMS):
                continue
            arm_rows = {arm: artifacts[(*block, arm)]["train"] for arm in stage.ARMS}
            for step in range(1, stage.TARGET_STEPS + 1):
                failures.extend(
                    f"{block}: {value}"
                    for value in mechanism.exact_values(arm_rows, step=step, keys=identity_keys)
                )
    results_payload = None
    if not failures:
        try:
            results_payload = builder.build(
                ledger_path,
                table_path=args.table.resolve(),
                output_path=args.results.resolve(),
                provenance={
                    "effective_contract_manifest_sha256": ledger["effective_contract_manifest_sha256"],
                    "snapshot_identity_sha256": ledger["snapshot_identity_sha256"],
                    "compute_accounting": compute,
                    "cost_enters_scientific_gate": False,
                    "confirmation_reserve_read": False,
                },
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures.append(f"development table/analyzer failed: {error}")
    payload = {
        "schema": "e117_stage1_development_audit_v1",
        "terminal": True,
        "passed": not failures,
        "development_analysis_run": results_payload is not None,
        "confirmation_reserve_read": False,
        "pointmaze": "excluded",
        "ledger": str(ledger_path),
        "ledger_sha256": sha256(ledger_path),
        "mechanism_blocks": [
            {"scale": key[0], "domain": key[1], "training_seed": key[2], "arm": key[3], "summary": value["mechanism"]}
            for key, value in sorted(artifacts.items())
        ],
        "compute_accounting": compute,
        "results": str(args.results.resolve()) if results_payload else None,
        "results_sha256": sha256(args.results.resolve()) if results_payload else None,
        "table": str(args.table.resolve()) if results_payload else None,
        "table_sha256": sha256(args.table.resolve()) if results_payload else None,
        "failures": failures,
    }
    stage.atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
