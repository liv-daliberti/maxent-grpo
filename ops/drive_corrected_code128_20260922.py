#!/usr/bin/env python3
"""Continue the explicitly authorized, frozen coding128 pilot across sessions.

The training pair must already be submitted. This driver materializes only the
thirteen registered endpoints, checks terminal scheduler success, and runs the
independent training/endpoint audits. It never retries an ambiguous submission,
changes the protocol, launches a wider study, or chooses outcomes to retain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "var/seed_paper_eval/paper310/bin/python"
TERMINAL_FAILURES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        temporary.write_text(content)
        temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    env = dict(os.environ)
    env.update(OAT_ZERO_REPO_ROOT=str(ROOT), OAT_ZERO_SOURCE_ROOT=str(ROOT / "src"),
               OAT_ZERO_TESTLIB_ROOT=str(ROOT / "third_party/testlib"),
               OAT_ZERO_SANDBOX_SOURCE=str(ROOT / "ops/constructive_code_sandbox.c"),
               PYTHONPATH=str(ROOT / "ops") + ":" + str(ROOT / "src"),
               LD_LIBRARY_PATH=str(PYTHON.parent.parent / "lib"), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
    return env


def command(args, log):
    with Path(log).open("a") as handle:
        proc = subprocess.run([str(x) for x in args], cwd=ROOT, env=environment(), stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"command exited {proc.returncode}; see {log}")


def verify_frozen_config(execution):
    freeze = read(execution / "protocol_freeze.json")
    if freeze.get("status") != "FROZEN_AUTHORIZED_READY" or digest(execution / "campaign_config.json") != freeze["campaign_config_sha256"]:
        raise ValueError("campaign configuration differs from final protocol freeze")
    config = read(execution / "campaign_config.json")
    if config["scope"] != "fixed-code128-one-paired-seed" or config["gpu_hour_envelope"] != 47:
        raise ValueError("driver scope or allocation envelope changed")
    for item in config["bindings"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("campaign dependency changed: " + item["path"])
    schedule = read(config["schedule_path"])
    names = [row["name"] for row in schedule["shards"]]
    if len(names) != 13 or len(set(names)) != 13 or sum(x["total_completions"] for x in schedule["shards"]) != 23040:
        raise ValueError("endpoint schedule no longer has the thirteen fixed shards")
    if set(config["training_runs"]) != {"maxrl", "remax"} or config["training_runs"] != schedule["training_runs"]:
        raise ValueError("driver and endpoint training pair differ")
    for receipt_path in config["launch_gate_receipts"]:
        receipt = read(receipt_path)
        if receipt.get("status", "").lower() not in {"pass", "preflight_pass"}:
            raise ValueError("launch gate did not pass: " + receipt_path)
    return config, schedule


def all_run_paths(config, schedule):
    return [Path(x) for x in config["validation_runs"] + config.get("historical_runs", [])] + [Path(x) for x in config["training_runs"].values()] + [Path(schedule["run_parent"]) / row["name"] for row in schedule["shards"]]


def scheduler_rows(runs):
    submissions = {int(read(run / "submission.json")["job_id"]): run for run in runs if (run / "submission.json").exists()}
    if not submissions:
        return {}
    argv = ["sacct", "-X", "-n", "-P", "-j", ",".join(map(str, sorted(submissions))), "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES"]
    proc = subprocess.run(argv, text=True, capture_output=True, check=True)
    result = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        job, state, exit_code, seconds, tres = line.split("|")
        state = state.split()[0].rstrip("+")
        allocation = dict(x.split("=", 1) for x in tres.split(",") if "=" in x)
        count = int(allocation.get("gres/gpu", 0))
        if count not in (0, 1):
            raise ValueError("unexpected allocated GPU count")
        result[int(job)] = {"job_id": int(job), "state": state, "exit_code": exit_code, "elapsed_seconds": int(seconds), "allocated_gpu_count": count, "allocated_gpu_hours": int(seconds) * count / 3600, "allocation": tres, "run": str(submissions[int(job)])}
    return result


def terminal_success(run, scheduler):
    receipt = read(Path(run) / "submission.json")
    row = scheduler.get(int(receipt["job_id"]))
    if not row:
        return False
    if row["state"] in TERMINAL_FAILURES or (row["state"] == "COMPLETED" and row["exit_code"] != "0:0"):
        raise RuntimeError(f"job {row['job_id']} {row['state']} {row['exit_code']}; retain evidence and review recovery")
    return row["state"] == "COMPLETED" and row["exit_code"] == "0:0"


def expected_argv(run, shard):
    return ["sbatch", "--parsable", "--no-requeue", "--nodes=1", "--ntasks=1", "--gres=gpu:a6000:1", "--cpus-per-task=8", "--mem=64G", f"--time={int(shard['gpu_hour_cap'] * 60)}", "--account=mltheory", "--partition=lowprio", "--job-name=" + shard["name"], f"--output={run}/slurm-%j.out", f"--error={run}/slurm-%j.err", str(run / "run.slurm")]


def materialization_bindings(run, config, schedule):
    matches = [row for row in schedule["shards"] if row["name"] == run.name]
    if len(matches) != 1 or run != Path(schedule["run_parent"]) / run.name:
        raise ValueError("run is not a unique planned endpoint")
    shard = matches[0]
    path = Path(config["schedule_path"]).parent / "materialized" / run.name / "materialization.json"
    if not path.exists():
        raise RuntimeError("materialization did not complete; retain partial freeze for review")
    receipt, identity = read(path), read(run / "identity.json")
    expected = {"run": str(run), "name": run.name, "submitted": False,
                "frozen_identity_sha256": digest(run / "identity.json"),
                "config_sha256": digest(run / "config.json"),
                "schedule_sha256": digest(config["schedule_path"]),
                "request_sha256": identity["request_sha256"],
                "gpu_hour_cap": shard["gpu_hour_cap"],
                "total_completions": shard["total_completions"]}
    if any(receipt.get(k) != v for k, v in expected.items()) or identity["config_sha256"] != expected["config_sha256"]:
        raise ValueError("completed materialization binding differs from run/config/schedule")
    intent = read(run / "submission_intent.json")
    if intent["argv"] != expected_argv(run, shard) or intent["authorized_gpu_hour_ceiling"] != shard["gpu_hour_cap"]:
        raise ValueError("submission argv/time differs from registered shard allocation")
    return {"materialization": {"path": str(path), "sha256": digest(path)},
            "files": {name: digest(run / name) for name in ("identity.json", "config.json", "run.slurm", "submission_intent.json")},
            "schedule_sha256": digest(config["schedule_path"]), "name": run.name}


def seal_materialization(run, config, schedule):
    # Called only immediately after a successful preparer process. An interrupted
    # call without this seal requires review, never blind restart submission.
    value = materialization_bindings(run, config, schedule)
    write(run / "controller_ready.json", {"status": "pass", "created_at": now(), **value}, exclusive=True)


def verify_materialization_ready(run, config, schedule):
    ready_path = run / "controller_ready.json"
    if not ready_path.exists():
        raise RuntimeError("missing controller materialization seal; review interrupted preparation before submission")
    ready = read(ready_path)
    if ready.get("status") != "pass":
        raise ValueError("materialization readiness did not pass")
    current = materialization_bindings(run, config, schedule)
    if any(ready.get(key) != value for key, value in current.items()):
        raise ValueError("materialized script, intent, source or configuration changed after readiness")


def submit(run, config, schedule):
    run = Path(run)
    verify_materialization_ready(run, config, schedule)
    if (run / "submission.json").exists():
        return read(run / "submission.json")
    if (run / "submission_started.json").exists():
        raise RuntimeError("ambiguous prior submission; inspect scheduler before retry: " + str(run))
    intent = read(run / "submission_intent.json")
    argv = intent["argv"]
    if argv[0] != "sbatch" or "--gres=gpu:a6000:1" not in argv or "--partition=lowprio" not in argv:
        raise ValueError("endpoint allocation differs from the frozen contract")
    if digest(run / "config.json") != read(run / "identity.json")["config_sha256"]:
        raise ValueError("endpoint configuration drift before submission")
    reserved = intent["authorized_gpu_hour_ceiling"]
    for other in all_run_paths(config, schedule):
        if (other / "submission.json").exists():
            reserved += read(other / "submission_intent.json")["authorized_gpu_hour_ceiling"]
    if reserved > config["gpu_hour_envelope"] + 1e-9:
        raise ValueError("sum of submitted allocation ceilings would exceed authorized envelope")
    attempt = {"created_at": now(), "argv": argv, "identity_sha256": digest(run / "identity.json"), "submission_intent_sha256": digest(run / "submission_intent.json")}
    write(run / "submission_started.json", attempt, exclusive=True)
    proc = subprocess.run(argv, text=True, capture_output=True)
    receipt = {**attempt, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    if proc.returncode == 0:
        receipt["job_id"] = int(proc.stdout.strip().split(";")[0])
    write(run / "submission.json", receipt, exclusive=True)
    if proc.returncode:
        raise RuntimeError("endpoint submission failed: " + proc.stderr)
    print(f"Submitted {run.name}: {receipt['job_id']}", flush=True)
    return receipt


def step(execution):
    config, schedule = verify_frozen_config(execution)
    audits = execution / "audits"
    audits.mkdir(exist_ok=True)
    runs = all_run_paths(config, schedule)
    scheduler = scheduler_rows(runs)
    train_runs = {a: Path(p) for a, p in config["training_runs"].items()}
    initialized = all((p / "training/identity.json").exists() and read(p / "training/identity.json").get("initial_trainable_parameters_sha256") for p in train_runs.values())
    training_terminal = [terminal_success(p, scheduler) for p in train_runs.values()]
    trained = all(training_terminal)
    training_audit = audits / "training_pair.json"
    if trained and not training_audit.exists():
        command([PYTHON, ROOT / "ops/audit_real_domains_corrected_training_20260921.py", "--maxrl-run", train_runs["maxrl"], "--remax-run", train_runs["remax"], "--output", training_audit], audits / "training_pair.log")
    if trained and read(training_audit).get("status", "").lower() != "pass":
        raise ValueError("completed training pair failed independent audit")
    endpoints_complete = True
    for shard in schedule["shards"]:
        run = Path(schedule["run_parent"]) / shard["name"]
        ready = initialized if shard["arm"] == "base" else trained
        if ready:
            if not (run / "identity.json").exists():
                command([PYTHON, ROOT / "ops/prepare_real_domains_native_endpoints_20260922.py", "materialize", "--schedule", config["schedule_path"], "--name", shard["name"]], execution / "materialization.log")
                seal_materialization(run, config, schedule)
            submit(run, config, schedule)
        if not (run / "submission.json").exists() or not terminal_success(run, scheduler):
            endpoints_complete = False
            continue
        audit = audits / (shard["name"] + ".json")
        if not audit.exists():
            command([PYTHON, ROOT / "ops/audit_real_domains_native_hf_20260922.py", "--run", run, "--output", audit], audits / (shard["name"] + ".log"))
        if read(audit).get("status", "").lower() != "pass":
            raise ValueError("endpoint failed independent audit: " + shard["name"])
    summary = execution / "summary.json"
    if trained and endpoints_complete and not summary.exists():
        command([PYTHON, ROOT / "ops/audit_real_domains_native_hf_20260922.py", "--schedule", config["schedule_path"], "--output", summary], execution / "summary.log")
    if summary.exists() and read(summary).get("status", "").lower() != "pass":
        raise ValueError("complete endpoint aggregation did not pass")
    stage = "complete" if summary.exists() and trained and endpoints_complete else "evaluating" if trained else "training" if initialized else "waiting_for_training_allocation"
    details = {"schema": "corrected-code128-driver-status-20260922-v1", "status": stage, "updated_at": now(), "training_audit_complete": training_audit.exists(), "summary": str(summary) if summary.exists() else None, "jobs": list(scheduler.values()), "allocated_gpu_hours_so_far": sum(r["allocated_gpu_hours"] for r in scheduler.values()), "allocation_envelope_gpu_hours": 47, "training": {}}
    for arm, run in train_runs.items():
        progress = run / "training/progress.json"
        details["training"][arm] = read(progress) if progress.exists() else None
    write(execution / "campaign_status.json", details)
    print(json.dumps({"at": details["updated_at"], "status": stage, "jobs": len(scheduler), "gpu_hours": details["allocated_gpu_hours_so_far"]}), flush=True)
    return stage == "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    execution = args.execution.resolve()
    with (execution / "controller.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                complete = step(execution)
                if complete or args.once:
                    return
                time.sleep(45)
        except Exception as exc:
            write(execution / "campaign_failure.json", {"status": "needs_review", "at": now(), "error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    main()
