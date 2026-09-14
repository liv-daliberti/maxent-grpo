#!/usr/bin/env python3
"""One-line E112-R1 presentation update with a coordinated six-CPU-guard handoff.

prepare writes a reviewed candidate and immutable plan only. stage creates idle
CPU successors. apply replaces the helper only after all old supervisors have
stopped at the shared ledger lock and relinquished their singleton locks.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import ExitStack
import copy
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
import time
import prioritize_e118_capacity_20260905 as base

ROOT = base.ROOT
ART = ROOT / "var/artifacts/supervisor_display_e112r1_20260909"
PLAN = ART / "handoff_plan.json"
TX = ART / "handoff_transaction.json"
CAMPAIGN = ROOT / "ops/exp_scaling/campaign_stats.py"
BEFORE = ART / "campaign_stats.before.py"
AFTER = ART / "campaign_stats.after.py"
LEDGER_LOCK = ROOT / "var/artifacts/e118_ledger_promotion.lock"
EXPECTED_BEFORE = "d7289f2601e795eeed4f594d288f9a506da63de1ecc2de9bb0d52062c0058a50"
TARGETS = (
    (31159945, "campaign_timeout_guard_20260908", "guard_campaign_timeouts_20260908.py"),
    (31159946, "campaign_timeout_guard_31048146_20260909", "guard_pantry_timeout_31048146_20260909.py"),
    (31159947, "campaign_timeout_guard_31048182_20260909", "guard_pantry_timeout_31048182_20260909.py"),
    (31159948, "mathir_hourly_timeout_guard_20260909", "guard_mathir_hourly_timeouts_20260909.py"),
    (31159949, "preemption_hourly_timeout_guard_20260909", "guard_preemption_hourly_timeouts_20260909.py"),
    (31159699, "e119_lowprio_completion_20260909/guard", "guard_e119_lowprio_20260909.py"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(tx, message):
    tx.setdefault("events", []).append({"at": base.now(), "message": message})
    base.atomic(TX, tx)
    print(json.dumps({"at": base.now(), "event": message}), flush=True)


def candidate(before):
    old = '        registry.by_tag("e113r4").label,\n'
    assert before.count(old) == 1
    return before.replace(old, old + '        registry.by_tag("e112r1").label,\n')


def presentation_audit(before, after):
    assert after == candidate(before), "Only the exact reviewed set entry may change"
    old, new = ast.parse(before), ast.parse(after)
    def normalize(tree):
        tree = copy.deepcopy(tree)
        found = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "stopped_scale_labels" for t in node.targets):
                assert isinstance(node.value, ast.Set)
                node.value.elts = []
                found += 1
        assert found == 1
        return ast.dump(tree, include_attributes=False)
    assert normalize(old) == normalize(new)
    return {"only_stopped_scale_set_entry_changed": True,
            "all_other_ast_equal": True,
            "history_and_running_redisplay_unchanged": True}


def singleton(controller):
    values = {"ROOT": ROOT}
    def resolve(node):
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return Path(resolve(node.left)) / resolve(node.right)
        raise ValueError("Unsupported static path expression")
    for node in ast.parse(Path(controller).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in {"ART", "LOCK"}:
                values[key] = resolve(node.value)
    assert "LOCK" in values
    return str(values["LOCK"])


def safe_transition(tx):
    rows = list(tx["jobs"].values()) if "jobs" in tx else [tx]
    for row in rows:
        assert not any(not a.get("released") for a in row.get("attempts", [])), "Unfinished science release"
        assert not row.get("retirement_in_progress") and not row.get("retirement_pending"), "Unfinished retirement"
        assert not row.get("fallback_pending_reason"), "Unfinished fallback"
        assert row.get("status") not in {"fallback_in_progress", "deadline_hold_handoff", "holding"}, "Unfinished transition"
    if "jobs" not in tx:
        assert tx["status"] == "watching", "Stopped E119 guard requires separate review"


def check_cpu(record, item, new=False):
    assert base.field(record, "JobId") == str(item["new_job_id"] if new else item["old_job_id"])
    assert base.field(record, "UserId").endswith(f"({os.getuid()})")
    assert base.field(record, "NodeList") in ({"", "(null)", "node915", "node916", "node917"} if new else {"node915"})
    assert "gres/gpu" not in base.field(record, "ReqTRES")
    assert base.field(record, "Account") == "mltheory" and base.field(record, "Partition") == ("lowprio" if new else "mltheory")
    assert base.field(record, "MinMemoryNode") == ("1G" if new else item["memory"])
    assert base.field(record, "TimeLimit") == item["time_limit"]
    assert base.field(record, "Command") == (item["successor_script"] if new else item["old_command"])
    if new:
        assert base.field(record, "JobName") == f"display-e112r1-{item['old_job_id']}"
        assert base.submit_tokens(record) == item["submission_command"]
        assert sha(item["successor_script"]) == item["successor_script_sha256"]


def prepare():
    assert not PLAN.exists() and not TX.exists(), "Never overwrite an existing handoff"
    assert sha(CAMPAIGN) == EXPECTED_BEFORE, "Concurrent helper edit requires review"
    before = CAMPAIGN.read_text()
    after = candidate(before)
    audit = presentation_audit(before, after)
    BEFORE.write_text(before)
    AFTER.write_text(after)
    (ART / "display_only.diff").write_text("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=str(CAMPAIGN), tofile=str(AFTER))))
    rows = []
    for job, directory, name in TARGETS:
        directory = ROOT / "var/artifacts" / directory
        ppath, tpath = directory / "plan.json", directory / "transaction.json"
        p, t = json.loads(ppath.read_text()), json.loads(tpath.read_text())
        assert sha(ppath) == t["plan_sha256"]
        safe_transition(t)
        controller = ROOT / "ops/exp_scaling" / name
        assert sha(controller) == p["controller_sha256"]
        assert p["helper_sha256"][str(CAMPAIGN)] == EXPECTED_BEFORE
        assert all(sha(path) == digest for path, digest in p["helper_sha256"].items())
        record = base.show(job)
        item = {"old_job_id": job, "new_job_id": None, "guard_plan": str(ppath), "guard_tx": str(tpath),
                "controller": str(controller), "controller_sha256": sha(controller), "singleton_lock": singleton(controller),
                "old_command": base.field(record, "Command"), "old_record": record,
                "memory": base.field(record, "MinMemoryNode"), "time_limit": base.field(record, "TimeLimit"),
                "old_plan_sha256": sha(ppath), "deadline_utc": t["deadline_utc"],
                "cleanup_deadline_utc": t.get("cleanup_deadline_utc"),
                "watch_args": ["--watch", "--apply"] if name == "guard_e119_lowprio_20260909.py" else ["watch", "--apply"],
                "ready_token": str(ART / f"ready_{job}.json"), "waiter_ready": str(ART / f"waiting_{job}.json"),
                "successor_script": str(ART / f"waiter_{job}.slurm")}
        check_cpu(record, item)
        assert base.field(record, "JobState") == "RUNNING"
        rows.append(item)
    value = {"schema": "supervisor-e112r1-display-handoff-v1", "prepared_at_utc": base.now(), "rows": rows,
             "controller_sha256": sha(__file__), "before_campaign_sha256": sha(BEFORE), "after_campaign_sha256": sha(AFTER),
             "presentation_audit": audit, "science_jobs_changed": False, "events": []}
    base.atomic(PLAN, value)
    print(json.dumps({"prepared": True, "guards": [r["old_job_id"] for r in rows], "plan": str(PLAN), "audit": audit}))


def verify(plan):
    assert sha(__file__) == plan["controller_sha256"]
    assert sha(BEFORE) == plan["before_campaign_sha256"] and sha(AFTER) == plan["after_campaign_sha256"]
    presentation_audit(BEFORE.read_text(), AFTER.read_text())
    assert sha(CAMPAIGN) in {sha(BEFORE), sha(AFTER)}
    for item in plan["rows"]:
        assert sha(item["controller"]) == item["controller_sha256"]


def stage():
    plan = json.loads(PLAN.read_text())
    verify(plan)
    assert sha(CAMPAIGN) == plan["before_campaign_sha256"], "Stage before applying helper change"
    tx = json.loads(TX.read_text()) if TX.exists() else copy.deepcopy(plan)
    for item in tx["rows"]:
        if item.get("waiter_released"):
            check_cpu(base.show(item["new_job_id"]), item, new=True)
            continue
        assert not item.get("submission_uncertain"), "Reconcile an uncertain submission manually; never repeat it"
        if item["new_job_id"] is None:
            old = base.show(item["old_job_id"])
            check_cpu(old, item)
            assert base.field(old, "JobState") == "RUNNING"
            script = Path(item["successor_script"])
            script.write_text("#!/bin/bash\nset -euo pipefail\nexport PATH=/usr/bin:/bin\nexport PYTHONDONTWRITEBYTECODE=1\ncd " + shlex.quote(str(ROOT)) + "\nexec /usr/local/anaconda3/2024.02/bin/python3 " + shlex.quote(str(Path(__file__).resolve())) + " wait " + str(item["old_job_id"]) + "\n")
            item["successor_script_sha256"] = sha(script)
            command = ["sbatch", "--parsable", "--hold", f"--job-name=display-e112r1-{item['old_job_id']}",
                       "--account=mltheory", "--partition=lowprio", "--nodelist=node915,node916,node917", "--nodes=1", "--ntasks=1",
                       "--cpus-per-task=1", "--mem=1G", "--gres=none", f"--time={item['time_limit']}", "--requeue",
                       f"--chdir={ROOT}", f"--output={ART}/guard-{item['old_job_id']}-%j.out", f"--error={ART}/guard-{item['old_job_id']}-%j.err",
                       "--export=NONE", f"--comment=e112r1-display-handoff-20260909-{item['old_job_id']}", str(script)]
            item["submission_command"] = command
            item["submission_uncertain"] = True
            save(tx, f"Intent to submit idle CPU successor for {item['old_job_id']}")
            result = base.command(command, check=False)
            item["submission_result"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            save(tx, f"CPU submission returned for {item['old_job_id']}")
            assert result.returncode == 0, result.stderr
            item["new_job_id"] = int(result.stdout.strip().split(";")[0])
            item["submission_uncertain"] = False
            save(tx, f"Recorded successor {item['new_job_id']}")
        record = base.show(item["new_job_id"])
        check_cpu(record, item, new=True)
        if item.get("release_requested") and base.field(record, "JobState") in {"PENDING", "RUNNING"} and base.field(record, "Reason") != "JobHeldUser":
            item["waiter_released"] = True
            save(tx, f"Reconciled CPU release {item['new_job_id']}")
            continue
        assert base.field(record, "JobState") == "PENDING" and base.field(record, "Reason") == "JobHeldUser"
        item["release_requested"] = True
        save(tx, f"Release idle CPU waiter {item['new_job_id']}; old guard remains authoritative")
        base.command(["scontrol", "release", str(item["new_job_id"])])
        item["waiter_released"] = True
        save(tx, f"CPU waiter {item['new_job_id']} released")
    tx["status"] = "waiters_staged"
    save(tx, "All six CPU successors staged; live helper and all old guards unchanged")


def images(tx, item):
    directory = ART / f"guard_{item['old_job_id']}"
    directory.mkdir(exist_ok=True)
    ppath, tpath = Path(item["guard_plan"]), Path(item["guard_tx"])
    p, t = json.loads(ppath.read_text()), json.loads(tpath.read_text())
    assert sha(ppath) == item["old_plan_sha256"] == t["plan_sha256"]
    safe_transition(t)
    assert t["deadline_utc"] == item["deadline_utc"] and t.get("cleanup_deadline_utc") == item["cleanup_deadline_utc"]
    (directory / "plan.before.json").write_bytes(ppath.read_bytes())
    (directory / "transaction.before.json").write_bytes(tpath.read_bytes())
    original_p, original_t = copy.deepcopy(p), copy.deepcopy(t)
    p["helper_sha256"][str(CAMPAIGN)] = tx["after_campaign_sha256"]
    after_p, after_t = directory / "plan.after.json", directory / "transaction.after.json"
    base.atomic(after_p, p)
    t["plan_sha256"] = sha(after_p)
    base.atomic(after_t, t)
    restored_p, restored_t = copy.deepcopy(p), copy.deepcopy(t)
    restored_p["helper_sha256"][str(CAMPAIGN)] = tx["before_campaign_sha256"]
    restored_t["plan_sha256"] = item["old_plan_sha256"]
    assert restored_p == original_p and restored_t == original_t, "Changes must be hash-only"
    item["images"] = {"plan_before_sha256": sha(ppath), "tx_before_sha256": sha(tpath),
                      "plan_after": str(after_p), "plan_after_sha256": sha(after_p),
                      "tx_after": str(after_t), "tx_after_sha256": sha(after_t)}
    save(tx, f"Staged exact hash-only receipt updates for {item['old_job_id']}")


def apply():
    tx = json.loads(TX.read_text())
    verify(tx)
    if tx.get("status") == "handed_off":
        print(json.dumps({"status": "handed_off", "successors": [i["new_job_id"] for i in tx["rows"]]}))
        return
    for item in tx["rows"]:
        record = base.show(item["new_job_id"])
        check_cpu(record, item, new=True)
        assert base.field(record, "JobState") == "RUNNING", "All successor waiters must be running first"
        ready = json.loads(Path(item["waiter_ready"]).read_text())
        assert ready["job_id"] == item["new_job_id"] and ready["controller_sha256"] == sha(__file__)
    with ExitStack() as stack:
        ledger = stack.enter_context(LEDGER_LOCK.open("a+"))
        fcntl.flock(ledger, fcntl.LOCK_EX)
        # Verify every guard before cancelling any CPU supervisor.
        for item in tx["rows"]:
            if not item.get("old_stop_requested"):
                record = base.show(item["old_job_id"])
                check_cpu(record, item)
                assert base.field(record, "JobState") == "RUNNING"
                assert sha(item["guard_plan"]) == item["old_plan_sha256"]
                safe_transition(json.loads(Path(item["guard_tx"]).read_text()))
        for item in tx["rows"]:
            if not item.get("old_stop_requested"):
                item["old_stop_requested"] = True
                save(tx, f"Retiring CPU guard {item['old_job_id']} at the locked transition boundary")
                base.command(["scancel", str(item["old_job_id"])])
        for item in tx["rows"]:
            lock = stack.enter_context(Path(item["singleton_lock"]).open("a+"))
            deadline = time.monotonic() + 45
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert time.monotonic() < deadline, "Old singleton still active; receipts are unchanged"
                    time.sleep(0.25)
        queued = base.queue()
        assert not any(i["old_job_id"] in queued for i in tx["rows"]), "Old CPU allocations still active"
        for item in tx["rows"]:
            if not item.get("images"):
                images(tx, item)
        assert sha(CAMPAIGN) in {tx["before_campaign_sha256"], tx["after_campaign_sha256"]}
        if sha(CAMPAIGN) == tx["before_campaign_sha256"]:
            temp = CAMPAIGN.with_name(CAMPAIGN.name + ".e112r1-handoff-tmp")
            assert not temp.exists()
            with temp.open("x") as handle:
                handle.write(AFTER.read_text())
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp, CAMPAIGN.stat().st_mode)
            os.replace(temp, CAMPAIGN)
            save(tx, "Applied the reviewed one-line E112-R1 display filter while all six old guards are stopped")
        for item in tx["rows"]:
            image = item["images"]
            for path, before, after, source in [(item["guard_plan"], image["plan_before_sha256"], image["plan_after_sha256"], image["plan_after"]),
                                               (item["guard_tx"], image["tx_before_sha256"], image["tx_after_sha256"], image["tx_after"])]:
                assert sha(path) in {before, after}, "Receipt changed outside this handoff"
                assert sha(source) == after
                if sha(path) != after:
                    base.atomic(Path(path), json.loads(Path(source).read_text()))
            item["receipts_updated"] = True
            save(tx, f"Updated only helper and plan hashes for {item['old_job_id']}; deadlines and retry state preserved")
        tx["status"] = "receipts_updated"
        save(tx, "All six receipt pairs match the reviewed display helper")
    # Every old guard is gone and every receipt is current before any successor starts.
    for item in tx["rows"]:
        token = {"new_job_id": item["new_job_id"], "guard_plan_sha256": item["images"]["plan_after_sha256"],
                 "after_campaign_sha256": tx["after_campaign_sha256"], "at": base.now()}
        base.atomic(Path(item["ready_token"]), token)
        item["handoff_complete"] = True
    tx["status"] = "handed_off"
    save(tx, "All six CPU successors may enter unchanged guard logic; science jobs were not altered")


def wait(old):
    plan = json.loads(PLAN.read_text())
    verify(plan)
    item = next(r for r in plan["rows"] if r["old_job_id"] == old)
    job = int(os.environ["SLURM_JOB_ID"])
    txitem = next(r for r in json.loads(TX.read_text())["rows"] if r["old_job_id"] == old)
    assert txitem["new_job_id"] == job
    base.atomic(Path(item["waiter_ready"]), {"job_id": job, "controller_sha256": sha(__file__), "at": base.now()})
    print(json.dumps({"waiting_for_handoff": old, "at": base.now()}), flush=True)
    deadline = time.monotonic() + 3600
    while not Path(item["ready_token"]).exists():
        assert time.monotonic() < deadline, "No handoff within one hour; old guard remains authoritative"
        time.sleep(1)
    ready = json.loads(Path(item["ready_token"]).read_text())
    assert ready["new_job_id"] == job
    assert sha(CAMPAIGN) == ready["after_campaign_sha256"] and sha(item["guard_plan"]) == ready["guard_plan_sha256"]
    assert sha(item["controller"]) == item["controller_sha256"]
    print(json.dumps({"handoff_accepted": old, "at": base.now(), "preserved_deadline_utc": item["deadline_utc"]}), flush=True)
    sys.argv = [item["controller"]] + item["watch_args"]
    runpy.run_path(item["controller"], run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "stage", "apply", "wait"))
    parser.add_argument("old_job_id", nargs="?", type=int)
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    if args.phase == "wait":
        assert args.old_job_id is not None
        wait(args.old_job_id)
    else:
        globals()[args.phase]()
