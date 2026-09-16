from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
import launch_e109r1_qwen3_python_continuation as recovery  # noqa: E402
import campaign_stats  # noqa: E402


def targets():
    ledger = json.loads(recovery.ORIGINAL_LEDGER.read_text(encoding="utf-8"))
    return recovery.select_targets(ledger)


def test_recovery_is_exactly_the_two_preempted_qwen3_cells():
    rows = targets()
    assert [(row["seed"], row["job_id"]) for row in rows] == [
        (73, 30659554),
        (74, 30659555),
    ]
    assert recovery.TARGETS == {
        73: {"original_job_id": 30659554, "progress": 2391, "checkpoint": 2304},
        74: {"original_job_id": 30659555, "progress": 2197, "checkpoint": 2112},
    }


def test_continuation_reuses_byte_exact_scientific_export():
    for row in targets():
        environment = recovery.scheduler.environment(row["held_scheduler_record"])
        command = recovery.build_command(row, environment)
        assert f"--export={environment}" in command
        assert "--partition=all" in command
        assert "--account=allcs" in command
        assert f"--nodelist={recovery.NODE_LIST}" in command
        assert "--gres=gpu:a6000:1" in command
        assert "--cpus-per-task=16" in command
        assert "--mem=128G" in command
        assert "--time=3-00:00:00" in command
        assert f"SAVE_PATH={row['run_dir']}" in environment
        assert f"OAT_ZERO_SEED={row['seed']}" in environment
        assert "OAT_ZERO_AUTO_RESUME=1" in environment


def test_held_audit_rejects_environment_drift():
    environment = "ALL,OAT_ZERO_SEED=73"
    record = (
        "JobName=e109r1-q3-python-s73 JobState=PENDING Reason=JobHeldUser "
        "RunTime=00:00:00 Restarts=0 Partition=all Account=allcs "
        f"ReqNodeList={recovery.NODE_LIST} NumCPUs=16 MinMemoryNode=128G "
        "TimeLimit=3-00:00:00 Nice=100 TresPerNode=gres/gpu:a6000:1 "
        "SubmitLine=sbatch --export=ALL,OAT_ZERO_SEED=999 --partition=all"
    )
    try:
        recovery.audit_held(
            record, seed=73, environment=environment, partition="all"
        )
    except RuntimeError as error:
        assert "environment_sha256" in str(error)
    else:
        raise AssertionError("environment drift was accepted")


def test_protocol_freezes_same_cell_resume_without_endpoint_reads():
    text = recovery.PROTOCOL.read_text(encoding="utf-8")
    for required in (
        "steps 2,391 and 2,197",
        "step 2,304",
        "step 2,112",
        "full `--export` block",
        "byte for",
        "existing registered run",
        "without reading an E109",
        "or E112 evaluation endpoint",
    ):
        assert required in text


def test_submit_route_repair_is_zero_runtime_and_held():
    text = recovery.ROUTING_AMENDMENT.read_text(encoding="utf-8")
    for required in (
        "30873997",
        "user-held",
        "zero runtime",
        "explicitly update",
        "partition to `all`",
        "No evaluation endpoint was",
    ):
        assert required in text


def test_campaign_stats_resolves_the_two_continuation_job_ids():
    assert campaign_stats.e109_continuation_jobs(
        recovery.ORIGINAL_LEDGER
    ) == {
        30659554: 30874012,
        30659555: 30874013,
    }
