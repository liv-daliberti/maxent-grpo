from __future__ import annotations

import importlib.util
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/exp_scaling"))
SPEC = importlib.util.spec_from_file_location("e120_node105_backfill", ROOT / "ops/exp_scaling/backfill_e120_node105_20260905.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate():
    original = ["sbatch", "--parsable", "--hold", "--job-name=e120-test",
                "--export=ALL,SAVE_PATH=/registered/run,RUN_STAMP=e120r1_falcon1b_graph_fresh_frequency_s56,OAT_ZERO_AUTO_RESUME=1,OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING=fresh_frequency",
                "--partition=all", "--account=allcs", "--nodelist=node202,node203",
                "--gres=gpu:1", "--cpus-per-task=8", "--mem=64G", "--time=3-00:00:00",
                "--nice=100", "--requeue", f"--exclude={MODULE.EXCLUSION}",
                "/frozen/ops/slurm/e120_resource_fenced_train.slurm"]
    return {"original_command": original, "command": MODULE.placed(original, 31048527)}


def scheduler_record(item, **overrides):
    values = {"JobId": "98765", "JobState": "PENDING", "Reason": "JobHeldUser",
              "RunTime": "00:00:00", "Restarts": "0", "Account": "mltheory",
              "Partition": "mltheory", "ReqNodeList": "node105", "MinMemoryNode": "64G",
              "NumCPUs": "8", "TimeLimit": "3-00:00:00", "Nice": "100",
              "ExcNodeList": MODULE.EXCLUSION, "Dependency": "(null)", "Requeue": "1",
              "ReqTRES": "cpu=8,mem=64G,node=1,gres/gpu=1,gres/gpu:a5000=1"}
    values.update(overrides)
    return " ".join(f"{k}={v}" for k, v in values.items()) + f" SubmitLine={shlex.join(item['command'])} WorkDir={ROOT} StdOut={ROOT}/slurm-98765.out"


def test_route_change_preserves_exact_scientific_export_and_resource_request():
    item = candidate()
    before, after = item["original_command"], item["command"]
    assert [x for x in before if x.startswith("--export=")] == [x for x in after if x.startswith("--export=")]
    for token in ("--mem=64G", "--cpus-per-task=8", "--time=3-00:00:00", "--nice=100", "--requeue", f"--exclude={MODULE.EXCLUSION}"):
        assert token in after
    assert after.count("--hold") == 1
    assert "--account=mltheory" in after and "--partition=mltheory" in after
    assert "--nodelist=node105" in after and "--gres=gpu:a5000:1" in after
    assert after[-1] == before[-1]


def test_complete_held_record_passes():
    item = candidate()
    MODULE.audit(scheduler_record(item), item)


@pytest.mark.parametrize("changed", [
    {"JobState": "RUNNING"}, {"Reason": "Priority"}, {"Restarts": "1"},
    {"MinMemoryNode": "40G"}, {"ReqNodeList": "node302"},
    {"ReqTRES": "cpu=8,mem=64G,gres/gpu=2,gres/gpu:a5000=2"},
    {"Dependency": "afterok:123(unfulfilled)"}, {"Account": "pvl"},
])
def test_held_audit_rejects_unsafe_state_or_placement(changed):
    item = candidate()
    with pytest.raises(RuntimeError):
        MODULE.audit(scheduler_record(item, **changed), item)


def test_held_audit_rejects_treatment_change():
    item = candidate()
    record = scheduler_record(item).replace("KEY_WEIGHTING=fresh_frequency", "KEY_WEIGHTING=uniform")
    with pytest.raises(RuntimeError, match="exports"):
        MODULE.audit(record, item)


def test_held_audit_rejects_wrong_watchdog_stdout():
    item = candidate()
    record = scheduler_record(item).replace(f"StdOut={ROOT}/slurm-98765.out", "StdOut=/other/log.out")
    with pytest.raises(RuntimeError, match="stdout"):
        MODULE.audit(record, item)


def node_record(**overrides):
    fields = dict(CfgTRES='cpu=96,mem=515096M,gres/gpu=10',
                  AllocTRES='cpu=72,mem=424G,gres/gpu=9',
                  RealMemory='515096', AllocMem='434176', CPUEfctv='96',
                  CPUAlloc='72', State='MIXED', Partitions='all,lowprio,mltheory')
    fields.update(overrides)
    return ' '.join(f'{k}={v}' for k, v in fields.items())


def test_busy_node_requires_explicit_queue_mode(monkeypatch):
    monkeypatch.setattr(MODULE, 'command', lambda _: SimpleNamespace(stdout=node_record()))
    with pytest.raises(RuntimeError, match='headroom changed'):
        MODULE.capacity()
    observed = MODULE.capacity(allow_queue=True)
    assert observed['allow_queue'] and not observed['joint_fits_now']
    assert observed['free_gpus'] == 1


@pytest.mark.parametrize('overrides', [
    dict(State='MIXED+DRAIN'), dict(Partitions='all,lowprio'),
    dict(RealMemory='131072'), dict(CPUEfctv='16'),
    dict(CfgTRES='cpu=96,mem=515096M,gres/gpu=2'),
])
def test_queue_mode_preserves_node_feasibility_guards(monkeypatch, overrides):
    monkeypatch.setattr(MODULE, 'command', lambda _: SimpleNamespace(stdout=node_record(**overrides)))
    with pytest.raises(RuntimeError):
        MODULE.capacity(allow_queue=True)
