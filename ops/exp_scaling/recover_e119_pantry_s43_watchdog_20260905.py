#!/usr/bin/env python3
"""Resume the exact E119 Pantry Re:Dr.GRPO s43 cell after retry exhaustion."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import campaign_stats as campaign
from recover_e119_health_20260905 import checkpoint, atomic
OLD = 31041178
ORIGINAL = 31014459
ART = ROOT / 'var/artifacts/campaign_health_capacity_20260905'
AUDIT = ART / 'e119_pantry_s43_recovery.json'
CAPTURE = ART / 'e119_job_31041178.json'


def call(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if AUDIT.exists():
        raise SystemExit(f'Existing recovery receipt requires inspection: {AUDIT}')
    mapping = campaign.e119_continuation_jobs(campaign.E119_LEDGER)
    assert mapping[ORIGINAL] == OLD
    ledger = json.loads(campaign.E119_LEDGER.read_text())
    run = next(r for r in ledger['runs'] if int(r['job_id']) == ORIGINAL)
    assert (run['domain'], run['arm'], int(run['seed'])) == ('pantry_plan', 'replay_drgrpo', 43)
    assert call('sacct', '-X', '-n', '-P', '-j', str(OLD), '--format=State').split('|')[0] == 'FAILED'
    assert 'e119-pantry-rd-s43' not in call('squeue', '-h', '-u', 'od2961', '-o', '%i|%j')
    cp = checkpoint(run)
    assert cp['step'] == 480
    captured = json.loads(CAPTURE.read_text())
    command = shlex.split(captured['submitted_command'])
    assert command[0] == 'sbatch' and '--hold' in command
    for part in ('--partition=mltheory', '--account=mltheory', '--nodelist=node302', '--mem=40G', '--gres=gpu:1', '--cpus-per-task=8'):
        assert part in command, part
    export = next(x for x in command if x.startswith('--export='))
    for value in ('OAT_ZERO_AUTO_RESUME=1', 'OAT_ZERO_WATCHDOG_STALE_SECONDS=7200', 'OAT_ZERO_WATCHDOG_MAX_RESTARTS=12', f'SAVE_PATH={run["run_dir"]}', f'RUN_STAMP={run["run_stamp"]}'):
        assert value in export, value
    # The new allocation must inherit the already audited artifact-aware guard.
    runtime = Path(captured['submitted_exports']['OAT_ZERO_OPS_SNAPSHOT_ROOT'])
    assert 'E119_WATCHDOG_PROGRESS_20260905' in (runtime / 'run_experiment.sh').read_text()
    assert 'E119_WATCHDOG_ARTIFACT_PROGRESS_20260905' in (runtime / 'train.sh').read_text()
    record = {'schema': 'e119-pantry-s43-watchdog-recovery-v1', 'created_at': datetime.now(timezone.utc).isoformat(),
              'original_job_id': ORIGINAL, 'failed_job_id': OLD, 'checkpoint': cp,
              'same_scientific_cell': True, 'same_run_directory': True, 'same_submitted_command': True,
              'optimizer_changed': False, 'evaluation_changed': False, 'placement': 'node302/mltheory,40G,1GPU',
              'cause': 'Old allocation exhausted12 retries after7277-second stale watchdog during sampled evaluation480',
              'command': command, 'released': False, 'applied': False,
              'runtime_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (runtime/'run_experiment.sh', runtime/'train.sh')}}
    atomic(ART / 'e119_pantry_s43_recovery_plan.json', record)
    if not args.apply:
        print(json.dumps({'plan_validated': True, 'checkpoint': cp['step'], 'failed_job': OLD})); return
    before = campaign.E119_CONTINUATIONS.read_bytes()
    new_id = int(call(*command).split(';', 1)[0])
    record['new_job_id'] = new_id
    atomic(AUDIT, record)
    committed = False
    released = False
    try:
        shown = call('scontrol', 'show', 'job', '-dd', '-o', str(new_id))
        for value in ('JobState=PENDING', 'Reason=JobHeldUser', 'ReqNodeList=node302', 'MinMemoryNode=40G', 'Account=mltheory', 'Partition=mltheory', export):
            assert value in shown, value
        record['held_scheduler_record'] = shown
        continuation = json.loads(before)
        row = next(r for r in continuation['continuations'] if int(r['original_job_id']) == ORIGINAL)
        assert int(row['continuation_job_id']) == OLD
        row['previous_continuation_job_ids'] = [*row.get('previous_continuation_job_ids', []), OLD]
        row['continuation_job_id'] = new_id
        row['repair_kind'] = '20260905_pantry_s43_artifact_aware_watchdog_continuation'
        continuation['operational_change'] = str(continuation.get('operational_change', '')) + '; 2026-09-05 Pantry Re:Dr.GRPO s43 continues from480 with current artifact-aware watchdog after prior12-retry exhaustion'
        assert campaign.E119_CONTINUATIONS.read_bytes() == before, 'Concurrent continuation-ledger mutation'
        (ART / 'e119_pantry_s43_continuation.before.json').write_bytes(before)
        atomic(campaign.E119_CONTINUATIONS, continuation)
        committed = True
        assert campaign.e119_continuation_jobs(campaign.E119_LEDGER)[ORIGINAL] == new_id
        atomic(AUDIT, record)
        call('scontrol', 'release', str(new_id))
        released = True
        record.update(released=True, applied=True)
        atomic(AUDIT, record)
        record['after_scheduler_record'] = call('scontrol', 'show', 'job', '-dd', '-o', str(new_id))
        atomic(AUDIT, record)
    except Exception as exc:
        record['error'] = repr(exc)
        if not released:
            if committed:
                campaign.E119_CONTINUATIONS.write_bytes(before)
            subprocess.run(['scancel', str(new_id)], check=False)
            record['held_replacement_cancelled'] = True
        atomic(AUDIT, record)
        raise
    print(json.dumps({'new_job_id': new_id, 'checkpoint': cp['step'], 'audit': str(AUDIT), 'released': True}))

if __name__ == '__main__':
    main()
