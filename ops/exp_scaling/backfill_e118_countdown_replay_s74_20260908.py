#!/usr/bin/env python3
"""Audited placement of the remaining Countdown replay seed-74 cell on node203."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import backfill_e118_owner_nodes_20260908 as c

OLD_JOB = 31048126
WRAPPER = Path(__file__).resolve()
PROTOCOL = c.ROOT / 'paper/preregistration/e118_countdown_replay_s74_backfill_20260908.md'
PROFILE = dict(node='node203', memory_gib=116, gpu='a5000', ratio='0.40',
               account='allcs', partition='lowprio', domain='countdown',
               arm='replay_maxrl', seed=74, fresh=True)


def capacity(profile):
    record = c.base.command(['scontrol', 'show', 'node', '-o', profile['node']]).stdout
    state = c.base.field(record, 'State').split('+')
    assert state[0] in {'IDLE', 'MIXED', 'ALLOCATED'}
    assert not set(state[1:]) - {'PLANNED'}, state
    assert int(c.base.field(record, 'RealMemory')) - int(c.base.field(record, 'AllocMem')) >= profile['memory_gib'] * 1024
    assert int(c.base.field(record, 'CPUEfctv')) - int(c.base.field(record, 'CPUAlloc')) >= 16
    def tres(key):
        return dict(x.split('=', 1) for x in c.base.field(record, key).split(',') if '=' in x)
    total, used = tres('CfgTRES'), tres('AllocTRES')
    for key in ('gres/gpu', 'gres/gpu:' + profile['gpu']):
        assert int(total[key]) - int(used.get(key, 0)) >= 1
    return record


def configure():
    c.PROTOCOL = PROTOCOL
    c.PROFILES[OLD_JOB] = PROFILE
    c.capacity = capacity


def preflight():
    configure()
    source = json.loads(c.base.LEDGER.read_text())
    row = next(r for r in source['runs'] if r['job_id'] == OLD_JOB)
    assert all(row[k] == PROFILE[k] for k in ('domain', 'arm', 'seed'))
    record = c.base.show(OLD_JOB)
    assert c.base.field(record, 'JobState') == 'PENDING'
    assert c.base.field(record, 'Dependency') == '(null)'
    item = dict(row, old_job_id=OLD_JOB, new_job_id=None, checkpoint=None, profile=PROFILE)
    c.safe_cell(item)
    before = c.base.submit_tokens(record)
    after = c.command(before, PROFILE, OLD_JOB)
    expected = c.base.nonroot_exports(before)
    assert expected['OAT_ZERO_VLLM_GPU_RATIO'] == '0.25'
    expected['OAT_ZERO_VLLM_GPU_RATIO'] = '0.40'
    assert c.base.nonroot_exports(after) == expected
    supporting = json.loads((c.ART / 'node202_first_optimizer_verified.json').read_text())
    assert supporting['all_verified']
    assert any(r['domain'] == 'countdown' and r['seed'] == 74 and r['first_optimizer_verified'] for r in supporting['rows'])
    c.admission(PROFILE)
    node = capacity(PROFILE)
    test = c.base.command(['sbatch', '--test-only', *[x for x in after[1:] if x != '--hold']])
    print(json.dumps(dict(old_job_id=OLD_JOB, profile=PROFILE, fresh=True,
                          runtime_changes={'OAT_ZERO_VLLM_GPU_RATIO': {'before': '0.25', 'after': '0.40'}},
                          node_record=node, test_only_stderr=test.stderr)), flush=True)


def apply():
    configure()
    with (c.ROOT / 'var/artifacts/e118_ledger_promotion.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = c.ART / f'{OLD_JOB}.json'
        if not path.exists():
            preflight()
            c.prepare(OLD_JOB)
            audit = json.loads(path.read_text())
            audit.update(authorization='User authorized more existing E118 jobs; reviewed remaining pending Countdown replay seed-74 cell for node203.',
                         wrapper_path=str(WRAPPER), wrapper_sha256=c.recovery.digest(WRAPPER),
                         supporting_admission=str(c.ART / 'node202_first_optimizer_verified.json'))
            c.base.atomic(path, audit)
        audit = json.loads(path.read_text())
        assert audit['wrapper_path'] == str(WRAPPER)
        assert audit['wrapper_sha256'] == c.recovery.digest(WRAPPER)
        assert audit['replacements'][0]['profile'] == PROFILE
        assert audit['runtime_changes'] == {'OAT_ZERO_VLLM_GPU_RATIO': {'before': '0.25', 'after': '0.40'}}
        c.apply(OLD_JOB)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    apply() if args.apply else preflight()
