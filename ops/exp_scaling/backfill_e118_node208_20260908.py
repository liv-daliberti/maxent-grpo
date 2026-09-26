#!/usr/bin/env python3
"""Place four existing E118 MathIR cells on node208 with unchanged runtime."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

import backfill_e118_owner_nodes_20260908 as c

PROFILE_IDS = {
    31048137: ('maxrl', 72),
    31048138: ('replay_maxrl', 72),
    31048139: ('maxrl', 73),
    31048140: ('replay_maxrl', 73),
}
WRAPPER = Path(__file__).resolve()
PROTOCOL = c.ROOT / 'paper/preregistration/e118_node208_backfill_20260908.md'
AUTHORIZATION = 'User requested additional existing E118/E119/E120 jobs anywhere after filling node302/node105; four existing MathIR cells placed on node208.'


def capacity(profile):
    """Apply the shared resource checks, including an idle node's empty AllocTRES."""
    record = c.base.command(['scontrol', 'show', 'node', '-o', profile['node']]).stdout
    assert c.base.field(record, 'State').split('+')[0] in {'IDLE', 'MIXED', 'ALLOCATED'}
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
    c.capacity = capacity
    for old, (arm, seed) in PROFILE_IDS.items():
        c.PROFILES[old] = dict(node='node208', memory_gib=116, gpu='a6000',
                              ratio='0.25', domain='mathir', arm=arm,
                              seed=seed, fresh=True, account='allcs', partition='lowprio')


def run(old):
    configure()
    # Both phases must share this lock: prepare captures the source ledger hash.
    with (c.ROOT / 'var/artifacts/e118_ledger_promotion.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = c.ART / f'{old}.json'
        if not path.exists():
            capacity(c.PROFILES[old])
            c.prepare(old)
            audit = json.loads(path.read_text())
            audit.update(authorization=AUTHORIZATION, wrapper_path=str(WRAPPER),
                         wrapper_sha256=c.recovery.digest(WRAPPER),
                         supporting_mathir_admission=str(c.ART / 'mathir_startup_verified.json'),
                         supporting_restore_admission=str(c.ART / 'node302_first_optimizer_verified.json'))
            c.base.atomic(path, audit)
        audit = json.loads(path.read_text())
        assert audit['wrapper_path'] == str(WRAPPER)
        assert audit['wrapper_sha256'] == c.recovery.digest(WRAPPER)
        assert audit['replacements'][0]['profile'] == c.PROFILES[old]
        assert audit['scheduler_only'] and not audit['runtime_changes']
        c.apply(old)
        audit = json.loads(path.read_text())
        item = audit['replacements'][0]
        print(json.dumps(dict(old_job_id=old, new_job_id=item['new_job_id'],
                              released=item['released'], transaction=str(path))), flush=True)
    # Ensure this release is accounted in the node capacity before another cell.
    deadline = time.monotonic() + 90
    while True:
        record = c.base.show(item['new_job_id'])
        state = c.base.field(record, 'JobState')
        if state == 'RUNNING':
            assert c.base.field(record, 'NodeList') == 'node208'
            return
        assert state in {'PENDING', 'CONFIGURING'}, (item['new_job_id'], state)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Released job {item['new_job_id']} has not started; inspect before admitting another cell")
        time.sleep(3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_job', type=int, choices=PROFILE_IDS)
    args = parser.parse_args()
    run(args.old_job)
