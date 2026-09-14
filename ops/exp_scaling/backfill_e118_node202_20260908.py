#!/usr/bin/env python3
"""Execute three existing E118 cells on verified node202 low-priority capacity."""
import argparse
import fcntl
import hashlib
from pathlib import Path

import backfill_e118_owner_nodes_20260908 as controller
import prioritize_e118_capacity_20260905 as base

TARGETS = {
    31048135: ('mathir', 'maxrl', 71),
    31048136: ('mathir', 'replay_maxrl', 71),
    31048125: ('countdown', 'maxrl', 74),
}
controller.PROTOCOL = base.ROOT / 'paper/preregistration/e118_node202_backfill_20260908.md'
for old, (domain, arm, seed) in TARGETS.items():
    controller.PROFILES[old] = dict(node='node202', memory_gib=116, gpu='a5000', ratio='0.40',
                                   account='allcs', partition='lowprio', domain=domain,
                                   arm=arm, seed=seed, fresh=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_job', type=int, choices=TARGETS)
    args = parser.parse_args()
    with (base.ROOT / 'var/artifacts/e118_ledger_promotion.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = controller.ART / f'{args.old_job}.json'
        if not path.exists():
            controller.prepare(args.old_job)
            import json
            audit = json.loads(path.read_text())
            audit.update(wrapper_path=str(Path(__file__).resolve()),
                         wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            base.atomic(path, audit)
        controller.apply(args.old_job)
