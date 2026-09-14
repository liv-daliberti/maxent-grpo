#!/usr/bin/env python3
"""Release the user's exact remaining 68 E122 cells through existing journals.

All jobs become scheduler eligible, with aggregate storage reserved for all 68.
Slurm controls concurrent allocations. Preserve frozen recipes and use the
previously qualified node208 route only after the original held-job audit.
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'ops/exp_scaling'), str(ROOT / 'ops'), str(ROOT / 'src')]
import expand_e122_after_cli_20260912 as base

ART = ROOT / 'var/artifacts/e122_all_remaining_20260913'
PLAN = ART / 'plan.json'
SOURCE = Path(__file__).resolve()
CAP = 68
c, m = base.c, base.m
read, sha, new, require, run = base.read, base.sha, base.new, base.require, base.run
base.ART = ART


def clean(status):
    require(not status['issues'] and not status['needs_operator_review_job_ids']
            and not status['unknown'], 'Campaign requires reconciliation')


def summary(status):
    return {k: status[k] for k in ('observed_at', 'successful_endpoints', 'running',
            'staged_held', 'released_nonterminal', 'blocked_reason', 'issues',
            'needs_operator_review_job_ids')}


def prepare():
    require(not PLAN.exists(), 'Retain existing plan; use its digest')
    campaign = base.setup(CAP, [])
    with m.admission_locks(), c.locked(m.JOURNAL):
        status = c.status(campaign, m.JOURNAL)
        clean(status)
        candidates = status['staged_held_job_ids']
        require(len(candidates) == CAP and len(set(candidates)) == CAP,
                'Expected exactly 68 remaining held cells')
        require(status['successful_endpoints'] == 32
                and status['reserved_unfinished_slots'] == 0, 'Initial census changed')
        jobs = {j['job_id']: j for j in campaign['jobs']}
        held = {jid: m.old.launcher.audit_held(int(jid), jobs[jid]['cell'])
                for jid in candidates}
        budget = m.budget_helper.storage_budget(campaign, candidates)
        require(budget['allowed'], 'Insufficient aggregate checkpoint storage')
        paths = [SOURCE, Path(base.__file__), Path(base.recovery.__file__),
                 base.recovery.REG, Path(m.__file__), m.PLAN, m.COMMIT,
                 Path(m.budget_helper.__file__), Path(base.capacity.__file__)]
        new(PLAN, {
            'schema': 'e122_release_all_remaining_v1', 'created_at': c.now(),
            'authorization': 'User explicitly requested all 68 remaining E122 cells set to train on September 13, 2026.',
            'candidate_job_ids': candidates, 'max_unfinished_slots': CAP,
            'binding': campaign['binding'], 'initial_status': status,
            'held_records': held, 'storage': budget,
            'nodes_before': run(['scontrol', 'show', 'node',
                                ','.join(sorted(base.capacity.POOL)), '-o']),
            'route_policy': 'After audited release, add previously qualified node208 to pending jobs; retain every scientific and resource setting.',
            'source_pins': {str(p): sha(p) for p in paths},
        })
    return {'plan_sha256': sha(PLAN), 'count': CAP,
            'storage_margin_gib': budget['margin_bytes'] / 1024**3}


def validate(expected):
    require(expected and sha(PLAN) == expected, 'Plan digest differs')
    plan = read(PLAN)
    for path, digest in plan['source_pins'].items():
        require(sha(Path(path)) == digest, 'Source changed: ' + path)
    require(plan['max_unfinished_slots'] == CAP
            and len(set(plan['candidate_job_ids'])) == CAP, 'Release bound differs')
    return plan


def release(expected):
    plan = validate(expected)
    require(not (ART / 'claim.json').exists(), 'Already claimed; reconcile journals before continuing')
    campaign = base.setup(CAP, plan['candidate_job_ids'])
    require(campaign['binding'] == plan['binding'], 'Campaign binding changed')
    jobs = {j['job_id']: j for j in campaign['jobs']}
    new(ART / 'claim.json', {'created_at': c.now(), 'plan_sha256': expected})
    released = []
    for index in range(CAP):
        validate(expected)
        # Retry only lock acquisition, never an ambiguous scheduler mutation.
        for attempt in range(10):
            try:
                with m.admission_locks():
                    result = c.advance_once(m.old.fresh_args(), m.old.launcher,
                                            root=m.JOURNAL)
                break
            except BlockingIOError:
                if attempt == 9:
                    raise
                print(json.dumps({'waiting_for_admission_lock': True}), flush=True)
                time.sleep(2)
        new(ART / f'release_{index:02d}.json', result)
        clean(result)
        jid = result.get('last_release_job_id')
        require(jid in plan['candidate_job_ids'] and jid not in released,
                'Release did not advance exactly one selected job')
        released.append(jid)
        base.route(jid, jobs[jid]['cell'])
        print(json.dumps({'released': jid, 'count': len(released),
                          **summary(result)}), flush=True)
    final = c.status(campaign, m.JOURNAL)
    clean(final)
    require(final['staged_held'] == 0 and len(released) == CAP, 'Some selected jobs remain held')
    new(ART / 'result.json', {'created_at': c.now(), 'plan_sha256': expected,
                            'released_job_ids': released, 'final_status': final})
    return {'released_count': len(released), **summary(final)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'release'])
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare'
                     else release(args.plan_sha256)), flush=True)


if __name__ == '__main__':
    main()
