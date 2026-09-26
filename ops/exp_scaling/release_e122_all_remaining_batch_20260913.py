#!/usr/bin/env python3
"""Execute the audited exact-68 release as one locked, journaled transaction.

Full historical qualification is checked at transaction admission. Live held
identity, storage, release acknowledgements, and placement remain per-job checks.
This avoids replaying the same historical calibration for every status query.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_e122_all_remaining_20260913 as batch

c, m = batch.c, batch.m
ART = batch.ART
SOURCE = Path(__file__).resolve()


def execute(expected):
    plan = batch.validate(expected)
    batch.require(not (ART / 'claim.json').exists(),
                  'Already claimed; reconcile existing journals before continuing')
    print(json.dumps({'phase': 'validating_frozen_campaign'}), flush=True)
    campaign = batch.base.setup(batch.CAP, plan['candidate_job_ids'])
    batch.require(campaign['binding'] == plan['binding'], 'Campaign binding changed')
    jobs = {j['job_id']: j for j in campaign['jobs']}
    candidates = plan['candidate_job_ids']
    released = []
    with m.admission_locks(), c.locked(m.JOURNAL):
        initial = c.status(campaign, m.JOURNAL)
        batch.clean(initial)
        batch.require(set(initial['staged_held_job_ids']) == set(candidates)
                      and initial['reserved_unfinished_slots'] == 0
                      and initial['successful_endpoints'] == 32,
                      'Initial exact-68 census changed')
        entries = c.read_journals(m.JOURNAL, campaign['binding'], set(jobs))
        for jid in candidates:
            batch.require(not (m.JOURNAL / 'jobs' / f'{jid}.intent.json').exists()
                          and not (m.JOURNAL / 'jobs' / f'{jid}.result.json').exists(),
                          'Candidate has previous release activity: ' + jid)
        budget = m.budget_helper.storage_budget(campaign, candidates)
        batch.require(budget['allowed'], 'Insufficient storage for all 68 writers')
        batch.new(ART / 'batch_execution.json', {
            'created_at': c.now(), 'plan_sha256': expected,
            'source': str(SOURCE), 'source_sha256': batch.sha(SOURCE),
            'authorization': plan['authorization'], 'candidate_job_ids': candidates,
            'policy': 'One shared-admission and journal transaction; full historical admission once, fresh held identity and full storage reservation before each release; standard immutable per-job release receipts.',
            'initial_status': initial, 'storage': budget,
        })
        batch.new(ART / 'claim.json', {'created_at': c.now(), 'plan_sha256': expected,
                                     'execution_sha256': batch.sha(ART / 'batch_execution.json')})
        print(json.dumps({'phase': 'releasing', 'count': len(candidates),
                          'storage_margin_gib': budget['margin_bytes'] / 1024**3}), flush=True)
        for index, jid in enumerate(candidates):
            batch.validate(expected)
            before = c.evaluate(campaign, c.scheduler_snapshot(list(jobs)), entries,
                                c.disk_space(batch.ROOT))
            batch.clean(before)
            batch.require(before['next_job_id'] == jid,
                          'Live campaign no longer admits the next selected job')
            held = m.old.launcher.audit_held(int(jid), jobs[jid]['cell'])
            budget = m.budget_helper.storage_budget(campaign, candidates[index:])
            batch.require(budget['allowed'], 'Shared storage changed; reconcile partial release')
            intent_path = m.JOURNAL / 'jobs' / f'{jid}.intent.json'
            argv = ['scontrol', 'release', jid]
            intent = {
                'schema': 'e122_level3_release_intent_v1', 'created_at': c.now(),
                'binding': campaign['binding'], 'job_id': jid, 'cell': jobs[jid]['cell'],
                'held_scheduler_record': held, 'command': argv,
                'decision': {**before, 'shared_storage': budget},
                'finite_batch': {'plan_sha256': expected, 'index': index,
                                 'count': batch.CAP, 'authorization': plan['authorization']},
            }
            c.immutable_json(intent_path, intent)
            receipt = {'schema': 'e122_level3_release_result_v1', 'job_id': jid,
                       'intent_sha256': c.digest(intent_path), 'command': argv}
            try:
                result = c.command(argv)
                receipt.update(returncode=result.returncode, stdout=result.stdout,
                               stderr=result.stderr, error=None)
            except Exception as exc:
                receipt.update(returncode=None, error=f'{type(exc).__name__}: {exc}')
            receipt['recorded_at'] = c.now()
            c.immutable_json(m.JOURNAL / 'jobs' / f'{jid}.result.json', receipt)
            entries[jid] = {'intent': intent, 'result': receipt}
            batch.require(receipt['returncode'] == 0 and not receipt.get('error'),
                          'Release failed or ambiguous; inspect durable receipt before continuing')
            released.append(jid)
            batch.base.route(jid, jobs[jid]['cell'])
            print(json.dumps({'released': jid, 'count': len(released)}), flush=True)
        batch.validate(expected)
        final = c.status(campaign, m.JOURNAL)
        batch.clean(final)
        batch.require(final['staged_held'] == 0 and len(released) == batch.CAP,
                      'Some selected jobs remain held')
        batch.new(ART / 'result.json', {'created_at': c.now(), 'plan_sha256': expected,
                                      'released_job_ids': released, 'final_status': final})
    return {'released_count': len(released), **batch.summary(final)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.plan_sha256)), flush=True)
