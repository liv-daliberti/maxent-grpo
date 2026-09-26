import sys
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'ops'),
                str(Path(__file__).resolve().parents[1] / 'ops/exp_scaling')]
import guard_e119_lowprio_20260909 as g

NOW = datetime(2026, 9, 9, 18, 0, tzinfo=timezone.utc)


class GuardBoundaryTests(unittest.TestCase):
    def tx(self, expired=False):
        deadline = NOW + timedelta(minutes=-1 if expired else 60)
        return {'attempts': [], 'last_resume_step': 192, 'deadline_utc': deadline.isoformat(),
                'cleanup_deadline_utc': (deadline + timedelta(minutes=65)).isoformat(),
                'status': 'watching'}

    def plan(self):
        return {'new_job_id': 99, 'old_job_id': 98, 'identity': {'run_dir': '/tmp/guard-fixture'},
                'submit_tokens': [], 'resources': {}}

    def fixture(self, state, tx, cp=288):
        stack = ExitStack()
        record = {'JobState': state, 'Reason': 'None', 'Restarts': '7'}
        for name in ['frozen', 'identity', 'stable', 'observations', 'no_other_writer', 'inactive_timeout', 'save']:
            stack.enter_context(patch.object(g, name))
        stack.enter_context(patch.object(g, 'utcnow', return_value=NOW))
        stack.enter_context(patch.object(g, 'show', return_value=record))
        stack.enter_context(patch.object(g.base, 'field', side_effect=lambda r, k: r[k]))
        stack.enter_context(patch.object(g.recovery, 'complete', return_value=False))
        stack.enter_context(patch.object(g, 'checkpoint', return_value={'step': cp}))
        fallback = stack.enter_context(patch.object(g, 'routing_fallback'))
        return stack, fallback, record

    def test_first_retry_requires_strict_advancement(self):
        tx = self.tx()
        self.assertFalse(g.may_retry(tx, 192, NOW))
        self.assertTrue(g.may_retry(tx, 288, NOW))

    def test_last_retry_floor_and_cap(self):
        tx = self.tx(); tx['last_resume_step'] = 288
        self.assertFalse(g.may_retry(tx, 288, NOW))
        tx['attempts'] = [{}] * 24
        self.assertFalse(g.may_retry(tx, 384, NOW))

    def test_deadline_never_authorizes_new_attempt(self):
        tx = self.tx(expired=True)
        self.assertFalse(g.may_retry(tx, 288, NOW))

    def test_memory_failure_blocks_retry(self):
        tx = self.tx(); tx['memory_failure'] = True
        self.assertFalse(g.may_retry(tx, 288, NOW))

    def test_memory_pressure_needs_live_limit_and_events(self):
        value = {'noncache_gib': 116.1, 'memory.high': str(116 * 2**30), 'events': {'high': 3}}
        self.assertTrue(g.memory_failure(value, {'events': {'high': 2}}))
        self.assertFalse(g.memory_failure(value, {'events': {'high': 3}}))
        value['noncache_gib'] = 80
        self.assertFalse(g.memory_failure(value))
        value['events']['oom'] = 1
        self.assertTrue(g.memory_failure(value))

    def test_running_survives_deadline_until_cleanup(self):
        tx = self.tx(expired=True); stack, fallback, _ = self.fixture('RUNNING', tx)
        with stack:
            self.assertEqual(g.observe_one(self.plan(), tx, apply=True), 'running')
        self.assertEqual(tx['status'], 'watching'); fallback.assert_not_called()

    def test_running_survives_cleanup_without_preemption(self):
        tx = self.tx(expired=True); tx['cleanup_deadline_utc'] = (NOW-timedelta(seconds=1)).isoformat()
        stack, fallback, _ = self.fixture('RUNNING', tx)
        with stack: g.observe_one(self.plan(), tx, apply=True)
        self.assertEqual(tx['status'], 'deadline_running_preserved'); fallback.assert_not_called()

    def test_no_progress_timeout_uses_exact_fallback_reason(self):
        tx = self.tx(); stack, fallback, _ = self.fixture('TIMEOUT', tx, cp=192)
        with stack: g.observe_one(self.plan(), tx, apply=True)
        fallback.assert_called_once_with(self.plan(), tx, 'no_progress')

    def test_inactive_deadline_timeout_falls_back(self):
        tx = self.tx(expired=True); stack, fallback, _ = self.fixture('TIMEOUT', tx)
        with stack: g.observe_one(self.plan(), tx, apply=True)
        fallback.assert_called_once_with(self.plan(), tx, 'deadline')

    def test_retry_cap_timeout_falls_back(self):
        tx = self.tx(); tx['attempts'] = [{'released': True}] * 24
        stack, fallback, _ = self.fixture('TIMEOUT', tx)
        with stack: g.observe_one(self.plan(), tx, apply=True)
        fallback.assert_called_once_with(self.plan(), tx, 'requeue_cap')

    def test_memory_failure_keeps_original_held(self):
        tx = self.tx(); tx['memory_failure'] = True
        stack, fallback, _ = self.fixture('OUT_OF_MEMORY', tx)
        with stack: g.observe_one(self.plan(), tx, apply=True)
        self.assertEqual(tx['status'], 'manual_stop'); fallback.assert_not_called()

    def test_unclassified_failure_never_falls_back(self):
        tx = self.tx(); stack, fallback, _ = self.fixture('FAILED', tx)
        with stack: g.observe_one(self.plan(), tx, apply=True)
        self.assertEqual(tx['status'], 'manual_stop'); fallback.assert_not_called()

    def test_owned_transition_gets_only_predeadline_grace(self):
        tx = self.tx(expired=True)
        action = {'before_restarts': 6, 'checkpoint': {'step': 288}, 'released': False,
                  'requested_at_utc': (NOW-timedelta(minutes=2)).isoformat()}
        tx['attempts'] = [action]
        record = {'JobState': 'PENDING', 'Reason': 'job_requeued_in_held_state', 'Restarts': '7'}
        def command(_): record['Reason'] = 'Priority'
        with patch.object(g,'utcnow',return_value=NOW), patch.object(g,'show',return_value=record), \
             patch.object(g,'stable'), patch.object(g,'no_other_writer'), patch.object(g,'save'), \
             patch.object(g,'checkpoint',return_value={'step':288}), \
             patch.object(g.base,'field',side_effect=lambda r,k:r[k]), patch.object(g.base,'command',side_effect=command) as mutate:
            g.reconcile(self.plan(),tx)
        self.assertTrue(action['released']); self.assertEqual(tx['last_resume_step'],288)
        self.assertEqual(mutate.call_args.args[0], ['scontrol','release','99'])

    def test_late_owned_hold_handoffs_instead_of_orphaning(self):
        tx = self.tx(expired=True); tx['deadline_utc'] = (NOW-timedelta(minutes=6)).isoformat()
        tx['attempts'] = [{'before_restarts':6,'released':False,'requested_at_utc':(NOW-timedelta(minutes=7)).isoformat()}]
        record = {'JobState':'PENDING','Reason':'job_requeued_in_held_state','Restarts':'7'}
        with patch.object(g,'utcnow',return_value=NOW), patch.object(g,'show',return_value=record), \
             patch.object(g,'stable'), patch.object(g.base,'field',side_effect=lambda r,k:r[k]), \
             patch.object(g,'handoff_deadline_hold') as handoff:
            g.reconcile(self.plan(),tx)
        handoff.assert_called_once()

    def test_handoff_rejects_unowned_user_hold(self):
        tx = self.tx(expired=True)
        action = {'before_restarts':6,'requested_at_utc':(NOW-timedelta(minutes=2)).isoformat()}
        record = {'JobState':'PENDING','Reason':'JobHeldUser','Restarts':'7'}
        with patch.object(g,'stable'),patch.object(g,'identity'),patch.object(g.base,'field',side_effect=lambda r,k:r[k]):
            with self.assertRaisesRegex(RuntimeError,'unrelated'):
                g.handoff_deadline_hold(self.plan(),tx,action,record)

    def test_lost_requeue_ack_reconciles_without_second_requeue(self):
        tx=self.tx();stack,_,record=self.fixture('TIMEOUT',tx)
        calls=[]
        def command(argv):
            calls.append(argv)
            if argv[1]=='requeuehold':
                record.update(JobState='PENDING',Reason='job_requeued_in_held_state',Restarts='8')
                raise RuntimeError('lost requeue acknowledgement')
            record['Reason']='Priority'
        with stack,patch.object(g.base,'command',side_effect=command):
            with self.assertRaisesRegex(RuntimeError,'lost requeue'):
                g.observe_one(self.plan(),tx,apply=True)
            self.assertEqual(len(tx['attempts']),1)
            g.observe_one(self.plan(),tx,apply=True)
        self.assertTrue(tx['attempts'][0]['released'])
        self.assertEqual([x[1] for x in calls],['requeuehold','release'])

    def test_lost_release_ack_reconciles_without_second_release(self):
        tx=self.tx();stack,_,record=self.fixture('TIMEOUT',tx)
        calls=[]
        def command(argv):
            calls.append(argv)
            if argv[1]=='requeuehold':
                record.update(JobState='PENDING',Reason='job_requeued_in_held_state',Restarts='8')
                return
            record['Reason']='Priority'
            raise RuntimeError('lost release acknowledgement')
        with stack,patch.object(g.base,'command',side_effect=command):
            with self.assertRaisesRegex(RuntimeError,'lost release'):
                g.observe_one(self.plan(),tx,apply=True)
            self.assertTrue(tx['attempts'][0]['release_requested'])
            g.observe_one(self.plan(),tx,apply=True)
        self.assertTrue(tx['attempts'][0]['released'])
        self.assertEqual([x[1] for x in calls],['requeuehold','release'])

    def test_deadline_pending_allocation_race_preserves_running(self):
        tx=self.tx(expired=True)
        ctl=SimpleNamespace(fallback=Mock(side_effect=RuntimeError('continuation allocated; defer fallback until inactive')))
        with patch.object(g,'capacity',return_value=ctl),patch.object(g,'save'), \
             patch.object(g,'show',return_value={'JobState':'RUNNING'}),patch.object(g.base,'field',side_effect=lambda r,k:r[k]):
            g.routing_fallback(self.plan(),tx,'deadline')
        self.assertEqual(tx['status'],'watching');self.assertNotIn('fallback_pending_reason',tx)


if __name__ == '__main__':
    unittest.main()
