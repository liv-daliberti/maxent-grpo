import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('code128_driver', Path(__file__).parents[1] / 'ops/drive_corrected_code128_20260922.py')
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class DriverGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'endpoint'
        self.run.mkdir()
        self.shard = {'name': 'endpoint', 'gpu_hour_cap': 3, 'total_completions': 4}
        self.schedule = {'run_parent': str(self.root), 'shards': [self.shard]}
        self.config = {'gpu_hour_envelope': 47, 'validation_runs': [], 'training_runs': {}, 'schedule_path': str(self.root / 'schedule.json')}
        driver.write(self.config['schedule_path'], self.schedule)
        driver.write(self.run / 'config.json', {'fixed': True})
        driver.write(self.run / 'identity.json', {'config_sha256': driver.digest(self.run / 'config.json'), 'request_sha256': 'f' * 64})
        (self.run / 'run.slurm').write_text('#!/bin/bash\ntrue\n')
        driver.write(self.run / 'submission_intent.json', {'argv': driver.expected_argv(self.run, self.shard), 'authorized_gpu_hour_ceiling': 3})
        self.materialization = self.root / 'materialized/endpoint/materialization.json'
        driver.write(self.materialization, {'run': str(self.run), 'name': 'endpoint', 'submitted': False,
            'frozen_identity_sha256': driver.digest(self.run / 'identity.json'), 'config_sha256': driver.digest(self.run / 'config.json'),
            'schedule_sha256': driver.digest(self.config['schedule_path']), 'request_sha256': 'f' * 64,
            'gpu_hour_cap': 3, 'total_completions': 4})
        driver.seal_materialization(self.run, self.config, self.schedule)

    def blocked(self, error, message):
        with patch.object(driver.subprocess, 'run') as launch:
            with self.assertRaisesRegex(error, message):
                driver.submit(self.run, self.config, self.schedule)
            launch.assert_not_called()

    def test_existing_receipt_is_idempotent(self):
        driver.write(self.run / 'submission.json', {'job_id': 71})
        with patch.object(driver.subprocess, 'run') as launch:
            self.assertEqual(driver.submit(self.run, self.config, self.schedule)['job_id'], 71)
            launch.assert_not_called()

    def test_ambiguous_attempt_never_submits_again(self):
        driver.write(self.run / 'submission_started.json', {'created_at': 'earlier'})
        self.blocked(RuntimeError, 'ambiguous')

    def test_allocation_envelope_blocks_before_submission(self):
        prior = self.root / 'prior'
        prior.mkdir()
        driver.write(prior / 'submission.json', {'job_id': 70})
        driver.write(prior / 'submission_intent.json', {'authorized_gpu_hour_ceiling': 45})
        self.config['validation_runs'] = [str(prior)]
        self.blocked(ValueError, 'envelope')
        self.assertFalse((self.run / 'submission_started.json').exists())

    def test_changed_frozen_config_blocks(self):
        driver.write(self.run / 'config.json', {'fixed': False})
        self.blocked(ValueError, 'materialization binding')

    def test_partial_materialization_never_submits_on_restart(self):
        self.materialization.unlink()
        self.blocked(RuntimeError, 'materialization did not complete')

    def test_interrupted_driver_before_ready_seal_requires_review(self):
        (self.run / 'controller_ready.json').unlink()
        self.blocked(RuntimeError, 'missing controller materialization seal')

    def test_failed_readiness_receipt_blocks(self):
        value = driver.read(self.run / 'controller_ready.json')
        value['status'] = 'failed'
        driver.write(self.run / 'controller_ready.json', value)
        self.blocked(ValueError, 'readiness did not pass')

    def test_wall_time_cannot_exceed_claimed_cap(self):
        value = driver.read(self.run / 'submission_intent.json')
        value['argv'] = [x.replace('--time=180', '--time=240') for x in value['argv']]
        driver.write(self.run / 'submission_intent.json', value)
        self.blocked(ValueError, 'argv/time')

    def test_job_script_mutation_blocks(self):
        (self.run / 'run.slurm').write_text('#!/bin/bash\nfalse\n')
        self.blocked(ValueError, 'changed after readiness')

    def test_submission_binds_receipt_and_terminal_failure_is_not_success(self):
        with patch.object(driver.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '72\n', '')) as launch:
            receipt = driver.submit(self.run, self.config, self.schedule)
            self.assertEqual(receipt['job_id'], 72)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(driver.read(self.run / 'submission.json'), receipt)
        self.assertFalse(driver.terminal_success(self.run, {}))
        self.assertFalse(driver.terminal_success(self.run, {72: {'job_id': 72, 'state': 'RUNNING', 'exit_code': '0:0'}}))
        self.assertTrue(driver.terminal_success(self.run, {72: {'job_id': 72, 'state': 'COMPLETED', 'exit_code': '0:0'}}))
        for state, code in [('TIMEOUT', '0:0'), ('COMPLETED', '1:0')]:
            with self.assertRaises(RuntimeError):
                driver.terminal_success(self.run, {72: {'job_id': 72, 'state': state, 'exit_code': code}})


if __name__ == '__main__':
    unittest.main()
