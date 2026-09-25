"""Independent integrity tests for source-admission exclusion regrading."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import regrade_real_domains_diagnostic_20260921 as regrade


def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()


class DiagnosticExclusionsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.original = self.root / 'original'
        self.original.mkdir()
        self.ids = ['keep_a', 'exclude_b', 'keep_c']
        self.config = {'adapter_module': 'old', 'adapter_config': {}, 'task_ids': self.ids,
                       'samples_per_task': 2, 'seed': 29, 'max_tokens': 16}
        self.prompts = [{'task_id': task_id, 'prompt': f'Prompt {task_id}',
                         'prompt_sha256': hashed(f'Prompt {task_id}')} for task_id in self.ids]
        self.rows = [{'task_id': task_id, 'sample_index': index,
                      'request_seed': 29 + 10000 * task_position + index,
                      'prompt_sha256': self.prompts[task_position]['prompt_sha256'],
                      'text': 'A', 'text_sha256': hashed('A'), 'token_ids': [65],
                      'token_count': 1, 'finish_reason': 'stop'}
                     for task_position, task_id in enumerate(self.ids) for index in range(2)]
        self.raw = self.original / 'responses.jsonl'
        self.receipt = {'status': 'complete', 'config': self.config,
                        'runner_sha256': '1' * 64, 'config_sha256': '2' * 64,
                        'job_id': '123', 'task_prompts': self.prompts,
                        'artifacts': {'responses': {'path': str(self.raw)}}}
        self.seal_raw()
        self.new_config = self.root / 'config.json'
        self.new_config.write_text(json.dumps({**self.config, 'adapter_module': 'hardened',
                                              'task_ids': ['keep_a', 'keep_c']}))
        self.failed_receipt = self.root / 'source_audit.json'
        self.failed_receipt.write_text(json.dumps({'source_problem_id': 'exclude_b', 'status': 'fail'}))
        self.exclusions = self.root / 'exclusions.json'
        self.seal_exclusion()
        self.calls = []
        def task(task_id):
            def verify(text):
                self.calls.append(task_id)
                return {'accepted': True, 'canonical_key': f'{task_id}:A',
                        'hard_violations': [], 'receipt': {'source_gate': 'pass'}}
            return SimpleNamespace(task_id=task_id, prompt=f'Prompt {task_id}',
                                   family='test', split='development', metadata={}, verify=verify)
        self.tasks = [task('keep_a'), task('keep_c')]
        loader = patch.object(regrade, 'load_task_selection',
                              return_value=(SimpleNamespace(__file__=__file__), self.tasks, {}))
        loader.start()
        self.addCleanup(loader.stop)

    def seal_raw(self):
        self.raw.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))
        self.receipt['artifacts']['responses']['sha256'] = regrade.sha256(self.raw)
        (self.original / 'evaluation.json').write_text(json.dumps(self.receipt))

    def seal_exclusion(self):
        self.exclusions.write_text(json.dumps({'excluded_tasks': {'exclude_b': {
            'reason': 'A source-positive program failed the fixed source gate',
            'receipt_path': str(self.failed_receipt),
            'receipt_sha256': regrade.sha256(self.failed_receipt)}}}))

    def invoke(self):
        return regrade.run(self.original, self.new_config, self.root / 'out.json', self.exclusions)

    def test_preserves_original_denominator_and_original_seed_indices(self):
        result = self.invoke()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['original_sampling_denominator'], 6)
        self.assertEqual(result['source_excluded_samples'], 2)
        self.assertEqual(result['summary']['samples'], 4)
        self.assertEqual(sorted(self.calls), ['keep_a', 'keep_a', 'keep_c', 'keep_c'])
        self.assertEqual([row['task_id'] for row in result['task_prompts']], ['keep_a', 'keep_c'])
        attempts = [json.loads(line) for line in Path(result['artifacts']['attempts']['path']).read_text().splitlines()]
        self.assertEqual(sorted(row['request_seed'] for row in attempts), [29, 30, 20029, 20030])
        self.assertFalse(result['primary_endpoint'])

    def test_rejects_missing_excluded_raw_sample_before_filtering(self):
        self.rows = [row for row in self.rows if not (row['task_id'] == 'exclude_b' and row['sample_index'] == 1)]
        self.seal_raw()
        with self.assertRaisesRegex(ValueError, 'incomplete or duplicated'):
            self.invoke()
        self.assertEqual(self.calls, [])

    def test_rejects_excluded_request_seed_corruption_before_filtering(self):
        self.rows[2]['request_seed'] += 1
        self.seal_raw()
        with self.assertRaisesRegex(ValueError, 'request seed mismatch'):
            self.invoke()
        self.assertEqual(self.calls, [])

    def test_rejects_excluded_text_corruption_before_filtering(self):
        self.rows[2]['text'] = 'B'
        self.seal_raw()
        with self.assertRaisesRegex(ValueError, 'raw text hash mismatch'):
            self.invoke()
        self.assertEqual(self.calls, [])

    def test_rejects_missing_explicit_exclusion_receipt(self):
        with self.assertRaisesRegex(ValueError, 'every excluded task'):
            regrade.run(self.original, self.new_config, self.root / 'out.json')

    def test_rejects_receipt_file_tampering(self):
        self.failed_receipt.write_text(self.failed_receipt.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'audit hash mismatch'):
            self.invoke()

    def test_rejects_receipt_naming_another_problem(self):
        self.failed_receipt.write_text(json.dumps({'source_problem_id': 'other', 'status': 'fail'}))
        self.seal_exclusion()
        with self.assertRaisesRegex(ValueError, 'must name the excluded task'):
            self.invoke()

    def test_rejects_passing_receipt_as_exclusion_reason(self):
        self.failed_receipt.write_text(json.dumps({'source_problem_id': 'exclude_b', 'status': 'pass'}))
        self.seal_exclusion()
        with self.assertRaisesRegex(ValueError, 'failed source gate'):
            self.invoke()

    def test_rejects_reordered_admitted_task_ids(self):
        config = json.loads(self.new_config.read_text())
        config['task_ids'] = ['keep_c', 'keep_a']
        self.new_config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'preserve original order'):
            self.invoke()

    def test_rejects_frozen_dependency_drift_before_verification(self):
        dependency = self.root / 'dependency.py'
        dependency.write_text('original = True\n')
        identity = {'config_sha256': regrade.sha256(self.new_config),
                    'files': [{'snapshot': str(dependency), 'sha256': regrade.sha256(dependency)}]}
        (self.root / 'identity.json').write_text(json.dumps(identity))
        dependency.write_text('original = False\n')
        with self.assertRaisesRegex(ValueError, 'frozen diagnostic dependency changed'):
            self.invoke()
        self.assertEqual(self.calls, [])

    def test_hard_verifier_failure_prevents_complete_status(self):
        self.tasks[1].verify = lambda text: {'accepted': False, 'canonical_key': None,
                                          'hard_violations': ['harness failed'], 'receipt': {}}
        result = self.invoke()
        self.assertNotEqual(result['status'], 'complete')
        self.assertEqual(result['original_sampling_denominator'], 6)
        self.assertEqual(result['summary']['samples'], 4)


if __name__ == '__main__':
    unittest.main()
