"""Coverage integrity tests for the frozen Figure 6 source reader."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops' / 'exp_scaling'))
from snapshot_evaluation_coverage import read_cell


def draw(index, step=96):
    return {'benchmark': 'multi_answer', 'evaluation_kind': 'fixed_seed_sampled_k_neutral',
            'sample_count': 8, 'draw_index': index, 'step': step, 'seed': 100 + index,
            'temperature': 1.0, 'prompts': [{}] * 128,
            'metrics': {'any_correct_at_k': 0.5, 'distinct_correct_modes_at_k': 1.25,
                        'mean_at_k': 0.125}}


class SnapshotIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, rows, job=100, final_newline=True):
        path = self.root / f'debug_job{job}' / 'eval_mode_coverage_draws.jsonl'
        path.parent.mkdir()
        content = '\n'.join(json.dumps(row) for row in rows) + ('\n' if final_newline else '')
        path.write_text(content)
        return path

    def test_four_draws_and_identical_repeats_preserve_provenance(self):
        rows = [draw(i) for i in range(4)]
        path = self.write(rows + [draw(0)])
        result = read_cell(self.root)
        self.assertEqual(result['complete_steps'], [96])
        checkpoint = result['complete_checkpoints']['96']
        self.assertEqual(checkpoint['draw_count'], 4)
        self.assertEqual(len(checkpoint['draws'][0]['origins']), 2)
        self.assertEqual(checkpoint['draws'][0]['metrics'], rows[0]['metrics'])
        self.assertEqual(result['source_files'][0]['sha256_read_prefix'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_conflict_in_other_metric_field_refuses_entire_checkpoint(self):
        conflict = draw(0)
        conflict['metrics']['mean_at_k'] = 0.25
        self.write([draw(i) for i in range(4)] + [conflict])
        result = read_cell(self.root)
        self.assertEqual(result['complete_steps'], [])
        self.assertEqual(result['invalid_or_conflicted_steps'], [96])
        self.assertEqual(result['issues'][0]['kind'], 'conflicting_duplicate')
        self.assertFalse(result['issues'][0]['metric_payload_equal'])

    def test_missing_draw_and_partial_final_record_are_not_admitted(self):
        self.write([draw(i) for i in range(4)], final_newline=False)
        result = read_cell(self.root)
        self.assertEqual(result['complete_steps'], [])
        self.assertEqual(result['incomplete_checkpoints']['96']['draw_indices'], [0, 1, 2])
        self.assertEqual(result['issues'][0]['kind'], 'incomplete_final_line')

    def test_incorrect_sample_count_invalidates_checkpoint(self):
        rows = [draw(i) for i in range(4)]
        invalid = draw(0)
        invalid['sample_count'] = 4
        self.write(rows + [invalid])
        result = read_cell(self.root)
        self.assertEqual(result['complete_steps'], [])
        self.assertEqual(result['invalid_or_conflicted_steps'], [96])

    def test_explicit_job_exclusion_does_not_pool_conflicting_attempt(self):
        self.write([draw(i) for i in range(4)])
        conflict = draw(0)
        conflict['metrics']['any_correct_at_k'] = 0.9
        self.write([conflict], job=101)
        result = read_cell(self.root, excluded_job_ids={101})
        self.assertEqual(result['complete_steps'], [96])
        self.assertEqual(result['issues'][0]['kind'], 'excluded_source')


if __name__ == '__main__':
    unittest.main()
