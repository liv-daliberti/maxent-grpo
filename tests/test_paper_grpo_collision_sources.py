"""Integrity checks for the frozen plain-GRPO saved-sample adapter."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'paper_grpo_collision_sources_tested',
    ROOT / 'ops/exp_scaling/load_paper_grpo_collision_sources.py')
LOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOADER)
NORMALIZE, ASSEMBLE = LOADER._normalizer(ROOT)


def draws():
    records = []
    for step in (0, 3072):
        for draw in range(4):
            records.append({
                'benchmark': 'fixture', 'evaluation_kind': LOADER.KINDS,
                'step': step, 'draw_index': draw, 'sample_count': 8,
                'temperature': 1.0, 'top_p': 1.0, 'seed': 610100 + draw,
                'metrics': {},
                'prompts': [{
                    'prompt_index': index, 'prompt': f'prompt {index}',
                    'reference': {'instance_id': f'fixture-{index}'},
                    'answer_keys': ['valid-key'] + ['invalid-key'] * 7,
                    'rewards': [1.0] + [0.0] * 7,
                    'responses': ['answer'] * 8,
                    'request_seeds_by_option': [], 'option_ids': [None] * 8,
                    'metrics': {},
                } for index in range(128)],
            })
    return records


class FrozenGRPOSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write_source(self, records):
        payload = b''.join(LOADER.compact(row) + b'\n' for row in records)
        (self.root / 'source.jsonl').write_bytes(payload)
        return {
            'level': 'level1', 'scale': 'qwen05b', 'domain': 'graph_coloring',
            'method': 'grpo', 'seed': 43, 'terminal_admitted': True,
            'source': {'path': 'source.jsonl', 'byte_length': len(payload),
                       'sha256': hashlib.sha256(payload).hexdigest()},
        }

    def load(self, descriptor):
        return LOADER._load_one(self.root, descriptor, NORMALIZE, ASSEMBLE)

    def test_preserves_failed_keys_but_gates_verified_keys(self):
        descriptor = self.write_source(draws())
        cell, receipt = self.load(descriptor)
        self.assertTrue(cell['before_after_available'])
        prompt = cell['checkpoints']['0']['draws'][0]['prompts'][0]
        self.assertEqual(prompt['answer_keys'][1], 'invalid-key')
        self.assertIsNone(prompt['verified_keys'][1])
        self.assertEqual(receipt['verified_sha256'], descriptor['source']['sha256'])

    def test_identical_duplicates_are_recorded_not_resampled(self):
        records = draws()
        records.append(deepcopy(records[0]))
        cell, receipt = self.load(self.write_source(records))
        self.assertTrue(cell['before_after_available'])
        self.assertEqual(receipt['identical_duplicate_rows'], 1)
        self.assertEqual(len(cell['checkpoints']['0']['draws'][0]['origins']), 2)

    def test_conflicting_response_payload_invalidates_only_its_checkpoint(self):
        records = draws()
        duplicate = deepcopy(records[0])
        duplicate['prompts'][0]['responses'][0] = 'different output, same logged metrics'
        records.append(duplicate)
        cell, _ = self.load(self.write_source(records))
        self.assertTrue(cell['terminal_admitted'])
        self.assertIsNone(cell['checkpoints']['0'])
        self.assertIsNotNone(cell['checkpoints']['3072'])
        self.assertFalse(cell['before_after_available'])
        self.assertTrue(any(issue['reason'] == 'conflicting_duplicate_draw' for issue in cell['sample_issues']))

    def test_appended_records_are_outside_frozen_evidence(self):
        records = draws()
        descriptor = self.write_source(records)
        duplicate = deepcopy(records[0])
        duplicate['prompts'][0]['responses'][0] = 'unfrozen later output'
        suffix = LOADER.compact(duplicate) + b'\n'
        with (self.root / 'source.jsonl').open('ab') as handle:
            handle.write(suffix)
        cell, receipt = self.load(descriptor)
        self.assertTrue(cell['before_after_available'])
        self.assertEqual(receipt['appended_bytes_ignored'], len(suffix))
        self.assertEqual(receipt['matching_endpoint_rows'], 8)

    def test_changed_frozen_bytes_fail_hash_validation(self):
        descriptor = self.write_source(draws())
        path = self.root / 'source.jsonl'
        path.write_bytes(path.read_bytes().replace(b'valid-key', b'valid-kez', 1))
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            self.load(descriptor)

    def test_changing_prompt_population_is_not_a_paired_before_after(self):
        records = draws()
        for row in records:
            if row['step'] == 3072:
                row['prompts'][0]['prompt'] = 'changed task surface'
        cell, _ = self.load(self.write_source(records))
        self.assertTrue(all(checkpoint is not None for checkpoint in cell['checkpoints'].values()))
        self.assertFalse(cell['before_after_available'])
        self.assertTrue(any(issue['reason'] == 'before_after_prompt_population_mismatch'
                            for issue in cell['sample_issues']))


if __name__ == '__main__':
    unittest.main()
