"""Scope and provenance checks for the complete fresh inference campaign."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import prepare_modebench_fresh_concentration as m


class PrepareFreshConcentrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.templates = {
            'recorded_graph': lambda text: '<graph>' + text + '</graph>',
            'recorded_pantry': lambda text: '<pantry>' + text + '</pantry>',
        }
        self.files = [
            {'name': 'config.json', 'bytes': 2, 'sha256': 'a' * 64},
            {'name': 'model.safetensors', 'bytes': 100, 'sha256': 'b' * 64},
        ]
        (self.base / 'ANALYSIS_PLAN.md').write_text('Fixed complete factorial and independent initial evaluation replicas.\n')
        self.write('analysis_registration.json', {'sha256': m.digest(self.base / 'ANALYSIS_PLAN.md')})
        self.write('prompt_population_audit.json', {'all_cells_match_exact_domain_representative': True})
        prompt_groups = []
        for domain in m.DOMAINS:
            path = self.base / f'{domain}_raw.jsonl'
            rows = [{'problem': f'{domain} problem {i}', 'answer': {'instance': i},
                     'row_index': i} for i in range(128)]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            prompt_groups.append({'domain': domain, 'raw_prompts': m.bound(path)})
        records = []
        for scale in m.SCALES:
            for domain in m.DOMAINS:
                pantry = domain == 'pantry_plan'
                sampling = {'n': 8, 'temperature': 1.0, 'top_p': 1.0,
                            'max_tokens': 6 if pantry else 192, 'min_tokens': 6 if pantry else 0,
                            'ignore_eos': pantry, 'stop': None, 'stop_token_ids': None,
                            'include_stop_str_in_output': False,
                            'allowed_token_ids': ([2039, 2040] if scale == 'falcon1b' else [15, 16]) if pantry else None}
                settings = {'prompt_template': 'recorded_pantry' if pantry else 'recorded_graph',
                            'syntax_profile': 'none', 'response_decoder': 'pantry_support_mask' if pantry else 'identity',
                            'prompt_encoding': 'rendered_text_vllm_v0'}
                path = self.base / f'{scale}_{domain}_source.json'
                path.write_text(json.dumps({**settings, 'effective_sampling': sampling,
                                            'max_model_len': 704 if pantry else 512}))
                for method in m.METHODS:
                    for seed in m.SEEDS[scale]:
                        cell = f'level1/{scale}/{domain}/{method}/{seed}'
                        records.append({'cell_id': cell, 'model_scale': scale, 'domain': domain,
                                        'method': method, 'training_seed': seed, **settings,
                                        'source_eval_config': m.bound(path), 'effective_sampling': copy.deepcopy(sampling),
                                        'model_path': str(self.base / 'originals' / cell.replace('/', '__')),
                                        'files': copy.deepcopy(self.files),
                                        'completion_receipt': {'path': '/preserved/receipt.json', 'sha256': 'c' * 64},
                                        'logged_evaluation_step': 3072, 'terminal_export_step': 3073,
                                        'archive_manifest': None})
        self.inventory = {'records': records, 'initial_models': [
            {'model_scale': scale, 'model_path': str(self.base / 'base_models' / scale),
             'revision': scale + '-pinned', 'files': copy.deepcopy(self.files)} for scale in m.SCALES],
            'prompt_groups': prompt_groups}
        self.save_inventory()
        self.template_patch = patch.object(m.collector, 'default_templates', return_value=self.templates)
        self.template_patch.start()
        self.addCleanup(self.template_patch.stop)

    def write(self, name, value):
        (self.base / name).write_text(json.dumps(value))

    def save_inventory(self):
        self.write('checkpoint_inventory.json', self.inventory)
        self.write('local_checkpoint_weight_hashes.json', {
            'inventory_sha256': m.digest(self.base / 'checkpoint_inventory.json'), 'files': []})

    def prepare(self):
        return m.prepare(self.base)

    def test_complete_plan_preserves_all_methods_and_distinguishes_initial_replicas(self):
        path, plan = self.prepare()
        self.assertEqual(len(plan['tasks']), 150)
        self.assertEqual(plan['expected_response_slots'], 1_228_800)
        initial = [t for t in plan['tasks'] if t['checkpoint_stage'] == 'initial']
        terminal = [t for t in plan['tasks'] if t['checkpoint_stage'] == 'terminal']
        self.assertEqual(len(initial), 30)
        self.assertEqual(len(terminal), 120)
        self.assertEqual(Counter(t['method'] for t in terminal), dict.fromkeys(m.METHODS, 30))
        for scale in m.SCALES:
            for domain in m.DOMAINS:
                group = [t for t in initial if t['model_scale'] == scale and t['domain'] == domain]
                self.assertEqual({t['eval_replica_id'] for t in group}, set(m.SEEDS[scale]))
                self.assertTrue(all(t['training_seed'] is None and t['method'] == 'initial' for t in group))
                self.assertEqual(len({t['model_path'] for t in group}), 1)
                self.assertTrue(all(t['source_identity']['training_replication'] is False for t in group))
        for task in plan['tasks']:
            self.assertEqual(task['engine']['dtype'], 'bfloat16')
            self.assertEqual(task['sampling']['max_tokens'], 6 if task['domain'] == 'pantry_plan' else 192)
            self.assertEqual(task['prompt_encoding'], 'rendered_text_vllm_v0')
        self.assertEqual(plan['rng_audit']['unique_child_seeds'], 1_228_800)
        self.assertEqual(self.prepare(), (path, plan))

    def test_missing_replay_maxrl_checkpoint_cannot_shrink_the_campaign(self):
        self.inventory['records'] = [r for r in self.inventory['records']
                                     if not (r['method'] == 'replay_maxrl' and r['model_scale'] == 'qwen3b'
                                             and r['domain'] == 'pantry_plan' and r['training_seed'] == 74)]
        self.save_inventory()
        with self.assertRaisesRegex(ValueError, 'inventory|checkpoint'):
            self.prepare()
        self.assertFalse((self.base / 'plan.json').exists())

    def test_duplicate_checkpoint_cannot_replace_a_missing_registered_seed(self):
        self.inventory['records'][-1] = copy.deepcopy(self.inventory['records'][-2])
        self.save_inventory()
        with self.assertRaisesRegex(ValueError, 'inventory|checkpoint'):
            self.prepare()

    def test_duplicate_initial_model_is_rejected_even_with_all_three_scales_present(self):
        self.inventory['initial_models'].append(copy.deepcopy(self.inventory['initial_models'][0]))
        self.save_inventory()
        with self.assertRaisesRegex(ValueError, 'initial|model'):
            self.prepare()

    def test_missing_initial_model_is_rejected_before_preparing_tasks(self):
        self.inventory['initial_models'].pop()
        self.save_inventory()
        with self.assertRaisesRegex(ValueError, 'initial|model'):
            self.prepare()

    def test_nonrepresentative_method_source_cannot_be_silently_overwritten(self):
        record = next(r for r in self.inventory['records'] if r['method'] == 'replay_maxrl')
        original = Path(record['source_eval_config']['path'])
        source = json.loads(original.read_text())
        source['effective_sampling']['temperature'] = 0.7
        alternate = self.base / 'different_replay_source.json'
        alternate.write_text(json.dumps(source))
        record['source_eval_config'] = m.bound(alternate)
        self.save_inventory()
        with self.assertRaisesRegex(ValueError, 'source|config|evaluation|sampling'):
            self.prepare()

    def test_nonrepresentative_prompt_template_cannot_be_silently_overwritten(self):
        record = next(r for r in self.inventory['records'] if r['method'] == 'maxrl')
        record['prompt_template'] = 'different_template'
        self.save_inventory()
        with self.assertRaisesRegex(ValueError, 'source|config|evaluation|template'):
            self.prepare()

    def test_changed_registered_protocol_blocks_preparation(self):
        (self.base / 'ANALYSIS_PLAN.md').write_text('Changed after registration.')
        with self.assertRaisesRegex(ValueError, 'protocol'):
            self.prepare()

    def test_stale_local_weight_hash_inventory_blocks_preparation(self):
        self.write('local_checkpoint_weight_hashes.json', {'inventory_sha256': '0' * 64, 'files': []})
        with self.assertRaisesRegex(ValueError, 'weight hash inventory'):
            self.prepare()

    def test_changed_raw_prompt_source_blocks_preparation(self):
        source = self.inventory['prompt_groups'][0]['raw_prompts']['path']
        with Path(source).open('a') as f:
            f.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'prompt source'):
            self.prepare()

    def test_existing_plan_cannot_substitute_an_initial_replica_for_a_terminal(self):
        _, plan = self.prepare()
        broken = copy.deepcopy(plan['tasks'])
        terminal = next(t for t in broken if t['checkpoint_stage'] == 'terminal')
        terminal.update(method='initial', checkpoint_stage='initial', training_seed=None,
                        eval_replica_id=m.SEEDS[terminal['model_scale']][0])
        with self.assertRaisesRegex(ValueError, '150|matrix|scope'):
            m.validate_coverage(broken)


if __name__ == '__main__':
    unittest.main()
