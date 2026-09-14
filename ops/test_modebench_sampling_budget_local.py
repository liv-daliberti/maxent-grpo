"""Pre-collection tests for the 64-draw collector's scientific invariants."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as S
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_modebench_sampling_budget_local as m


class SamplingBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Path(__file__).resolve().parents[1] / 'artifacts/modebench_prompt_ablation_20260911'
        cls.prior_rows = m.read_jsonl(cls.base / 'rows.jsonl')
        cls.prior_prompts = m.read_jsonl(cls.base / 'prompts.jsonl')
        cls.selection = json.loads((cls.base / 'selection.json').read_text())
        cls.keys = m.selected_keys(cls.selection)
        cls.rows = [r for r in cls.prior_rows if m.row_key(r) in cls.keys]
        cls.prompts = [p for p in cls.prior_prompts if m.row_key(p) in cls.keys]
        cls.lookup, cls.pairs = m.validate_inputs(cls.rows, cls.prompts, cls.prior_rows, cls.prior_prompts, cls.selection)

    def test_exact_outcome_independent_selection(self):
        self.assertEqual(len(self.keys), 96)
        self.assertEqual(len(self.rows), 96)
        self.assertEqual(len(self.prompts), 192)

    def test_changed_user_prompt_rejected_even_if_hash_recomputed(self):
        prompts = copy.deepcopy(self.prompts)
        prompts[0]['messages'][1]['content'] += ' changed'
        prompts[0]['messages_sha256'] = m.sha(prompts[0]['messages'])
        with self.assertRaisesRegex(ValueError, 'immutable parent'):
            m.validate_inputs(self.rows, prompts, self.prior_rows, self.prior_prompts, self.selection)

    def test_nonselected_row_substitution_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0] = next(r for r in self.prior_rows if m.row_key(r) not in self.keys)
        with self.assertRaisesRegex(ValueError, 'preselected'):
            m.validate_inputs(rows, self.prompts, self.prior_rows, self.prior_prompts, self.selection)

    def test_missing_or_duplicate_arm_rejected(self):
        for prompts in (self.prompts[:-1], self.prompts[:-1] + [self.prompts[0]]):
            with self.assertRaises(ValueError):
                m.validate_inputs(self.rows, prompts, self.prior_rows, self.prior_prompts, self.selection)

    def test_all_768_candidate_child_seed_sets_disjoint_and_fresh(self):
        import evaluate_modebench_prompt_ablation_local_v2 as old
        keys = [(level, domain, index) for level in (2, 3) for domain in m.DOMAINS for index in range(128)]
        new_seeds = [m.problem_seed(k, block) + child for k in keys for block in range(8) for child in range(8)]
        old_seeds = {old.problem_seed(k) + child for k in keys for child in range(8)}
        self.assertEqual(len(set(new_seeds)), 768 * 64)
        self.assertFalse(set(new_seeds) & old_seeds)
        self.assertTrue(all(0 <= s < 2**31 for s in new_seeds))

    def test_each_problem_order_balanced_across_blocks_and_each_cell(self):
        selected = sorted(self.keys)
        starts = {key: [] for key in selected}
        for block in range(8):
            ordered = m.ordered_items(selected, block)
            for i in range(0, len(ordered), 2):
                a, b = ordered[i:i+2]
                self.assertEqual(a[0], b[0])
                self.assertNotEqual(a[1], b[1])
                self.assertEqual(i // 8, (i+1) // 8)
                starts[a[0]].append(a[1])
            for level in (2, 3):
                for domain in m.DOMAINS:
                    self.assertEqual(sum(ordered[i][1] == 'original' for i in range(0, len(ordered), 2)
                                         if ordered[i][0][:2] == (level, domain)), 8)
        self.assertTrue(all(v.count('original') == 4 for v in starts.values()))

    def test_64_flat_draw_records_have_exact_order_and_paired_rng(self):
        checkpoint = {'label': 'test', 'training_method': 'drgrpo', 'training_seed': 43, 'trained_on_level': 2}
        row = self.rows[0]
        draw = {'attempts': [{'text': 'x', 'verified': False, 'canonical_key': None, 'token_count': 1, 'finish_reason': 'stop'} for _ in range(8)]}
        by_arm = {}
        for arm in ('original', 'neutral'):
            by_arm[arm] = [record for block in range(8) for record in m.completion_records(checkpoint, row, self.pairs[m.row_key(row)][arm], draw, block)]
            self.assertEqual([r['draw_index'] for r in by_arm[arm]], list(range(64)))
            self.assertEqual(len({r['child_sampling_seed'] for r in by_arm[arm]}), 64)
        self.assertEqual([r['child_sampling_seed'] for r in by_arm['original']], [r['child_sampling_seed'] for r in by_arm['neutral']])

    def test_n8_grading_keeps_semantic_keys(self):
        generated = S(outputs=[S(text=t, token_ids=[1], finish_reason='stop') for t in ['a', 'b', 'bad', 'a', 'bad', 'bad', 'bad', 'bad']])
        draw = m.grade_samples({'answer': 'spec'}, generated, lambda t,a: {'a': ('x',), 'b': ('x',)}.get(t))
        self.assertEqual((len(draw['attempts']), draw['pass1'], draw['pass8'], draw['distinct8']), (8, 3/8, 1, 1))

    def test_resumed_batch_cannot_change_block_or_seed_with_recomputed_hash(self):
        key = sorted(self.keys)[0]
        items = [(key, 'original'), (key, 'neutral')]
        records = [{'pair_id': self.pairs[key][arm]['pair_id'], 'arm': arm, 'draw_block': 0,
                    'sampling_seed': m.problem_seed(key, 0), 'level': key[0], 'domain': key[1], 'row_index': key[2],
                    'attempts': [{'token_count': 1} for _ in range(8)]} for _,arm in items]
        batch = {'identity_sha256': 'id', 'draw_block': 0, 'start': 0, 'records': records, 'records_sha256': m.sha(records)}
        m.validate_batch(batch, 'id', 0, 0, items, self.pairs, self.lookup)
        altered = copy.deepcopy(batch)
        altered['records'][0]['sampling_seed'] += 8
        altered['records_sha256'] = m.sha(altered['records'])
        with self.assertRaisesRegex(ValueError, 'seed identity'):
            m.validate_batch(altered, 'id', 0, 0, items, self.pairs, self.lookup)
        with self.assertRaisesRegex(ValueError, 'immutable panel'):
            m.validate_batch(batch, 'id', 1, 0, items, self.pairs, self.lookup)

    def test_same_size_checkpoint_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'model.safetensors'
            p.write_bytes(b'good')
            checkpoint = {'model_path': directory, 'files': [{'name': p.name, 'bytes': 4, 'sha256': m.file_sha(p)}]}
            m.verify_checkpoint(checkpoint)
            p.write_bytes(b'evil')
            with self.assertRaisesRegex(ValueError, 'changed'):
                m.verify_checkpoint(checkpoint)


if __name__ == '__main__':
    unittest.main()
