import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_modebench_prompt_ablation_local_v2 as m

class LocalAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parents[1] / 'artifacts/modebench_prompt_ablation_20260911'
        cls.rows = m.read_jsonl(base / 'rows.jsonl')
        cls.prompts = m.read_jsonl(base / 'prompts.jsonl')

    def test_actual_complete_contract_matches_training_templates(self):
        rows, pairs = m.validate_inputs(self.rows, self.prompts)
        self.assertEqual(len(rows),192)
        self.assertEqual(len(pairs),192)

    def test_user_problem_edit_rejected(self):
        prompts = copy.deepcopy(self.prompts)
        prompts[0]['messages'][1]['content'] += ' edited'
        prompts[0]['messages_sha256'] = m.sha(prompts[0]['messages'])
        with self.assertRaisesRegex(ValueError,'altered'):
            m.validate_inputs(self.rows,prompts)

    def test_missing_arm_rejected(self):
        with self.assertRaises(ValueError):
            m.validate_inputs(self.rows,self.prompts[:-1])

    def test_duplicate_arm_rejected(self):
        prompts = copy.deepcopy(self.prompts)
        prompts[-1] = prompts[0]
        with self.assertRaises(ValueError):
            m.validate_inputs(self.rows,prompts)

    def test_spec_edit_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['answer'] += ' '
        with self.assertRaisesRegex(ValueError,'digest'):
            m.validate_inputs(rows,self.prompts)

    def test_digest_edit_rejected(self):
        prompts = copy.deepcopy(self.prompts)
        prompts[0]['messages_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'digest'):
            m.validate_inputs(self.rows,prompts)

    def test_all_rng_problem_seeds_unique(self):
        values = [m.problem_seed(m.row_key(row)) for row in self.rows]
        self.assertEqual(len(set(values)),192)
        self.assertTrue(all(0 <= v < 2**32 for v in values))

    def test_all_expanded_child_seeds_disjoint_across_problems(self):
        child_seeds = [m.problem_seed(m.row_key(row)) + i for row in self.rows for i in range(8)]
        self.assertEqual(len(set(child_seeds)),1536)

    def test_all_candidate_child_seeds_disjoint(self):
        child_seeds = [m.problem_seed((level,domain,index)) + draw
                       for level in (2,3) for domain in m.DOMAINS for index in range(128) for draw in range(8)]
        self.assertEqual(len(set(child_seeds)),6144)

    def test_arm_order_balanced_by_row(self):
        cells = {}
        for position, key in enumerate(sorted(m.row_key(r) for r in self.rows)):
            cells.setdefault((key[0],key[1]),[]).append(position % 2)
        self.assertTrue(all(sum(v)==16 for v in cells.values()))

    def test_checkpoint_same_size_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'model.safetensors'
            p.write_bytes(b'good')
            c = {'model_path': directory, 'files': [{'name':p.name,'bytes':4,'sha256':m.file_sha(p)}]}
            m.verify_checkpoint(c)
            p.write_bytes(b'evil')
            with self.assertRaisesRegex(ValueError,'changed'):
                m.verify_checkpoint(c)

    def test_grader_records_semantic_distinctness(self):
        from types import SimpleNamespace as S
        generated = S(outputs=[S(text=t,token_ids=[1],finish_reason='stop') for t in ['a','b','bad','a','bad','bad','bad','bad']])
        draw = m.grade_samples({'answer':'spec'},generated,lambda t,a: {'a':('x',),'b':('x',)}.get(t))
        self.assertEqual(draw['pass1'],3/8)
        self.assertEqual(draw['pass8'],1)
        self.assertEqual(draw['distinct8'],1)

if __name__ == '__main__':unittest.main()
