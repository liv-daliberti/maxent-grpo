"""Scientific and recovery invariants for fresh independent concentration draws."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as S
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops'))
import evaluate_modebench_fresh_concentration as m


def template(problem):
    return '<recorded>' + problem + '</recorded>'


def grader(text, reference):
    return 'mode:' + text[-1] if text.startswith('good') else None


def params_factory(task, row, seed):
    return S(**task['sampling'], seed=seed)


class FakeEngine:
    def __init__(self, fail_after=None):
        self.calls = []
        self.fail_after = fail_after

    def get_tokenizer(self):
        return S(encode=lambda text: [1, 2])

    def generate(self, prompts, params, use_tqdm=False):
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            raise RuntimeError('simulated worker interruption')
        self.calls.append([p.seed for p in params])
        return [S(prompt=prompt, prompt_token_ids=[1, 2], outputs=[
            S(index=i, text=' good' + str((param.seed+i) % 3), token_ids=[10+i],
              finish_reason='stop', stop_reason=None)
            for i in reversed(range(8))]) for prompt, param in zip(prompts, params)]


class FreshConcentrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root/'model'
        self.model.mkdir()
        (self.model/'config.json').write_text('{}')
        (self.model/'model.safetensors').write_bytes(b'weights')
        self.rows = [{'prompt_id': f'graph-{i}', 'row_index': i, 'problem': f'problem {i}',
                      'answer': 'reference', 'rendered_prompt': template(f'problem {i}')} for i in range(128)]
        self.prompts = self.root/'prompts.jsonl'
        self.prompts.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        sampling = {'n': 8, 'temperature': 1.0, 'top_p': 1.0, 'max_tokens': 192,
                    'min_tokens': 0, 'ignore_eos': False, 'stop': None,
                    'stop_token_ids': None, 'include_stop_str_in_output': False, 'allowed_token_ids': None}
        self.task = {'task_id': 'graph-qwen05b-drgrpo-s43', 'domain': 'graph_coloring',
                     'level': 1, 'model_scale': 'qwen05b', 'method': 'drgrpo', 'training_seed': 43,
                     'checkpoint_stage': 'terminal', 'model_path': str(self.model),
                     'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': m.file_sha(p)}
                               for p in sorted(self.model.iterdir())],
                     'prompts_path': str(self.prompts), 'prompt_template': 'recorded',
                     'syntax_profile': 'none', 'response_decoder': 'identity', 'sampling': sampling,
                     'prompt_encoding': m.PROMPT_ENCODING,
                     'engine': {'dtype': 'bfloat16', 'max_model_len': 512, 'tensor_parallel_size': 1,
                                'gpu_memory_utilization': .65, 'swap_space': 4,
                                'enable_prefix_caching': True, 'enforce_eager': True}}
        self.source = self.root/'source.json'
        self.source.write_text(json.dumps(m.source_contract(self.task)))
        self.task['source_eval_config'] = {'path': str(self.source), 'sha256': m.file_sha(self.source)}
        self.plan = {'schema': m.SCHEMA, 'campaign_id': 'fresh-test', 'seed_namespace': 'test-fresh-20260912',
                     'output_root': str(self.root/'outputs'), 'draws_per_prompt': 64, 'prompts_per_task': 128,
                     'draw_labels': list(range(8)), 'batch_size': 128,
                     'input_sha256': {str(self.prompts): m.file_sha(self.prompts), str(self.source): m.file_sha(self.source)},
                     'code_sha256': m.code_identity(), 'tasks': [self.task]}
        self.plan_path = self.root/'plan.json'
        self.save_plan()
        self.runtime = {'engine_contract': dict(m.ENGINE_CONTRACT), 'versions': {'vllm': '0.8.4'},
                        'environment': {'VLLM_USE_V1': '0', 'VLLM_ATTENTION_BACKEND': 'XFORMERS',
                                        'OMP_NUM_THREADS': '4', 'CUDA_VISIBLE_DEVICES': '0'}}

    def save_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def validate(self):
        return m.validate_plan(self.plan, templates={'recorded': template})

    def collect(self, engine, runtime=None):
        with redirect_stdout(io.StringIO()):
            return m.collect_task(self.plan_path, 0, templates={'recorded': template},
                                  engine_factory=lambda task: engine,
                                  runtime_factory=lambda: self.runtime if runtime is None else runtime,
                                  params_factory=params_factory, grader=grader)

    def test_preflight_never_hashes_or_requires_large_weight_files(self):
        (self.model/'model.safetensors').unlink()
        self.assertEqual(len(self.validate()[self.task['task_id']]), 128)
        with self.assertRaisesRegex(ValueError, 'checkpoint missing or changed'):
            m.validate_file_manifest(self.task, hash_files=True)

    def test_same_size_checkpoint_corruption_rejected_before_generation(self):
        (self.model/'model.safetensors').write_bytes(b'changed')
        engine = FakeEngine()
        with self.assertRaisesRegex(ValueError, 'checkpoint missing or changed'):
            self.collect(engine)
        self.assertEqual(engine.calls, [])

    def test_all_task_prompt_child_streams_are_disjoint_and_order_invariant(self):
        other = copy.deepcopy(self.task)
        other['task_id'] = 'graph-qwen05b-initial-rep43'
        other['checkpoint_stage'] = 'initial'
        other['method'] = 'initial'
        other['training_seed'] = None
        other['eval_replica_id'] = 43
        self.plan['tasks'].append(other)
        self.validate()
        one = m.task_schedule(self.plan, self.task, self.rows)
        two = m.task_schedule(self.plan, other, self.rows)
        children = [seed+i for schedule in (one, two) for row in schedule for seed in row for i in range(8)]
        self.assertEqual(len(children), 2*128*64)
        self.assertEqual(len(set(children)), len(children))
        self.assertEqual(m.task_schedule(self.plan, self.task, self.rows[::-1]), one[::-1])
        self.assertEqual(m.task_schedule(self.plan, self.task, self.rows[:2]), one[:2])
        self.assertTrue(all(0 <= seed < 2**63 for seed in children))

    def test_initial_replicas_cannot_be_mislabeled_as_independent_training_runs(self):
        self.task.update(checkpoint_stage='initial', method='initial', training_seed=None, eval_replica_id=43)
        self.validate()
        for patch in ({'method': 'drgrpo'}, {'training_seed': 43}, {'eval_replica_id': None},
                      {'eval_replica_id': True}, {'eval_replica_id': '43'}):
            original = copy.deepcopy(self.task)
            with self.subTest(patch=patch):
                self.task.update(patch)
                with self.assertRaisesRegex(ValueError, 'initial tasks require'):
                    self.validate()
                self.task.clear()
                self.task.update(original)

    def test_terminal_tasks_require_real_training_seeds_and_registered_methods(self):
        for patch in ({'method': 'initial'}, {'method': 'unknown'}, {'training_seed': None},
                      {'training_seed': True}, {'training_seed': '43'}, {'eval_replica_id': 43}):
            original = copy.deepcopy(self.task)
            with self.subTest(patch=patch):
                self.task.update(patch)
                with self.assertRaisesRegex(ValueError, 'terminal tasks require'):
                    self.validate()
                self.task.clear()
                self.task.update(original)
        self.task['eval_replica_id'] = None
        self.validate()

    def test_cross_task_collision_rejected_before_generation(self):
        from unittest.mock import patch
        self.plan['tasks'].append(dict(self.task, task_id='another'))
        with patch.object(m, 'task_schedule', return_value=[[8*i for i in range(8)] for _ in range(128)]):
            with self.assertRaisesRegex(ValueError, 'RNG collision across'):
                self.validate()

    def test_rendering_and_source_settings_cannot_silently_change(self):
        self.rows[0]['rendered_prompt'] += ' use a different strategy'
        self.prompts.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        self.plan['input_sha256'][str(self.prompts)] = m.file_sha(self.prompts)
        with self.assertRaisesRegex(ValueError, 'recorded training template'):
            self.validate()
        self.task['sampling']['max_tokens'] = 8
        with self.assertRaisesRegex(ValueError, 'normalized source evaluation evidence'):
            self.validate()

    def test_exactly_8192_slots_and_finished_retry_does_not_load_engine(self):
        engine = FakeEngine()
        result = self.collect(engine)
        self.assertEqual(len(engine.calls), 8)
        records = m.read_jsonl(result['responses_path'])
        self.assertEqual(len(records), 8192)
        self.assertEqual(len({(r['prompt_id'],r['draw_index']) for r in records}), 8192)
        self.assertEqual(len({r['child_sampling_seed'] for r in records}), 8192)
        self.assertEqual({r['draw_index'] for r in records if r['prompt_id'] == 'graph-0'}, set(range(64)))
        self.assertTrue(all(r['text'].startswith(' good') and r['verifier_text'].startswith('good')
                            and r['verified'] and r['canonical_key'].startswith('mode:') for r in records))
        fresh = FakeEngine(fail_after=0)
        self.assertEqual(self.collect(fresh), result)
        self.assertEqual(fresh.calls, [])

    def test_interrupted_retry_reuses_committed_batches_only(self):
        interrupted = FakeEngine(fail_after=2)
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.collect(interrupted)
        output = self.root/'outputs'/self.task['task_id']
        before = {p.name: m.file_sha(p) for p in output.glob('batch_*.json')}
        self.assertEqual(len(before), 2)
        resumed = FakeEngine()
        self.collect(resumed)
        self.assertEqual(len(resumed.calls), 6)
        self.assertEqual(before, {name: m.file_sha(output/name) for name in before})
        original_requests = {seed for call in interrupted.calls for seed in call}
        resumed_requests = {seed for call in resumed.calls for seed in call}
        self.assertFalse(original_requests & resumed_requests)

    def test_runtime_change_rejected_on_partial_retry(self):
        with self.assertRaises(RuntimeError):
            self.collect(FakeEngine(fail_after=1))
        runtime = copy.deepcopy(self.runtime)
        runtime['versions']['torch'] = 'different'
        with self.assertRaisesRegex(ValueError, 'runtime versions/hardware changed'):
            self.collect(FakeEngine(), runtime)

    def test_attention_backend_and_thread_settings_are_bound_but_device_allocation_is_not(self):
        for field, value in [('VLLM_ATTENTION_BACKEND', 'FLASH_ATTN'), ('OMP_NUM_THREADS', '8'),
                             ('VLLM_USE_V1', '1')]:
            changed = copy.deepcopy(self.runtime)
            changed['environment'][field] = value
            self.assertNotEqual(m.runtime_fingerprint(changed), m.runtime_fingerprint(self.runtime))
        changed = copy.deepcopy(self.runtime)
        changed['environment']['CUDA_VISIBLE_DEVICES'] = '3'
        self.assertEqual(m.runtime_fingerprint(changed), m.runtime_fingerprint(self.runtime))

    def test_changed_physical_gpu_assignment_can_resume_and_is_recorded_per_attempt(self):
        with self.assertRaises(RuntimeError):
            self.collect(FakeEngine(fail_after=1))
        runtime = copy.deepcopy(self.runtime)
        runtime['environment']['CUDA_VISIBLE_DEVICES'] = '3'
        self.collect(FakeEngine(), runtime)
        output = self.root/'outputs'/self.task['task_id']
        attempts = [json.loads(path.read_text()) for path in (output/'attempts').glob('*.json')]
        self.assertEqual({row['environment']['CUDA_VISIBLE_DEVICES'] for row in attempts}, {'0', '3'})

    def test_saved_seed_or_key_tampering_rejected_even_with_new_payload_hash(self):
        engine = FakeEngine(fail_after=1)
        with self.assertRaises(RuntimeError):
            self.collect(engine)
        output = self.root/'outputs'/self.task['task_id']
        path = next(output.glob('batch_*.json'))
        original = json.loads(path.read_text())
        for field, value, message in [('child_sampling_seed', 1, 'child slot or RNG'),
                                       ('canonical_key', 'fabricated-key', 'verifier key')]:
            altered = copy.deepcopy(original)
            altered['requests'][0]['attempts'][0][field] = value
            altered['requests_sha256'] = m.sha(altered['requests'])
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, message):
                self.collect(FakeEngine())
        path.write_text(json.dumps(original))

    def test_duplicate_engine_child_index_is_not_committed(self):
        expected = m.request_identity(self.task, self.rows[0], 0, 0, 104)
        engine = FakeEngine()
        draw = engine.generate([self.rows[0]['rendered_prompt']], [S(seed=104)])[0]
        draw.outputs[0].index = draw.outputs[1].index
        with self.assertRaisesRegex(ValueError, 'duplicate/missing child'):
            m.grade_request(self.task, self.rows[0], draw, expected, grader)

    def test_missing_completed_batch_fails_instead_of_regenerating(self):
        self.collect(FakeEngine())
        output = self.root/'outputs'/self.task['task_id']
        next(output.glob('batch_*.json')).unlink()
        with self.assertRaisesRegex(ValueError, 'missing committed batch'):
            self.collect(FakeEngine())

    def test_pantry_mask_uses_six_token_horizon_and_decoded_verifier_text(self):
        task = copy.deepcopy(self.task)
        task.update(domain='pantry_plan', response_decoder='pantry_support_mask')
        task['sampling'].update(max_tokens=6, min_tokens=6, ignore_eos=True, allowed_token_ids=[15,16])
        expected = m.request_identity(task, self.rows[0], 0, 0, 104)
        generated = S(prompt=self.rows[0]['rendered_prompt'], prompt_token_ids=[1], outputs=[
            S(index=i, text='010101', token_ids=[15,16]*3, finish_reason='length') for i in range(8)])
        decode = lambda task, raw, ref: 'good0' if raw == '010101' else 'bad'
        record = m.grade_request(task, self.rows[0], generated, expected, grader, decode)
        self.assertEqual(record['attempts'][0]['text'], '010101')
        self.assertEqual(record['attempts'][0]['verifier_text'], 'good0')
        self.assertTrue(record['attempts'][0]['verified'])
        generated.outputs[0].token_ids.append(15)
        with self.assertRaisesRegex(ValueError, 'token budget'):
            m.grade_request(task, self.rows[0], generated, expected, grader, decode)


if __name__ == '__main__':
    unittest.main()

