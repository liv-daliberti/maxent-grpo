"""Empirical Level 3 receipts must preserve interfaces, sampling, and resumability."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'ops/evaluate_modebench_level3.py'
SPEC = spec_from_file_location('modebench_level3_evaluator_test', PATH)
evaluator = module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def encode(self, prompt, **kwargs):
        return list(prompt)


class LLM:
    def __init__(self, fail_call=None):
        self.calls = []
        self.fail_call = fail_call

    def generate(self, prompts, params, use_tqdm=False):
        self.calls.append((prompts, params))
        if len(self.calls) == self.fail_call:
            raise RuntimeError('simulated worker interruption')
        return [SimpleNamespace(prompt=prompt, outputs=[
            SimpleNamespace(text=text, token_ids=[1, 2], finish_reason='stop')
            for text in ('a', 'a', 'b', 'wrong', 'wrong', 'wrong', 'wrong', 'wrong')
        ]) for prompt in prompts]


def make_task(tmp_path, count=3):
    rows = [{'problem': f'Problem {index}', 'answer': json.dumps({'target': index}),
             'answer_mode_count': 4, 'candidate_id': str(index)} for index in range(count)]
    source = tmp_path / 'pool.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return {'domain': 'graph_coloring', 'level': 'level3', 'split': 'dev',
            'rows_jsonl': str(source), 'output': str(tmp_path / 'scores.json'),
            'seeds': [41, 42], 'batch_size': 2}


def evaluate(llm, task, **kwargs):
    return evaluator.evaluate_task(llm, Tokenizer(), task, model={'label': '3b'},
                                   code={'test': 'frozen'}, grader=lambda text, _: text if text != 'wrong' else None,
                                   params_factory=lambda args, domain, row: {'seed': args.seed, 'profile': args.prompt_profile},
                                   **kwargs)


def test_original_interface_is_fixed_across_all_five_domains():
    for domain in evaluator.DOMAINS:
        interface = evaluator.frozen_interface(domain)
        assert interface['name'] == 'original_level1'
        assert interface['prompt_profile'] == 'boxed_direct_v1'
        assert interface['syntax_profile'] == 'none'
        assert interface['sample_count'] == 8
        assert interface['max_tokens'] == 192
        assert interface['temperature'] == interface['top_p'] == 1
        assert interface['max_model_len'] == 1024


def test_admission_interface_preserves_qwen_r5_domain_profiles():
    expected = {'countdown': 'countdown_legal_v3', 'graph_coloring': 'none',
                'python_factors': 'domain_legal_v1', 'mathir': 'domain_legal_v1', 'pantry': 'domain_legal_v1'}
    for domain, syntax in expected.items():
        interface = evaluator.frozen_interface(domain, 'level2_qwen_r5')
        assert interface['syntax_profile'] == syntax
        assert interface['prompt_profile'] == ('boxed_direct_v1' if domain == 'graph_coloring' else 'hybrid_solver_v4')


def test_distinct8_counts_verified_semantic_modes_and_pass1_uses_all_draws(tmp_path):
    task = make_task(tmp_path)
    llm = LLM()
    result = evaluate(llm, task)
    assert len(llm.calls) == 4
    assert result['metrics']['pass1'] == 3 / 8
    assert result['metrics']['pass8'] == 1
    assert result['metrics']['distinct8'] == 2
    assert len(result['prompt_results']) == 3
    for row in result['prompt_results']:
        assert [draw['seed'] for draw in row['draws']] == [41, 42]
        assert len(row['spec_sha256']) == len(row['row_sha256']) == 64
        assert 'candidate_id' in row['row_metadata']
    assert result['identity']['interface_sha256'] == evaluator.sha(evaluator.frozen_interface('graph_coloring'))
    assert json.loads(Path(task['output']).read_text()) == result


def test_interruption_resumes_only_missing_batches_and_never_replaces_final(tmp_path):
    task = make_task(tmp_path)
    failed = LLM(fail_call=2)
    with pytest.raises(RuntimeError, match='simulated worker'):
        evaluate(failed, task)
    assert not Path(task['output']).exists()
    first_batch = Path(task['output'] + '.batches/seed-41__rows-000000-000002.json')
    first_bytes = first_batch.read_bytes()
    resumed = LLM()
    result = evaluate(resumed, task, resume=True)
    assert len(resumed.calls) == 3
    assert first_batch.read_bytes() == first_bytes
    completed_bytes = Path(task['output']).read_bytes()
    completed = LLM()
    assert evaluate(completed, task, resume=True) == result
    assert not completed.calls
    assert Path(task['output']).read_bytes() == completed_bytes
    with pytest.raises(FileExistsError):
        evaluate(LLM(), task)


def test_resume_rejects_changed_sampling_and_source(tmp_path):
    task = make_task(tmp_path)
    with pytest.raises(RuntimeError):
        evaluate(LLM(fail_call=2), task)
    changed_seed = dict(task, seeds=[50])
    with pytest.raises(ValueError, match='identity mismatch'):
        evaluate(LLM(), changed_seed, resume=True)
    source = Path(task['rows_jsonl'])
    source.write_text(source.read_text().replace('Problem 0', 'Changed problem'))
    with pytest.raises(ValueError, match='identity mismatch'):
        evaluate(LLM(), task, resume=True)


def test_corrupted_batch_is_rejected_before_reuse(tmp_path):
    task = make_task(tmp_path)
    with pytest.raises(RuntimeError):
        evaluate(LLM(fail_call=2), task)
    batch_path = Path(task['output'] + '.batches/seed-41__rows-000000-000002.json')
    batch = json.loads(batch_path.read_text())
    batch['draws'][0]['pass1'] = 1
    batch_path.write_text(json.dumps(batch))
    with pytest.raises(ValueError, match='invalid resumed batch'):
        evaluate(LLM(), task, resume=True)


def test_confirmation_requires_explicit_flag_and_full_split(tmp_path):
    task = dict(make_task(tmp_path), split='eval')
    with pytest.raises(ValueError, match='--confirm-eval'):
        evaluate(LLM(), task)
    with pytest.raises(ValueError, match='full split'):
        evaluate(LLM(), dict(task, row_limit=1), confirm_eval=True)
    result = evaluate(LLM(), task, confirm_eval=True)
    assert result['information_boundary']['evaluation_prompts_loaded']
    assert result['information_boundary']['confirmation_explicitly_authorized']


def test_invalid_sample_count_cannot_publish_receipt():
    output = SimpleNamespace(outputs=[])
    with pytest.raises(RuntimeError, match='expected 8 samples'):
        evaluator.grade_samples({'answer': '{}'}, output, lambda *_: 'valid')


def test_atomic_receipt_does_not_overwrite(tmp_path):
    path = tmp_path / 'receipt.json'
    evaluator.atomic_new(path, {'first': True})
    with pytest.raises(FileExistsError):
        evaluator.atomic_new(path, {'first': False})
    assert json.loads(path.read_text()) == {'first': True}
