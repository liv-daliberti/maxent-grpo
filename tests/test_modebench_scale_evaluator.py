"""Scale receipts preserve the frozen scientific interface and own their provenance."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import evaluate_modebench_scale as evaluator


class Tokenizer:
    def __init__(self, prefix=''):
        self.prefix = prefix

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        return self.prefix + json.dumps(messages)

    def encode(self, prompt, **kwargs):
        return list(prompt)


class LLM:
    def __init__(self, fail_call=None, runtime=None):
        self.calls = []
        self.fail_call = fail_call
        self._modebench_runtime = runtime or evaluator.runtime_settings(max_model_len=2048)

    def generate(self, prompts, params, use_tqdm=False):
        self.calls.append((prompts, params))
        if len(self.calls) == self.fail_call:
            raise RuntimeError('simulated worker interruption')
        return [SimpleNamespace(prompt=prompt, outputs=[
            SimpleNamespace(text=text, token_ids=[1, 2], finish_reason='stop')
            for text in ('a', 'a', 'b', 'wrong', 'wrong', 'wrong', 'wrong', 'wrong')
        ]) for prompt in prompts]


def make_task(tmp_path, *, count=3):
    rows = [{'problem': f'Problem {index}', 'answer': json.dumps({'target': index}),
             'answer_mode_count': 4, 'candidate_id': str(index)} for index in range(count)]
    source = tmp_path / 'pool.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return {'domain': 'graph_coloring', 'level': 'level4', 'split': 'dev',
            'rows_jsonl': str(source), 'output': str(tmp_path / 'run.json'),
            'seeds': [9418000, 9418001, 9418002, 9418003], 'batch_size': 2}


def evaluate(llm, task, **kwargs):
    return evaluator.evaluate_task(
        llm, kwargs.pop('tokenizer', Tokenizer()), task,
        model=kwargs.pop('model', {'label': '7b', 'vllm_version': '0.8.4'}),
        runtime=kwargs.pop('runtime', llm._modebench_runtime),
        code=kwargs.pop('code', evaluator.code_identity()),
        grader=lambda text, _: text if text != 'wrong' else None,
        params_factory=kwargs.pop('params_factory', lambda args, *_: SimpleNamespace(seed=args.seed, n=8)),
        **kwargs)


def test_interface_reuses_all_frozen_settings_without_changing_old_namespace():
    for domain in evaluator.DOMAINS:
        old = evaluator.frozen.frozen_interface(domain)
        new = evaluator.frozen_interface(domain)
        assert new == {**old, 'name': evaluator.INTERFACE}
        assert old['name'] == 'level2_qwen_r5_independent_v2'
    assert evaluator.SCHEMA != evaluator.frozen.SCHEMA
    code = evaluator.code_identity()
    assert set(evaluator.frozen.code_identity()) < set(code)
    assert code['ops/evaluate_modebench_scale.py'] == evaluator.frozen.file_sha(Path(evaluator.__file__))


@pytest.mark.parametrize(('level', 'label'), [('level1', '05b'), ('level3', '05b'),
                                          ('level4', '7b'), ('level5', '14b')])
def test_complete_scale_receipt_records_real_level_runtime_and_frozen_seeds(tmp_path, level, label):
    task = dict(make_task(tmp_path), level=level)
    runtime = evaluator.runtime_settings(max_model_len=4096, tensor_parallel_size=2,
                                         gpu_memory_utilization=.9, enable_prefix_caching=False)
    llm = LLM(runtime=runtime)
    receipt = evaluate(llm, task, model={'label': label, 'vllm_version': '0.8.4'})
    identity = receipt['identity']
    assert receipt['schema'] == identity['schema'] == evaluator.SCHEMA
    assert receipt['level'] == identity['level'] == level
    assert receipt['model_label'] == label
    assert identity['runtime'] == runtime
    assert identity['interface']['max_model_len'] == 1024  # Actual engine context does not enlarge prompt budget.
    assert identity['interface']['max_tokens'] == 192
    rows, source = evaluator.load_rows(task)
    assert identity['source'] == source
    schedule = evaluator.frozen.schedule_record(task['domain'], rows, task['seeds'])
    assert identity['seed_schedule'] == schedule
    forwarded = [params.seed for _, batch in llm.calls for params in batch]
    assert forwarded == [row[index] for index in range(4) for row in schedule['request_seeds']]
    assert receipt['metrics']['pass1'] == 3 / 8
    assert receipt['metrics']['pass8'] == 1
    assert receipt['metrics']['distinct8'] == 2
    assert evaluator.validate_seed_receipt(receipt)['distinct_child_seeds'] == 96
    assert json.loads(Path(task['output']).read_text()) == receipt
    manifest = json.loads(Path(task['output'] + '.batches/run.json').read_text())
    assert manifest['identity'] == identity
    with pytest.raises(ValueError, match='independent v2 complete receipt'):
        evaluator.frozen.validate_seed_receipt(receipt, rows)


def test_interrupted_scale_run_resumes_only_missing_batches_and_reuses_complete_receipt(tmp_path):
    task = make_task(tmp_path)
    with pytest.raises(RuntimeError, match='worker interruption'):
        evaluate(LLM(fail_call=2), task)
    path = next(Path(task['output'] + '.batches').glob('seed-*'))
    saved = path.read_bytes()
    with pytest.raises(FileExistsError, match='partial run'):
        evaluate(LLM(), task)
    resumed = LLM()
    receipt = evaluate(resumed, task, resume=True)
    assert len(resumed.calls) == 7
    assert path.read_bytes() == saved
    complete = LLM()
    assert evaluate(complete, task, resume=True) == receipt
    assert not complete.calls
    with pytest.raises(FileExistsError, match='fresh final receipt'):
        evaluate(LLM(), task)


@pytest.mark.parametrize('change', ['runtime', 'model', 'source', 'prompt', 'level'])
def test_resume_refuses_provenance_changes_before_sampling(tmp_path, change):
    task = make_task(tmp_path)
    with pytest.raises(RuntimeError):
        evaluate(LLM(fail_call=2), task)
    llm, kwargs = LLM(), {}
    if change == 'runtime':
        llm._modebench_runtime['tensor_parallel_size'] = 2
    elif change == 'model':
        kwargs['model'] = {'label': '14b', 'vllm_version': '0.8.4'}
    elif change == 'source':
        source = Path(task['rows_jsonl'])
        source.write_text(source.read_text().replace('Problem 1', 'Changed problem'))
    elif change == 'prompt':
        kwargs['tokenizer'] = Tokenizer('changed template')
    else:
        task['level'] = 'level5'
    with pytest.raises(ValueError, match='resume identity mismatch'):
        evaluate(llm, task, resume=True, **kwargs)
    assert not llm.calls


@pytest.mark.parametrize('change', ['old_schema', 'old_identity_schema', 'old_interface', 'missing_code',
                                  'runtime', 'engine', 'request_seed', 'schedule', 'source', 'level', 'labels'])
def test_public_validator_rejects_wrong_schema_or_changed_provenance(tmp_path, change):
    receipt = evaluate(LLM(), make_task(tmp_path))
    identity = receipt['identity']
    if change == 'old_schema':
        receipt['schema'] = evaluator.frozen.SCHEMA
    elif change == 'old_identity_schema':
        identity['schema'] = evaluator.frozen.SCHEMA
    elif change == 'old_interface':
        identity['interface'] = evaluator.frozen.frozen_interface('graph_coloring')
        identity['interface_sha256'] = evaluator.sha(identity['interface'])
    elif change == 'missing_code':
        del identity['code_sha256']['ops/evaluate_modebench_scale.py']
    elif change == 'runtime':
        del identity['runtime']['tensor_parallel_size']
    elif change == 'engine':
        identity['sampling_engine']['engine'] = 'V1'
    elif change == 'request_seed':
        receipt['prompt_results'][0]['draws'][0]['request_seed'] += 8
    elif change == 'schedule':
        identity['seed_schedule']['request_seeds'][0][0] += 8
        identity['seed_schedule_sha256'] = evaluator.sha(identity['seed_schedule'])
    elif change == 'source':
        identity['source']['rows_sha256'] = '0' * 64
    elif change == 'level':
        receipt['level'] = 'level3'
    else:
        identity['seeds'] = identity['seeds'][:1]
    receipt['identity_sha256'] = evaluator.sha(identity)
    with pytest.raises(ValueError):
        evaluator.validate_seed_receipt(receipt)


def test_resume_rejects_rehashed_rng_tampering(tmp_path):
    task = make_task(tmp_path)
    with pytest.raises(RuntimeError):
        evaluate(LLM(fail_call=2), task)
    path = next(Path(task['output'] + '.batches').glob('seed-*'))
    batch = json.loads(path.read_text())
    batch['draws'][0]['child_seeds'][0] += 8
    batch['draws_sha256'] = evaluator.sha(batch['draws'])
    path.write_text(json.dumps(batch))
    llm = LLM()
    with pytest.raises(ValueError, match='draw RNG metadata'):
        evaluate(llm, task, resume=True)
    assert not llm.calls


@pytest.mark.parametrize('change', ['draw', 'row', 'summary', 'attempt_count', 'histogram', 'boundary'])
def test_public_validator_recomputes_metric_evidence_and_boundary(tmp_path, change):
    receipt = evaluate(LLM(), dict(make_task(tmp_path), split='eval'), confirm_eval=True)
    if change == 'draw':
        receipt['prompt_results'][0]['draws'][0]['pass1'] = .9
    elif change == 'row':
        receipt['prompt_results'][0]['distinct8'] = 8
    elif change == 'summary':
        receipt['metrics']['pass8'] = .9
    elif change == 'attempt_count':
        receipt['prompt_results'][0]['draws'][0]['attempts'].pop()
    elif change == 'histogram':
        receipt['answer_mode_histogram'] = {}
    else:
        receipt['information_boundary']['confirmation_explicitly_authorized'] = False
    with pytest.raises(ValueError):
        evaluator.validate_seed_receipt(receipt)


def test_eval_confirmation_requires_full_split(tmp_path):
    task = dict(make_task(tmp_path), split='eval')
    with pytest.raises(ValueError, match='--confirm-eval'):
        evaluate(LLM(), task)
    with pytest.raises(ValueError, match='full split'):
        evaluate(LLM(), dict(task, row_offset=1), confirm_eval=True)
    receipt = evaluate(LLM(), task, confirm_eval=True)
    assert receipt['information_boundary'] == {
        'evaluation_prompts_loaded': True, 'confirmation_explicitly_authorized': True,
        'treatment_training_started': False}


@pytest.mark.parametrize('labels', [[1], [1, 2, 3, 3], [True, 2, 3, 4], [1, 2, 3, -1]])
def test_task_requires_four_registered_integer_draw_labels(tmp_path, labels):
    llm = LLM()
    with pytest.raises(ValueError, match='exactly four'):
        evaluate(llm, dict(make_task(tmp_path), seeds=labels))
    assert not llm.calls


def test_frozen_context_budget_is_preserved_with_larger_engine_context(tmp_path):
    llm = LLM(runtime=evaluator.runtime_settings(max_model_len=8192))
    with pytest.raises(ValueError, match='frozen context budget'):
        evaluate(llm, make_task(tmp_path), tokenizer=Tokenizer('x' * 1024))
    assert not llm.calls
    assert not list(tmp_path.glob('*.batches'))


def test_runtime_mismatch_or_wrong_sampling_seed_stops_before_generation(tmp_path):
    task = make_task(tmp_path)
    llm = LLM()
    changed = {**llm._modebench_runtime, 'tensor_parallel_size': 2}
    with pytest.raises(ValueError, match='loaded model runtime differs'):
        evaluate(llm, task, runtime=changed)
    with pytest.raises(ValueError, match='scheduled seed and n=8'):
        evaluate(llm, task, params_factory=lambda args, *_: SimpleNamespace(seed=args.seed + 8, n=8))
    assert not llm.calls


def test_saved_dataset_source_uses_frozen_loader(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    rows, _ = evaluator.load_rows(task)
    dataset_path = tmp_path / 'dataset'
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(
        load_from_disk=lambda path: {'multi_answer': rows} if path == str(dataset_path) else None))
    del task['rows_jsonl']
    task['dataset'] = str(dataset_path)
    receipt = evaluate(LLM(), task)
    assert receipt['identity']['source']['kind'] == 'saved_dataset'
    assert evaluator.validate_seed_receipt(receipt)['prompts'] == 3


def test_cli_passes_scale_settings_to_model_and_every_task(tmp_path, monkeypatch):
    first = make_task(tmp_path)
    second = {**first, 'domain': 'countdown', 'level': 'level5', 'output': str(tmp_path / 'second.json')}
    manifest = tmp_path / 'tasks.json'
    manifest.write_text(json.dumps([first, second]))
    loaded, evaluated = [], []
    def create_llm(**kwargs):
        loaded.append(kwargs)
        return SimpleNamespace(get_tokenizer=lambda: Tokenizer())
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=create_llm))
    monkeypatch.setattr(evaluator.frozen, 'validate_runtime_contract', lambda: '0.8.4')
    monkeypatch.setattr(evaluator, 'model_identity', lambda path, label: {'label': label, 'path': str(path)})
    monkeypatch.setattr(evaluator, 'evaluate_task', lambda llm, tokenizer, task, **kwargs:
                        evaluated.append((llm, task, kwargs)))
    evaluator.main(['--model', str(tmp_path), '--model-label', '14b', '--tasks-json', str(manifest),
                    '--tensor-parallel-size', '2', '--gpu-memory-utilization', '.9',
                    '--max-model-len', '4096', '--swap-space', '8', '--no-enable-prefix-caching', '--resume'])
    expected = evaluator.runtime_settings(max_model_len=4096, tensor_parallel_size=2,
                                         gpu_memory_utilization=.9, swap_space=8., enable_prefix_caching=False)
    assert loaded == [{'model': str(tmp_path), **expected}]
    assert len(evaluated) == 2
    for llm, task, kwargs in evaluated:
        assert llm._modebench_runtime == kwargs['runtime'] == expected
        assert kwargs['model']['label'] == '14b'
        assert kwargs['resume'] is True
        assert task['interface'] == evaluator.INTERFACE


def test_cli_rejects_cross_task_rng_overlap_before_loading(tmp_path, monkeypatch):
    first = make_task(tmp_path)
    manifest = tmp_path / 'tasks.json'
    manifest.write_text(json.dumps([first, {**first, 'output': str(tmp_path / 'other.json')}]))
    monkeypatch.setattr(evaluator, 'model_identity', lambda *_: pytest.fail('model metadata reached'))
    with pytest.raises(ValueError, match='across task manifest'):
        evaluator.main(['--model', str(tmp_path), '--model-label', '7b', '--tasks-json', str(manifest)])


@pytest.mark.parametrize(('flag', 'value'), [('--tensor-parallel-size', '0'),
                                          ('--gpu-memory-utilization', 'nan'),
                                          ('--gpu-memory-utilization', '1.1'),
                                          ('--max-model-len', '512')])
def test_cli_rejects_invalid_runtime_before_loading(tmp_path, monkeypatch, flag, value):
    task = make_task(tmp_path)
    manifest = tmp_path / 'tasks.json'
    manifest.write_text(json.dumps([task]))
    monkeypatch.setattr(evaluator, 'model_identity', lambda *_: pytest.fail('model metadata reached'))
    with pytest.raises(ValueError):
        evaluator.main(['--model', str(tmp_path), '--model-label', '7b', '--tasks-json', str(manifest), flag, value])


def test_cli_checks_pinned_engine_before_model_metadata(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    monkeypatch.delenv('VLLM_USE_V1', raising=False)
    monkeypatch.setattr(evaluator, 'model_identity', lambda *_: pytest.fail('model metadata reached'))
    with pytest.raises(ValueError, match='VLLM_USE_V1=0'):
        evaluator.main(['--model', str(tmp_path), '--model-label', '7b', '--domain', task['domain'],
                        '--rows-jsonl', task['rows_jsonl'], '--output', task['output'],
                        '--seeds', *map(str, task['seeds'])])
