"""The frozen-base grid rejects unregistered cells and incomplete evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import evaluate_modebench_base_grid as evaluator
import evaluate_modebench_scale as sealed


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def make_model(tmp_path, label='3b'):
    size, revision, dimensions = evaluator.MODEL_SPECS[label]
    path = tmp_path / f'models--Qwen--Qwen2.5-{size}-Instruct/snapshots' / revision
    write_json(path / 'config.json', {'model_type': 'qwen2', 'architectures': ['Qwen2ForCausalLM'],
                                     **dict(zip(evaluator.ARCHITECTURE_FIELDS, dimensions))})
    write_json(path / 'tokenizer.json', {})
    write_json(path / 'tokenizer_config.json', {'chat_template': 'native template'})
    (path / 'model.safetensors').write_bytes(b'test-only upstream fixture')
    return {**evaluator.model_identity(path, label), 'vllm_version': '0.8.4'}


def make_task(tmp_path, level='level2', domain='graph_coloring', count=128):
    rows = [{'problem': f'{domain} problem {level} {index}', 'answer': json.dumps({'target': index}),
             'answer_mode_count': 4, 'candidate_id': str(index)} for index in range(count)]
    path = tmp_path / f'{level}_{domain}.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    original = tmp_path / f'original_{level}_{domain}'
    original.mkdir()
    (original / 'data.arrow').write_bytes(b'fixture original dataset bytes')
    admission = tmp_path / f'admission_{level}_{domain}.json'
    write_json(admission, {'level': level, 'domain': domain, 'status': 'admitted'})
    entry = {'level': level, 'domain': domain, 'split': 'eval', 'status': 'admitted',
             'dataset_path': str(original), 'dataset_sha256': evaluator.sha(evaluator.dataset_inventory(original)),
             'rows': count, 'rows_sha256': evaluator.sha(rows),
             'rows_jsonl': str(path), 'rows_jsonl_sha256': evaluator.frozen.file_sha(path),
             'admission_evidence': [{'path': str(admission), 'sha256': evaluator.frozen.file_sha(admission),
                                     'role': 'frozen construction admission'}]}
    registry = tmp_path / f'registry_{level}_{domain}.json'
    write_json(registry, {'schema': evaluator.REGISTRY_SCHEMA, 'datasets': [entry]})
    binding = {**entry, 'source_manifest_path': str(registry),
               'source_manifest_sha256': evaluator.frozen.file_sha(registry)}
    return {'domain': domain, 'level': level, 'split': 'eval', 'rows_jsonl': str(path),
            'dataset_binding': binding, 'output': str(tmp_path / f'{level}_{domain}_run.json'),
            'seeds': [91212000, 91212001, 91212002, 91212003], 'batch_size': 64}


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        return json.dumps(messages)

    def encode(self, prompt, **kwargs):
        return list(prompt)


class LLM:
    def __init__(self, model, fail_call=None):
        self._modebench_runtime = evaluator.runtime_settings(max_model_len=2048)
        self._modebench_model_identity = model
        self.fail_call = fail_call
        self.calls = []

    def generate(self, prompts, params, use_tqdm=False):
        self.calls.append((prompts, params))
        if len(self.calls) == self.fail_call:
            raise RuntimeError('simulated worker interruption')
        return [SimpleNamespace(prompt=prompt, outputs=[
            SimpleNamespace(text=text, token_ids=[1, 2], finish_reason='stop')
            for text in ('a', 'a', 'b', 'wrong', 'wrong', 'wrong', 'wrong', 'wrong')
        ]) for prompt in prompts]


def evaluate(llm, task, **kwargs):
    return evaluator.evaluate_task(
        llm, Tokenizer(), task, model=llm._modebench_model_identity,
        runtime=llm._modebench_runtime, code=evaluator.code_identity(),
        confirm_eval=kwargs.pop('confirm_eval', True),
        grader=lambda text, _: {'native_key': [text, 1]} if text != 'wrong' else None,
        params_factory=lambda args, *_: SimpleNamespace(seed=args.seed, n=8), **kwargs)


def test_namespace_changes_without_changing_frozen_sampling_or_model_level_scope():
    assert evaluator.SCHEMA not in (sealed.SCHEMA, evaluator.frozen.SCHEMA)
    assert evaluator.MODEL_LABELS == ('05b', '3b', '7b', '14b')
    assert evaluator.LEVELS == ('level1', 'level2', 'level3', 'level4', 'level5')
    for domain in evaluator.DOMAINS:
        old = evaluator.frozen.frozen_interface(domain)
        assert evaluator.frozen_interface(domain) == {**old, 'name': evaluator.INTERFACE}
        assert old['max_tokens'] == 192 and old['temperature'] == old['top_p'] == 1
    assert evaluator.code_identity()['ops/evaluate_modebench_base_grid.py'] == evaluator.frozen.file_sha(
        Path(evaluator.__file__))


def test_complete_registered_cell_keeps_native_keys_and_all_independent_draws(tmp_path):
    task, model = make_task(tmp_path), make_model(tmp_path)
    llm = LLM(model)
    receipt = evaluate(llm, task)
    assert receipt['identity']['purpose'] == 'frozen_base_model_benchmarking'
    assert receipt['identity']['dataset_binding'] == task['dataset_binding']
    assert receipt['metrics']['rows'] == 128
    assert receipt['metrics']['pass1'] == 3 / 8
    assert receipt['metrics']['pass8'] == 1
    assert receipt['metrics']['distinct8'] == 2
    assert receipt['prompt_results'][0]['draws'][0]['attempts'][0]['canonical_key'] == {'native_key': ['a', 1]}
    checked = evaluator.validate_seed_receipt(receipt)
    assert checked['distinct_request_blocks'] == 512 and checked['distinct_child_seeds'] == 4096
    forwarded = [params.seed for _, batch in llm.calls for params in batch]
    schedule = receipt['identity']['seed_schedule']
    assert forwarded == [row[draw] for draw in range(4) for row in schedule['request_seeds']]
    assert json.loads(Path(task['output']).read_text()) == receipt
    with pytest.raises(ValueError, match='complete scale receipt'):
        sealed.validate_seed_receipt(receipt)


def test_interrupted_cell_resumes_only_missing_batches_and_validates_completed_cell(tmp_path):
    task, model = make_task(tmp_path), make_model(tmp_path)
    with pytest.raises(RuntimeError, match='worker interruption'):
        evaluate(LLM(model, fail_call=2), task)
    batch = next(Path(task['output'] + '.batches').glob('seed-*'))
    saved = batch.read_bytes()
    resumed = LLM(model)
    receipt = evaluate(resumed, task, resume=True)
    assert len(resumed.calls) == 7 and batch.read_bytes() == saved
    completed = LLM(model)
    assert evaluate(completed, task, resume=True) == receipt and not completed.calls
    with pytest.raises(FileExistsError):
        evaluate(LLM(model), task)


@pytest.mark.parametrize('change', ['level', 'domain', 'registry_entry', 'registry_hash', 'rows',
                                  'original', 'admission', 'pending', 'missing', 'slice', 'dev'])
def test_unregistered_or_changed_dataset_fails_before_generation(tmp_path, change):
    task, model = make_task(tmp_path), make_model(tmp_path)
    binding = task['dataset_binding']
    if change == 'level':
        task['level'] = 'level5'
    elif change == 'domain':
        task['domain'] = 'countdown'
    elif change == 'registry_entry':
        binding['level'] = task['level'] = 'level5'
    elif change == 'registry_hash':
        binding['source_manifest_sha256'] = '0' * 64
    elif change == 'rows':
        Path(task['rows_jsonl']).write_text('{}\n')
    elif change == 'original':
        (Path(binding['dataset_path']) / 'data.arrow').write_bytes(b'changed')
    elif change == 'admission':
        Path(binding['admission_evidence'][0]['path']).write_text('{}\n')
    elif change == 'pending':
        binding['status'] = 'pending'
    elif change == 'missing':
        del task['dataset_binding']
    elif change == 'slice':
        task['row_limit'] = 1
    else:
        task['split'] = 'dev'
    llm = LLM(model)
    with pytest.raises(ValueError):
        evaluate(llm, task)
    assert not llm.calls


def test_even_registered_partial_split_and_unconfirmed_eval_are_rejected(tmp_path):
    model = make_model(tmp_path)
    task = make_task(tmp_path, count=127)
    with pytest.raises(ValueError, match='exactly 128'):
        evaluate(LLM(model), task)
    with pytest.raises(ValueError, match='--confirm-eval'):
        evaluate(LLM(model), task, confirm_eval=False)


@pytest.mark.parametrize('change', ['missing_prompt', 'missing_draw', 'missing_attempt', 'draw_metric',
                                  'summary', 'key', 'child_seed', 'model', 'purpose', 'level'])
def test_completed_receipts_fail_closed_for_incomplete_or_changed_evidence(tmp_path, change):
    task, model = make_task(tmp_path), make_model(tmp_path)
    receipt = evaluate(LLM(model), task)
    identity = receipt['identity']
    draw = receipt['prompt_results'][0]['draws'][0]
    if change == 'missing_prompt':
        receipt['prompt_results'].pop()
    elif change == 'missing_draw':
        receipt['prompt_results'][0]['draws'].pop()
    elif change == 'missing_attempt':
        draw['attempts'].pop()
    elif change == 'draw_metric':
        draw['distinct8'] = 7
    elif change == 'summary':
        receipt['metrics']['pass8'] = .5
    elif change == 'key':
        draw['attempts'][0]['canonical_key'] = None
    elif change == 'child_seed':
        draw['child_seeds'][0] += 8
    elif change == 'model':
        identity['model']['revision'] = '0' * 40
    elif change == 'purpose':
        identity['purpose'] = 'difficulty_calibration'
    else:
        identity['level'] = receipt['level'] = 'level5'
    receipt['identity_sha256'] = evaluator.sha(identity)
    with pytest.raises(ValueError):
        evaluator.validate_seed_receipt(receipt)


def test_rehashed_resumed_batch_rng_tampering_fails_before_sampling(tmp_path):
    task, model = make_task(tmp_path), make_model(tmp_path)
    with pytest.raises(RuntimeError):
        evaluate(LLM(model, fail_call=2), task)
    path = next(Path(task['output'] + '.batches').glob('seed-*'))
    batch = json.loads(path.read_text())
    batch['draws'][0]['child_seeds'][0] += 8
    batch['draws_sha256'] = evaluator.sha(batch['draws'])
    write_json(path, batch)
    llm = LLM(model)
    with pytest.raises(ValueError, match='draw RNG metadata'):
        evaluate(llm, task, resume=True)
    assert not llm.calls


@pytest.mark.parametrize('label', evaluator.MODEL_LABELS)
def test_base_model_identity_verifies_architecture_and_original_snapshot(tmp_path, label):
    identity = make_model(tmp_path, label)
    evaluator.validate_model_identity(identity)
    assert identity['model_role'] == 'frozen_base_model_before_modebench_training'
    path = Path(identity['path'])
    config = json.loads((path / 'config.json').read_text())
    config['hidden_size'] += 1
    write_json(path / 'config.json', config)
    with pytest.raises(ValueError, match='architecture'):
        evaluator.model_identity(path, label)
    with pytest.raises(ValueError, match='upstream'):
        evaluator.model_identity(tmp_path / 'fine_tuned_checkpoint', label)


def test_cli_rejects_duplicate_rng_blocks_before_loading_a_model(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    tasks = tmp_path / 'tasks.json'
    write_json(tasks, [task, {**task, 'output': str(tmp_path / 'other.json')}])
    monkeypatch.setattr(evaluator.frozen, 'validate_runtime_contract',
                        lambda: pytest.fail('model runtime check reached'))
    with pytest.raises(ValueError, match='across task manifest'):
        evaluator.main(['--model', str(tmp_path), '--model-label', '3b', '--tasks-json', str(tasks),
                        '--confirm-eval'])


def test_cli_preserves_registered_task_and_actual_engine_settings(tmp_path, monkeypatch):
    task, model = make_task(tmp_path), make_model(tmp_path)
    tasks = tmp_path / 'tasks.json'
    write_json(tasks, [task])
    loaded, evaluated = [], []
    monkeypatch.setattr(evaluator.frozen, 'validate_runtime_contract', lambda: '0.8.4')
    def make_llm(**kwargs):
        loaded.append(kwargs)
        return SimpleNamespace(get_tokenizer=lambda: Tokenizer())
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=make_llm))
    monkeypatch.setattr(evaluator, 'evaluate_task', lambda llm, tokenizer, task, **kwargs:
                        evaluated.append((llm, task, kwargs)))
    evaluator.main(['--model', model['path'], '--model-label', '3b', '--tasks-json', str(tasks),
                    '--confirm-eval', '--tensor-parallel-size', '2', '--max-model-len', '4096', '--resume'])
    assert loaded[0]['model'] == model['path'] and loaded[0]['tensor_parallel_size'] == 2
    llm, retained, kwargs = evaluated[0]
    assert retained['dataset_binding'] == task['dataset_binding']
    assert kwargs['model'] == llm._modebench_model_identity == model
    assert kwargs['runtime'] == llm._modebench_runtime
    assert kwargs['confirm_eval'] is True and kwargs['resume'] is True


def test_supplied_rows_cannot_hide_changed_receipt_source_metadata(tmp_path):
    task, model = make_task(tmp_path), make_model(tmp_path)
    receipt = evaluate(LLM(model), task)
    rows, _ = evaluator.load_rows(task)
    receipt['identity']['source']['file_sha256'] = '0' * 64
    receipt['identity_sha256'] = evaluator.sha(receipt['identity'])
    with pytest.raises(ValueError, match='saved source identity changed'):
        evaluator.validate_seed_receipt(receipt, rows)
