"""Independent-stream receipts preserve the frozen interface and reject RNG drift."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'ops/evaluate_modebench_level3_independent.py'
SPEC = spec_from_file_location('modebench_independent_evaluator_test', PATH)
evaluator = module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


class Tokenizer:
    def __init__(self, prefix=''):
        self.prefix = prefix

    def apply_chat_template(self, messages, **kwargs):
        return self.prefix + json.dumps(messages)

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


def make_task(tmp_path, *, count=3, name='run'):
    rows = [{'problem': f'Problem {index}', 'answer': json.dumps({'target': index}),
             'answer_mode_count': 4, 'candidate_id': str(index)} for index in range(count)]
    source = tmp_path / f'{name}.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return {'domain': 'graph_coloring', 'level': 'level3', 'split': 'dev',
            'rows_jsonl': str(source), 'output': str(tmp_path / f'{name}.json'),
            'seeds': [6318000, 6318001, 6318002, 6318003], 'batch_size': 2}


def evaluate(llm, task, **kwargs):
    return evaluator.evaluate_task(llm, kwargs.pop('tokenizer', Tokenizer()), task,
        model={'label': '3b', 'vllm_version': '0.8.4'}, code=evaluator.code_identity(),
        grader=lambda text, _: text if text != 'wrong' else None,
        params_factory=kwargs.pop('params_factory', lambda args, domain, row: SimpleNamespace(seed=args.seed, n=8)),
        **kwargs)


def rehash_identity(receipt):
    receipt['identity_sha256'] = evaluator.sha(receipt['identity'])


def test_interface_changes_only_namespace_and_rng_policy():
    for domain in evaluator.DOMAINS:
        old = evaluator.original.frozen_interface(domain, 'level2_qwen_r5')
        new = evaluator.frozen_interface(domain)
        assert new['name'] == evaluator.INTERFACE
        assert new['seed_policy'] == evaluator.POLICY
        assert {k: v for k, v in old.items() if k not in ('name', 'seed_policy')} == {
            k: v for k, v in new.items() if k not in ('name', 'seed_policy')}
    with pytest.raises(ValueError, match='independent v2 interface'):
        evaluator.frozen_interface('graph_coloring', 'level2_qwen_r5')


def test_all_effective_seeds_are_forwarded_recorded_and_disjoint(tmp_path):
    task = make_task(tmp_path)
    llm = LLM()
    receipt = evaluate(llm, task)
    schedule = receipt['identity']['seed_schedule']
    forwarded = [params.seed for _, batch in llm.calls for params in batch]
    expected_order = [row[index] for index in range(4) for row in schedule['request_seeds']]
    assert forwarded == expected_order
    children = [seed for row in schedule['child_seeds'] for draw in row for seed in draw]
    assert len(children) == len(set(children)) == 3 * 4 * 8
    assert receipt['identity']['seed_schedule_sha256'] == evaluator.sha(schedule)
    for index, row in enumerate(receipt['prompt_results']):
        assert [draw['seed'] for draw in row['draws']] == task['seeds']
        for draw_index, draw in enumerate(row['draws']):
            assert draw['request_seed'] == schedule['request_seeds'][index][draw_index]
            assert draw['child_seeds'] == schedule['child_seeds'][index][draw_index]
    assert receipt['metrics']['pass1'] == 3 / 8
    assert receipt['metrics']['pass8'] == 1
    assert receipt['metrics']['distinct8'] == 2
    assert evaluator.validate_seed_receipt(receipt)['distinct_child_seeds'] == 96


def test_seed_schedule_uses_raw_problem_and_survives_renderer_batching_sharding(tmp_path):
    task = make_task(tmp_path)
    complete = evaluate(LLM(), task)
    changed = dict(task, output=str(tmp_path / 'slice.json'), row_offset=1, row_limit=1, batch_size=1)
    sliced = evaluate(LLM(), changed, tokenizer=Tokenizer('new renderer'))
    assert sliced['identity']['seed_schedule']['request_seeds'] == complete['identity']['seed_schedule']['request_seeds'][1:2]
    assert sliced['identity']['seed_schedule']['child_seeds'] == complete['identity']['seed_schedule']['child_seeds'][1:2]


def test_params_factory_receives_a_fresh_namespace_per_row(tmp_path):
    task = make_task(tmp_path)
    arguments = []
    def factory(args, *_):
        arguments.append(args)
        result = SimpleNamespace(seed=args.seed, n=8)
        args.seed = -1
        return result
    evaluate(LLM(), task, params_factory=factory)
    assert len({id(args) for args in arguments}) == 12


def test_interruption_preserves_batches_and_resumes_only_missing_work(tmp_path):
    task = make_task(tmp_path)
    with pytest.raises(RuntimeError, match='simulated worker'):
        evaluate(LLM(fail_call=2), task)
    batch = next(Path(task['output'] + '.batches').glob('seed-*'))
    before = batch.read_bytes()
    resumed = LLM()
    result = evaluate(resumed, task, resume=True)
    assert len(resumed.calls) == 7
    assert batch.read_bytes() == before
    completed = LLM()
    assert evaluate(completed, task, resume=True) == result
    assert not completed.calls
    with pytest.raises(FileExistsError):
        evaluate(LLM(), task)


@pytest.mark.parametrize('field', ['seed', 'request_seed', 'child_seeds'])
def test_resumed_batch_rejects_changed_seed_metadata_even_with_recomputed_digest(tmp_path, field):
    task = make_task(tmp_path)
    with pytest.raises(RuntimeError):
        evaluate(LLM(fail_call=2), task)
    path = next(Path(task['output'] + '.batches').glob('seed-*'))
    batch = json.loads(path.read_text())
    if field == 'child_seeds':
        batch['draws'][0][field][1] += 8
    else:
        batch['draws'][0][field] += 8
    batch['draws_sha256'] = evaluator.sha(batch['draws'])
    path.write_text(json.dumps(batch))
    llm = LLM()
    with pytest.raises(ValueError, match='draw RNG metadata'):
        evaluate(llm, task, resume=True)
    assert not llm.calls


def test_completed_receipt_seed_tampering_is_rejected_on_resume(tmp_path):
    task = make_task(tmp_path)
    receipt = evaluate(LLM(), task)
    receipt['prompt_results'][0]['draws'][0]['request_seed'] += 8
    Path(task['output']).write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='draw RNG metadata'):
        evaluate(LLM(), task, resume=True)


@pytest.mark.parametrize('mutation', ['duplicate_block', 'wrong_effective_seed', 'missing_schedule', 'missing_code_hash', 'old_schema', 'old_interface'])
def test_public_seed_validator_rejects_forged_metadata(tmp_path, mutation):
    receipt = evaluate(LLM(), make_task(tmp_path))
    if mutation == 'duplicate_block':
        schedule = receipt['identity']['seed_schedule']
        schedule['request_seeds'][0][1] = schedule['request_seeds'][0][0]
        schedule['child_seeds'][0][1] = schedule['child_seeds'][0][0]
        receipt['identity']['seed_schedule_sha256'] = evaluator.sha(schedule)
    elif mutation == 'wrong_effective_seed':
        receipt['prompt_results'][0]['draws'][0]['request_seed'] += 8
    elif mutation == 'missing_schedule':
        del receipt['identity']['seed_schedule']
    elif mutation == 'missing_code_hash':
        del receipt['identity']['code_sha256']['ops/modebench_independent_seeds.py']
    elif mutation == 'old_schema':
        receipt['schema'] = evaluator.original.SCHEMA
    else:
        receipt['identity']['interface'] = evaluator.original.frozen_interface('graph_coloring', 'level2_qwen_r5')
        receipt['identity']['interface_sha256'] = evaluator.sha(receipt['identity']['interface'])
    rehash_identity(receipt)
    with pytest.raises(ValueError):
        evaluator.validate_seed_receipt(receipt)


def test_public_validator_checks_supplied_rows_and_saved_source(tmp_path):
    task = make_task(tmp_path)
    receipt = evaluate(LLM(), task)
    rows, _ = evaluator.load_rows(task)
    assert evaluator.validate_seed_receipt(receipt, rows)['distinct_request_blocks'] == 12
    rows[0]['problem'] = 'Changed problem'
    with pytest.raises(ValueError, match='selected source rows'):
        evaluator.validate_seed_receipt(receipt, rows)
    source = Path(task['rows_jsonl'])
    source.write_text(source.read_text().replace('Problem 0', 'Changed problem'))
    with pytest.raises(ValueError, match='saved source identity changed'):
        evaluator.validate_seed_receipt(receipt)


def test_duplicate_problem_rejected_before_sampling_or_run_manifest(tmp_path):
    task = make_task(tmp_path)
    source = Path(task['rows_jsonl'])
    source.write_text(source.read_text().replace('Problem 1', 'Problem 0'))
    llm = LLM()
    with pytest.raises(ValueError, match='duplicate prompt'):
        evaluate(llm, task)
    assert not llm.calls
    assert not Path(task['output'] + '.batches').exists()


def test_manifest_overlap_is_rejected_before_model_loading(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    tasks = [task, dict(task, output=str(tmp_path / 'duplicate.json'))]
    manifest = tmp_path / 'tasks.json'
    manifest.write_text(json.dumps(tasks))
    monkeypatch.setattr(evaluator, 'model_identity', lambda *_: pytest.fail('model loading was reached'))
    with pytest.raises(ValueError, match='across task manifest'):
        evaluator.main(['--model', str(tmp_path), '--model-label', '3b', '--tasks-json', str(manifest)])


def test_confirmation_gate_and_complete_source_rule_are_unchanged(tmp_path):
    task = dict(make_task(tmp_path), split='eval')
    with pytest.raises(ValueError, match='--confirm-eval'):
        evaluate(LLM(), task)
    with pytest.raises(ValueError, match='full split'):
        evaluate(LLM(), dict(task, row_limit=1), confirm_eval=True)
    receipt = evaluate(LLM(), task, confirm_eval=True)
    assert receipt['information_boundary']['confirmation_explicitly_authorized']
    assert receipt['information_boundary']['evaluation_prompts_loaded']


def test_wrong_actual_sampling_seed_is_rejected(tmp_path):
    task = make_task(tmp_path)
    llm = LLM()
    with pytest.raises(ValueError, match='sampling parameters must retain'):
        evaluate(llm, task, params_factory=lambda args, *_: SimpleNamespace(seed=args.seed + 1, n=8))
    assert not llm.calls


def test_code_identity_includes_all_implementation_dependencies():
    code = evaluator.code_identity()
    assert set(evaluator.original.code_identity()) <= set(code)
    assert code['ops/evaluate_modebench_level3_independent.py'] == evaluator.file_sha(PATH)
    assert code['ops/modebench_independent_seeds.py'] == evaluator.file_sha(PATH.with_name('modebench_independent_seeds.py'))


@pytest.mark.parametrize('engine_setting', [None, '1'])
def test_cli_rejects_unverified_engine_before_model_metadata(tmp_path, monkeypatch, engine_setting):
    if engine_setting is None:
        monkeypatch.delenv('VLLM_USE_V1', raising=False)
    else:
        monkeypatch.setenv('VLLM_USE_V1', engine_setting)
    task = make_task(tmp_path)
    monkeypatch.setattr(evaluator, 'model_identity', lambda *_: pytest.fail('model construction was reached'))
    with pytest.raises(ValueError, match='VLLM_USE_V1=0'):
        evaluator.main(['--model', str(tmp_path), '--model-label', '3b', '--domain', 'graph_coloring',
                        '--rows-jsonl', task['rows_jsonl'], '--output', task['output'], '--seeds', '6318000'])


def test_cli_rejects_unverified_vllm_version_before_model_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv('VLLM_USE_V1', '0')
    monkeypatch.setattr(evaluator.importlib.metadata, 'version', lambda _: '0.9.0')
    monkeypatch.setattr(evaluator, 'model_identity', lambda *_: pytest.fail('model construction was reached'))
    task = make_task(tmp_path)
    with pytest.raises(ValueError, match='requires vLLM 0.8.4'):
        evaluator.main(['--model', str(tmp_path), '--model-label', '3b', '--domain', 'graph_coloring',
                        '--rows-jsonl', task['rows_jsonl'], '--output', task['output'], '--seeds', '6318000'])


def test_seed_validator_rejects_changed_engine_contract(tmp_path):
    receipt = evaluate(LLM(), make_task(tmp_path))
    receipt['identity']['sampling_engine']['engine'] = 'V1'
    rehash_identity(receipt)
    with pytest.raises(ValueError, match='V0 child-seed contract'):
        evaluator.validate_seed_receipt(receipt)


def test_seed_validator_rejects_boolean_label_in_schedule_even_when_python_equality_matches(tmp_path):
    task = dict(make_task(tmp_path), seeds=[0])
    receipt = evaluate(LLM(), task)
    receipt['identity']['seed_schedule']['draw_labels'][0] = False
    rehash_identity(receipt)
    with pytest.raises(ValueError, match='RNG schedule mismatch'):
        evaluator.validate_seed_receipt(receipt)
