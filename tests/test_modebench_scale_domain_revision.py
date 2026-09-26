"""Isolated revision integration: synthetic graphs, mock inference, real evidence gates.

These tests never open campaign data or model checkpoints. The candidate provider
and parent registration are explicit fixtures; prompt rendering, graph support and
grading, four-draw receipts, mixture fitting, and Arrow round trips are real.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ops/exp_scaling'))
import modebench_scale_domain_revision as revision
import modebench_scale_candidates as candidates
from oat_drgrpo.math_grader import validated_modebench_outcome_key


DOMAIN = 'graph_coloring'
LEVEL = 'level4'
VALID = r'\boxed{1212121222}'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


class MockTokenizer:
    """Deterministic fixture tokenizer; this is not a checkpoint token-budget test."""
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        return json.dumps(messages)

    def encode(self, prompt, **kwargs):
        return [1] * 64


class MockLLM:
    def __init__(self, successes=2, runtime=None, success_text=VALID):
        self.successes = successes
        self.success_text = success_text
        self._modebench_runtime = runtime or revision.evaluator.runtime_settings(
            max_model_len=1024, tensor_parallel_size=2, swap_space=4)

    def generate(self, prompts, params, use_tqdm=False):
        assert all(param.n == 8 for param in params)
        texts = [self.success_text] * self.successes + ['wrong'] * (8 - self.successes)
        return [SimpleNamespace(prompt=prompt, outputs=[
            SimpleNamespace(text=text, token_ids=[1, 2], finish_reason='stop')
            for text in texts]) for prompt in prompts]


class MockGraphProvider:
    """Many distinct exact-support-four graphs with two independently hidden colors."""
    PROFILES = {DOMAIN: [{'mock_law': tier} for tier in range(4)]}

    def __init__(self, source):
        self.source = source
        self.calls = []

    def source_paths(self):
        return [self.source]

    def build_pool(self, domain, target, excluded, seed, tag, tier, multiplier=1):
        assert domain == DOMAIN and set(target) == {4} and multiplier == 1
        self.calls.append((tag, tier, dict(target)))
        partial = [1, 2] * 4 + [None, None]
        optional = [(u, v) for u in range(1, 9) for v in range(u + 1, 9)
                    if partial[u - 1] != partial[v - 1]]
        blocked, rows = set(excluded), []
        for step in range(1 << len(optional)):
            mask = (seed + step) % (1 << len(optional))
            edges = sorted([[1, 9], [1, 10]] + [list(edge) for bit, edge in enumerate(optional)
                                               if mask & (1 << bit)])
            spec = {'verifier': DOMAIN, 'n': 10, 'edges': edges, 'partial_colors': partial,
                    'num_completions': 4, 'source': 'EXPLICIT_MOCK_FIXTURE',
                    'instance_id': f'{tag}-{tier}-{mask}'}
            row = {'problem': candidates._graph_prompt(10, edges, partial),
                   'answer': json.dumps(spec, sort_keys=True), 'answer_mode_count': 4,
                   'modebench_task': DOMAIN, 'answer_mode_split': tag,
                   'scale_candidate_tier': tier, 'scale_candidate_generator': 'EXPLICIT_MOCK_FIXTURE'}
            key = candidates.identity(domain, row)
            if key in blocked:
                continue
            blocked.add(key)
            rows.append(row)
            if len(rows) == target[4]:
                return rows
        raise AssertionError('fixture graph support exhausted')

    def verify_rows(self, domain, rows):
        # This independently counts all completions, checks the original prompt,
        # and executes the original graph verifier for every completion.
        return candidates.verify_rows(domain, rows)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source = tmp_path / 'mock_candidate_source.py'
    source.write_text('# explicit synthetic provider pin, never a production law\n')
    parent = tmp_path / 'parent'
    parent_recipe = parent / LEVEL / 'recipes' / (DOMAIN + '.json')
    parent_evidence = tmp_path / 'mock_parent_development.json'
    write_json(parent_evidence, {'fixture_only': True})
    failed_recipe = {'development_fit_pass': False,
                     'input_sha256': {str(parent_evidence): revision.file_sha(parent_evidence)}}
    write_json(parent_recipe, failed_recipe)
    histograms = {split: revision.original.serialize_cells(Counter({(4,): count}))
                  for split, count in [('train', 384), ('dev', 128), ('eval', 128)]}
    protocol = {'files_sha256': {},
                'models': {'7b': {'label': '7b', 'path': str(tmp_path / 'MOCK_NO_CHECKPOINT'),
                                   'fixture_only': True}},
                'targets': {DOMAIN: {'metrics': {'pass1': .25, 'pass8': 1.0}}},
                'histograms': {DOMAIN: histograms},
                'split_sizes': {'train': 384, 'dev': 128, 'eval': 128},
                'tolerances': revision.original.TOLERANCES,
                'sampling': {'fixture_only': True}, 'fit': {'fixture_only': True}}
    write_json(parent / 'protocol.json', protocol)
    provider = MockGraphProvider(source)
    monkeypatch.setattr(revision, 'candidate_provider', lambda name: provider)

    def parent_authenticate(path):
        assert Path(path) == parent / 'protocol.json', 'must never authenticate a real campaign'
        return copy.deepcopy(protocol)

    def parent_fit(path, level, domain, publish=False):
        assert Path(path) == parent and (level, domain, publish) == (LEVEL, DOMAIN, False)
        return copy.deepcopy(failed_recipe)

    monkeypatch.setattr(revision.original, 'authenticate', parent_authenticate)
    monkeypatch.setattr(revision.common_fit, 'fit_domain', parent_fit)

    def empty_history(domain, root):
        assert domain == DOMAIN and tmp_path in Path(root).parents
        return set(), set(), {}

    monkeypatch.setattr(revision.original, 'history', empty_history)
    # Cross-source scanner has its own tests. Keep this integration strictly in
    # tmp_path while asserting that both public held-out entry points invoke it.
    import modebench_scale_source_disjointness as disjointness
    cross_source_calls = []

    def mock_verify_dataset(root, level, domain):
        assert tmp_path in Path(root).parents and (level, domain) == (LEVEL, DOMAIN)
        cross_source_calls.append((Path(root), level, domain))
        return {'fixture_only': True, 'files_sha256': {str(source): revision.file_sha(source)}}

    monkeypatch.setattr(disjointness, 'verify_dataset', mock_verify_dataset)
    return SimpleNamespace(root=tmp_path / 'revision', parent=parent, parent_recipe=parent_recipe,
                           protocol=protocol, failed_recipe=failed_recipe, provider=provider,
                           source=source, cross_source_calls=cross_source_calls)


def register(campaign):
    return revision.register(campaign.root, LEVEL, DOMAIN, parent=campaign.parent,
                             candidate_module='modebench_scale_mock_fixture')


def evaluate(campaign, task, successes=2, *, model=None, runtime=None, grader=None, success_text=VALID):
    llm = MockLLM(successes, runtime, success_text)
    return revision.evaluator.evaluate_task(
        llm, MockTokenizer(), task,
        model=model or {**campaign.protocol['models']['7b'], 'vllm_version': '0.8.4'},
        code=revision.evaluator.code_identity(), runtime=llm._modebench_runtime,
        confirm_eval=task['split'] == 'eval', grader=grader,
        params_factory=lambda args, *_: SimpleNamespace(seed=args.seed, n=8))


def develop(campaign, successes=(1, 2, 3, 4)):
    register(campaign)
    revision.materialize_pools(campaign.root)
    cell, _ = revision.launch_inputs(campaign.root, 'dev')
    for task, count in zip(cell['tasks'], successes):
        evaluate(campaign, task, count)
    return revision.fit_domain(campaign.root)


@pytest.mark.parametrize('failure', ['passing', 'irreproducible', 'observed_holdout'])
def test_registration_requires_reproducible_failed_parent_without_observed_holdout(campaign, failure):
    if failure == 'passing':
        campaign.failed_recipe['development_fit_pass'] = True
        write_json(campaign.parent_recipe, campaign.failed_recipe)
        match = 'failed development fit'
    elif failure == 'irreproducible':
        write_json(campaign.parent_recipe, {**campaign.failed_recipe, 'changed': True})
        match = 'not reproducible'
    else:
        write_json(campaign.parent / LEVEL / 'results/confirmation' / (DOMAIN + '.json'),
                   {'EXPLICIT_MOCK_HOLDOUT_MARKER': True})
        match = 'observed parent holdout'
    with pytest.raises(ValueError, match=match):
        register(campaign)
    assert not campaign.root.exists()


@pytest.mark.parametrize('field', ['models', 'targets', 'histograms', 'draw_labels', 'candidate_profiles'])
def test_protocol_seal_rejects_changes_before_any_pool_exists(campaign, field):
    protocol = register(campaign)
    protocol[field] = {'tampered': True}
    write_json(campaign.root / 'protocol.json', protocol)
    with pytest.raises(ValueError, match='revision protocol changed'):
        revision.materialize_pools(campaign.root)
    assert not (campaign.root / LEVEL).exists()


def test_registration_seals_source_and_is_one_time(campaign):
    protocol = register(campaign)
    assert protocol['candidate_module'] == 'modebench_scale_mock_fixture'
    assert protocol['models'] == campaign.protocol['models']
    assert protocol['draw_labels']['dev'] != protocol['draw_labels']['eval']
    assert not set(protocol['draw_labels']['dev']) & set(revision.original.labels(LEVEL, 'dev'))
    with pytest.raises(ValueError, match='fresh revision root'):
        register(campaign)
    campaign.source.write_text('# altered mock source\n')
    with pytest.raises(ValueError, match='registered revision input changed'):
        revision.materialize_pools(campaign.root)


def test_complete_pool_certificate_disjointness_and_launch_binding(campaign):
    protocol = register(campaign)
    identity = revision.materialize_pools(campaign.root)
    cell, pins = revision.launch_inputs(campaign.root, 'dev')
    assert len(cell['tasks']) == 4 and cell['source_kind'] == 'domain_revision_v1'
    seen, prompts = set(), set()
    for tier, task in enumerate(cell['tasks']):
        rows = revision.rows_from_jsonl(task['rows_jsonl'])
        ids = revision.identity_set(DOMAIN, rows)
        texts = {revision.sha(row['problem']) for row in rows}
        assert not ids & seen and not texts & prompts
        seen |= ids
        prompts |= texts
        assert task['seeds'] == protocol['draw_labels']['dev']
        assert task['batch_size'] == 8 and task['row_offset'] == task['row_limit'] == 0
        assert identity['tiers'][str(tier)]['verification']['witnesses_verified'] == 128 * 4
        assert pins[task['rows_jsonl']] == revision.file_sha(task['rows_jsonl'])
    assert len(seen) == 512
    with pytest.raises(ValueError, match='never overwrite'):
        revision.materialize_pools(campaign.root)


@pytest.mark.parametrize('tamper', ['row', 'model', 'seed', 'source_path'])
def test_authenticated_receipts_still_require_registered_pool_model_and_draws(campaign, tamper):
    register(campaign)
    revision.materialize_pools(campaign.root)
    task = revision.launch_inputs(campaign.root, 'dev')[0]['tasks'][0]
    if tamper == 'row':
        path = Path(task['rows_jsonl'])
        rows = revision.rows_from_jsonl(path)
        rows[0]['problem'] += ' tampered'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        match = 'changed revision pool'
    else:
        kwargs = {}
        if tamper == 'model':
            kwargs['model'] = {**campaign.protocol['models']['7b'], 'path': 'WRONG_MOCK_CHECKPOINT',
                               'vllm_version': '0.8.4'}
            match = 'wrong revision checkpoint'
        elif tamper == 'seed':
            task = {**task, 'seeds': [label + 10000 for label in task['seeds']]}
            match = 'wrong revision draw labels'
        else:
            duplicate = campaign.root / 'wrong_source.jsonl'
            duplicate.write_bytes(Path(task['rows_jsonl']).read_bytes())
            task = {**task, 'rows_jsonl': str(duplicate)}
            match = 'wrong revision source path'
        receipt = evaluate(campaign, task, **kwargs)
        # These are internally authenticated mock receipts, so the revision's
        # registration binding must reject them beyond the generic validator.
        revision.evaluator.validate_seed_receipt(receipt)
    with pytest.raises(ValueError, match=match):
        revision.fit_domain(campaign.root)
    assert not revision.paths(campaign.root)['recipe'].exists()


def test_fit_requires_all_four_tiers(campaign):
    register(campaign)
    revision.materialize_pools(campaign.root)
    cell, _ = revision.launch_inputs(campaign.root, 'dev')
    evaluate(campaign, cell['tasks'][0])
    with pytest.raises(FileNotFoundError):
        revision.fit_domain(campaign.root)
    assert not revision.paths(campaign.root)['recipe'].exists()


def test_full_grid_fit_freeze_exact_selected_dev_and_original_grader_confirmation(campaign, monkeypatch):
    recipe = develop(campaign)
    assert recipe['development_fit_pass']
    assert recipe['weight_combinations_considered'] == 1771
    assert recipe['selected_development_sets_scored'] == 1
    assert recipe['selected_evaluation_sets_scored'] == 0
    assert len(campaign.provider.calls) == 4  # Only development pools existed during fitting.
    dataset_path = Path(revision.freeze_dataset(campaign.root))
    identity = revision.read(dataset_path / 'identity.json')
    assert identity['status'] == 'frozen_pending_heldout_confirmation'
    assert identity['difficulty_matched'] is False
    sets = {}
    for split, count in [('train', 384), ('dev', 128), ('eval', 128)]:
        rows = revision.rows_from_jsonl(dataset_path / (split + '.jsonl'))
        assert len(rows) == count
        assert revision.cell_histogram(DOMAIN, rows) == Counter({(4,): count})
        assert revision.load_rows(dataset_path / split, revision.SPLITS[split][1]) == rows
        assert identity['splits'][split]['rows_sha256'] == revision.sha(rows)
        sets[split] = revision.identity_set(DOMAIN, rows)
        if split == 'dev':
            assert [revision.sha(row) for row in rows] == recipe['selected_dev_row_sha256']
    assert not sets['train'] & sets['dev'] and not sets['train'] & sets['eval'] and not sets['dev'] & sets['eval']
    pool_ids = set().union(*(revision.identity_set(DOMAIN, revision.rows_from_jsonl(
        revision.paths(campaign.root)['pools'] / f'difficulty_{tier}.jsonl')) for tier in range(4)))
    assert sets['dev'] <= pool_ids
    assert not (sets['train'] | sets['eval']) & pool_ids
    cell, pins = revision.launch_inputs(campaign.root, 'eval')
    assert len(cell['tasks']) == 1 and campaign.cross_source_calls
    assert str(campaign.source) in pins
    evaluate(campaign, cell['tasks'][0])
    import oat_drgrpo.math_grader as grader_module
    replayed = []

    def real_counted_grader(text, answer):
        replayed.append((text, answer))
        return validated_modebench_outcome_key(text, answer)

    monkeypatch.setattr(grader_module, 'validated_modebench_outcome_key', real_counted_grader)
    audit = revision.confirm_domain(campaign.root)
    assert audit['difficulty_matched'] and all(audit['gates'].values())
    assert audit['original_grader_replayed_attempts'] == len(replayed) == 4096
    assert len(campaign.cross_source_calls) == 2
    assert revision.read(revision.paths(campaign.root)['audit']) == audit


def test_failed_fit_cannot_freeze_or_launch_confirmation(campaign):
    recipe = develop(campaign, successes=(0, 0, 0, 0))
    assert not recipe['development_fit_pass']
    assert recipe['selected_evaluation_sets_scored'] == 0
    with pytest.raises(ValueError, match='passing reproducible'):
        revision.freeze_dataset(campaign.root)
    with pytest.raises(ValueError, match='passing revision fit'):
        revision.launch_inputs(campaign.root, 'eval')
    assert not revision.paths(campaign.root)['dataset'].exists()


@pytest.mark.parametrize('mismatch', ['forged_grading', 'runtime'])
def test_heldout_replay_rejects_self_consistent_forgery_or_changed_runtime(campaign, mismatch):
    develop(campaign)
    revision.freeze_dataset(campaign.root)
    task = revision.launch_inputs(campaign.root, 'eval')[0]['tasks'][0]
    kwargs = {}
    if mismatch == 'forged_grading':
        kwargs = {'success_text': 'forged success',
                  'grader': lambda text, answer: validated_modebench_outcome_key(VALID, answer)
                  if text == 'forged success' else None}
        match = 'original grader disagrees'
    else:
        kwargs['runtime'] = revision.evaluator.runtime_settings(
            max_model_len=1024, tensor_parallel_size=1, swap_space=4)
        match = 'runtime differs'
    receipt = evaluate(campaign, task, **kwargs)
    revision.evaluator.validate_seed_receipt(receipt)
    assert receipt['metrics']['pass1'] == .25 and receipt['metrics']['pass8'] == 1
    with pytest.raises(ValueError, match=match):
        revision.confirm_domain(campaign.root)
    assert not revision.paths(campaign.root)['audit'].exists()
