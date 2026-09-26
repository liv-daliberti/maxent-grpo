"""E124 pairing, fixed benchmark interfaces, physical isolation and admission gates."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'ops/exp_scaling/e124_qwen7b_recipes.py'


@pytest.fixture(scope='module')
def recipe():
    spec = importlib.util.spec_from_file_location('e124_recipe_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def cells(recipe):
    return recipe.build_cells('/reviewed/e124/source', '/reviewed/qwen7b')


def test_complete_factorial_and_distinct_identities(recipe, cells):
    assert len(cells) == 30
    assert {(c['level'], c['domain'], c['arm'], c['seed']) for c in cells} == {
        (level, domain, arm, 70) for level in (1, 2, 3)
        for domain in recipe.DOMAINS for arm in ('maxrl', 'replay_maxrl')}
    for key in ('run_dir', 'run_stamp', 'scientific_sha256'):
        assert len({c[key] for c in cells}) == 30
    assert {c['target_steps'] for c in cells} == {3072}
    assert sum(c['target_steps'] for c in cells) == 92160
    assert len({c['constants_sha256'] for c in cells}) == 1
    assert all(c['proof']['physical_qualification_required'] for c in cells)


def test_exact_paired_controls(recipe, cells):
    for level in recipe.LEVELS:
        for domain in recipe.DOMAINS:
            pair = [c for c in cells if c['level'] == level and c['domain'] == domain]
            left, right = (c['environment'] for c in pair)
            changed = {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}
            assert changed == recipe.PAIR_DIFFERENCE_KEYS
            assert left['OAT_ZERO_MAXRL_TASK_OBJECTIVE'] == right['OAT_ZERO_MAXRL_TASK_OBJECTIVE'] == '1'
            assert left['OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY'] == '1'
            assert right['OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY'] == '0'
            for env in (left, right):
                assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY'] == '1'
                assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE'] == 'verified_likelihood_per_rollout'
                assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING'] == 'uniform'
                assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY'] == '16'
                assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA'] == '0.1'
                assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA'] == '0.1'
                assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP'] == '1'
                assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS'] == '0'


def test_common_e80_optimizer_and_exact_evaluation(cells):
    expected = {'OAT_ZERO_LEARNING_RATE': '1e-07', 'OAT_ZERO_LR_SCHEDULER': 'cosine_with_min_lr',
        'OAT_ZERO_LR_WARMUP_RATIO': '0.1', 'OAT_ZERO_MAX_STEP_ADJUSTMENT': '16.0',
        'OAT_ZERO_ADAM_BETA_1': '0.9', 'OAT_ZERO_ADAM_BETA_2': '0.999',
        'OAT_ZERO_NUM_SAMPLES': '16', 'OAT_ZERO_TRAIN_BATCH_SIZE': '16',
        'OAT_ZERO_ROLLOUT_BATCH_SIZE': '1', 'OAT_ZERO_NUM_PPO_EPOCHS': '1',
        'OAT_ZERO_NUM_PROMPT_EPOCH': '8', 'OAT_ZERO_MAX_PROMPT_EPOCHS': '8',
        'OAT_ZERO_MAX_TRAIN': '384', 'OAT_ZERO_BETA': '0.0', 'OAT_ZERO_MAX_NORM': '1.0',
        'OAT_ZERO_EVAL_PROMPT_INTERVAL': '192', 'OAT_ZERO_ALLOW_SPARSE_EVAL': '1',
        'OAT_ZERO_EVAL_BATCH_SIZE': '32', 'OAT_ZERO_EVAL_MODE_COVERAGE_K': '8',
        'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS': '4', 'OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE': '1.0',
        'OAT_ZERO_EVAL_MODE_COVERAGE_TOP_P': '1.0', 'OAT_ZERO_EVAL_TEMPERATURE': '0.0'}
    for cell in cells:
        assert {key: cell['environment'][key] for key in expected} == expected
        assert cell['scientific_identity']['model_revision'] == 'a09a35458c702b33eeacc393d103063234e8bc28'


def test_checkpoint_recovery_is_uniform_and_compact(cells):
    for cell in cells:
        env = cell['environment']
        assert {env[key] for key in ('OAT_ZERO_SAVE_STEPS', 'OAT_ZERO_SAVE_FROM',
                                     'OAT_ZERO_RESUME_STEPS', 'OAT_ZERO_RESUME_FROM')} == {'96'}
        assert {env[key] for key in ('OAT_ZERO_MAX_SAVE_NUM', 'OAT_ZERO_MAX_RESUME_NUM',
                                     'OAT_ZERO_MAX_EXPORT_NUM', 'OAT_ZERO_PRUNE_RESUME_ON_SUCCESS',
                                     'OAT_ZERO_AUTO_RESUME')} == {'1'}
        assert env['OAT_ZERO_EXPORT_STEPS'] == '0'


def test_native_level1_pantry_is_preserved(cells):
    for cell in cells:
        if cell['level'] != 1 or cell['domain'] != 'pantry_plan':
            continue
        env = cell['environment']
        assert env['OAT_ZERO_PROMPT_TEMPLATE'] == 'qwen_pantry_support_mask'
        assert env['OAT_ZERO_CANONICAL_ACTION_TASK'] == 'pantry_support_mask'
        assert env['OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT'] == '6'
        assert env['OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING'] == '1'
        assert env['OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING'] == '1'
        assert (env['OAT_ZERO_PROMPT_MAX_LENGTH'], env['OAT_ZERO_GENERATE_MAX_LENGTH'],
                env['OAT_ZERO_MAX_MODEL_LEN']) == ('640', '8', '704')
        # Physical microbatch is explicitly reduced, preserving logical batch 16.
        assert env['OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE'] == '1'
        assert env['OAT_ZERO_TRAIN_BATCH_SIZE'] == '16'
        assert env['OAT_ZERO_EVAL_MODE_COVERAGE_SEED'] == '76299'


def test_native_level1_matrix_matches_archived_e118(recipe):
    proof = recipe.native_level1_proof()
    assert set(proof['semantic_templates']) == set(recipe.DOMAINS)
    mathir = proof['semantic_templates']['mathir']['interface']
    assert (mathir['OAT_ZERO_PROMPT_MAX_LENGTH'], mathir['OAT_ZERO_GENERATE_MAX_LENGTH'],
            mathir['OAT_ZERO_MAX_MODEL_LEN']) == ('256', '64', '384')
    assert mathir['OAT_ZERO_PROMPT_TEMPLATE'] == 'qwen_boxed'


def test_harder_levels_use_ordinary_generation_and_no_dev(cells):
    for cell in cells:
        if cell['level'] == 1:
            continue
        env = cell['environment']
        assert env['OAT_ZERO_CANONICAL_ACTION_TASK'] == 'none'
        assert env['OAT_ZERO_CANONICAL_GRAPH_ACTIONS'] == '0'
        assert env['OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT'] == '3'
        assert env['OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING'] == '0'
        assert env['OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING'] == '0'
        assert (env['OAT_ZERO_PROMPT_MAX_LENGTH'], env['OAT_ZERO_GENERATE_MAX_LENGTH'],
                env['OAT_ZERO_MAX_MODEL_LEN']) == ('1024', '192', '2048')
        assert Path(env['OAT_ZERO_PROMPT_DATA']).name == 'train'
        assert Path(env['OAT_ZERO_EVAL_DATA']).name == 'eval'
        assert '/dev' not in env['OAT_ZERO_PROMPT_DATA'] + env['OAT_ZERO_EVAL_DATA']
        if cell['domain'] == 'pantry_plan':
            assert env['OAT_ZERO_PROMPT_TEMPLATE'] == 'qwen_level2_pantry'
            assert env['OAT_ZERO_MODEBENCH_SYNTAX_PROFILE'] == 'domain_legal_v1'
            assert Path(env['OAT_ZERO_DATA_ROOT']).name == 'pantry'


def test_physical_profiles_cannot_change_science(recipe, cells):
    alternate = recipe.build_cells('/different/source', '/different/model', gpu_class='a6000',
                                   output_root='/different/output')
    for original, other in zip(cells, alternate):
        assert original['scientific_sha256'] == other['scientific_sha256']
        assert original['scientific_identity'] == other['scientific_identity']
        changed = {key for key in original['environment'] if original['environment'][key] != other['environment'][key]}
        assert changed <= recipe.PHYSICAL_ENV_KEYS | recipe.LOCATION_ENV_KEYS
        assert original['environment']['OAT_ZERO_VLLM_GPU_RATIO'] == '0.25'
        assert other['environment']['OAT_ZERO_VLLM_GPU_RATIO'] == '0.40'
        assert original['resources']['memory_gib'] == other['resources']['memory_gib'] == 256
        assert original['resources']['gpus'] == other['resources']['gpus'] == 1
        assert not original['runtime_profile']['qualified']


def test_no_shared_mutable_cell_state(recipe):
    cells = recipe.build_cells('/source', '/model')
    cells[0]['runtime_profile']['resources']['memory_gib'] = 1
    cells[0]['environment']['OAT_ZERO_LEARNING_RATE'] = '9'
    assert cells[1]['runtime_profile']['resources']['memory_gib'] == 256
    assert cells[1]['environment']['OAT_ZERO_LEARNING_RATE'] == '1e-07'
    assert recipe.build_cells('/source', '/model')[0]['environment']['OAT_ZERO_LEARNING_RATE'] == '1e-07'


def test_preview_makes_no_directories(recipe, tmp_path):
    target = tmp_path / 'uncreated'
    cells = recipe.build_cells(tmp_path / 'source', tmp_path / 'model', output_root=target)
    assert not target.exists()
    assert cells[0]['run_dir'] == str(target / 'level1/graph_coloring/maxrl/s70')


@pytest.mark.parametrize('gpu', ['a5000', 'h100', 'a100-40gb', '', None])
def test_unqualified_gpu_class_rejected(recipe, gpu):
    with pytest.raises(ValueError, match='only A100'):
        recipe.build_cells('/source', '/model', gpu_class=gpu)


def test_export_delimiter_rejected(recipe):
    with pytest.raises(ValueError, match='delimiter'):
        recipe.build_cells('/source', '/model,malformed')


def test_physical_helper_cannot_silently_override_evaluation(recipe, monkeypatch):
    original = recipe.e80.memory_env
    monkeypatch.setattr(recipe.e80, 'memory_env', lambda: original() | {'OAT_ZERO_NUM_SAMPLES': '8'})
    with pytest.raises(ValueError, match='scientific settings'):
        recipe.build_cells('/source', '/model')


@pytest.fixture
def synthetic_splits(recipe, monkeypatch):
    rows = {split: [{'identity': f'{split}-{i}', 'answer_mode_count': 4} for i in range(count)]
            for split, count in [('train', 384), ('dev', 128), ('eval', 128)]}
    records = {split: {'rows': len(items), 'rows_sha256': recipe._row_hash(items),
                       'checks': {'row_count': True, 'historical_disjointness': True},
                       'answer_mode_count_histogram': {'4': len(items)}} for split, items in rows.items()}
    monkeypatch.setattr(recipe, '_rows', lambda path, subset: deepcopy(rows[Path(path).name]))
    monkeypatch.setattr(recipe, '_identities', lambda domain, data: {r['identity'] for r in data})
    monkeypatch.setattr(recipe, '_tree', lambda path: {'data.arrow': 'a' * 64})
    return rows, records


def test_row_hash_and_canonical_checks_pass(recipe, synthetic_splits):
    _, records = synthetic_splits
    proof = recipe._validate_splits(2, 'countdown', records)
    assert {split: v['rows'] for split, v in proof.items()} == {'train': 384, 'dev': 128, 'eval': 128}


def test_changed_published_rows_rejected(recipe, synthetic_splits):
    rows, records = synthetic_splits
    rows['train'][0]['answer_mode_count'] = 5
    with pytest.raises(ValueError, match='row hash drift'):
        recipe._validate_splits(2, 'countdown', records)


def test_false_historical_check_rejected(recipe, synthetic_splits):
    _, records = synthetic_splits
    records['dev']['checks']['historical_disjointness'] = False
    with pytest.raises(ValueError, match='structural checks failed'):
        recipe._validate_splits(2, 'countdown', records)


def test_cross_split_identity_leak_rejected(recipe, synthetic_splits):
    rows, records = synthetic_splits
    rows['eval'][0]['identity'] = rows['train'][0]['identity']
    records['eval']['rows_sha256'] = recipe._row_hash(rows['eval'])
    with pytest.raises(ValueError, match='splits overlap'):
        recipe._validate_splits(2, 'countdown', records)


def test_support_histogram_drift_rejected(recipe, synthetic_splits):
    _, records = synthetic_splits
    records['eval']['answer_mode_count_histogram'] = {'4': 127, '5': 1}
    with pytest.raises(ValueError, match='support histogram drift'):
        recipe._validate_splits(2, 'countdown', records)


def test_pinned_l1_inventory_drift_rejected(recipe, monkeypatch):
    monkeypatch.setattr(recipe, 'native_level1_proof', lambda: {})
    monkeypatch.setattr(recipe, '_tree', lambda path: {'unexpected.arrow': 'b' * 64})
    with pytest.raises(ValueError, match='Level 1 graph_coloring dataset inventory drift'):
        recipe.dataset_admission()


def test_missing_fifth_domain_admission_rejected(recipe, monkeypatch):
    fake_native = {'native.arrow': 'a' * 64}
    tree_hash = recipe.sha({f'{split}/{key}': value for split in ('train', 'eval') for key, value in fake_native.items()})
    monkeypatch.setattr(recipe, 'native_level1_proof', lambda: {})
    monkeypatch.setattr(recipe, '_tree', lambda path: fake_native)
    monkeypatch.setattr(recipe, '_validate_splits', lambda *args: {})
    monkeypatch.setattr(recipe, 'L1_INVENTORY_SHA256', {d: tree_hash for d in recipe.DOMAINS})
    monkeypatch.setattr(recipe, 'digest', lambda path: recipe.L2_IDENTITY_SHA256 if Path(path).name == 'identity.json'
                        else recipe.L2_REPORTS[Path(path)])
    def read(path):
        if Path(path).name == 'identity.json':
            return {'split_sizes': {'train': 384, 'dev': 128, 'eval': 128},
                    'domains': {d: {} for d in recipe.DOMAIN_DIR.values()}}
        return {'status': 'pass', 'decision': 'admit_all_domains_for_treatment_training',
                'domain_decisions': {d: 'admit' for d in list(recipe.DOMAIN_DIR.values())[:-1]}}
    monkeypatch.setattr(recipe, 'read', read)
    with pytest.raises(ValueError, match='both all-five admission reports'):
        recipe.dataset_admission()


def test_no_false_7b_matching_claim(recipe):
    boundary = recipe.scientific_constants()['information_boundary']
    assert boundary == {'treatment_uses_dev': False, 'rematch_for_7b': False,
                        'level3_adaptive_confirmation_round': 2, 'statistical_equivalence_claimed': False}


def test_flat_inventory_preserves_canonical_pins_and_uses_absolute_paths(recipe, monkeypatch, tmp_path):
    native = {'data.arrow': 'a' * 64}
    native_hash = recipe.sha({f'{split}/{key}': value for split in ('train', 'eval') for key, value in native.items()})
    monkeypatch.setattr(recipe, 'native_level1_proof', lambda: {})
    monkeypatch.setattr(recipe, '_tree', lambda path: native)
    monkeypatch.setattr(recipe, 'L1_INVENTORY_SHA256', {d: native_hash for d in recipe.DOMAINS})
    def split_proof(level, domain, records=None):
        return {split: {'path': str(tmp_path / f'l{level}' / domain / split),
                        'rows': 384 if split == 'train' else 128, 'files_sha256': native}
                for split in ('train', 'eval') if level == 1} if level == 1 else {
                    split: {'path': str(tmp_path / f'l{level}' / domain / split),
                            'rows': 384 if split == 'train' else 128, 'files_sha256': native}
                    for split in ('train', 'dev', 'eval')}
    monkeypatch.setattr(recipe, '_validate_splits', split_proof)
    monkeypatch.setattr(recipe, 'digest', lambda path: (
        recipe.L3_IDENTITY_SHA256 if Path(path).parent == recipe.DATA_ROOTS[3] else recipe.L2_IDENTITY_SHA256)
        if Path(path).name == 'identity.json' else recipe.L2_REPORTS[Path(path)])
    def read(path):
        if Path(path).name == 'identity.json':
            return {'split_sizes': {'train': 384, 'dev': 128, 'eval': 128},
                    'domains': {d: {} for d in recipe.DOMAIN_DIR.values()}}
        return {'status': 'pass', 'decision': 'admit_all_domains_for_treatment_training',
                'domain_decisions': {d: 'admit' for d in recipe.DOMAIN_DIR.values()}}
    monkeypatch.setattr(recipe, 'read', read)
    canonical_path = str(tmp_path / 'canonical/report.json')
    canonical = {'status': 'matched_fixed_reference', 'sha256': recipe.L3_REPORT_SHA256,
                 'files_sha256': {canonical_path: recipe.L3_REPORT_SHA256},
                 'directory_files': {str(tmp_path / 'canonical'): [canonical_path]}}
    monkeypatch.setitem(sys.modules, 'launch_e122_level3_factorial',
                        SimpleNamespace(admission_proof=lambda required: deepcopy(canonical)))
    proof = recipe.dataset_admission()
    assert proof['files_sha256'][canonical_path] == recipe.L3_REPORT_SHA256
    assert proof['directory_files'][str(tmp_path / 'canonical')] == [canonical_path]
    for directory, files in proof['directory_files'].items():
        assert all(Path(path).is_absolute() and path in proof['files_sha256'] for path in files)
        assert files == sorted(files)
        assert all(Path(path).is_relative_to(directory) for path in files)
    assert proof['sha256'] == recipe.sha({key: value for key, value in proof.items() if key != 'sha256'})
    assert not proof['information_boundary']['new_7b_viability_or_matching_claimed']
