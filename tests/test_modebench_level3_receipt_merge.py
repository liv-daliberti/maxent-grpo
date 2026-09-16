"""Seed receipt merges retain prompt units and reject mismatched or corrupt evidence."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'ops/merge_modebench_level3_receipts.py'
SPEC = spec_from_file_location('modebench_level3_receipt_merge_test', PATH)
merge = module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


def make_receipt(tmp_path, name, seeds, *, correct=4, split='dev', count=3):
    rows = [{'problem': f'problem {index}', 'answer': json.dumps({'target': index}),
             'answer_mode_count': 4} for index in range(count)]
    interface = merge.frozen_interface('countdown')
    prompt_results = []
    for index, row in enumerate(rows):
        metrics = {'pass1': correct / 8, 'pass8': float(correct > 0), 'distinct8': float(correct > 0)}
        attempts = [{'text': str(sample), 'verified': sample < correct,
                     'canonical_key': 'valid' if sample < correct else None, 'token_count': 4}
                    for sample in range(8)]
        prompt_results.append({'row_index': index, 'row_sha256': merge.sha(row),
                               'spec_sha256': merge.sha(json.loads(row['answer'])),
                               'problem_sha256': merge.sha(row['problem']),
                               'row_metadata': {'answer_mode_count': 4},
                               'draws': [{'seed': seed, 'attempts': deepcopy(attempts),
                                          'verified_count': correct, **metrics} for seed in seeds], **metrics})
    identity = {'schema': merge.SCHEMA, 'domain': 'countdown', 'split': split, 'level': 'level3',
                'interface': interface, 'interface_sha256': merge.sha(interface), 'seeds': seeds,
                'model': {'label': '3b', 'path': '/checkpoint', 'vllm_version': 'test'},
                'batch_size': 8, 'code_sha256': {'verifier': 'frozen'},
                'source': {'kind': 'jsonl', 'path': str(tmp_path / 'shared.jsonl'),
                           'rows_sha256': merge.sha(rows), 'all_rows_sha256': merge.sha(rows),
                           'selected_rows': count, 'total_rows': count, 'row_offset': 0, 'row_limit': 0}}
    result = {'schema': merge.SCHEMA, 'status': 'complete', 'domain': 'countdown', 'level': 'level3',
              'model_label': '3b', 'split': split, 'identity': identity, 'identity_sha256': merge.sha(identity),
              'sampling': {**interface, 'seeds': seeds}, 'prompt_results': prompt_results,
              'metrics': merge.summarize(prompt_results), 'answer_mode_histogram': {'4': count},
              'information_boundary': {'evaluation_prompts_loaded': split == 'eval',
                                       'confirmation_explicitly_authorized': split == 'eval',
                                       'treatment_training_started': False}}
    path = tmp_path / (name + '.json')
    path.write_text(json.dumps(result))
    return path


def change(path, transform):
    value = json.loads(path.read_text())
    transform(value)
    value['identity_sha256'] = merge.sha(value['identity'])
    path.write_text(json.dumps(value))


def test_unequal_seed_group_sizes_weight_draws_not_receipts(tmp_path):
    first = make_receipt(tmp_path, 'first', [30], correct=0)
    second = make_receipt(tmp_path, 'second', [20, 10], correct=8)
    output = tmp_path / 'merged.json'
    result = merge.merge_receipts([first, second], output)
    assert result['sampling']['seeds'] == [10, 20, 30]
    assert result['metrics']['pass1'] == pytest.approx(2 / 3)
    assert result['metrics']['pass8'] == pytest.approx(2 / 3)
    assert result['metrics']['rows'] == 3
    assert all([draw['seed'] for draw in row['draws']] == [10, 20, 30] for row in result['prompt_results'])
    assert result['aggregation']['input_receipts'][0]['sha256'] == merge.file_sha(first)
    assert result['identity_sha256'] == merge.sha(result['identity'])
    assert json.loads(output.read_text()) == result


def test_nested_merges_retain_provenance_and_input_bytes(tmp_path):
    paths = [make_receipt(tmp_path, str(seed), [seed], correct=seed) for seed in (1, 2, 3)]
    originals = [path.read_bytes() for path in paths]
    prior = tmp_path / 'prior.json'
    merge.merge_receipts(paths[:2], prior)
    result = merge.merge_receipts([prior, paths[2]])
    assert result['metrics']['pass1'] == .25
    assert result['sampling']['seeds'] == [1, 2, 3]
    assert result['aggregation']['input_receipts'][0]['aggregation']['input_receipts'][0]['path'] == str(paths[0])
    assert [path.read_bytes() for path in paths] == originals


def test_overlap_cannot_double_count_evidence(tmp_path):
    paths = [make_receipt(tmp_path, 'first', [10, 20]), make_receipt(tmp_path, 'second', [20, 30])]
    with pytest.raises(ValueError, match='overlapping sampling seeds'):
        merge.merge_receipts(paths)


@pytest.mark.parametrize('field,new', [('batch_size', 16), ('code_sha256', {'verifier': 'different'}),
                                      ('model', {'label': '3b', 'path': '/different', 'vllm_version': 'test'})])
def test_settings_model_or_code_changes_are_rejected(tmp_path, field, new):
    paths = [make_receipt(tmp_path, 'first', [1]), make_receipt(tmp_path, 'second', [2])]
    change(paths[1], lambda value: value['identity'].__setitem__(field, new))
    with pytest.raises(ValueError, match='identities differ'):
        merge.merge_receipts(paths)


def test_row_spec_changes_are_rejected_even_with_matching_source_claim(tmp_path):
    paths = [make_receipt(tmp_path, 'first', [1]), make_receipt(tmp_path, 'second', [2])]
    change(paths[1], lambda value: value['prompt_results'][0].__setitem__('spec_sha256', '0' * 64))
    with pytest.raises(ValueError, match='rows/specs/metadata differ'):
        merge.merge_receipts(paths)


def test_tampered_draw_and_summary_metrics_are_rejected(tmp_path):
    paths = [make_receipt(tmp_path, 'first', [1]), make_receipt(tmp_path, 'second', [2])]
    change(paths[1], lambda value: value['prompt_results'][0]['draws'][0].__setitem__('pass1', .99))
    with pytest.raises(ValueError, match='metric mismatch'):
        merge.merge_receipts(paths)
    paths[1] = make_receipt(tmp_path, 'second_repaired', [2])
    change(paths[1], lambda value: value['metrics'].__setitem__('pass8', .99))
    with pytest.raises(ValueError, match='metric mismatch'):
        merge.merge_receipts(paths)


def test_sample_budget_and_histogram_corruption_are_rejected(tmp_path):
    paths = [make_receipt(tmp_path, 'first', [1]), make_receipt(tmp_path, 'second', [2])]
    change(paths[1], lambda value: value['prompt_results'][0]['draws'][0]['attempts'][0].__setitem__('token_count', 193))
    with pytest.raises(ValueError, match='token budget violation'):
        merge.merge_receipts(paths)
    paths[1] = make_receipt(tmp_path, 'second_repaired', [2])
    change(paths[1], lambda value: value.__setitem__('answer_mode_histogram', {'4': 500}))
    with pytest.raises(ValueError, match='support histogram'):
        merge.merge_receipts(paths)


def test_partial_confirmation_receipts_merge_into_valid_four_draw_audit_input(tmp_path):
    from audit_modebench_level3_match import validate_receipt
    paths = [make_receipt(tmp_path, str(seed), [seed], split='eval', count=128) for seed in (1, 2, 3, 4)]
    result = merge.merge_receipts(paths)
    assert len(validate_receipt(result, domain='countdown', role='candidate')) == 128
    assert result['sampling']['seeds'] == [1, 2, 3, 4]


def test_completed_merge_cannot_be_overwritten(tmp_path):
    paths = [make_receipt(tmp_path, 'first', [1]), make_receipt(tmp_path, 'second', [2])]
    output = tmp_path / 'merged.json'
    merge.merge_receipts(paths, output)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        merge.merge_receipts(paths, output)
    assert output.read_bytes() == before
