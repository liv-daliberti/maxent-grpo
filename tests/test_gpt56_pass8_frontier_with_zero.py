"""Additional complete-cohort and paired-inference checks for the zero extension."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from test_gpt56_pass8_frontier import source, block, prompt_records

spec = importlib.util.spec_from_file_location('gpt_pass8_zero', Path(__file__).resolve().parents[1] /
                                             'ops/analyze_gpt56_pass8_frontier_with_zero.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def extended_source(source):
    source['schema'] = 'gpt56-none-temperature-curve-v2'
    source.update(temperatures=[0., .5, 1., 1.5, 2.], total_registered_responses=4800,
                  responses_per_condition=960, prompts_per_condition=120, draws_per_prompt=8)
    source['validation'].pop('all_3840_native_receipts_authenticated')
    source['validation'].update(all_native_receipts_authenticated=True,
        zero_extension_gate_authenticated=True, authenticated_native_receipt_count=4800)
    for data in source['analyses'].values():
        data['temperatures']['0.0'] = block(prompt_records([8] * 8))
        data['paired_contrasts_vs_t1p0']['0.0'] = m.empty_metric_block()
        data['paired_endpoint_contrast']['comparison'] = 'T2.0-T0.0'
    return source


def test_zero_extension_recomputes_endpoint_and_all_pairwise_contrasts(extended_source):
    before = deepcopy(extended_source)
    result = m.analyze(extended_source)
    assert extended_source == before
    assert result['schema'] == 'gpt56-pass8-temperature-frontier-v2'
    assert result['total_registered_responses'] == 4800
    for data in result['analyses'].values():
        assert len(data['pass8_pairwise_contrasts']) == 10
        assert data['temperatures']['0.0']['groups']['five_domain_macro']['overall']['pass8']['estimate'] == 1.
        endpoint = data['paired_endpoint_contrast']
        assert endpoint['comparison'] == 'T2.0-T0.0'
        assert endpoint['groups']['five_domain_macro']['overall']['pass8']['estimate'] == -.75
        assert endpoint['cells']['level1/countdown']['pass8'] == data['pass8_pairwise_contrasts']['2.0-0.0']['cells']['level1/countdown']['pass8']
    assert len(list(m.table_rows(result))) == 2 * 7 * 2 * 4
    assert '4,800 saved temperature-sweep responses' in m.markdown(result)
    assert 'Paired T2.0 − T0.0' in m.markdown(result)


@pytest.mark.parametrize('change,match', [
    ('zero_missing', 'incomplete temperature'), ('old_endpoint', 'Changed endpoint'),
    ('gate_missing', 'complete-cohort validation'), ('wrong_count', 'inventory is incomplete'),
    ('unsupported_temperature', 'registered temperature grid')])
def test_zero_extension_rejects_incomplete_or_unregistered_data(extended_source, change, match):
    if change == 'zero_missing':
        extended_source['analyses']['strict']['temperatures'].pop('0.0')
    elif change == 'old_endpoint':
        extended_source['analyses']['strict']['paired_endpoint_contrast']['comparison'] = 'T2.0-T0.5'
    elif change == 'gate_missing':
        extended_source['validation'].pop('zero_extension_gate_authenticated')
    elif change == 'wrong_count':
        extended_source['total_registered_responses'] = 3840
    else:
        extended_source['temperatures'].append(2.5)
    with pytest.raises(ValueError, match=match):
        m.analyze(extended_source)


def test_original_four_temperature_report_remains_supported(source):
    result = m.analyze(source)
    assert result['schema'] == 'gpt56-pass8-temperature-frontier-v1'
    assert result['analyses']['strict']['paired_endpoint_contrast']['comparison'] == 'T2.0-T0.5'
