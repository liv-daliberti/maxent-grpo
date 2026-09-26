"""Scientific presentation keeps sampling profiles and complete fixed draws distinct."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('gpt56_paper_curve',ROOT/'ops/plot_paper_gpt56_temperature_curve.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@pytest.fixture(scope='module')
def source():return json.loads((ROOT/'artifacts/frontier_temperature_20260911/GPT56_PASS8_FRONTIER_WITH_ZERO.json').read_text())
@pytest.fixture(scope='module')
def record():return m.build_record(ROOT/'artifacts/frontier_temperature_20260911/GPT56_PASS8_FRONTIER_WITH_ZERO.json')
def test_all_fixed_points_match_independent_source(source,record):
    assert record['sampling']['total_responses']==960*len(source['temperatures'])
    assert record['sampling']['temperatures']==sorted(source['temperatures'])
    assert {(p['level'],p['temperature']) for p in record['display']['points']}=={(l,t) for l in (1,2,3) for t in source['temperatures']}
    for point in record['display']['points']:
        node=source
        for part in point['source_group'].strip('/').split('/'):node=node[part]
        assert point['metrics']=={key:node[key] for key in ('pass8','distinct8')}
    assert all(record['analyses']['normalized_secondary']['temperatures'][t]['counts']['responses']==960 for t in record['analyses']['normalized_secondary']['temperatures'])
def test_historical_reference_retained_only_in_appendix(source,record):
    reference=record['matched_medium_reference']
    assert reference['reasoning_effort']=='medium' and reference['requested_temperature'] is None
    assert reference['connect_to_temperature_curve'] is False
    assert 'reference_points' not in record and 'overall_reference_point' not in record
    assert 'omitted from the figure' in m.render_appendix(record)
    fig=m.build_figure(record)
    assert len(fig.axes)==2
    assert len(fig.axes[0].lines)==1 and len(fig.axes[1].lines)==3
    assert all(len(line.get_xdata())==len(source['temperatures']) for ax in fig.axes for line in ax.lines)
    assert all(not ax.collections for ax in fig.axes)
    assert all('historical medium' not in text.get_text().lower() for text in fig.texts)
    assert all(ax.get_xlabel()=='pass@8 (%)' for ax in fig.axes)
    import matplotlib.pyplot as plt
    plt.close(fig)
@pytest.mark.parametrize('change',('reasoning','missing_temperature','connected_reference','failed_audit','stale_source'))
def test_rejects_profile_or_source_corruption(source,tmp_path,change):
    altered=deepcopy(source)
    if change=='reasoning':altered['reasoning_effort']='medium'
    elif change=='missing_temperature':del altered['analyses']['normalized_secondary']['temperatures']['2.0']
    elif change=='connected_reference':altered['matched_medium_reference']['connect_to_temperature_curve']=True
    elif change=='failed_audit':
        key='all_3840_native_receipts_authenticated' if source['schema'].endswith('-v1') else 'all_native_receipts_authenticated'
        altered['validation'][key]=False
    else:altered['collection_gate']['sha256']='0'*64
    path=tmp_path/'source.json';path.write_text(json.dumps(altered))
    with pytest.raises(ValueError):m.build_record(path)

def test_plot_coordinates_use_empirical_pass8_not_per_response_accuracy(source,record):
    fig=m.build_figure(record)
    points=record['display']['overall_points']
    assert list(fig.axes[0].lines[0].get_xdata())==[p['metrics']['pass8']['estimate']*100 for p in points]
    original=source['analyses']['normalized_secondary']['temperatures']
    assert list(fig.axes[0].lines[0].get_xdata())!=[original[t]['groups']['five_domain_macro']['overall']['accuracy']['estimate']*100 for t in (str(float(value)) for value in record['sampling']['temperatures'])]
    assert 'P is solved prompts' in m.render_appendix(record)
    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.fixture
def synthetic_extended_source(source):
    """In-memory-only fixture; it is never written to publication artifacts."""
    extended=deepcopy(source)
    extended.update(schema='gpt56-pass8-temperature-frontier-v2',
                    temperatures=[0.,.5,1.,1.5,2.],total_registered_responses=4800,
                    responses_per_condition=960,prompts_per_condition=120,draws_per_prompt=8)
    extended['validation'].pop('all_3840_native_receipts_authenticated',None)
    extended['validation'].update(all_native_receipts_authenticated=True,authenticated_native_receipt_count=4800)
    extended['conditions']['0.0']=deepcopy(extended['conditions']['0.5'])
    for data in extended['analyses'].values():
        data['temperatures']['0.0']=deepcopy(data['temperatures']['0.5'])
        data['paired_endpoint_contrast']['comparison']='T2.0-T0.0'
    return extended


def test_extended_grid_retains_and_labels_all_five_temperatures(synthetic_extended_source,tmp_path,monkeypatch):
    path=tmp_path/'synthetic_extended.json';path.write_text(json.dumps(synthetic_extended_source))
    monkeypatch.setattr(m,'relative',lambda value:str(Path(value).resolve()))
    record=m.build_record(path)
    assert record['schema']=='paper-gpt56-pass8-temperature-curve-v2'
    assert record['sampling']['total_responses']==4800
    assert len(record['display']['points'])==15 and len(record['display']['overall_points'])==5
    fig=m.build_figure(record)
    assert all(len(line.get_xdata())==5 for ax in fig.axes for line in ax.lines)
    assert sum(text.get_text()=='0' for ax in fig.axes for text in ax.texts)==4
    assert all(not ax.collections for ax in fig.axes)
    appendix=m.render_appendix(record)
    assert '4,800 in total' in appendix and 'T=2.0$ minus $T=0.0' in appendix
    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.parametrize('change',('missing_zero','duplicate_zero','incomplete_strict','incomplete_draws',
                                   'unauthenticated_draws','incorrect_total','stale_endpoint'))
def test_extended_grid_rejects_incomplete_or_inconsistent_evidence(synthetic_extended_source,tmp_path,change):
    altered=synthetic_extended_source
    if change=='missing_zero':del altered['analyses']['normalized_secondary']['temperatures']['0.0']
    elif change=='duplicate_zero':altered['temperatures'].append(0.)
    elif change=='incomplete_strict':del altered['analyses']['strict']['temperatures']['0.0']
    elif change=='incomplete_draws':altered['analyses']['strict']['temperatures']['0.0']['counts']['responses']=959
    elif change=='unauthenticated_draws':altered['validation']['authenticated_native_receipt_count']=4799
    elif change=='incorrect_total':altered['total_registered_responses']=3840
    else:altered['analyses']['strict']['paired_endpoint_contrast']['comparison']='T2.0-T0.5'
    path=tmp_path/'synthetic_corrupt.json';path.write_text(json.dumps(altered))
    with pytest.raises(ValueError):m.build_record(path)


def test_appendix_best_observed_temperature_comes_from_results(synthetic_extended_source,tmp_path,monkeypatch):
    group=synthetic_extended_source['analyses']['normalized_secondary']['temperatures']['0.0']['groups']['five_domain_macro']['overall']
    for metric,value in [('pass8',1.),('distinct8',8.)]:
        group[metric]={'estimate':value,'ci95':[value,value],'defined_replicates':20000}
    path=tmp_path/'synthetic_best_at_zero.json';path.write_text(json.dumps(synthetic_extended_source))
    monkeypatch.setattr(m,'relative',lambda value:str(Path(value).resolve()))
    appendix=m.render_appendix(m.build_record(path))
    assert r'\texttt{pass@8} occurs at $T=0.0$' in appendix
    assert r'\texttt{distinct@8}'+'\noccurs at $T=0.0$' in appendix
    assert 'these 5 measured temperatures' in appendix

@pytest.fixture
def synthetic_expanded480_source(source):
    """Count/renderer fixture only; never used as experimental measurements."""
    expanded=deepcopy(source)
    expanded.update(schema='gpt56-pass8-temperature-frontier-expanded480-v1',
                    total_registered_responses=19200,responses_per_condition=3840,
                    prompts_per_condition=480,draws_per_prompt=8)
    expanded['validation'].update(all_native_receipts_authenticated=True,
        authenticated_native_receipt_count=19200,same_480_prompts_all_eight_slots=True,
        expansion_plan_authenticated=True,original_4800_measurements_retained_unchanged=True)
    for analysis in expanded['analyses'].values():
        for block in analysis['temperatures'].values():
            block['counts'].update(responses=3840,prompts=480)
    for analysis in expanded['analyses'].values():
        analysis['paired_high_temperature_contrast']=deepcopy(analysis['paired_endpoint_contrast'])
        analysis['paired_high_temperature_contrast']['comparison']='T2.0-T1.5'
    expanded['historical_medium_reference_subset120']={'prompts_per_condition':120,
        'reference':expanded.pop('matched_medium_reference')}
    expanded['cohort_sensitivity']={}
    for key,n in [('original_120',120),('additional_360',360)]:
        analyses=deepcopy(expanded['analyses'])
        for analysis in analyses.values():
            analysis['paired_high_temperature_contrast']=deepcopy(analysis['paired_endpoint_contrast'])
            analysis['paired_high_temperature_contrast']['comparison']='T2.0-T1.5'
        expanded['cohort_sensitivity'][key]={'prompts_per_condition':n,'analyses':analyses}
    return expanded


def test_expanded480_figure_and_appendix_use_complete_new_cohort(synthetic_expanded480_source,tmp_path,monkeypatch):
    path=tmp_path/'synthetic_480.json';path.write_text(json.dumps(synthetic_expanded480_source))
    monkeypatch.setattr(m,'relative',lambda value:str(Path(value).resolve()))
    record=m.build_record(path)
    assert record['sampling']['total_responses']==19200
    assert record['sampling']['responses_per_temperature']==3840
    assert record['sampling']['prompts_per_domain_level']==32
    fig=m.build_figure(record)
    assert any('480 matched prompts' in text.get_text() for text in fig.texts)
    appendix=m.render_appendix(record)
    assert '480 held-out prompts' in appendix and '19,200 in total' in appendix
    assert 'Original 120 prompts' in appendix and 'Additional 360 prompts' in appendix
    assert 'M &' not in appendix
    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.parametrize('change',('old_count','missing_retained_audit','missing_plan_audit','missing_expanded_group_audit','wrong_native_count'))
def test_expanded480_cannot_relabel_old_or_unauthenticated_data(synthetic_expanded480_source,tmp_path,change):
    source=synthetic_expanded480_source
    if change=='old_count':source['analyses']['strict']['temperatures']['1.0']['counts']['prompts']=120
    elif change=='missing_retained_audit':source['validation']['original_4800_measurements_retained_unchanged']=False
    elif change=='missing_plan_audit':source['validation']['expansion_plan_authenticated']=False
    elif change=='missing_expanded_group_audit':source['validation']['same_480_prompts_all_eight_slots']=False
    else:source['validation']['authenticated_native_receipt_count']=4800
    path=tmp_path/'bad_480.json';path.write_text(json.dumps(source))
    with pytest.raises(ValueError):m.build_record(path)


def test_expanded480_requires_zero_even_if_all_totals_are_adjusted(synthetic_expanded480_source,tmp_path):
    source=synthetic_expanded480_source
    source['temperatures'].remove(0.)
    source['conditions'].pop('0.0')
    source['total_registered_responses']=15360
    source['validation']['authenticated_native_receipt_count']=15360
    for analysis in source['analyses'].values():
        analysis['temperatures'].pop('0.0')
        analysis['paired_endpoint_contrast']['comparison']='T2.0-T0.5'
    path=tmp_path/'missing_zero_480.json';path.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='all five registered temperatures'):
        m.build_record(path)
