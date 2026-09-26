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
def source():return json.loads(m.SOURCE.read_text())
@pytest.fixture(scope='module')
def record():return m.build_record()
def test_all_fixed_points_match_independent_source(source,record):
    assert record['sampling']['total_responses']==3840
    assert {(p['level'],p['temperature']) for p in record['display']['points']}=={(l,t) for l in (1,2,3) for t in (.5,1.,1.5,2.)}
    for point in record['display']['points']:
        node=source
        for part in point['source_group'].strip('/').split('/'):node=node[part]
        assert point['metrics']=={key:node[key] for key in ('accuracy','distinct8')}
    assert all(record['analyses']['normalized_secondary']['temperatures'][t]['counts']['responses']==960 for t in m.TEMPERATURES)
def test_historical_reference_remains_separate(source,record):
    reference=record['matched_medium_reference']
    assert reference['reasoning_effort']=='medium' and reference['requested_temperature'] is None
    assert reference['connect_to_temperature_curve'] is False
    assert len(record['reference_points'])==3 and all(not p['connected'] for p in record['reference_points'])
    fig=m.build_figure(record)
    assert len(fig.axes[0].lines)==3
    assert all(len(line.get_xdata())==4 for line in fig.axes[0].lines)
    assert len(fig.axes[0].collections)==3
    import matplotlib.pyplot as plt
    plt.close(fig)
@pytest.mark.parametrize('change',('reasoning','missing_temperature','connected_reference','failed_audit','stale_source'))
def test_rejects_profile_or_source_corruption(source,tmp_path,change):
    altered=deepcopy(source)
    if change=='reasoning':altered['reasoning_effort']='medium'
    elif change=='missing_temperature':del altered['analyses']['normalized_secondary']['temperatures']['2.0']
    elif change=='connected_reference':altered['matched_medium_reference']['connect_to_temperature_curve']=True
    elif change=='failed_audit':altered['validation']['all_3840_native_receipts_authenticated']=False
    else:altered['collection_gate']['sha256']='0'*64
    path=tmp_path/'source.json';path.write_text(json.dumps(altered))
    with pytest.raises(ValueError):m.build_record(path)
