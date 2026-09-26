"""Small offline checks for the stable expanded paper-to-archive guide."""
import importlib.util
from pathlib import Path
import re
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('archive_experiment_guide_under_test',ROOT/'ops/archive_experiment_guide.py')
guide=importlib.util.module_from_spec(spec);spec.loader.exec_module(guide)

def test_paper_anchor_targets_and_original_families():
    text=guide.render_experiment_guide()
    for anchor in ('direct-comparators','semantic-ablations','core-replay','frequency-ablation','mechanism-studies','reproducibility'):
        assert f'<a id="{anchor}"></a>' in text
    for family in ('E95','E114','E97','E99','E115','E98-R1','E100','E116','E81','E82','E83','E85','E86','E112-R1','E109','E87','E121'):
        assert family in text
    assert 'one paired seed' in text and 'not a multi-seed estimate' in text
    assert 'Adaptive controllers' in text and 'excluded from that reported E112-R1' in text

def test_unpublished_new_cohorts_and_data_are_not_linked():
    text=guide.render_experiment_guide()
    links=re.findall(r'\]\(([^)]+)\)',text)
    for target in links:
        if target.startswith('experiments/'):
            assert target.split('/')[1] in guide.INITIAL_STUDIES
    assert 'Data — in preparation' in text
    assert 'experiments/E95/' not in text
    assert '/tree/' not in text

def test_explicit_publication_creates_exact_new_links():
    text=guide.render_experiment_guide(guide.INITIAL_STUDIES+('E95',),available_sections={'Data':'data/README.md'})
    assert '[E95](experiments/E95/README.md#models)' in text
    assert '[Data](data/README.md)' in text
    assert 'experiments/E99/' not in text
    with pytest.raises(ValueError):guide.render_experiment_guide(('E95/../../secret',))
    with pytest.raises(ValueError):guide.render_experiment_guide(available_sections={'Data':'../private'})

def test_comparator_counts_and_scope_have_no_private_paths():
    text=guide.render_experiment_guide()
    assert '45 uniform comparators' in text
    assert '25 from E78' in text and '10 from E79' in text and '10 from E80-R1' in text
    assert 'not a 3073rd optimizer update' in text
    assert 'initial fixed selection contains 418' in text
    for value in ('/n/fs/','debug_job','job_id','run_stamp','token-file'):
        assert value not in text
