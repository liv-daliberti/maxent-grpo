from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'ops'),str(ROOT/'src')]
import modebench_current_contract as current
import frontier_modebench_contract as old
from prepare_modebench_prompt_ablation import neutral_messages
from oat_drgrpo.templates import TEMPLATE_FACTORY,PROMPT_TEMPLATE_ROLES,CHAT_SURFACES


def row():return {'problem':'Return one allowed lambda for the listed inputs.', 'answer':'private verifier specification'}


def test_default_matches_frozen_neutral_ablation_exactly():
    r=row();expected=neutral_messages(3,'python_factors',old.make_messages(3,'python_factors',r))
    assert current.make_messages(3,'python_factors',r)==expected
    assert current.make_messages(3,'python_factors',r,current.HISTORICAL)==old.make_messages(3,'python_factors',r)
    assert current.prompt_sha256(3,'python_factors',r)!=current.prompt_sha256(3,'python_factors',r,current.HISTORICAL)
    assert 'private verifier' not in str(expected)


@pytest.mark.parametrize('level,domain',[(l,d) for l in (1,2,3) for d in old.DOMAINS if (l,d)!=(3,'python_factors')])
def test_other_interfaces_unchanged(level,domain):
    # Pantry L1's template validates a task-specific ending; metadata alone
    # establishes that the same historical renderer remains selected.
    assert current.profile_metadata(level,domain)['template_name']==old.profile_metadata(level,domain)['template_name']


def test_training_and_inference_use_identical_new_system():
    r=row();env={'OAT_ZERO_PROMPT_TEMPLATE':'qwen_level2_python_factors','OAT_ZERO_GENERATE_MAX_LENGTH':'192','OAT_ZERO_MODEBENCH_SYNTAX_PROFILE':'domain_legal_v1'}
    changed=current.training_environment(3,'python_factors',env)
    assert changed['OAT_ZERO_GENERATE_MAX_LENGTH']=='192' and changed['OAT_ZERO_MODEBENCH_SYNTAX_PROFILE']=='domain_legal_v1'
    assert env['OAT_ZERO_PROMPT_TEMPLATE']=='qwen_level2_python_factors'
    key=changed['OAT_ZERO_PROMPT_TEMPLATE'];system,user=current.make_messages(3,'python_factors',r)
    sm,um,am=CHAT_SURFACES['qwen'];assert TEMPLATE_FACTORY[key](r['problem'])==sm+system['content']+um+user['content']+am
    assert TEMPLATE_FACTORY['qwen_level3_python_factors'](r['problem'])==TEMPLATE_FACTORY[key](r['problem'])
    assert PROMPT_TEMPLATE_ROLES[key]=='boxed'
    p=current.profile_metadata(3,'python_factors');assert p['choice_informed_by_existing_evaluation'] and not p['original_difficulty_calibration_applies_to_prompt']


def test_unrecognized_conditions_and_stale_interfaces_fail():
    with pytest.raises(ValueError):current.make_messages(3,'python_factors',row(),'not_a_version')
    with pytest.raises(ValueError):current.training_environment(3,'python_factors',{'OAT_ZERO_PROMPT_TEMPLATE':'unrelated'})
