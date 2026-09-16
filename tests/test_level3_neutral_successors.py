from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"ops"),str(ROOT/"src")]

def test_successor_environment_has_fresh_namespace_and_consistent_pair():
    from prepare_level3_neutral_successors import new_cell
    base={'domain':'python_factors','arm':'drgrpo','seed':43,'target_steps':3072,'resources':{},'run_stamp':'old_s43','run_dir':'/old/s43','environment':{'OAT_ZERO_PROMPT_TEMPLATE':'qwen_level2_python_factors','OAT_ZERO_GENERATE_MAX_LENGTH':'192','SAVE_PATH':'/old/s43','RUN_STAMP':'old_s43'}}
    a=new_cell(base,'e122',Path('/new/runtime'));b=new_cell({**base,'arm':'replay_drgrpo'},'e122',Path('/new/runtime'))
    assert a['environment']['OAT_ZERO_PROMPT_TEMPLATE']==b['environment']['OAT_ZERO_PROMPT_TEMPLATE']=='qwen_level3_python_factors_neutral_v1'
    assert a['run_dir']!=base['run_dir'] and a['run_dir']!=b['run_dir']
    assert a['environment']['OAT_ZERO_SOURCE_ROOT']=='/new/runtime/src'
    assert base['environment']['OAT_ZERO_PROMPT_TEMPLATE']=='qwen_level2_python_factors'
    assert not a['source_difficulty_calibration_applies'] and not a['previous_run_may_be_resumed']
