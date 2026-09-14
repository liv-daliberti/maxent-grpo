"""Admission-gated data defaults for new runs; frozen historical plans stay versioned."""
import hashlib
import json
from pathlib import Path
import modebench_current_contract as prompts

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'var/artifacts/modebench_level3_neutral_v5'
DATA=ROOT/'var/data/modebench_level3_matched_neutral_v5'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def neutral_python_dataset():
    """Return the current Python L3 root only after full fresh confirmation passes."""
    path=ART/'admission.json'
    if not path.is_file():raise ValueError('Neutral Python Level3 data awaits passing fresh confirmation')
    admission=read(path)
    if not (admission['admitted'] is True and admission['status']=='observed_approximate_match'
            and admission['gates']=={'pass1':True,'pass8':True}
            and admission['tolerances']=={'pass1':.04,'pass8':.08}
            and admission['registration_sha256']==digest(ART/'registration.json')
            and admission['dataset_identity_sha256']==digest(DATA/'identity.json')
            and Path(admission['dataset'])==DATA
            and admission['confirmation_sha256']==digest(ROOT/'var/results/modebench_level3_neutral_v5/confirmation.json')):
        raise ValueError('Neutral Level3 admission binding mismatch')
    identity=read(DATA/'identity.json')
    for name,expected in identity['files_sha256'].items():
        if digest(Path(name))!=expected:raise ValueError('Admitted dataset file changed: '+name)
    return DATA/'python_factors',{'dataset_identity_sha256':digest(DATA/'identity.json'),
             'admission_sha256':digest(path),'prompt_condition':prompts.CURRENT,
             'historical_reference_uses_original_hints':True,'statistical_equivalence_claimed':False}


def training_environment(level,domain,environment,condition=prompts.DEFAULT_CONDITION):
    """Bind both neutral wording and admitted data for new Python Level3 runs."""
    result=prompts.training_environment(level,domain,environment,condition)
    profile=prompts.profile_metadata(level,domain,condition)
    if profile['system_wording']=='neutral':
        root,identity=neutral_python_dataset()
        result['OAT_ZERO_PROMPT_DATA']=str(root/'train')
        result['OAT_ZERO_EVAL_DATA']=str(root/'eval')
        if 'OAT_ZERO_DATA_ROOT' in result:result['OAT_ZERO_DATA_ROOT']=str(root)
        result['OAT_ZERO_MODEBENCH_DATASET_IDENTITY_SHA256']=identity['dataset_identity_sha256']
        result['OAT_ZERO_MODEBENCH_ADMISSION_SHA256']=identity['admission_sha256']
    return result
