#!/usr/bin/env python3
"""Create successors to frozen runtimes with neutral-template CLI admission."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'var/artifacts/python_level3_cli_recovery_20260912'
NEUTRAL='qwen_level3_python_factors_neutral_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def amend(text):
    old='        "qwen_level2_python_factors",\n'
    assert text.count(old)==1
    text=text.replace(old,old+'        "'+NEUTRAL+'",\n')
    old='        required_template, required_syntax = level2_contracts[modebench_domain]\n'
    assert text.count(old)==1
    text=text.replace(old,old+'        if modebench_domain == "python_factors" and args.prompt_template == "'+NEUTRAL+'":\n            required_template = "'+NEUTRAL+'"\n')
    old='    if modebench_domain == "none":\n'
    assert text.count(old)==1
    text=text.replace(old,'    if args.prompt_template == "'+NEUTRAL+'" and modebench_domain != "python_factors":\n        raise ValueError("Neutral Python template requires the Python domain")\n'+old)
    return text

def main():
    rows=[]
    for campaign in ['e122','e124']:
        old=ROOT/'artifacts/modebench_level3_neutral_default_20260911/runtime'/campaign
        new=ART/'runtime_v2'/campaign
        assert not new.exists(),'Runtime successor already exists'
        shutil.copytree(old,new,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for directory in [new]+[p for p in new.rglob('*') if p.is_dir()]:
            directory.chmod(directory.stat().st_mode | 0o200)
        relative='src/oat_drgrpo/args.py'
        (new/relative).chmod((new/relative).stat().st_mode | 0o200)
        (new/relative).write_text(amend((old/relative).read_text()))
        before={str(p.relative_to(old)):sha(p) for p in old.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
        after={str(p.relative_to(new)):sha(p) for p in new.rglob('*') if p.is_file()}
        assert set(before)==set(after) and [p for p in before if before[p]!=after[p]]==[relative]
        identity={'campaign':campaign,'parent':str(old),'parent_inventory':before,'inventory_sha256':after,'only_changed_file':relative,'change':'Accept the existing neutral Python renderer in CLI Literal and Python/domain_legal_v1 validation; reject other domains. No renderer, dataset, verifier, objective, optimizer, or sampling changes.','builder_sha256':sha(Path(__file__))}
        (new/'CLI_AMENDMENT_IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n')
        rows.append({'campaign':campaign,'runtime':str(new),'identity_sha256':sha(new/'CLI_AMENDMENT_IDENTITY.json')})
    (ART/'runtime_prepared.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
