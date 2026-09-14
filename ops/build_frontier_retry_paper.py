#!/usr/bin/env python3
"""Render an independently audited, selected-validity Python retry diagnostic."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'artifacts/frontier_modebench_opus5_python_first_valid_20260911'
ORIGINAL=ROOT/'artifacts/frontier_modebench_claude_opus5_python_plain_20260911/summary.json'
STEM='frontier_python_retry_20260911'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def build_record():
    audit_path=DIRECTORY/'independent_audit.json';audit=json.loads(audit_path.read_text())
    if audit.get('status')!='pass' or not all(audit.get(k) is True for k in ('all_level_statistics_and_bootstraps_reconstructed','all_original_evidence_unchanged','all_retained_retries_independently_regraded')):
        raise ValueError('Python first-valid condition lacks a passing independent audit.')
    if (audit['new_terminal_responses'],audit['original_source_slots_authenticated'],audit['valid_selected_slots'])!=(3,3072,3072):
        raise ValueError('Unexpected selected retry population.')
    bindings={str(audit_path.relative_to(ROOT)):sha(audit_path),str(ORIGINAL.relative_to(ROOT)):sha(ORIGINAL)}
    for name,digest in audit['evidence_sha256'].items():
        path=DIRECTORY/name
        if sha(path)!=digest:raise ValueError(f'Stale retry audit evidence: {path}')
        bindings[str(path.relative_to(ROOT))]=digest
    summary=json.loads((DIRECTORY/'summary.json').read_text());original=json.loads(ORIGINAL.read_text())
    levels={level:{'fixed_draw':original['normalized_secondary']['cells'][f'level{level}/python_factors']['metrics'],
                   'first_valid':summary['levels'][level]['metrics']} for level in ('1','2','3')}
    return {'schema':'paper-selected-python-retry-v1','builder_sha256':sha(__file__),'source_sha256':bindings,
            'independent_audit':audit,'levels':levels,'main_fixed_draw_cohort_unchanged':True}
def render(record):
    values=[record['levels'][l]['first_valid']['distinct8']['estimate'] for l in ('1','2','3')]
    before=record['levels']['3']['fixed_draw']['correct_pair_collision']['estimate']*100
    after=record['levels']['3']['first_valid']['correct_pair_collision']['estimate']*100
    return r'''\paragraph{Separate first-valid retry diagnostic.}
After completing the fixed 3,072-response revised-wording cohort, a separate
condition retries only its three remaining refused slots, retaining the
first verified-valid response. All three succeed on their first additional
request, and all reproduce a mode already seen for that prompt. This uses
3,075 requests to select 3,072 valid responses; its 100\% accepted-set
accuracy follows from selection and is not unconditional model accuracy.
The selected set's \texttt{distinct@8} is unchanged at '''+', '.join(f'{v:.3f}' for v in values)+r''' at
Levels 1--3. Level-3 correct-pair collision changes from '''+f'{before:.4f}'+r'\% to '+f'{after:.4f}'+r'''\%;
the other levels are unchanged. All initial and retry responses remain
preserved. The main figure continues to use the complete fixed eight-draw
condition, including its three refusals; these selected-validity results
are only a separate diagnostic with variable request effort.
'''
def main():
    record=build_record();out=ROOT/'paper/results';out.mkdir(exist_ok=True)
    (out/(STEM+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    (out/(STEM+'.tex')).write_text(render(record))
    print(json.dumps({'retry_attempts':3,'authenticated_bindings':len(record['source_sha256'])}))
if __name__=='__main__':main()
