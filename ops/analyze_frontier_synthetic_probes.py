#!/usr/bin/env python3
"""Grade complete synthetic probe batches offline; preserve every native receipt."""
from __future__ import annotations
import argparse,ast,hashlib,json,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'artifacts/frontier_modebench_claude_opus5_20260911'

def sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def diagnostic_grade(fixture,text,remove_boxed,last_boxed):
    boxed=last_boxed(text)
    candidate=remove_boxed(boxed) if boxed else None
    verified=False;key=None
    if candidate is not None and len(candidate)<4096:
        interface=fixture.get('task_interface')
        if interface=='direct_integer_vector_diagnostic':
            try:vector=ast.literal_eval(candidate)
            except (ValueError,SyntaxError):vector=None
            cases=fixture['answer']['cases']
            if isinstance(vector,(list,tuple)) and len(vector)==len(cases) and all(
                type(d) is int and 1<d<n and n%d==0 for n,d in zip(cases,vector)):
                verified=True;key='python_factor:'+','.join(map(str,vector))
        elif interface=='two_argument_addition_lambda_control':
            try:
                expression=ast.parse(candidate,mode='eval').body
                valid_args=(isinstance(expression,ast.Lambda) and
                    [arg.arg for arg in expression.args.args]==['a','b'] and
                    not expression.args.posonlyargs and not expression.args.kwonlyargs and
                    not expression.args.defaults and not expression.args.kw_defaults and
                    expression.args.vararg is None and expression.args.kwarg is None)
                body=expression.body if valid_args else None
                verified=(isinstance(body,ast.BinOp) and isinstance(body.op,ast.Add) and
                    isinstance(body.left,ast.Name) and isinstance(body.right,ast.Name) and
                    {body.left.id,body.right.id}=={'a','b'})
                if verified:key='addition_control:sum'
            except (SyntaxError,ValueError):pass
    return {'verified':bool(verified),'canonical_key':key,'graded_text':text}

def analyze(directory):
    directory=Path(directory).resolve()
    sys.path[:0]=[str(REFERENCE/'code/ops'),str(REFERENCE/'code/src'),str(REFERENCE/'secondary_code/ops'),
                  str(ROOT/'artifacts/frontier_models_comparison_20260911/provider_outcome_code_legacy_digest_compat/ops')]
    from frontier_modebench_contract import grade_response
    from evaluate_claude_modebench import warm_python_worker,response_text,response_status
    from frontier_modebench_normalization import normalize_and_grade
    from oat_drgrpo.math_grader import last_boxed_only_string,remove_boxed
    from audit_hosted_provider_outcomes import classify_native
    import frontier_modebench_normalization as normalizer
    source_manifest=json.loads((REFERENCE/'manifest.json').read_text())
    for name,value in source_manifest['code_sha256'].items():
        if digest(REFERENCE/'code'/name)!=value:raise ValueError('Frozen verifier source changed: '+name)
    expected_normalizer=json.loads((REFERENCE/'summary.json').read_text())['normalized_secondary']['normalization_source_sha256']
    if digest(Path(normalizer.__file__))!=expected_normalizer:raise ValueError('Frozen normalizer changed')
    warmup=warm_python_worker()
    requests=read_jsonl(directory/'probe_requests.jsonl')
    if len({row['probe_id'] for row in requests})!=len(requests):raise ValueError('Duplicate planned probe identity')
    fixtures={row['fixture_id']:row for row in json.loads((directory/'fixtures.json').read_text())}
    by_probe=defaultdict(list)
    raw_inventory={}
    for path in (directory/'raw_responses').glob('*.json'):
        raw=json.loads(path.read_text());by_probe[raw['probe_id']].append((path,raw))
        raw_inventory[str(path.relative_to(directory))]=digest(path)
    if set(by_probe)-{row['probe_id'] for row in requests}:raise ValueError('Unexpected raw probe identity')
    records=[];response_ids=set();missing=[]
    for request in requests:
        if sha(request['request'])!=request['request_sha256']:raise ValueError('Planned request digest mismatch')
        attempts=sorted(by_probe[request['probe_id']],key=lambda pair:pair[1]['attempt'])
        for path,raw in attempts:
            for key in ('probe_id','condition','fixture_id','domain','replicate','request_sha256'):
                if raw[key]!=request[key]:raise ValueError('Raw identity mismatch: '+key)
        successes=[pair for pair in attempts if pair[1]['http_status']==200]
        if len(successes)!=1:
            missing.append(request['probe_id']);continue
        path,raw=successes[0];body=raw['response']
        if body.get('model')!=request['request']['model'] or response_status(body) is None:raise ValueError('Unexpected native model or stop status')
        if not body.get('id') or body['id'] in response_ids:raise ValueError('Missing or repeated model response ID')
        response_ids.add(body['id'])
        fixture=fixtures[request['fixture_id']]
        if request.get('fixture_sha256') and request['fixture_sha256']!=sha(fixture):raise ValueError('Fixture digest mismatch')
        text=response_text(body)
        interface=fixture.get('task_interface','restricted_python_factor_function' if fixture['domain']=='python_factors' else 'mathir_action_menu')
        if interface in ('direct_integer_vector_diagnostic','two_argument_addition_lambda_control'):
            strict=diagnostic_grade(fixture,text,remove_boxed,last_boxed_only_string)
            normalized=dict(strict)
            normalized['transformations']=[]
            normalization_applicable=False
        else:
            strict=grade_response(fixture['level'],fixture['domain'],fixture,text)
            normalized=normalize_and_grade(fixture,text,strict_grade=strict,grader=grade_response)
            normalization_applicable=True
        outcomes=classify_native(body,'anthropic_messages')
        records.append({'probe_id':request['probe_id'],'condition':request['condition'],'fixture_id':request['fixture_id'],
            'domain':request['domain'],'replicate':request['replicate'],'model':body['model'],'task_interface':interface,
            'request_sha256':request['request_sha256'],'fixture_sha256':sha(fixture),
            'raw_receipt':str(path.relative_to(directory)),'raw_receipt_sha256':sha(raw),'raw_receipt_file_sha256':digest(path),
            'response_id':body['id'],'text':text,'strict':strict,'normalized':normalized,
            'format_normalization_applicable':normalization_applicable,'provider_outcome':outcomes,'native_usage':body.get('usage')})
    if missing:raise ValueError('Incomplete or duplicate-success probe batch: '+str(missing))
    cells=defaultdict(list)
    for record in records:cells[record['condition']+'/'+record['domain']].append(record)
    summary_cells={}
    for cell,rows in sorted(cells.items()):
        summary_cells[cell]={'responses':len(rows),'native_refusals':sum(row['provider_outcome']['refusal'] for row in rows),
            'empty_answers':sum(row['provider_outcome']['answer_text_empty'] for row in rows),
            'category_counts':dict(Counter(label for row in rows for label in row['provider_outcome']['category_labels'])),
            'strict_valid':sum(row['strict']['verified'] for row in rows),
            'normalized_valid':sum(row['normalized']['verified'] for row in rows),
            'normalization_rescues':sum(row['normalized']['verified'] and not row['strict']['verified'] for row in rows),
            'observed_strict_modes':len({row['strict']['canonical_key'] for row in rows if row['strict']['verified']}),
            'observed_normalized_modes':len({row['normalized']['canonical_key'] for row in rows if row['normalized']['verified']}),
            'task_interfaces':sorted({row['task_interface'] for row in rows}),
            'format_normalization_applicable':all(row['format_normalization_applicable'] for row in rows)}
    (directory/'probe_grades.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in records))
    report={'schema':'synthetic_hosted_probe_grade_audit_v1','status':'complete','generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'planned_responses':len(requests),'saved_responses':len(records),'native_models':dict(Counter(row['model'] for row in records)),
        'cells':summary_cells,'frozen_python_warmup':warmup,'raw_file_sha256':raw_inventory,
        'source_sha256':{name:digest(directory/name) for name in ('probe_requests.jsonl','fixtures.json','probe_grades.jsonl')},
        'frozen_verifier_manifest_sha256':digest(REFERENCE/'manifest.json'),'normalization_source_sha256':expected_normalizer,
        'analyzer_source_sha256':digest(Path(__file__)),'api_calls_by_analyzer':0,
        'limitations':['One synthetic fixture per task/condition; counts do not estimate benchmark-wide rates.',
                       'Original and clarified prompt conditions remain separate; no responses replace original benchmark samples.',
                       'Addition control accepts the exact a+b or b+a AST form; it is a simple diagnostic, not a benchmark score.',
                       'Direct divisor vectors change the programming interface and are graded directly, not by formatting normalization.']}
    atomic(directory/'probe_analysis.json',report)
    lines=['# Synthetic development probe results','',f"All {len(records)} planned responses received and independently graded offline.",'',
        '| Condition / task | Responses | Native refusals | Strict valid | Formatting-normalized valid | Observed valid modes |',
        '|---|---:|---:|---:|---:|---:|']
    for cell,values in summary_cells.items():
        normalized=str(values['normalized_valid']) if values['format_normalization_applicable'] else 'not applicable'
        lines.append(f"| {cell} | {values['responses']} | {values['native_refusals']} | {values['strict_valid']} | {normalized} | {values['observed_strict_modes']} |")
    lines+=['',*report['limitations'],'']
    (directory/'PROBE_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'complete','responses':len(records),'cells':summary_cells},sort_keys=True))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();analyze(args.directory)
