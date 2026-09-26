"""Match the current protocol prose without rerunning record generation."""
from pathlib import Path
import ast,copy,hashlib,importlib,json,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));A=Path(__file__).resolve().parent
source=ROOT/'ops/build_frontier_paper_comparison.py';target=ROOT/'paper/results/frontier_comparison_20260911_protocol.tex';sidecar=ROOT/'paper/results/frontier_comparison_20260911.json'
expected=target.read_text();before=source.read_text();record_before=json.loads(sidecar.read_text())
old="""            unknown = collection['interruption_accounting']['registered_interrupted_attempts_with_unknown_outcome']
            malformed = collection['nonterminal_protocol_receipts']
            protocol += ['', r'\paragraph{Response and cost coverage: ' + tex(run['label']) + '.}',
                         'The 15,360 terminal responses include verification failures and truncations.',
                         f'Outside this cohort, {malformed} HTTP-200 responses have neither a',
                         'terminal finish reason nor a visible answer, and',
                         f'{unknown} request attempts have unknown outcomes and usage. These incomplete',
                         'attempts are not scored as model responses. Token totals cover known usage',
                         'only and can therefore understate the usage and cost of all attempted requests.']"""
new="""            protocol += ['', 'The 15,360 scored ' + tex(run['label'])
                         + ' responses include verification failures',
                         'and truncations. Usage totals omit requests without reported usage and',
                         'therefore give a lower bound on the cost of all attempted requests.']"""
assert before.count(old)==1,'Live emitter block differs; stop without overwriting'
assert expected.endswith('The 15,360 scored DeepSeek V4 Pro responses include verification failures\nand truncations. Usage totals omit requests without reported usage and\ntherefore give a lower bound on the cost of all attempted requests.\n'),'Live paragraph differs; stop without overwriting'
(A/'protocol_sync_before.py').write_text(before);(A/'protocol_sync_target.tex').write_text(expected)
assert target.read_text()==expected,'Target changed before mutation'
source.write_text(before.replace(old,new))
# Execute only the protocol-construction statements from export, ending in a
# return of its rendered TeX expression. No export, plot, or data builder runs.
mod=importlib.import_module('ops.build_frontier_paper_comparison');tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='export')
start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='names' for t in n.targets))
end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and '_protocol.tex' in ast.unparse(n))
body=copy.deepcopy(fn.body[start:end]);body.append(ast.Return(value=copy.deepcopy(fn.body[end].value.args[0])))
check_fn=ast.FunctionDef(name='_protocol_only_for_check',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='record')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
code=ast.fix_missing_locations(ast.Module(body=[check_fn],type_ignores=[]));namespace=dict(mod.__dict__);exec(compile(code,str(source),'exec'),namespace)
rendered=namespace['_protocol_only_for_check'](record_before)
assert target.read_text()==expected,'Target changed during rendering; stop before metadata mutation'
assert rendered==expected,'Protocol rendering differs from current TeX'
record=copy.deepcopy(record_before);record['builder_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
assert json.loads(sidecar.read_text())==record_before,'Sidecar changed concurrently; stop without overwriting'
sidecar.write_text(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
assert {k:v for k,v in record.items() if k!='builder_sha256'}=={k:v for k,v in record_before.items() if k!='builder_sha256'}
result={'status':'passed','protocol_render_byte_identical':True,'protocol_tex_unchanged':target.read_text()==expected,'only_sidecar_field_changed':'builder_sha256','builder_sha256':record['builder_sha256'],'sidecar_sha256':hashlib.sha256(sidecar.read_bytes()).hexdigest(),'record_generation_plots_or_experiments_run':False}
(A/'protocol_sync_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
