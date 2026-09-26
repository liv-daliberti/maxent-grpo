from pathlib import Path
import ast,json,shutil
ROOT=Path(__file__).resolve().parents[4]
AUDIT=Path(__file__).resolve().parent
SPECS={
 'ops/build_frontier_paper_comparison.py':['cell_tables','python_sensitivity_tex'],
 'ops/build_frontier_temperature_paper.py':['render'],
 'ops/build_paper_gpt56_all_levels32_discovery.py':['render_tex'],
 'ops/build_paper_gpt56sol_python_parity.py':['render'],
 'ops/build_paper_hosted_level_averages.py':['render_table','render_parity_table'],
 'ops/build_paper_hosted_reasoning_off.py':['render_main_table','render_appendix'],
}
SIDECARS=['frontier_comparison_20260911','frontier_temperature_20260911','gpt56_all_levels32_sampling_20260913','gpt56sol_python_parity_20260918','hosted_level_averages_20260911','hosted_reasoning_off_20260912']
inv=json.loads(Path('/tmp/paper-domain-typewriter-20260921/included-domain-files.json').read_text())
texpaths=[p for p in inv if Path(p).name.startswith(('frontier_','hosted_','gpt56','discovery_hosted_interpretation_'))]
for rel in list(SPECS)+texpaths+['paper/results/'+s+'.json' for s in SIDECARS]:
 src=ROOT/rel;dest=AUDIT/'before'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():shutil.copy2(src,dest)
for rel,names in SPECS.items():
 p=ROOT/rel;s=p.read_text();tree=ast.parse(s);lines=s.splitlines(keepends=True);offsets=[0]
 for line in lines:offsets.append(offsets[-1]+len(line))
 def span(n):return offsets[n.lineno-1]+n.col_offset,offsets[n.end_lineno-1]+n.end_col_offset
 edits=[]
 for fn in tree.body:
  if isinstance(fn,ast.FunctionDef) and fn.name in names:
   ret=max((n for n in ast.walk(fn) if isinstance(n,ast.Return)),key=lambda x:x.lineno)
   a,b=span(ret.value);edits.append((a,b,'format_domain_names('+s[a:b]+', exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS)'))
  if isinstance(fn,ast.FunctionDef) and fn.name=='export' and 'comparison' in rel:
   for n in ast.walk(fn):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text':
     a,b=span(n.func.value)
     if '.tex' in s[a:b]:
      a,b=span(n.args[0]);edits.append((a,b,'format_domain_names('+s[a:b]+', exclude_phrases=_DOMAIN_LANGUAGE_EXCEPTIONS)'))
 for a,b,val in sorted(edits,reverse=True):s=s[:a]+val+s[b:]
 tree=ast.parse(s)
 # Insert after imports, before the first module assignment/function. All imports
 # needed by the formatter itself remain in the shared, dependency-free module.
 last_import=max(n.end_lineno for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
 lines=s.splitlines(keepends=True)
 addition='\ntry:\n    from ops.paper_domain_typography import format_domain_names\nexcept ModuleNotFoundError:\n    from paper_domain_typography import format_domain_names\n\n_DOMAIN_LANGUAGE_EXCEPTIONS = (\n    "Python lambda", r"Python \\texttt{lambda}", "Python modulo",\n)\n'
 lines.insert(last_import,addition);p.write_text(''.join(lines))
(AUDIT/'owned_files.json').write_text(json.dumps({'tex':texpaths,'emitters':SPECS,'sidecars':SIDECARS},indent=2)+'\n')
print('Patched',len(SPECS),'emitters; saved',len(texpaths),'fragment backups.')
