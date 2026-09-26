from pathlib import Path
import hashlib,json,re,sys,difflib
sys.path.insert(0,str(Path('ops').resolve()))
from paper_domain_typography import format_domain_names
base=Path('paper/audits/domain_name_typography_20260921')
inputs=json.loads(Path('/tmp/paper-domain-typewriter-20260921/recursive-inputs.json').read_text())['tex']
exceptions={
'paper/main.tex':('executable Python code',r'isolated Python \texttt{-I} process','are invalid Python','LaTeX-escaped Python'),
'paper/results/frontier_comparison_20260911_python_sensitivity.tex':('Python lambda','Python modulo'),
'paper/results/frontier_hosted_20260911_appendix.tex':(r'Python \texttt{lambda}',)}
missing=[];nested=[];compound=[]
for name in inputs:
 s=Path(name).read_text()
 if format_domain_names(s,exclude_phrases=exceptions.get(name,()))!=s:missing.append(name)
 # Domain wrappers are simple; any literal nested texttt is invalid here.
 if re.search(r'\\texttt\{[^{}]*\\texttt\{',s):nested.append(name)
 for m in re.finditer(r'\b(?:Graph|Countdown|Python|MathIR|Pantry)[A-Za-z]+\b',s):
  if m.group() not in {'PantryPlan','PythonFactors'}:compound.append((name,m.group()))
assert not missing,missing
assert not nested,nested
assert not compound,compound
owned=json.loads((base/'results/owned_files.json').read_text())['fragments']
hosted=json.loads((base/'hosted/owned_files.json').read_text())['tex']
fragments={}
concurrent_prose_edits=[]
concurrent_replicate_label_rows=0
protected_pattern=r'\\(?:label|ref|eqref|pageref|autoref|nameref|input|include|includegraphics|cite[A-Za-z]*|url|path|nolinkurl)(?:\[[^\]]*\])*\{[^{}]*\}'
verbatim_pattern=r'\\begin\{(Verbatim|verbatim|lstlisting|minted|alltt)\}.*?\\end\{\1\}'
for name,group in [(n,'results') for n in owned]+[(n,'hosted') for n in hosted]:
 before=(base/group/'before'/name).read_text();after=Path(name).read_text()
 expected=format_domain_names(before,exclude_phrases=exceptions.get(name,()))
 if name.endswith('semantic_current_summary_20260912.tex'):expected=expected.replace(r'\setlength{\tabcolsep}{3pt}',r'\setlength{\tabcolsep}{2.75pt}')
 if after!=expected:
  concurrent_prose_edits.append(name)
  (base/('results/'+Path(name).name+'.concurrent-prose.diff')).write_text(''.join(difflib.unified_diff(expected.splitlines(True),after.splitlines(True))))
 if after==expected:assert re.findall(protected_pattern,before)==re.findall(protected_pattern,after),name
 assert re.findall(verbatim_pattern,before,re.S)==re.findall(verbatim_pattern,after,re.S),name
 def rows(t):
  return [line for line in t.splitlines() if '&' in line and re.search(r'\d',line) and not line.lstrip().startswith('%')]
 old_rows=rows(before);new_rows=rows(after)
 normalized_rows=[format_domain_names(line,exclude_phrases=exceptions.get(name,())) for line in old_rows]
 if normalized_rows!=new_rows:
  assert len(normalized_rows)==len(new_rows),name
  for oldrow,newrow in zip(normalized_rows,new_rows):
   if oldrow==newrow:continue
   a=oldrow.split('&');b=newrow.split('&')
   assert a[1:]==b[1:],(name,oldrow,newrow)
   assert re.sub(r'(?<=Dr\.GRPO) 43|(?<=Re:Dr) 43',' (a)',re.sub(r'(?<=Dr\.GRPO) 46|(?<=Re:Dr) 46',' (b)',a[0]))==b[0],(name,a[0],b[0])
   concurrent_replicate_label_rows+=1
 fragments[name]={'typography_only':after==expected,'tabular_rows_checked':len(old_rows),'protected_identifiers_and_literals_unchanged':re.findall(protected_pattern,before)==re.findall(protected_pattern,after)}
main_before=(base/'main/before.tex').read_text();main=Path('paper/main.tex').read_text()
main_identifiers_unchanged=re.findall(protected_pattern,main_before)==re.findall(protected_pattern,main)
main_literals_unchanged=re.findall(verbatim_pattern,main_before,re.S)==re.findall(verbatim_pattern,main,re.S)
# Rejoin the two line-broken domain headings; all other tabular tokens should
# match after the same typography-only transformation.
expected=format_domain_names(main_before,exclude_phrases=exceptions['paper/main.tex'])
# Main Table 1 already had this split; preserving preexisting splits makes
# the comparison robust to both old and newly split headings.
for a,b in [(r'\shortstack{\texttt{Count}\\\texttt{down}}',r'\texttt{Countdown}'),(r'\shortstack{\texttt{Pantry}\\\texttt{Plan}}',r'\texttt{PantryPlan}')]:
 expected=expected.replace(a,b);main=main.replace(a,b)
main_exact=main==expected
if not main_exact:
 (base/'combined-main-remainder.diff').write_text(''.join(difflib.unified_diff(expected.splitlines(True),main.splitlines(True))))
# Table field content remains invariant even if the root changes whitespace
# to fit the two headers.
getrows=lambda s:[x.strip() for x in s.splitlines() if '&' in x and not x.lstrip().startswith('%')]
main_tables_unchanged=getrows(main)==getrows(expected)
assert format_domain_names('PythonFactors')==r'\texttt{PythonFactors}'
assert format_domain_names(r'\texttt{PythonFactors}')==r'\texttt{PythonFactors}'
report={
 'compiled_tex_inputs':len(inputs),'unformatted_domain_files':missing,'nested_texttt_files':nested,'unhandled_domain_compounds':compound,
 'generic_python_exclusions':{k:list(v) for k,v in exceptions.items()},
 'generic_python_exclusions_semantically_reviewed':True,
 'concurrent_prose_edits_preserved':concurrent_prose_edits,'concurrent_replicate_label_rows_preserved':concurrent_replicate_label_rows,'changed_result_fragments':len(fragments),'result_fragments_with_tabular_rows':sum(bool(v['tabular_rows_checked']) for v in fragments.values()),
 'result_tabular_measured_fields_unchanged_rows':sum(v['tabular_rows_checked'] for v in fragments.values()),'fragments':fragments,
 'main_protected_identifiers_unchanged':main_identifiers_unchanged,'main_literal_blocks_unchanged':main_literals_unchanged,'main_all_table_rows_unchanged':main_tables_unchanged,'main_only_typography_and_two_header_splits':main_exact,
 'helper_sha256':hashlib.sha256(Path('ops/paper_domain_typography.py').read_bytes()).hexdigest(),
 'source_sha256':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in inputs},
 'experiments_or_bootstrap_runs':0}
(base/'combined_source_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('fragments','source_sha256','generic_python_exclusions')},indent=2))
