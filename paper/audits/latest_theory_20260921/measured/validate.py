from pathlib import Path
import ast, contextlib, hashlib, io, json, math, re, statistics, sys
import numpy as np
root=Path('/n/fs/similarity/maxent-grpo')
audit=Path('/tmp/paper-latest-theory-20260921/measured')
sys.path.insert(0,str(root/'ops'))
import paper_style as style
import matplotlib.pyplot as plt
checks=[]
def check(name, value):
 assert value, name
 checks.append(name)
p=json.loads((root/'paper/results/reference_kl_comparison.json').read_text())
for short,live in [('comparison.json','reference_kl_comparison.json'),('macros.tex','reference_kl_macros.tex'),('table.tex','reference_kl_table_body.tex')]:
 check('latest_'+short+'_unchanged',(audit/('before_'+short)).read_bytes()==(root/'paper/results'/live).read_bytes())
# Only the builder's module docstring and comments changed.
def tree(text):
 t=ast.parse(text)
 if isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str): t.body=t.body[1:]
 return ast.dump(t,include_attributes=False)
check('builder_algorithm_AST_unchanged',tree((audit/'before_build.py').read_text())==tree((root/'ops/build_reference_kl_comparison.py').read_text()))

def capture(text,actual):
 result=[]
 def save(fig,*args,**kwargs):
  fig.canvas.draw()
  result.append({'size':fig.get_size_inches().tolist(),'axes':[{
   'position':ax.get_position().bounds,
   'xlim':ax.get_xlim(),'ylim':ax.get_ylim(),'xscale':ax.get_xscale(),'yscale':ax.get_yscale(),
   'lines':[{'x':np.asarray(line.get_xdata()).tolist(),'y':np.asarray(line.get_ydata()).tolist(),'color':line.get_color(),'marker':line.get_marker(),'ls':line.get_linestyle()} for line in ax.lines],
   'collections':[[v.vertices.tolist() for v in coll.get_paths()] for coll in ax.collections],
   'patches':[patch.get_path().vertices.tolist() for patch in ax.patches],
   'annotations':[(ann.xy,ann.xyann) for ann in ax.texts if hasattr(ann,'xy')]
  } for ax in fig.axes]})
  plt.close(fig)
 original=style.save;style.save=save
 ns={'__file__':str(actual),'__name__':'coordinate_audit'}
 try:
  with contextlib.redirect_stdout(io.StringIO()):
   exec(compile(text,str(actual),'exec'),ns)
   ns['main']()
 finally: style.save=original
 return result
for name in ['plane','knee']:
 actual=root/f'ops/plot_paper_reference_kl_{name}.py'
 before=capture((audit/f'before_{name}.py').read_text(),actual)
 after=capture(actual.read_text(),actual)
 check(name+'_all_plot_coordinates_and_geometry_unchanged',before==after)
 (audit/f'{name}_coordinates.json').write_text(json.dumps(after,indent=2)+'\n')
# Check each numerical table field against the latest JSON rather than an old PDF.
labels={'Graph':'graph_coloring','Countdown':'countdown','Python':'python_factors','MathIR':'mathir','Pantry':'pantry_plan'}
rows=p['domains'];betas=p['betas'];seen=[]
def num(v,n=3):return f'{v:.{n}f}' if v is not None else '---'
for line in (root/'paper/results/reference_kl_table_body.tex').read_text().splitlines():
 if '&' not in line:continue
 cells=[c.strip() for c in line.strip().removesuffix('\\\\').split('&')]
 d=labels[cells[0]];r=rows[d];seen.append(d)
 expected=[num(r['control']['pmd']),num(r['maxrl']['pmd'])]
 for b in betas:
  v=r['kl'].get(str(b));expected.append(num(v['pmd'] if v else None)+(f"$^{{{v['seeds']}}}$" if v and v['seeds']<5 else ''))
 expected += [num(r['replay']['pmd']),num(r['remax']['pmd']),num(r['frozen_pmd'],2),num(r['replay_ceiling'],2),num(r['remax_ceiling'],2),num(r['beta_star'],2)]
 check(d+'_all_14_numerical_table_entries',cells[1:]==expected)
 check(d+'_occupancy_transforms',math.isclose(r['replay_ceiling'],1-1/r['bank_occupancy']) and math.isclose(r['remax_ceiling'],1-1/r['remax_bank_occupancy']))
 if r['frozen_correctness']:
  check(d+'_beta_star_formula',math.isclose(r['beta_star'],.9375/-math.log(r['frozen_correctness']/(1-r['frozen_correctness']))))
check('all_five_domains_present',set(seen)==set(rows))
allcells=[(d,b,e) for d,r in rows.items() for b,e in r['kl'].items()]
check('six_latest_coefficients',betas==[.001,.01,.04,.1,.2,.3])
check('27_coefficient_cells',len(allcells)==27)
check('two_reduced_seed_cells',[(d,b,e['seeds']) for d,b,e in allcells if e['seeds']<5]==[('mathir','0.3',4),('python_factors','0.1',3)])
check('three_missing_cells',[(d,str(b)) for d,r in rows.items() for b in betas if str(b) not in r['kl']]==[('pantry_plan','0.3'),('python_factors','0.2'),('python_factors','0.3')])
check('Graph_MathIR_decrease_beta_point2_to_point3',all(rows[d]['kl']['0.3']['pmd']<rows[d]['kl']['0.2']['pmd'] for d in ['graph_coloring','mathir']))
check('Python_decreases_at_point1',rows['python_factors']['kl']['0.1']['pmd']<rows['python_factors']['kl']['0.04']['pmd'])
for d in ['python_factors','pantry_plan']:
 r=rows[d];check(d+'_some_KL_cells_no_lower_than_both_replay_arms',any(all(e['pmd']>r[a]['pmd'] and e['pass8']>=r[a]['pass8'] for a in ['replay','remax']) for e in r['kl'].values()))
for d in ['countdown','mathir']:
 r=rows[d];over=[e for e in r['kl'].values() if all(e['pmd']>r[a]['pmd'] for a in ['replay','remax'])]
 check(d+'_every_larger_PCMD_has_lower_pass8_than_both',over and all(all(e['pass8']<r[a]['pass8'] for a in ['replay','remax']) for e in over))
r=rows['graph_coloring'];check('graph_every_KL_cell_below_both_replay_arms',all(all(e[k]<r[a][k] for a in ['replay','remax'] for k in ['pmd','pass8']) for e in r['kl'].values()))
check('Countdown_occupancy_does_not_order_PCMD',rows['countdown']['remax_bank_occupancy']<rows['countdown']['bank_occupancy'] and rows['countdown']['remax']['pmd']>rows['countdown']['replay']['pmd'])
keys=[d for d,r in rows.items() if r['frozen_pmd'] is not None and r['frozen_defined']>=30]
check('plane_three_domains',keys==['graph_coloring','mathir','pantry_plan'])
common=[b for b in betas if all(str(b) in rows[d]['kl'] for d in keys)]
check('plane_common_coefficients_exclude_point3',common==[.001,.01,.04,.1,.2])
means={'KLpoint04':{k:statistics.fmean(rows[d]['kl']['0.04'][k] for d in keys) for k in ['pmd','pass8']},'ReDr':{k:statistics.fmean(rows[d]['replay'][k] for d in keys) for k in ['pmd','pass8']}}
check('main_paragraph_numbers',[num(means[a][k]) for a,k in [('KLpoint04','pmd'),('ReDr','pmd'),('KLpoint04','pass8'),('ReDr','pass8')]]==['0.432','0.427','0.614','0.846'])
replacement=(audit/'replacement.tex').read_text()
check('15_column_tabular','@{}l'+'r'*14+'@{}' in replacement)
check('six_beta_header','\\multicolumn{6}{c}{\\textbf{$+$ reference KL, by $\\beta$}}' in replacement)
check('latest_beta_heading','& $.2$ & $.3$' in replacement)
check('replacement_removes_process_claims',not re.search(r'withdraw|post.hoc|what survives|what remains of|still filling|will carry|not part of the frozen|what would change this reading',replacement,re.I))
check('zero_initial_success_not_zero_population','does not establish zero population correctness' in replacement)
check('compute_limitation_explicit','do not equalize total computation' in replacement)
check('no_numbered_equation_changed','\\begin{equation}' not in replacement)
result={'checks':len(checks),'passed':checks,'latest_coefficient_cells':len(allcells),'betas':betas,'measured_seed_runs':sum(e['seeds'] for _,_,e in allcells),'main_means':means,'data_sha256':hashlib.sha256((root/'paper/results/reference_kl_comparison.json').read_bytes()).hexdigest()}
(audit/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
