from pathlib import Path
import copy,importlib,json,sys,time
ROOT=Path('/n/fs/similarity/maxent-grpo');OUT=ROOT/'paper/figures';BACK=Path('/tmp/paper-domain-typography-20260921/before/paper/figures')
for p in [ROOT,ROOT/'ops',ROOT/'ops/exp_scaling']:sys.path.insert(0,str(p))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_domain_figure_typography import apply_domain_typography

def load(n):return json.loads((BACK/(n+'.json')).read_text())
def cli(module,args):
 sys.argv=[module,*args];importlib.import_module(module).main()
def main():
 group=sys.argv[1]
 if group=='retained':
  m=importlib.import_module('plot_paper_baseline_collapse_precheck');payload=json.loads(m.DEFAULT_INPUT.read_text());m.render(payload,OUT/'baseline_collapse_precheck')
  m=importlib.import_module('plot_paper_aligned_domain_strips');m.render_ucpo(snapshot=load('direct_baseline_learning_curves_pass8'),metric='pass8',output=OUT/'direct_baseline_learning_curves_pass8')
  m=importlib.import_module('plot_paper_e118_all_scale_progress');r=load('e118_scale_extensions_appendix');m.render_appendix_figure(r,r['absolute_cross_domain_average'])
 elif group=='concentration':
  cli('plot_paper_concentration_levels',['--height','6.6','--xlow','-90','--xhigh','40'])
  cli('plot_paper_concentration_levels',['--output',str(OUT/'concentration_story_resampled'),'--scale','qwen3b','--level','level1','--width','3.15','--xlow','-80','--xhigh','40'])
  cli('plot_paper_concentration_story',['--output',str(OUT/'concentration_story_all_scales'),'--scales','all','--height','6','--xlow','-130','--xhigh','40'])
 elif group=='curves':
  m=importlib.import_module('plot_paper_training_curves')
  for name in ['factorial_training_curves_pass8','level2_factorial_training_curves']:
   old=load(name);source=ROOT/old['source_snapshot'];payload=json.loads(source.read_text())
   level='level2' if name.startswith('level2') else 'level1';metric=None if level=='level2' else 'pass8'
   fig,audit=m.build_figure(payload,level=level,metric=metric);apply_domain_typography(fig)
   fig.savefig((OUT/name).with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None});fig.savefig((OUT/name).with_suffix('.png'),dpi=220);plt.close(fig)
   for k in audit:
    if k in old and old[k]!=audit[k]:raise AssertionError((name,k,'measurement metadata changed'))
  cli('plot_paper_mode_diversity_curves',[])
  cli('plot_paper_mode_diversity_curves',['--level','level2','--output',str(OUT/'level2_training_curves_pmd')])
 elif group=='levels':cli('plot_paper_mode_diversity_levels',[])
 elif group=='standard':
  for mod in ['plot_paper_decoding_objection','plot_paper_reference_kl_knee','plot_paper_replay_bank_decomposition','plot_paper_reorganized_results']:
   cli(mod,[])
 elif group=='examples':importlib.import_module('plot_paper_modebench_examples').render()
 elif group=='withdrawal':
  m=importlib.import_module('build_paper_portfolio_withdrawals');r=json.loads((ROOT/'paper/results/portfolio_withdrawals_20260917.json').read_text())
  print('withdrawal keys',list(r))
  result=r['results'] if 'results' in r else r
  print(m.render_agreement_figure(result,OUT/'withdrawal_pmd_agreement'))
 else:raise ValueError(group)
 print('DONE',group,flush=True)
if __name__=='__main__':main()
