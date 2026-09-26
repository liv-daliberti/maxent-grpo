"""Independent semantic checks and a readable presentation of completed followups."""
from pathlib import Path
import ast,itertools,json,sys
from fractions import Fraction
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'ops'))
from followup_metrics import atomic_new,file_sha,coarse_key,portfolio
BASE=ROOT/'artifacts/modebench_inference_followups_20260911/offline'


def evaluate(node):
    if isinstance(node,ast.Constant):return Fraction(node.value),Counter([node.value])
    args=[evaluate(n) for n in node.args];v=[x[0] for x in args];leaves=sum((x[1] for x in args),Counter());op=node.func.id
    value={'add':lambda:sum(v),'mul':lambda:v[0]*v[1],'sub':lambda:v[0]-v[1],'div':lambda:v[0]/v[1],'neg':lambda:-v[0]}[op]()
    return value,leaves


def evaluate_coarse(node):
    op=node[0]
    if op=='num':return Fraction(node[1]),Counter([node[1]])
    args=[evaluate_coarse(n) for n in node[1:]];v=[x[0] for x in args];leaves=sum((x[1] for x in args),Counter())
    from functools import reduce
    value={'add':lambda:sum(v),'mul':lambda:reduce(lambda a,b:a*b,v,Fraction(1)),'sub':lambda:v[0]-v[1],'div':lambda:v[0]/v[1],'neg':lambda:-v[0]}[op]()
    return value,leaves


def main():
    r=json.loads((BASE/'results.json').read_text());cache=json.loads((BASE/'authenticated_keys.json').read_text());assert file_sha(BASE/'authenticated_keys.json')==r['cache_sha256'] and file_sha(BASE/'per_prompt.jsonl')==r['per_prompt_sha256']
    freeze=json.loads((BASE/'implementation_freeze.json').read_text())
    for p,h in freeze['code_sha256'].items():assert file_sha(p)==h
    rows=cache['rows'];seen=set();totals=Counter();merged=defaultdict(list)
    for model,gradings in cache['models'].items():
        for grading,pools in gradings.items():
            for i,(row,keys) in enumerate(zip(rows,pools)):
                spec=json.loads(row['answer']);fine={k for k in keys if k is not None};coarse={coarse_key(k,spec) for k in fine}
                merged[grading,model,row['level'],row['domain']].append({'fine':len(fine),'coarse':len(coarse)})
                for key in fine:
                    identity=(i,key)
                    if identity in seen:continue
                    seen.add(identity);new=coarse_key(key,spec)
                    if row['domain']=='countdown':
                        a=evaluate(ast.parse(key.split(':',1)[1],mode='eval').body);b=evaluate_coarse(json.loads(new.split(':',1)[1]));assert a==b
                    elif row['domain']=='graph_coloring':
                        vector=list(map(int,new.split(':',1)[1]));assert all(c is None or c==v for c,v in zip(spec['partial_colors'],vector));assert all(vector[u-1]!=vector[v-1] for u,v in spec['edges'])
                    elif row['domain']=='python_factors':
                        pairs=[list(map(int,p.split('*'))) for p in new.split(':',1)[1].split(',')]
                        assert all(a*b==n and 1<a<=b<n for n,(a,b) in zip(spec['cases'],pairs))
                    else:assert new==key
                    totals[row['domain']]+=1
    # Directly enumerate all 4+4 subsets for a deterministic, outcome-blind sample of prompt/model pairs.
    tests=0;names=list(cache['models']);indices=np.random.default_rng(177).choice(len(rows),size=12,replace=False)
    for ma,mb in itertools.combinations(names,2):
        for i in indices:
            a=cache['models'][ma]['strict'][i];b=cache['models'][mb]['strict'][i];ds=[];ps=[]
            for ca,cb in itertools.product(itertools.combinations(range(8),4),repeat=2):
                vals={a[j] for j in ca}|{b[j] for j in cb};vals.discard(None);ds.append(len(vals));ps.append(bool(vals))
            got=portfolio([a,b],[4,4]);assert abs(got['distinct8']-sum(ds)/len(ds))<1e-12 and abs(got['pass8']-sum(ps)/len(ps))<1e-12;tests+=1
    merge_stats=[]
    for (grading,model,l,d),vals in merged.items():
        n=sum(v['fine'] for v in vals);c=sum(v['coarse'] for v in vals)
        merge_stats.append({'grading':grading,'model':model,'level':l,'domain':d,'observed_prompt_mode_keys_fine':n,'observed_prompt_mode_keys_coarse':c,'fraction_merged':(n-c)/n if n else None,'prompts_changed':sum(v['fine']!=v['coarse'] for v in vals)})
    audit={'status':'pass','result_sha256':file_sha(BASE/'results.json'),'audit_code_sha256':file_sha(__file__),'unique_prompt_key_checks':dict(totals),'exact_4900_subset_checks':tests,'coarse_semantics_and_operand_multiplicity_preserved':True,'merged':merge_stats}
    atomic_new(BASE/'independent_audit.json',audit)
    p=BASE/'REPORT.md';text=p.read_text()
    for pair in r['cross_model']['strict']['fine']:text=text.replace('| '+pair+' |','| '+pair.replace(' | ',' + ')+' |')
    text+='\nIndependent audit: all unique successful prompt/key combinations preserve the stipulated task equivalences; 252 deterministic pair/prompt checks exactly match enumeration of all 4,900 possible 4+4 subsets.\n'
    p.write_text(text)
    # Both constituents are displayed explicitly; no best-pair selection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs=list(r['cross_model']['strict']['fine']);names=r['models'];short={'DeepSeek-V4-Pro':'DeepSeek','FW-Kimi-K3':'Kimi','claude-opus-4-8':'Opus 4.8','claude-opus-5':'Opus 5','gpt-5.4':'GPT-5.4','gpt-5.6-sol':'GPT-5.6','grok-4.3':'Grok'}
    fig,axs=plt.subplots(1,2,figsize=(12,8.5),sharey=True)
    for ax,keydef,title in zip(axs,('fine','coarse'),('Original outcome keys','Coarser outcome keys')):
        for j,(metric,label,color,offset) in enumerate([('mix_minus_a_distinct8','vs first model','#146c94',-.16),('mix_minus_b_distinct8','vs second model','#d97732',.16)]):
            vv=[r['cross_model']['strict'][keydef][p]['all'][metric] for p in pairs];y=np.arange(len(pairs))+offset;x=np.array([v['estimate'] for v in vv]);lo=np.array([v['ci95'][0] for v in vv]);hi=np.array([v['ci95'][1] for v in vv]);ax.errorbar(x,y,xerr=[x-lo,hi-x],fmt='o',markersize=4,color=color,label=label,elinewidth=1,capsize=2)
        ax.axvline(0,color='0.5',lw=.8);ax.set_title(title);ax.set_xlabel('Additional distinct correct modes in 4+4 portfolio');ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
    axs[0].set_yticks(range(len(pairs)),[short[a]+' + '+short[b] for a,b in (p.split(' | ') for p in pairs)],fontsize=8);axs[0].invert_yaxis();axs[1].legend(loc='lower right',fontsize=9)
    fig.suptitle('Mixed-model portfolios: every pair compared with eight responses from each constituent',fontsize=12);fig.tight_layout();fig.savefig(BASE/'mixed_portfolios.pdf',bbox_inches='tight');fig.savefig(BASE/'mixed_portfolios.png',dpi=170,bbox_inches='tight');plt.close(fig)
    print(json.dumps({k:v for k,v in audit.items() if k!='merged'}))

if __name__=='__main__':main()
