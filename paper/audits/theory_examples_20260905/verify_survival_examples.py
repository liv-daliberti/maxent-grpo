"""Verify hypothetical replay/survival examples; no training or cluster access."""
from pathlib import Path
import hashlib
import json
import math
import re

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]

def kl(a, r):
    return a * math.log(a/r) + (1-a)*math.log((1-a)/(1-r))

def lower_kl(a, d):
    if d == 0: return a
    lo, hi = 1e-300, a
    for _ in range(250):
        mid=(lo+hi)/2
        if kl(a,mid)>d: lo=mid
        else: hi=mid
    return hi

def binom_ge(n, x, p):
    return sum(math.comb(n,k)*p**k*(1-p)**(n-k) for k in range(x,n+1))

def binom_lower(n,x,alpha):
    lo,hi=0.,1.
    for _ in range(150):
        mid=(lo+hi)/2
        if binom_ge(n,x,mid)<alpha:lo=mid
        else:hi=mid
    return hi

def formal_blocks(s):
    return re.findall(r'\\begin\{(theorem|lemma|corollary|proof|remark)\}(.*?)\\end\{\1\}',s,re.S)

def extract(s,label):
    pos=s.index('\\label{'+label+'}')
    start=s.rfind('\\subsection{',0,pos)
    next_section=re.search(r'\n\\(?:subsection|section)\{',s[pos:])
    end=pos+next_section.start()+1 if next_section else len(s)
    return s[start:end]

def calculations():
    p=[.30,.15,.05,.50]; bank=[0,2]; rho=.3; c=.75; correct=.5
    r=-sum(math.log(p[b]) for b in bank)/2
    cap=r+c*(1-correct)/rho
    gradient=[rho*(v-(.5 if i in bank else 0))-c*v*((1 if i<3 else 0)-correct) for i,v in enumerate(p)]
    new=[v*math.exp(-g) for v,g in zip(p,gradient)]; new=[v/sum(new) for v in new]
    potential=lambda v:rho*(-sum(math.log(v[b]) for b in bank)/2)-c*sum(v[:3])
    entropy=lambda w:-sum(v*math.log(v) for v in w if v)
    w=[.25,.25,.5]; ep=[.2,.2,.4]
    cr=-sum(a*math.log(b) for a,b in zip(w,ep)); d=cr-entropy(w)
    out={'all_values_hypothetical':True,
      'bank_barrier':{'initial_p':p,'bank_indices':bank,'G':4,'rho':rho,'R_initial':r,'C':cap,'termwise_floor':math.exp(-2*cap),'rare_p_candidate':1e-4,'candidate_single_term':-math.log(1e-4)/2},
      'inverse_lengths':{'lengths':[10,20,40],'A':sum(1/n for n in [10,20,40])/3,'target':[4/7,2/7,1/7]},
      'deterministic_step':{'L':rho/2+c/2,'eta':1,'eta_upper_exclusive':2/(rho/2+c/2),'gradient':gradient,'p_after':new,'F_before':potential(p),'F_after':potential(new),'incomplete_bank_limit':[.5,0,.5,0]},
      'sharp_16':{'C':math.log(16)+.01,'D':.01,'sharp_floor':lower_kl(1/16,.01),'old_floor':math.exp(-16*(math.log(16)+.01)),'bank_mass_floor':math.exp(-.01)},
      'entropy_missing':{'m':16,'H':math.log(15),'normalized_entropy':math.log(15)/math.log(16),'forward_KL':math.log(16/15),'zero_coordinate':0},
      'response_aggregation':{'target':w,'response_p':ep,'other_invalid_p':.2,'H':entropy(w),'C':cr,'D':d,'key_A_p':.4,'key_A_target':.5,'key_A_floor':lower_kl(.5,d),'individual_response_floor':lower_kl(.25,d),'sum_individual_A_floors':2*lower_kl(.25,d)},
      'binomial':{'N':32,'zero_upper95':1-.05**(1/32),'zero_upper95_16keys':1-(.05/16)**(1/32),'two_lower95':binom_lower(32,2,.05),'two_plugin':2/32,'zero_upper95_160statements':1-(.05/160)**(1/32),'alpha_160':.05/160},
      'missing_mass':{'single_unseen_p':.1,'many_unseen_count':100,'each_unseen_p':.001}}
    assert out['bank_barrier']['candidate_single_term']>cap
    assert potential(new)<potential(p)
    assert all(new[b]>=out['bank_barrier']['termwise_floor'] for b in bank)
    assert abs(out['inverse_lengths']['A']-7/120)<1e-14
    assert abs(d-math.log(1.25))<1e-14
    assert abs(out['response_aggregation']['key_A_floor']-.2)<1e-14
    assert abs(binom_ge(32,2,out['binomial']['two_lower95'])-.05)<1e-13
    assert abs(out['missing_mass']['many_unseen_count']*out['missing_mass']['each_unseen_p']-.1)<1e-14
    return out

def main():
    out=calculations()
    current=(ROOT/'paper/main.tex').read_text()
    for name,label in [('replay_section.tex','app:theory-replay'),('survival_section.tex','app:theory-survival-certificates')]:
        p=BASE/name
        if p.exists():
            before=extract(current,label);after=p.read_text()
            assert formal_blocks(before)==formal_blocks(after),f'Formal blocks changed: {name}'
            for cite in re.findall(r'\\cite\w*\*?(?:\[[^]]*\])*\{[^}]*\}',before):assert cite in after
            out.setdefault('preservation',{})[name]={'formal_blocks_unchanged':True,'citations_preserved':True,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (BASE/'verify_survival_examples.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
