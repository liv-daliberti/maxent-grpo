from fractions import Fraction as F
from itertools import product
from math import comb
from pathlib import Path
import json, re, hashlib
base=Path('/tmp/paper-appendix-eighth10-20260921')
out=base/'binary'
source=(base/'before.tex').read_text()
start=source.index(r'\subsection{Conditional trajectories of outcome-binary mean flows}')
end=source.index(r'\subsection{Neural score identities and the limits of conditional inertness}',start)
before=source[start:end]
after=(out/'replacement.tex').read_text()
(out/'before-block.tex').write_text(before)
counts={}
def check(group, condition):
    assert condition, group
    counts[group]=counts.get(group,0)+1
labels=lambda t: re.findall(r'\\label\{([^}]+)\}',t)
check('structure',labels(before)==labels(after))
check('structure',re.findall(r'\\cite\w*(?:\[[^\]]*\])*\{([^}]+)\}',before)==re.findall(r'\\cite\w*(?:\[[^\]]*\])*\{([^}]+)\}',after))
for env in ['proof','corollary','remark','equation']:
    check('structure',after.count('\\begin{'+env+'}')==after.count('\\end{'+env+'}')==before.count('\\begin{'+env+'}'))
check('structure',not re.search(r'(?im)^\s*%|post[- ]?hoc|the question|one line of conditioning|by inspection|never about|does buy|fragment|Insert after',after))

def coefficients(G,P,a):
    expectation=sum(F(comb(G,r))*P**r*(1-P)**(G-r)*(F(r)*a(1,r)/P-F(G-r)*a(0,r)/(1-P))/G for r in range(G+1))
    bernstein=sum(F(comb(G-1,j))*P**j*(1-P)**(G-1-j)*(a(1,j+1)-a(0,j)) for j in range(G))
    return expectation,bernstein
for G in range(2,33):
    dr=lambda r,R:F(r)-F(R,G)
    mx=lambda r,R:F(G*r,R)-1 if R else F(0)
    synthetic=lambda r,R:F((R*R+3*r*R+7*r)%19-9,11)
    check('maxrl_endpoints',mx(1,G)==0 and mx(0,0)==0)
    for j in range(G):
        check('drgrpo_bernstein_gaps',dr(1,j+1)-dr(0,j)==F(G-1,G))
        check('maxrl_bernstein_gaps',mx(1,j+1)-mx(0,j)==(F(G-1) if j==0 else F(G,j+1)))
    prior=None
    for P in [F(k,20) for k in range(1,20)]:
        for name,a in [('drgrpo',dr),('maxrl',mx),('general_reward_map',synthetic)]:
            e,b=coefficients(G,P,a)
            check(name+'_binomial_identity',e==b)
            if name=='drgrpo':check('drgrpo_closed_form',e==F(G-1,G))
            if name=='maxrl':check('maxrl_closed_form',e==(1-(1-P)**(G-1))/P==sum((1-P)**j for j in range(G-1)))
        ratio=(1-(1-P)**(G-1))/P/F(G-1,G)
        if prior is not None:
            check('maxrl_ratio_monotonicity',ratio==prior==2 if G==2 else ratio<prior)
        prior=ratio
    check('ratio_endpoints',F(G,G-1)*(G-1)==G and F(G,G-1)>=1)

# Direct exact expectation over finite categorical groups, including all-equal groups.
p=[F(1,10),F(2,10),F(3,10),F(4,10)]
P=sum(p[:2]); q=[x/P for x in p[:2]]
grad=[p[k]*(int(k<2)-P) for k in range(4)]
for G in range(2,6):
    estimators={'drgrpo':lambda r,R:F(r)-F(R,G),
                'maxrl':lambda r,R:F(G*r,R)-1 if R else F(0),
                'general_reward_map':lambda r,R:F((R*R+3*r*R+7*r)%19-9,11)}
    for name,a in estimators.items():
        eg=[F(0) for _ in p]
        for group in product(range(4),repeat=G):
            prob=F(1)
            for z in group:prob*=p[z]
            R=sum(z<2 for z in group)
            for z in group:
                for k in range(4):eg[k]+=prob*a(int(z<2),R)*(int(z==k)-p[k])/G
        c,_=coefficients(G,P,a)
        for k in range(4):check('direct_categorical_mean',eg[k]==c*grad[k])
        dz=[c*x for x in grad]
        if c:
            check('conditional_log_ratio', (dz[0]-dz[1])/(c*P*(1-P))==q[0]-q[1])

for G in range(2,33):
    for P in [F(k,20) for k in range(1,20)]:
        h=1-P**G-(1-P)**G
        check('mixed_group_probability',0<h<=G*min(P,1-P))
        check('mixed_group_symmetry',h==1-(1-P)**G-P**G)

summary={
    'scope':'P.3: conditional trajectories of outcome-binary mean flows; no build and no data/model calls',
    'checks':counts,'total_checks':sum(counts.values()),
    'maxrl_last_bernstein_gap':'Unchanged and verified: a(1,G)=0; a(0,G-1)=-1; d_(G-1)=1.',
    'substantive_corrections':['G=2 coefficient ratio is constant at 2; strict decrease requires G>2.', 'Fresh mixed-reward groups become rare near both P=0 and P=1.'],
    'preserved_labels':labels(after),
    'sha256':{'before_block':hashlib.sha256(before.encode()).hexdigest(),'replacement':hashlib.sha256(after.encode()).hexdigest()},
}
(out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
