"""Deterministic exact-arithmetic checks for Appendix P.4; no experiments."""
from fractions import Fraction as F
from itertools import product
from math import comb, prod, sqrt
from pathlib import Path
import hashlib, json, re

out = Path(__file__).parent
checks = 0

def check(condition, label):
    global checks
    assert condition, label
    checks += 1

def vecsum(vs):
    return tuple(sum(v[i] for v in vs) for i in range(2))

def scale(a, v):
    return tuple(a*x for x in v)

def outer(v, w):
    return tuple(v[i]*w[j] for i in range(2) for j in range(2))

def matsum(ms):
    return tuple(sum(m[i] for m in ms) for i in range(4))

def mat_scale(a, m):
    return tuple(a*x for x in m)

# Shared log-linear neural parametrization: scores f_y-E_pi f_y.
p = (F(1,10), F(2,10), F(3,10), F(4,10))
features = ((F(2),F(-1)), (F(-3),F(2)), (F(1),F(4)), (F(0),F(-2)))
fbar = vecsum(scale(w,f) for w,f in zip(p,features))
scores = tuple(tuple(f[i]-fbar[i] for i in range(2)) for f in features)
rewards = (1,1,0,0)
P = p[0]+p[1]
v = vecsum(scale(p[y],scores[y]) for y in (0,1))
mu = {1: scale(1/P,v), 0: scale(-1/(1-P),v)}
cov = {}
for r in (0,1):
    mass = P if r else 1-P
    rows = [y for y in range(4) if rewards[y] == r]
    actual = vecsum(scale(p[y]/mass,scores[y]) for y in rows)
    check(actual == mu[r], f'conditional-score identity {r}')
    cov[r] = matsum(mat_scale(p[y]/mass,outer(tuple(scores[y][i]-mu[r][i] for i in range(2)),tuple(scores[y][i]-mu[r][i] for i in range(2)))) for y in rows)

cases=[]
for G in (2,3,4):
    advantages={
        'constant':lambda r,R:F(1),
        'reward':lambda r,R:F(r),
        'negative_reward':lambda r,R:F(-r),
        'centered':lambda r,R:F(r)-F(R,G),
        'maxrl':lambda r,R: (F(G,R)-1 if r else F(-1)) if R else F(0),
        'arbitrary_count':lambda r,R:F((2*r-1)*(R+1)*(R+2)+3*r,G+2),
    }
    rows=[]
    for ys in product(range(4),repeat=G):
        pr=prod(p[y] for y in ys)
        R=sum(rewards[y] for y in ys)
        rows.append((ys,pr,R))
    for name,a in advantages.items():
        gs=[(pr,vecsum(scale(a(rewards[y],R)/G,scores[y]) for y in ys)) for ys,pr,R in rows]
        actual_mean=vecsum(scale(pr,g) for pr,g in gs)
        actual_cov=matsum(mat_scale(pr,outer(tuple(g[i]-actual_mean[i] for i in range(2)),tuple(g[i]-actual_mean[i] for i in range(2)))) for pr,g in gs)
        prob=lambda R:F(comb(G,R))*P**R*(1-P)**(G-R)
        beta=lambda R:(R*a(1,R)/P-(G-R)*a(0,R)/(1-P))/G
        c=sum(prob(R)*beta(R) for R in range(G+1))
        beta_var=sum(prob(R)*(beta(R)-c)**2 for R in range(G+1))
        expected_cov=matsum([mat_scale(beta_var,outer(v,v)),matsum(mat_scale(prob(R)/G**2,matsum([mat_scale(R*a(1,R)**2,cov[1]),mat_scale((G-R)*a(0,R)**2,cov[0])])) for R in range(G+1))])
        check(actual_mean==scale(c,v), f'mean G={G} {name}')
        check(actual_cov==expected_cov, f'covariance G={G} {name}')
        if c>0:
            # f(theta)=||theta||^2/2 saturates the absolute Taylor bound.
            actual_quadratic=sum(pr*sum(x*x for x in g)/(2*c*c) for pr,g in gs)
            theoretical=(sum(x*x for x in v)+(actual_cov[0]+actual_cov[3])/(c*c))/2
            check(actual_quadratic==theoretical, f'finite-step sharp quadratic G={G} {name}')
        omega=(F(1,3),F(1,5),F(2),F(-1,7))
        weighted_mean=vecsum(scale(pr,vecsum(scale(a(rewards[y],R)*omega[y]/G,scores[y]) for y in ys)) for ys,pr,R in rows)
        A={1:sum(prob(R)*R*a(1,R)/G for R in range(G+1)),0:sum(prob(R)*(G-R)*a(0,R)/G for R in range(G+1))}
        wbar={}
        K={}
        svar={}
        for r in (0,1):
            mass=P if r else 1-P
            ids=[y for y in range(4) if rewards[y]==r]
            wbar[r]=sum(p[y]*omega[y]/mass for y in ids)
            K[r]=vecsum(scale(p[y]*(omega[y]-wbar[r])/mass,tuple(scores[y][i]-mu[r][i] for i in range(2))) for y in ids)
            svar[r]=sum(p[y]*(omega[y]-wbar[r])**2/mass for y in ids)
        ctilde=A[1]*wbar[1]/P-A[0]*wbar[0]/(1-P)
        residual=vecsum(scale(A[r],K[r]) for r in (0,1))
        predicted=vecsum([scale(ctilde,v),residual])
        check(weighted_mean==predicted, f'response weighting residual G={G} {name}')
        bound=sum(float(abs(A[r]))*sqrt(float(svar[r]*(cov[r][0]+cov[r][3]))) for r in (0,1))
        check(sqrt(float(sum(z*z for z in residual)))<=bound+1e-12,f'residual Cauchy bound G={G} {name}')
        cases.append({'G':G,'advantage':name,'enumerated_groups':len(rows),'c':str(c)})

# Two-prompt example; derivatives of logistic at sigma(v)=3/4 are rational.
PA,PB=F(3,4),F(1,4)
gA,gB=(F(3,16),F(3,16)),(F(3,16),F(-3,16))
dr=vecsum([scale(F(1,2)*F(2,3),gA),scale(F(1,2)*F(2,3),gB)])
mx=vecsum([scale(F(1,2)*(2-PA),gA),scale(F(1,2)*(2-PB),gB)])
check(dr==(F(1,8),F(0)),'shared-prompt Dr flow')
check(mx==(F(9,32),F(-3,64)),'shared-prompt Max flow')
Dprime=2*F(3,4)*F(1,4)*(1-2*F(3,4))
check(Dprime*dr[1]==0,'Dr disagreement derivative')
check(Dprime*mx[1]==F(9,1024),'Max disagreement derivative')
for flow in (dr,mx):
    for pg in (gA,gB):
        check(sum(x*y for x,y in zip(flow,pg))>0,'both correctness derivatives positive')

# SetPO influence and gradient; two keys divide the correct response event.
q=(p[0]/P,p[1]/P)
S2=sum(x*x for x in q)
influence=tuple(2*(S2-x) for x in q)
check(sum(q[i]*influence[i] for i in range(2))==0,'influence mean zero')
for key in range(2):
    d_q=tuple(F(i==key)-q[i] for i in range(2))
    direct=-2*sum(q[i]*d_q[i] for i in range(2))
    check(direct==influence[key],f'mixture influence key={key}')
grad_direct=vecsum(scale(-2*q[y],tuple((p[y]*scores[y][i]*P-p[y]*v[i])/(P*P) for i in range(2))) for y in (0,1))
grad_influence=vecsum(scale(q[y]*influence[y],scores[y]) for y in (0,1))
check(grad_direct==grad_influence,'neural verified-key gradient')

before=(out/'before.tex').read_text()
after=(out/'replacement.tex').read_text()
pattern=r'\\\[.*?\\\]|\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}'
check(re.findall(pattern,before,re.S)==re.findall(pattern,after,re.S),'all displayed equations unchanged')
check(re.findall(r'\\label\{[^}]+\}',before)==re.findall(r'\\label\{[^}]+\}',after),'all labels unchanged')
check(re.findall(r'\\cite\w+(?:\[[^]]*\])?\{[^}]+\}',before)==re.findall(r'\\cite\w+(?:\[[^]]*\])?\{[^}]+\}',after),'all citations and version locators unchanged')
check(not re.search(r'(?i)post.?hoc|this subsection replaces|extension here|reproduc|registered|admitted|retained evidence|reasoning modes?',after),'target process and terminology scan')
report={'checks_passed':checks,'exact_arithmetic_group_cases':cases,'all_displayed_equations_unchanged':True,'all_labels_unchanged':True,'all_citations_unchanged':True,'experiments_or_model_calls':False,'block_sha256':hashlib.sha256(after.encode()).hexdigest()}
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks_passed':checks,'enumerated_group_cases':len(cases),'block_sha256':report['block_sha256']}))
