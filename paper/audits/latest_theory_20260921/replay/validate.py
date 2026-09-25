"""Deterministic mathematical and preservation checks for the replay review.

Checks transcription, exact finite-category identities, and representative
bounds. These checks supplement the proofs; they are not training results.
"""
from pathlib import Path
from collections import Counter
from itertools import product
import hashlib, json, math, re
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parent
before=(ROOT/'before-block.tex').read_text()
after=(ROOT/'replacement.tex').read_text()
counts=Counter(); max_error={}
def verify(group,condition):
    counts[group]+=1
    assert condition, (group,counts[group])
def close(group,a,b,tol=1e-10):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    max_error[group]=max(max_error.get(group,0.),error)
    verify(group,error<=tol)
def softmax(z):
    z=np.asarray(z,dtype=float); e=np.exp(z-z.max()); return e/e.sum()
def kl(a,r):
    return (a*math.log(a/r) if a else 0.)+((1-a)*math.log((1-a)/(1-r)) if a<1 else 0.)
def lower(a,d):
    if d==0:return a
    if a==1:return math.exp(-d)
    log_r=brentq(lambda x:kl(a,math.exp(x))-d,-(d/a+abs(math.log(a))+10),math.log(a),xtol=1e-12)
    return math.exp(log_r)
def labels(s):return re.findall(r'\\label\{([^}]+)\}',s)
def environments(s):return re.findall(r'\\(begin|end)\{([^}]+)\}',s)
def displays(s):
    pattern=r'\\\[.*?\\\]|\\begin\{(equation\*?|align\*?|gather\*?)\}.*?\\end\{\1\}'
    return [m.group(0) for m in re.finditer(pattern,s,re.S)]
verify('preservation',labels(before)==labels(after))
verify('preservation',environments(before)==environments(after))
verify('preservation',displays(before)==displays(after))
formal_names=('theorem','corollary','proposition','lemma','proof','remark')
formal_before=Counter(name for token,name in environments(before) if token=='begin' and name in formal_names)
formal_after=Counter(name for token,name in environments(after) if token=='begin' and name in formal_names)
verify('preservation',formal_before==formal_after)
stack=[]
for token,name in environments(after):
    if token=='begin':stack.append(name)
    else: verify('environment_balance',stack.pop()==name)
verify('environment_balance',not stack)
for phrase in ('post-hoc','post hoc','Insert after','new contribution','frozen policy','frozen bank','admitted banked','reasoning modes','reasoning mode'):
    verify('process_language',phrase.lower() not in after.lower())

# Fresh groups: independently enumerate every binary reward vector.
for G in range(2,9):
    for P in (.001,.01,.1,.25,.5,.75,.9,.99,.999):
        exact=0.
        for rewards in product((0,1),repeat=G):
            r=sum(rewards)
            if 0<r<G:exact+=P**r*(1-P)**(G-r)
        h=1-P**G-(1-P)**G
        close('mixed_group_enumeration',h,exact)
        verify('mixed_group_union_bound',h<=G*min(P,1-P)+1e-14)

# Softmax gradients, conditional allocation, and discrete steps are evaluated
# from the complete probability vector, independently of the reduced formula.
probability_cases=[]
for dimension,correct,k in ((4,3,2),(5,3,3),(8,5,3),(18,16,16)):
    for scale in (0.,.01,.3,1.,3.):
        for shift in (0.,.7):
            z=scale*np.sin(np.arange(dimension)*1.73+shift)
            z[:correct]+=shift
            p=softmax(z); P=p[:correct].sum(); bank=np.arange(k)
            w=np.zeros(dimension); w[bank]=1/k
            M=p[bank].sum(); r=p[bank]/M; C=np.dot(r,r)
            gradP=p*(np.arange(dimension)<correct)-P*p
            probability_cases.append((z,correct,bank))
            for coefficient in (-.4,0.,2/3,2.):
                for margin in (-.25,0.,.1,1.):
                    rho=coefficient*(1-P)+margin
                    if rho<=0:continue
                    zdot=coefficient*gradP+rho*(w-p)
                    pdot=p*(zdot-np.dot(p,zdot))
                    rdot=(pdot[bank]-r*pdot[bank].sum())/M
                    close('uniform_conditional_equation',rdot,-margin*M*r*(r-C))
                    close('bank_ratio_equation',zdot[bank][:,None]-zdot[bank][None,:],-margin*M*(r[:,None]-r[None,:]))
                    diversity_derivative=-2*np.dot(r,rdot)
                    expected=2*margin*M*(np.sum(r**3)-C*C)
                    close('pcmd_derivative',diversity_derivative,expected)
                    if np.ptp(r)>1e-10 and margin!=0:
                        verify('pcmd_derivative_sign',diversity_derivative*margin>0)
                    if margin>0:
                        for gamma in (0.,.01,.1,.5,1.,1.5,2.):
                            eta=gamma/(M*margin)
                            next_z=z+eta*zdot
                            D=np.ptp(z[bank]); Dnext=np.ptp(next_z[bank])
                            rhs=(1-(1-math.exp(-gamma))/k)*math.expm1(D)
                            verify('discrete_contraction',math.expm1(Dnext)<=rhs+1e-11*max(1,rhs))
                            order=np.argsort(z[bank])
                            verify('discrete_order',np.min(np.diff(next_z[bank][order]))>=-1e-12)
            weights=np.arange(1,k+1,dtype=float); weights/=weights.sum()
            w[bank]=weights
            c=.8; rho=.5; a=rho-c*(1-P)
            zdot=c*gradP+rho*(w-p)
            actual=r*(zdot[bank]-np.dot(r,zdot[bank]))
            rhs=r*(rho*(weights-np.dot(r,weights))-a*M*(r-C))
            close('weighted_conditional_equation',actual,rhs)
            for coordinate in range(dimension):
                eps=1e-5
                zp=z.copy(); zm=z.copy(); zp[coordinate]+=eps;zm[coordinate]-=eps
                loss=lambda v:-np.dot(weights,np.log(softmax(v)[bank]))
                fd=(loss(zp)-loss(zm))/(2*eps)
                close('cross_entropy_gradient',fd,p[coordinate]-w[coordinate],3e-10)
            D=np.ptp(z[bank]); rmin=r.min();rmax=r.max()
            verify('recovery_span_inequality',rmax-rmin >= (1-math.exp(-D))/k-1e-14)
            verify('recovery_minimum_inequality',rmin >= 1/(1+(k-1)*math.exp(D))-1e-14)
# The gamma <= 2 restriction matters near a binary tie.
d=.001; r=softmax([d,0.]); next_d=d-2.1*(r[0]-r[1])
verify('discrete_step_size_guard',next_d<0)

# Numerical ODE paths check energy and retention for positive and signed fresh
# coefficients and full/partial coverage. Integrating this finite categorical
# ODE does not sample a model or recompute empirical results.
for k,coefficient in ((2,.7),(3,.7),(2,-.7),(3,-.7),(2,0.)):
    z0=np.array([1.,-.3,-1.,.2]); correct=3;rho=.4; w=np.zeros(4);w[:k]=1/k
    def field(t,z):
        p=softmax(z);P=p[:correct].sum();gradP=p*(np.arange(4)<correct)-P*p
        return coefficient*gradP+rho*(w-p)
    solution=solve_ivp(field,(0,200),z0,rtol=1e-9,atol=1e-11,t_eval=np.linspace(0,200,101))
    verify('ode_integration',solution.success)
    p=np.array([softmax(z) for z in solution.y.T]);P=p[:,:correct].sum(axis=1)
    R=-np.log(p[:,:k]).mean(axis=1);F=rho*R-coefficient*P
    C=R[0]+(max(0.,coefficient)-coefficient*P[0])/rho
    verify('potential_descent',np.max(np.diff(F))<=1e-10)
    verify('energy_retention',np.max(R)<=C+1e-10)
    verify('energy_probability_floor',np.min(p[:,:k])>=math.exp(-k*C)-1e-10)

# Sharp normalized KL certificate: equality constructions and simple floors.
for a in (1/16,.1,.25,.5,.9,1.):
    for d in (0.,.001,.01,.1,1.,4.):
        r=lower(a,d)
        close('binary_kl_inversion',kl(a,r),d,2e-10)
        verify('binary_kl_range',0<r<=a+1e-14)
        verify('binary_kl_pinsker',r>=max(0,a-math.sqrt(d/2))-1e-12)
        if a<1:
            w=np.array([a,(1-a)*.3,(1-a)*.7])
            p=np.array([r,(1-r)*.3,(1-r)*.7])
            H=-np.dot(w,np.log(w));R=-np.dot(w,np.log(p))
            close('sharp_equality_construction',R-H,d,2e-10)
            verify('sharp_vs_termwise',r>=math.exp(-(H+d)/a)-1e-12)
        else:
            close('single_bank_target',r,math.exp(-d))
example_sharp=lower(1/16,.01);example_simple=math.exp(-16*(math.log(16)+.01))
close('displayed_sharp_example',example_sharp,.03397,5e-6)
close('displayed_termwise_example',example_simple/1e-20,4.62,.005)
zero_upper=1-.05**(1/32)
close('displayed_zero_count_example',zero_upper,.0894,.00005)

# Exact occupancy tails on a small alphabet with failure as the null coupon.
def coverage_tails(q,P,K):
    m=len(q);law=np.zeros(1<<m);law[0]=1
    for _ in range(K):
        nxt=(1-P)*law
        for mask in range(1<<m):
            for j in range(m):nxt[mask|(1<<j)]+=law[mask]*P*q[j]
        law=nxt
    return np.array([sum(law[mask] for mask in range(1<<m) if mask.bit_count()>=r) for r in range(1,m+1)])
for m in (2,3,4):
    q_cases=[np.arange(1,m+1,dtype=float),np.array([1.]+[0.]*(m-1)),np.exp(np.arange(m,dtype=float))]
    for P in (.1,.5,1.):
        for K in (1,2,4,8):
            uniform=coverage_tails(np.ones(m)/m,P,K)
            for q in q_cases:
                q=q/q.sum();other=coverage_tails(q,P,K)
                verify('uniform_coverage_tails',np.min(uniform-other)>=-1e-12)
                expected=sum(1-(1-P*qj)**K for qj in q)
                close('occupancy_expectation',other.sum(),expected)

# The displayed stochastic drift estimate is checked on a smooth quadratic
# with a predictable SPD metric and exactly centered finite gradient noise.
for L in (.3,1.,3.):
    for kappa_low,kappa_high in ((.1,.8),(.7,.7),(1.,2.)):
        H=np.diag([kappa_low,kappa_high]);theta=np.array([.3,-.6]);grad=L*theta
        noise=np.array([.4,.7]);sigma2=np.dot(noise,noise)
        for fraction in (.1,.5,1.):
            eta=fraction*kappa_low/(L*kappa_high**2)
            values=[.5*L*np.dot(theta-eta*H@(grad+sign*noise),theta-eta*H@(grad+sign*noise)) for sign in (-1,1)]
            expected=np.mean(values);current=.5*L*np.dot(theta,theta)
            drift=L*eta**2*kappa_high**2*sigma2/2
            verify('stochastic_smooth_drift',expected<=current+drift+1e-14)
            stronger=current-.5*eta*kappa_low*np.dot(grad,grad)+drift
            verify('stochastic_smooth_descent',expected<=stronger+1e-14)
# Finite exact nonnegative-supermartingale tree with a deterministic drift.
N=8;d=np.array([.01*(t+1) for t in range(N)]);D0=.2;initial=D0+d.sum()
paths=[]
for signs in product((.4,1.6),repeat=N):
    D=D0;values=[initial]
    for t,factor in enumerate(signs):
        D=factor*D+d[t];values.append(D+d[t+1:].sum())
    paths.append(max(values))
for delta in (.05,.1,.25,.5,.9):
    tail=np.mean(np.array(paths)>=initial/delta)
    verify('ville_finite_tree',tail<=delta+1e-14)

# Shared-parameter example: gradients from exact categorical probabilities,
# and a Taylor likelihood bound using the global variance bound for slopes.
slopes=np.array([1.,-2.,0.]);B=(slopes.max()-slopes.min())**2/4
for s in (.001,.01,.1,.24):
    p=np.array([s,s,1-2*s]);z=np.log(p)
    h=slopes-np.dot(p,slopes);hbar=h[:2].mean()
    close('neural_counterexample_gradients',h[:2],[1+s,-2+s])
    close('neural_counterexample_mean',hbar,s-.5)
    verify('neural_counterexample_harmed_key',h[0]*hbar<0)
    verify('neural_counterexample_mean_improves',hbar*hbar>0)
    for eta in (.001,.01,.1):
        Delta=eta*hbar;new_p=softmax(z+slopes*Delta)
        for b in (0,1):
            margin=h[b]*Delta-B*Delta**2/2
            actual=math.log(new_p[b]/p[b])
            verify('neural_taylor_lower_bound',actual>=margin-1e-13)
            verify('neural_probability_lower_bound',new_p[b]>=p[b]*math.exp(margin)-1e-14)

result={
 'passed':True,
 'scope':'Five consecutive replay-related subsections, from gradient availability through neural replay response.',
 'before_sha256':hashlib.sha256(before.encode()).hexdigest(),
 'replacement_sha256':hashlib.sha256(after.encode()).hexdigest(),
 'labels_preserved':len(labels(before)),
 'display_math_blocks_preserved_byte_for_byte':len(displays(before)),
 'formal_blocks_preserved':dict(formal_before),
 'assertions':sum(counts.values()),
 'checks_by_category':dict(counts),
 'maximum_absolute_identity_residuals':max_error,
 'illustrative_constants':{'sharp_binary_kl_floor_k16_D001':example_sharp,'elementary_floor_k16_D001':example_simple,'zero_sightings_32_one_sided_95_upper':zero_upper},
 'limitations':'Deterministic finite-category and analytic-example checks supplement the proofs; no neural experiment, model call, bootstrap, or empirical table recomputation was run.'
}
(ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
