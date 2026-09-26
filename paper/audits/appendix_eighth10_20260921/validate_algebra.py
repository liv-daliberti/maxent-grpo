#!/usr/bin/env python3
"""Exact small-state checks of the appendix identities; no model or sampling runs.

The finite-step cases corroborate, and do not prove, the stated general theorem.
Requires SymPy. The source snapshot and manifest are saved beside this script.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import re
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COUNTS = Counter()

def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    COUNTS[name] += 1

def exact(name, lhs, rhs):
    check(name, sp.expand(lhs-rhs) == 0 or sp.simplify(lhs-rhs) == 0)

def rational(x):
    x = F(x)
    return sp.Rational(x.numerator, x.denominator)

def zeros(d):
    return [[F(0) for _ in range(d)] for _ in range(d)]

def dot(x,y):
    return sum(a*b for a,b in zip(x,y))

def conditional(p, scores, reward, omega):
    mass = sum(p[i] for i in range(4) if int(i < 2) == reward)
    inds = [i for i in range(4) if int(i < 2) == reward]
    d=len(scores[0])
    mean = [sum(p[i]*scores[i][k] for i in inds)/mass for k in range(d)]
    cov = [[sum(p[i]*(scores[i][k]-mean[k])*(scores[i][l]-mean[l]) for i in inds)/mass for l in range(d)] for k in range(d)]
    om = sum(p[i]*omega[i] for i in inds)/mass
    ov = sum(p[i]*(omega[i]-om)**2 for i in inds)/mass
    K = [sum(p[i]*(omega[i]-om)*(scores[i][k]-mean[k]) for i in inds)/mass for k in range(d)]
    return mean,cov,om,ov,K

def enumerate_groups(p,scores,G,omega):
    """Sufficient moments from every ordered group, using exact rationals."""
    d=len(scores[0]); moments=[]
    for _ in range(G+1):
        moments.append({'mass':F(0), 'first':[[F(0)]*d for _ in range(2)],
                        'weighted':[[F(0)]*d for _ in range(2)],
                        'second':[[zeros(d) for _ in range(2)] for _ in range(2)]})
    for group in product(range(4),repeat=G):
        prob=math.prod(p[i] for i in group); R=sum(i<2 for i in group)
        m=moments[R]; m['mass']+=prob
        sums=[[sum((scores[i][k] for i in group if int(i<2)==r),F(0)) for k in range(d)] for r in range(2)]
        wsums=[[sum((omega[i]*scores[i][k] for i in group if int(i<2)==r),F(0)) for k in range(d)] for r in range(2)]
        for r in range(2):
            for k in range(d):
                m['first'][r][k]+=prob*sums[r][k]
                m['weighted'][r][k]+=prob*wsums[r][k]
            for t in range(2):
                for k in range(d):
                    for l in range(d):
                        m['second'][r][t][k][l]+=prob*sums[r][k]*sums[t][l]
    return moments

def advantage(name,G,r,R):
    if name=='DrGRPO': return sp.Rational(r)-sp.Rational(R,G)
    if name=='MaxRL': return sp.Rational(G*r,R)-1 if R else sp.S.Zero
    if name=='standardized_GRPO':
        return (sp.Rational(r)-sp.Rational(R,G))*G/sp.sqrt(R*(G-R)) if 0<R<G else sp.S.Zero
    if name=='leave_one_out': return sp.Rational(G*r-R,G-1)
    if name=='mixed_sign': return sp.Rational((3*R+2*r+1)%7-3,5)
    raise ValueError(name)

def exact_moments():
    case_count=0; sequence_count=0
    policies=[tuple(map(F,('1/2','1/4','1/6','1/12'))),tuple(map(F,('1/12','1/6','1/4','1/2')))]
    features=[[[F(i==j) for j in range(4)] for i in range(4)],
              [[F(2),F(-1)],[F(-1),F(3)],[F(0),F(2)],[F(3),F(-2)]]]
    omega=list(map(F,('1','1/3','1/2','1/5')))
    for p in policies:
        P=sum(p[:2]); Q=1-P
        for feat in features:
            d=len(feat[0]); avg=[sum(p[i]*feat[i][k] for i in range(4)) for k in range(d)]
            scores=[[feat[i][k]-avg[k] for k in range(d)] for i in range(4)]
            v=[sum(p[i]*scores[i][k] for i in range(2)) for k in range(d)]
            con=[conditional(p,scores,r,omega) for r in range(2)]
            for r in range(2):
                mean,cov,om,ov,K=con[r]
                for k in range(d): exact('conditional_score',mean[k],v[k]/P if r else -v[k]/Q)
                check('vector_Cauchy_Schwarz',dot(K,K)<=ov*sum(cov[k][k] for k in range(d)))
            for G in (2,3,4):
                ms=enumerate_groups(p,scores,G,omega); sequence_count+=4**G
                for R,m in enumerate(ms):
                    exact('enumerated_binomial_mass',m['mass'],F(math.comb(G,R))*P**R*Q**(G-R))
                for name in ('DrGRPO','MaxRL','standardized_GRPO','leave_one_out','mixed_sign'):
                    case_count+=1
                    a=[[advantage(name,G,r,R) for r in range(2)] for R in range(G+1)]
                    beta=[(R*a[R][1]/rational(P)-(G-R)*a[R][0]/rational(Q))/G for R in range(G+1)]
                    c=sum(rational(ms[R]['mass'])*beta[R] for R in range(G+1))
                    varbeta=sum(rational(ms[R]['mass'])*beta[R]**2 for R in range(G+1))-c**2
                    mu=[sum(a[R][r]*rational(ms[R]['first'][r][k])/G for R in range(G+1) for r in range(2)) for k in range(d)]
                    wmu=[sum(a[R][r]*rational(ms[R]['weighted'][r][k])/G for R in range(G+1) for r in range(2)) for k in range(d)]
                    raw=[[sum(a[R][r]*a[R][t]*rational(ms[R]['second'][r][t][k][l])/G**2 for R in range(G+1) for r in range(2) for t in range(2)) for l in range(d)] for k in range(d)]
                    for k in range(d): exact('enumerated_mean',mu[k],c*rational(v[k]))
                    for k in range(d):
                        for l in range(d):
                            predicted=varbeta*rational(v[k]*v[l])+sum(rational(ms[R]['mass'])*(R*a[R][1]**2*rational(con[1][1][k][l])+(G-R)*a[R][0]**2*rational(con[0][1][k][l]))/G**2 for R in range(G+1))
                            exact('enumerated_covariance',raw[k][l]-mu[k]*mu[l],predicted)
                    bern=sum(sp.binomial(G-1,j)*rational(P)**j*rational(Q)**(G-1-j)*(advantage(name,G,1,j+1)-advantage(name,G,0,j)) for j in range(G))
                    exact('Bernstein_coefficient',c,bern)
                    A=[sum(rational(ms[R]['mass'])*(R if r else G-R)*a[R][r]/G for R in range(G+1)) for r in range(2)]
                    ct=A[1]*rational(con[1][2]/P)-A[0]*rational(con[0][2]/Q)
                    residual=[sum(A[r]*rational(con[r][4][k]) for r in range(2)) for k in range(d)]
                    for k in range(d): exact('weighted_residual_identity',wmu[k],ct*rational(v[k])+residual[k])
                    lhs2=sum(x*x for x in residual)
                    rhs=sum(sp.Abs(A[r])*sp.sqrt(rational(con[r][3]*sum(con[r][1][k][k] for k in range(d)))) for r in range(2))
                    check('aggregate_residual_bound',sp.simplify(rhs**2-lhs2).is_nonnegative)
                    if name=='MaxRL': exact('MaxRL_closed_coefficient',c,(1-rational(Q)**(G-1))/rational(P))
                    if name=='DrGRPO': exact('DrGRPO_closed_coefficient',c,sp.Rational(G-1,G))
                    if name=='MaxRL' and G==2: exact('MaxRL_G2_constant',c,1)
    for G in range(2,13):
        x=sp.Symbol('P')
        series=sum((1-x)**j for j in range(G-1))
        exact('MaxRL_lower_endpoint',series.subs(x,0),G-1)
        exact('MaxRL_upper_endpoint',series.subs(x,1),1)
        exact('MaxRL_polynomial_identity',sp.expand(x*series),1-(1-x)**(G-1))
        if G==2:
            exact('MaxRL_G2_ratio',series/sp.Rational(G-1,G),2)
        exact('MaxRL_first_gap',advantage('MaxRL',G,1,1)-advantage('MaxRL',G,0,0),G-1)
        exact('MaxRL_last_gap',advantage('MaxRL',G,1,G)-advantage('MaxRL',G,0,G-1),1)
        for j in range(1,G): exact('MaxRL_interior_gaps',advantage('MaxRL',G,1,j+1)-advantage('MaxRL',G,0,j),sp.Rational(G,j+1))
    return {'exact_enumeration_cases':case_count,'ordered_groups_enumerated':sequence_count,'policy_correctness':['3/4','1/4'],'score_geometries':['four independent logits','two shared parameters with four distinct feature vectors'],'advantages':['DrGRPO','MaxRL','standardized_GRPO (population standard deviation, exact radicals)','leave_one_out','arbitrary mixed_sign'],'group_sizes':[2,3,4]}

def finite_steps():
    cases=[('nonuniform',[0.6,0.3,0.1]),('uniform',[1/3]*3),('tied_maxima',[0.45,0.45,0.1]),('two_modes',[0.75,0.25]),('uniform_two',[0.5,0.5])]
    records=[]
    for name,q in cases:
        D=1-sum(x*x for x in q)
        for h in (0.001,0.1,1.0,20.0,1000.0):
            maximum=max(q); raw=[x*math.exp(h*(x-maximum)) for x in q]; total=sum(raw); u=[x/total for x in raw]
            Dnext=1-sum(x*x for x in u); uniform=max(q)==min(q)
            check('finite_step_diversity',abs(Dnext-D)<2e-15 if uniform else Dnext<D)
            check('finite_step_leader',max(u)>=max(q)-1e-15)
            for i in range(len(q)):
                for j in range(i):
                    check('finite_step_order_and_ties',abs(u[i]-u[j])<1e-15 if q[i]==q[j] else (u[i]-u[j])*(q[i]-q[j])>=0)
            if h<=20:
                def tilt(s):
                    v=[x*math.exp(s*x) for x in q]; z=sum(v); return [x/z for x in v]
                mid=tilt(h/2); ave=sum(mid[i]*q[i] for i in range(len(q)))
                chain=2*sum(mid[i]**2*(q[i]-ave) for i in range(len(q)))
                pair=sum(mid[i]*mid[j]*(mid[i]-mid[j])*(q[i]-q[j]) for i in range(len(q)) for j in range(len(q)))
                check('finite_step_derivative_identity',math.isclose(chain,pair,rel_tol=1e-11,abs_tol=1e-15))
                check('finite_step_derivative_sign',pair>=-1e-15 if uniform else pair>0)
            r=[0.25,0.75]
            logcorrect=h*maximum+math.log(total)
            neg=[math.log(x)-h*x for x in r]; shift=max(neg); logincorrect=shift+math.log(sum(math.exp(x-shift) for x in neg))
            odds_gain=logcorrect-logincorrect
            check('finite_step_odds_bound',odds_gain+1e-12>=h*sum(x*x for x in q))
            records.append({'case':name,'h':h,'D_before':D,'D_after':Dnext,'log_odds_gain':odds_gain})
    return {'case_count':len(records),'examples':records,'interpretation':'Illustrative deterministic arithmetic checks; not proofs, training, or sampled experiments.'}

def equation_signatures(source):
    start=source.index(r'\section{Mode Collapse and Verified Replay}')
    end=source.index(r'\subsection{A replay signal when fresh groups supply none}',start)
    scope=source[start:end]
    pat=re.compile(r'\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}.*?\\end\{\1\}|\\\[.*?\\\]',re.S)
    records=[]
    for i,m in enumerate(pat.finditer(scope),1):
        text=m.group(); normalized=re.sub(r'\s+','',text)
        records.append({'index':i,'labels':re.findall(r'\\label\{([^}]+)\}',text),'sha256':sha256(text.encode()).hexdigest(),'whitespace_normalized_sha256':sha256(normalized.encode()).hexdigest()})
    return {'scope':'Theory introduction through complete P.5; excludes P.6','display_count':len(records),'displays':records}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,default=HERE/'algebra_snapshot.tex')
    parser.add_argument('--live',type=Path,default=ROOT/'paper'/'main.tex')
    args=parser.parse_args()
    if not args.snapshot.exists():
        raise SystemExit(f'Missing snapshot: {args.snapshot}')
    out={'status':'passed','method':'Exact rational enumeration plus exact algebraic radicals; deterministic floating-point finite-step examples.'}
    out['moments']=exact_moments();out['finite_steps']=finite_steps()
    snap=equation_signatures(args.snapshot.read_text());live=equation_signatures(args.live.read_text())
    stable=[r['whitespace_normalized_sha256'] for r in snap['displays']]==[r['whitespace_normalized_sha256'] for r in live['displays']]
    out['equation_comparison']={'snapshot':str(args.snapshot),'live':str(args.live),'snapshot_display_count':snap['display_count'],'live_display_count':live['display_count'],'whitespace_normalized_equations_unchanged':stable,'byte_identical_equations':[r['sha256'] for r in snap['displays']]==[r['sha256'] for r in live['displays']]}
    out['checks']=dict(COUNTS);out['total_checks']=sum(COUNTS.values())
    out['no_training_sampling_or_experiments']=True
    (HERE/'algebra_equation_signatures.json').write_text(json.dumps(snap,indent=2)+'\n')
    (HERE/'algebra_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'total_checks':out['total_checks'],'moments':out['moments'],'equation_comparison':out['equation_comparison']},indent=2))

if __name__=='__main__': main()
