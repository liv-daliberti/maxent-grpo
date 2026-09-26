from pathlib import Path
from fractions import Fraction as F
from itertools import product
from math import comb
import re, json, hashlib
out=Path('/tmp/paper-appendix-eighth10-20260921/categorical')
a=(out/'original.tex').read_text();b=(out/'replacement.tex').read_text()
checks=[]
def check(ok,label):
    assert ok,label
    checks.append(label)
displays=lambda x:re.findall(r'\\begin\{equation\}.*?\\end\{equation\}|\\\[.*?\\\]',x,re.S)
check(displays(a)==displays(b),'All display equations byte-identical')
labels=lambda x:re.findall(r'\\label\{([^}]+)\}',x)
check(labels(a)==labels(b),'All labels preserved in order')
check(re.findall(r'\\(?:begin|end)\{[^}]+\}',a)==re.findall(r'\\(?:begin|end)\{[^}]+\}',b),'All LaTeX environments preserved in order')
check(set(re.findall(r'\\cite\w*(?:\[[^\]]*\])*\{([^}]+)\}',a)) <= set(re.findall(r'\\cite\w*(?:\[[^\]]*\])*\{([^}]+)\}',b))|{'shao2024deepseekmath,liu2025understanding'},'Attribution retained with GRPO/Dr.GRPO split explicitly')
check(not re.search(r'post[- ]?hoc|we invoke|published convergence|not a new general theorem|our derivations|we apply established',b,re.I),'No target process/history phrases')
check('(A1)--(A5)' in b and 'fixed length-normalization factor' in b,'Correct assumption scope')
# Exact expectation over every category sequence, without stochastic simulation.
cases=0;sequences=0
for probs,m in [([F(1,2),F(1,3),F(1,6)],2),([F(1,5),F(1,10),F(3,10),F(1,4),F(3,20)],3)]:
    P=sum(probs[:m]);n=len(probs)
    grad=[p*((int(j<m))-P) for j,p in enumerate(probs)]
    for G in range(2,5):
        for kind in ['dr','max','positive_count']:
            w=lambda r:F(1) if kind=='dr' else (F(G,r) if kind=='max' else F(r+1))
            exact=[F(0)]*n
            for seq in product(range(n),repeat=G):
                seqp=F(1)
                for j in seq:seqp*=probs[j]
                R=sum(j<m for j in seq)
                if 0<R<G:
                    for j in seq:
                        adv=w(R)*(F(j<m)-F(R,G))
                        for k,p in enumerate(probs):
                            exact[k]+=seqp*adv*(F(j==k)-p)/G
                sequences+=1
            coeff=sum(F(comb(G,r))*P**r*(1-P)**(G-r)*w(r)*r*(G-r) for r in range(1,G))/(G*G*P*(1-P))
            check(exact==[coeff*x for x in grad],f'Exact mean identity {m} correct categories G={G} {kind}')
            if kind=='dr':check(coeff==F(G-1,G),f'Dr.GRPO coefficient G={G} m={m}')
            if kind=='max':check(coeff==sum((1-P)**j for j in range(G-1)),f'MaxRL coefficient G={G} m={m}')
            cases+=1
    # Exact softmax-chain-rule dynamics and gradient lower bound.
    zdot=grad
    meanz=sum(p*d for p,d in zip(probs,zdot))
    pdot=[p*(d-meanz) for p,d in zip(probs,zdot)]
    Pdot=sum(pdot[:m]);q=[p/P for p in probs[:m]]
    check(Pdot==sum(d*d for d in grad),f'Correctness-gradient identity m={m}')
    check(Pdot>=P*P*(1-P)**2*(F(1,m)+F(1,n-m)),f'Gradient lower bound m={m}')
    qdot=[(d*P-p*Pdot)/(P*P) for p,d in zip(probs[:m],pdot[:m])]
    rate=P*(1-P);sq=sum(x*x for x in q)
    check(qdot==[rate*x*(x-sq) for x in q],f'Replicator chain rule m={m}')
    for c in range(m):
        for d in range(c):
            check(qdot[c]/q[c]-qdot[d]/q[d]==rate*(q[c]-q[d]),f'Log ratio dynamics m={m} c={c} d={d}')
for G in range(2,13):
    for P in [F(1,101),F(1,7),F(1,2),F(9,10),F(100,101)]:
        for kind in ['dr','max','positive_count']:
            w=lambda r:F(1) if kind=='dr' else (F(G,r) if kind=='max' else F(r+1))
            direct=sum(F(comb(G,r))*P**r*(1-P)**(G-r)*w(r)*r*(G-r) for r in range(1,G))/(G*G*P*(1-P))
            polynomial=F(G-1,G)*sum(F(comb(G-2,j))*w(j+1)*P**j*(1-P)**(G-2-j) for j in range(G-1))
            check(direct==polynomial,f'Denominator-free identity G={G} P={P} {kind}')
            check(F(G-1,G)*min(w(j+1) for j in range(G-1))<=direct<=F(G-1,G)*max(w(j+1) for j in range(G-1)),f'Coefficient bounds G={G} P={P} {kind}')
        geom=sum((1-P)**j for j in range(G-1))
        check(geom==(1-(1-P)**(G-1))/P,f'MaxRL geometric sum G={G} P={P}')
        derivative=sum((1-P)**(k-1) for k in range(1,G))
        check(derivative==geom,f'MaxRL potential derivative G={G} P={P}')
    check(sum(F(1) for j in range(G-1))==G-1,f'MaxRL lower endpoint G={G}')
    check(sum(F(0)**j for j in range(G-1))==1,f'MaxRL upper endpoint G={G}')
# Explicit counterexample motivating fixed, rather than sample-dependent, common length scaling.
p=[F(1,3)]*3;G=2;m=2
weighted=[F(0)]*3
for seq in product(range(3),repeat=2):
    R=sum(j<m for j in seq)
    if R==1:
        common=F(2) if 0 in seq else F(1)
        for j in seq:
            adv=F(j<m)-F(R,G)
            for k,pk in enumerate(p):weighted[k]+=F(1,9)*common*adv*(F(j==k)-pk)/G
check(weighted[0]!=weighted[1],'A sample-dependent common factor need not preserve direction even at equal correct probabilities')
report={'status':'passed','checks':len(checks),'display_equations_preserved':len(displays(a)),'labels_preserved':len(labels(a)),'exact_enumeration_cases':cases,'exact_enumerated_sequences':sequences,'no_training_or_sampling_runs':True,'counterexample_common_random_factor_mean':[str(x) for x in weighted],'source_sha256':hashlib.sha256(a.encode()).hexdigest(),'replacement_sha256':hashlib.sha256(b.encode()).hexdigest(),'check_labels':checks}
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='check_labels'},indent=2))
