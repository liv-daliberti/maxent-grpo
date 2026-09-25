from pathlib import Path
from fractions import Fraction
import hashlib,itertools,json,math,re
import numpy as np
p=Path(__file__).resolve().parent
rng=np.random.default_rng(72491)
def softmax(z):
    y=np.exp(z-np.max(z));return y/y.sum()
checks=0;max_tilt_error=0.;max_drift_error=0.
for m,n,P in itertools.product([2,3,7],[1,4],[.0001,.01,.1,.5,.9,.9999]):
    for uniform in [False,True]:
        q=np.full(m,1/m) if uniform else rng.dirichlet(np.ones(m))
        r=rng.dirichlet(np.ones(n));p0=np.r_[P*q,(1-P)*r];z=np.log(p0)
        v=np.r_[p0[:m]*(1-P),-p0[m:]*P]
        dp=p0*(v-p0@v);dP=dp[:m].sum();dq=(dp[:m]-q*dP)/P
        S2=q@q;T2=r@r;V=(q**3).sum()-S2**2
        drift=(-2*q@dq)/(P*(1-P));expected=-2*V
        max_drift_error=max(max_drift_error,abs(drift-expected))
        assert np.isclose(drift,expected,rtol=1e-9,atol=1e-12)
        assert np.isclose(dP,P**2*(1-P)**2*(S2+T2),rtol=1e-9,atol=1e-15)
        checks+=2
        for h in [1e-6,.01,.5,3.,50.]:
            new=softmax(z+h*v/(P*(1-P)));qnew=new[:m]/new[:m].sum()
            tilted=softmax(np.log(q)+h*q)
            max_tilt_error=max(max_tilt_error,float(np.max(np.abs(qnew-tilted))))
            assert np.allclose(qnew,tilted,rtol=1e-12,atol=1e-13)
            assert qnew@qnew+1e-13>=S2
            assert qnew[np.argmax(q)]+1e-13>=q.max()
            logratio=math.log(np.dot(q,np.exp(h*q)))-math.log(np.dot(r,np.exp(-h*r)))
            assert logratio+1e-13>=h*S2>=h/m-1e-13
            checks+=4
coefficient_checks=0
for G in range(2,25):
    for P in [Fraction(1,10),Fraction(1,3),Fraction(1,2),Fraction(9,10)]:
        exact=sum(Fraction(math.comb(G,k))*P**k*(1-P)**(G-k)*Fraction(k*(G-k)**2,G**4) for k in range(G+1))
        formula=Fraction(G-1,G**3)*P*(1-P)*(1+(G-2)*(1-P))
        assert exact==formula;coefficient_checks+=1
counterexample_P=Fraction(1,10)
coeff=lambda G:Fraction(G-1,G**3)*counterexample_P*(1-counterexample_P)*(1+(G-2)*(1-counterexample_P))
assert coeff(3)>coeff(2)
# An independent ordered-group enumeration verifies the one-step symmetry loss.
stochastic=[]
for G in [2,3,4]:
    m=3;P=.4;pr=np.array([P/m]*m+[1-P]);eta_values=[.01,.005,.0025]
    predicted=(m-1)/m**3*((G-1)/G**3*P*(1-P)*(1+(G-2)*(1-P)))
    losses=[]
    for eta in eta_values:
        expected_D=0.;expected_q=np.zeros(m)
        for group in itertools.product(range(m+1),repeat=G):
            weights=pr[list(group)].prod();counts=np.bincount(group,minlength=m+1);R=counts[:m].sum()
            adv=np.array([float(y<m)-R/G for y in group])
            scores=np.eye(m+1)[list(group)]-pr
            update=(adv[:,None]*scores).mean(axis=0)
            qnew=softmax(np.log(pr[:m])+eta*update[:m])
            expected_D+=weights*(1-qnew@qnew);expected_q+=weights*qnew
        assert np.allclose(expected_q,np.full(m,1/m),atol=1e-12)
        assert expected_D<1-1/m
        losses.append((1-1/m-expected_D)/eta**2)
    assert abs(losses[-1]-predicted)<abs(losses[0]-predicted)
    assert abs(losses[-1]-predicted)<2e-6
    stochastic.append({'G':G,'quadratic_coefficient':predicted,'enumerated_loss_over_eta_squared':losses})
# All displayed identities and numbered labels remain unchanged.
a=(p/'before.tex').read_text();b=(p/'replacement.tex').read_text()
pattern=r'\\\[.*?\\\]|\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}'
assert re.findall(pattern,a,re.S)==re.findall(pattern,b,re.S)
assert re.findall(r'\\label\{([^}]+)\}',a)==re.findall(r'\\label\{([^}]+)\}',b)
v={'finite_step_and_drift_checks':checks,'max_tilt_absolute_error':max_tilt_error,'max_drift_absolute_error':max_drift_error,'exact_binomial_coefficient_checks':coefficient_checks,'finite_group_nonmonotonicity_counterexample':{'P':str(counterexample_P),'G2':str(coeff(2)),'G3':str(coeff(3))},'stochastic_enumeration':stochastic,'displayed_equations_unchanged':True,'labels_unchanged':True,'model_calls':0,'experimental_data_changes':0}
(p/'validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
