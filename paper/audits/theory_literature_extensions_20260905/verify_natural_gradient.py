"""Independent full-simplex ODE versus reduced equations for the new Fisher result."""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
rng=np.random.default_rng(20260905)
records=[]
for m,n in [(2,1),(3,2),(5,3),(7,1)]:
 for kind in ['DrGRPO','MaxRL']:
  for rho in [0.,.03,.7]:
   d=m+n;G=16;p0=rng.dirichlet(np.full(d,2.));w=np.zeros(d)
   k=max(1,m-1);w[:k]=rng.dirichlet(np.full(k,2.));R=np.r_[np.ones(m),np.zeros(n)]
   def c(P): return (G-1)/G if kind=='DrGRPO' else sum((1-P)**j for j in range(G-1))
   def full_rhs(t,p):
    P=p[:m].sum();return c(P)*p*(R-P)+rho*(w-p)
   P0=p0[:m].sum();q0=p0[:m]/P0
   def reduced_rhs(t,y):
    P=y[0];return [(1-P)*(c(P)*P+rho),rho/P]
   ts=np.linspace(0,8,81)
   sol=solve_ivp(full_rhs,(0,8),p0,t_eval=ts,rtol=2e-11,atol=2e-13)
   red=solve_ivp(reduced_rhs,(0,8),[P0,0],t_eval=ts,rtol=2e-11,atol=2e-13)
   assert sol.success and red.success
   P=sol.y[:m].sum(axis=0);q=sol.y[:m]/P
   q_expected=w[:m,None]+(q0-w[:m])[:,None]*np.exp(-red.y[1])
   qerr=float(np.max(abs(q-q_expected)));Perr=float(np.max(abs(P-red.y[0])))
   assert qerr<2e-9 and Perr<2e-9
   # Build natural gradients through the singular logit Fisher pseudoinverse.
   Fisher=np.diag(p0)-np.outer(p0,p0)
   gz=c(P0)*Fisher@R+rho*(w-p0)
   vel=Fisher@np.linalg.pinv(Fisher,hermitian=True)@gz
   ferr=float(np.max(abs(vel-full_rhs(0,p0))))
   assert ferr<1e-12
   if rho>0:
    floor=P0*np.minimum(q0[:k],w[:k])
    assert np.all(sol.y[:k]>=floor[:,None]-1e-10)
    assert np.all(1-P<=(1-P0)*np.exp(-rho*ts)+1e-10)
   # Check both KL decompositions independently.
   Hq=-sum(q0*np.log(q0));u=np.ones(m)/m
   reverse=sum(q0*np.log(q0/u));forward=sum(u*np.log(u/q0))
   ce=-sum(u*np.log(p0[:m]))
   assert abs(Hq-(np.log(m)-reverse))<1e-12
   assert abs(ce-(np.log(m)-np.log(P0)+forward))<1e-12
   records.append({'m':m,'n':n,'bank_size':k,'kind':kind,'rho':rho,'q_closed_form_max_error':qerr,'P_reduction_max_error':Perr,'fisher_pseudoinverse_error':ferr})
out={'status':'PASS','cases':len(records),'scope':'Numerical corroboration of independent full-probability ODE/reduced closed-form solutions, not a proof or neural-optimizer experiment.','max_q_error':max(r['q_closed_form_max_error'] for r in records),'max_P_error':max(r['P_reduction_max_error'] for r in records),'max_fisher_identity_error':max(r['fisher_pseudoinverse_error'] for r in records),'records':records}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
