#!/usr/bin/env python3
"""Finite enumeration of the proposed drift bound, not a stochastic proof."""
from pathlib import Path
import json
import numpy as np
from scipy.special import logsumexp,softmax

rho=.3; G=4; w=np.array([.7,.3,0.,0.,0.]); reward=np.array([1.,1.,1.,0.,0.])
v=np.arange(1.,6.);v/=np.linalg.norm(v);Q=np.eye(5)-2*np.outer(v,v)
matrices={'identity':np.eye(5),'diagonal_singular':np.diag([0.,.2,.7,1.,2.]),'dense_singular':Q@np.diag([0.,.2,.7,1.,2.])@Q.T}
zs=[np.zeros(5),np.array([-3.,1.,2.,0.,-1.]),np.array([3.,-3.,.1,-2.,2.]),np.array([-5.,-4.,-3.,2.,1.])]
u=np.array([.2,-.3,.1,.4,-.2]);v2=np.array([-.1,.2,.3,-.2,.1]);noises=[u,-u,v2,-v2]
assert np.max(np.abs(sum(noises)))==0
records=[]; max_fd=0.; min_slack=float('inf')
for method in ['DrGRPO','MaxRL']:
 L=rho/2+((G-1)/G)/2 if method=='DrGRPO' else rho/2+(G-1)/2+(G-1)*(G-2)/16
 def fg(z):
  p=softmax(z);P=float(p@reward)
  if method=='DrGRPO':c=(G-1)/G;psi=c*P
  else:c=sum((1-P)**j for j in range(G-1));psi=sum((1-(1-P)**k)/k for k in range(1,G))
  F=rho*(logsumexp(z)-w@z)-psi
  gradient=rho*(p-w)-c*p*(reward-P)
  return float(F),gradient
 for zi,z in enumerate(zs):
  F,g=fg(z)
  for j in range(5):
   e=np.eye(5)[j]*1e-5;fd=(fg(z+e)[0]-fg(z-e)[0])/(2e-5)
   max_fd=max(max_fd,float(abs(fd-g[j])))
  for hn,H in matrices.items():
   M=float(np.linalg.eigvalsh(H).max()); assert np.linalg.eigvalsh(H).min()>-1e-14
   for bn,b in [('zero',np.zeros(5)),('fixed',np.array([.15,-.1,.05,.2,-.05])),('adversarial',-2*g)]:
    for fraction in [.1,.5,1.]:
     eta=fraction/(L*M)
     expected=sum(fg(z-eta*H@(g+b+xi))[0] for xi in noises)/len(noises)
     noise_second=sum(float((H@xi)@(H@xi)) for xi in noises)/len(noises)
     upper=F-eta/2*float(g@H@g)+eta/2*float(b@H@b)+L*eta*eta/2*noise_second
     slack=upper-expected;min_slack=min(min_slack,slack)
     assert slack>=-2e-12,(method,zi,hn,bn,fraction,slack)
     records.append(dict(method=method,state=zi,preconditioner=hn,bias=bn,step_fraction=fraction,smoothness=L,eta=eta,expected_next_energy=expected,drift_upper_bound=upper,slack=slack))
assert max_fd<1e-8
# Assumption-sensitivity witness: unbiased noise but H uses that current noise.
eta=.05; g=1.; coupled=[(-2.,10.),(2.,1.)]
actual=sum(.5*(1-eta*H*(g+xi))**2 for xi,H in coupled)/2
invalid_bound=.5-eta/2*sum(H for _,H in coupled)/2+eta**2/2*sum((H*xi)**2 for xi,H in coupled)/2
assert actual>invalid_bound
report=dict(passed=True,case_count=len(records),noise_outcomes_per_case=4,gradient_finite_difference_max_error=max_fd,minimum_drift_slack=min_slack,scope='Exact finite-support noise enumeration, including noncommuting/singular PSD matrices, nonzero and adversarial biases, and the maximum allowed step. Not Monte Carlo and not a proof.',predictability_counterexample=dict(objective='F(theta)=theta^2/2',theta=1.,noise=[-2.,2.],noise_probabilities=[.5,.5],current_noise_dependent_H=[10.,1.],eta=eta,L=1.,M=10.,actual_expected_next_energy=actual,invalid_predictable_drift_bound=invalid_bound,explanation='The inequality fails if the predictable-H assumption is removed, despite conditional-unbiased raw noise and eta*L*M<=1.'),records=records)
out=Path(__file__).with_suffix('.json');out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
