#!/usr/bin/env python3
"""Stable effective-time numerical checks of cor:collapse-error-rate.

This is finite-horizon evidence, not an asymptotic proof. We evolve logits
for q and r and logit(P), never computing 1-P by subtraction. The exact
constant-geometry reduction makes q,r independent of lambda, so logit(P)
is reconstructed as logit(P0)+integral(Sq+Sr)+lambda*tau.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp, softmax, logit

OUT=Path(__file__).with_suffix('.json')
FIXTURES=[
 dict(name='two_correct_one_incorrect',q=[.65,.35],r=[1.],P0=.2),
 dict(name='three_correct_two_incorrect',q=[.60,.25,.15],r=[.8,.2],P0=.08),
 dict(name='four_correct_seven_incorrect',q=[.47,.29,.17,.07],r=[.35,.20,.15,.10,.08,.07,.05],P0=.4),
 dict(name='near_tie_two_correct_three_incorrect',q=[.5001,.4999],r=[.60,.25,.15],P0=.15),
]
LAMBDAS=[0.,.25,4.,16.,64.]
TIMES=np.array([0.,48.,96.,192.,288.,384.])
records=[]
max_slope_error=0.
min_log_q=0.
min_log_failure=0.
for fixture in FIXTURES:
 q0=np.asarray(fixture['q']);r0=np.asarray(fixture['r']);m=len(q0);n=len(r0)
 assert np.isclose(sum(q0),1) and np.isclose(sum(r0),1)
 assert np.count_nonzero(q0==max(q0))==1 and min(q0)>0 and min(r0)>0
 def rhs(t,y):
  q=softmax(y[:m]);r=softmax(y[m:m+n])
  return np.r_[q,-r,q@q+r@r]
 initial=np.r_[np.log(q0),np.log(r0),0.]
 sol=solve_ivp(rhs,(0.,float(TIMES[-1])),initial,t_eval=TIMES,method='DOP853',rtol=2e-12,atol=2e-13,max_step=1.)
 assert sol.success,sol.message
 states=sol.y
 logq=states[:m]-logsumexp(states[:m],axis=0,keepdims=True)
 logr=states[m:m+n]-logsumexp(states[m:m+n],axis=0,keepdims=True)
 q=np.exp(logq);r=np.exp(logr)
 assert np.all(q>0) and np.all(r>0),'Choose shorter effective time to avoid underflow'
 winner=int(np.argmax(q0));minorities=[i for i in range(m) if i!=winner]
 tail_q_error=float(1-q[winner,-1])
 tail_r_error=float(np.max(np.abs(r[:,-1]-1/n)))
 # Positive entries can be much smaller than machine epsilon but remain
 # representable. logq, not log of a rounded 1-winner probability, is used.
 min_log_q=min(min_log_q,float(np.min(logq)))
 for lam in LAMBDAS:
  correct_log_odds=logit(fixture['P0'])+states[-1]+lam*TIMES
  log_failure=-np.logaddexp(0.,correct_log_odds)
  assert np.all(np.isfinite(log_failure))
  min_log_failure=min(min_log_failure,float(min(log_failure)))
  beta=1/(lam+1+1/n)
  # The consecutive-log ratio cancels finite-time intercepts. The theorem
  # concerns logq/log_failure itself, which is separately reported below.
  local_slope=(logq[minorities,-1]-logq[minorities,-2])/(log_failure[-1]-log_failure[-2])
  slope_error=float(np.max(np.abs(local_slope-beta)))
  max_slope_error=max(max_slope_error,slope_error)
  ratios=logq[minorities,1:]/log_failure[None,1:]
  first_error=float(np.max(np.abs(ratios[:,0]-beta)))
  final_error=float(np.max(np.abs(ratios[:,-1]-beta)))
  assert slope_error<2e-9,(fixture['name'],lam,slope_error)
  assert final_error<first_error,(fixture['name'],lam,first_error,final_error)
  records.append(dict(fixture=fixture['name'],m=m,n=n,lambda_value=lam,P0=fixture['P0'],q0=q0.tolist(),r0=r0.tolist(),predicted_exponent=beta,minority_indices=minorities,effective_times=TIMES[1:].tolist(),log_probability_ratios=ratios.tolist(),late_interval_log_slopes=local_slope.tolist(),maximum_slope_absolute_error=slope_error,first_ratio_absolute_error=first_error,final_ratio_absolute_error=final_error,final_winner_mass_shortfall=tail_q_error,final_incorrect_uniform_error=tail_r_error,smallest_correct_log_probability=float(logq.min()),final_log_failure=float(log_failure[-1])))
report=dict(passed=True,description='Finite-horizon stable-logit effective-time check of the collapse exponent; not a proof.',record_count=len(records),fixture_count=len(FIXTURES),lambdas=LAMBDAS,rtol=2e-12,atol=2e-13,max_step=1.,max_effective_time=float(TIMES[-1]),maximum_late_interval_slope_error=max_slope_error,minimum_correct_log_probability=min_log_q,minimum_log_failure=min_log_failure,underflow_avoided='q,r remained strictly representable and positive. Failure probability was never formed: log(1-P)=-logaddexp(0,logitP), permitting log failures below -25000.',finite_horizon_caveat='Raw logq/log(1-P) ratios retain O(1/tau) intercept error; only their approach and late-interval slope are tested. No exact finite-accuracy power law or physical-time exponential claim.',records=records)
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
