#!/usr/bin/env python3
"""Numerical check of the shared-correctness result; not a proof.

Run: python paper/audits/verify_model_size_theorem_20260904.py
Writes the adjacent verify_model_size_theorem_20260904.json.
Copied from the earlier numerical verifier; only output paths were changed.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit, logit, softmax

OUTPUT_PATH = Path(__file__).with_suffix(".json")

RTOL = 2e-11
ATOL = 2e-13
G = 16
LAMBDAS = [0.0, 0.25, 1.0, 4.0, 16.0]
FIXTURES = [
    dict(name='four_correct_three_incorrect', q=[.47,.29,.17,.07], r=[.60,.25,.15], P0=.20, target=.83),
    dict(name='three_correct_one_incorrect', q=[.61,.27,.12], r=[1.0], P0=.08, target=.93),
    dict(name='near_tie_two_correct_five_incorrect', q=[.5001,.4999], r=[.44,.23,.17,.11,.05], P0=.44, target=.88),
]

def c_value(P, method):
    if method == 'DrGRPO':
        return (G-1)/G
    if method == 'MaxRL':
        return -np.expm1((G-1)*np.log1p(-P))/P
    raise ValueError(method)


def logit_gradient(z, m):
    p = softmax(z)
    P = float(p[:m].sum())
    rewards = np.concatenate((np.ones(m),np.zeros(len(z)-m)))
    return P, p*(rewards-P)


def integrate_full(fixture, lam, method):
    q0, r0 = np.array(fixture['q']), np.array(fixture['r'])
    m, n = len(q0), len(r0)
    z0 = np.log(np.r_[fixture['P0']*q0, (1-fixture['P0'])*r0])
    v = np.r_[np.ones(m),np.zeros(n)]
    def flow(t, y):
        P, gradient = logit_gradient(y[:-1],m)
        c = c_value(P,method)
        dz = c*(gradient+lam*v*np.dot(v,gradient))
        return np.r_[dz,c*P*(1-P)]
    def event(t,y):
        return softmax(y[:-1])[:m].sum()-fixture['target']
    event.terminal = True
    event.direction = 1
    sol = solve_ivp(flow,(0,10000),np.r_[z0,0.0],events=event,method='DOP853',rtol=RTOL,atol=ATOL,dense_output=True)
    assert sol.success and len(sol.t_events[0])==1, sol.message
    t = float(sol.t_events[0][0])
    y = sol.sol(t)
    p = softmax(y[:-1])
    P = float(p[:m].sum())
    q, r = p[:m]/P, p[m:]/(1-P)
    tau = float(y[-1])
    L = float(logit(fixture['target'])-logit(fixture['P0']))
    lower = q0*np.exp(-L/(lam+1/m+1/n))
    return sol, dict(method=method,lambda_value=lam,physical_time=t,tau=tau,P=P,q=q.tolist(),r=r.tolist(),q_max=float(q.max()),S_q=float(q@q),distinct8=float(np.sum(-np.expm1(8*np.log1p(-P*q)))),pass8=float(-np.expm1(8*np.log1p(-P))),lower_bound=lower.tolist(),minimum_bound_slack=float(np.min(q-lower)),tau_upper=L/(lam+1/m+1/n),target_residual=abs(P-fixture['target']))


records=[]
max_universal_error=0.0
max_clock_error=0.0
max_objective_endpoint_error=0.0
min_bound_slack=float('inf')
max_target_residual=0.0
max_monotonicity_violation=0.0
for fixture in FIXTURES:
    full=[]
    for method in ['DrGRPO','MaxRL']:
        for lam in LAMBDAS:
            sol,record=integrate_full(fixture,lam,method)
            full.append((sol,record))
    m,n=len(fixture['q']),len(fixture['r'])
    tau_max=max(record['tau'] for _,record in full)
    def universal(tau,y):
        q,r=y[:m],y[m:m+n]
        Sq,Sr=q@q,r@r
        return np.r_[q*(q-Sq),-r*(r-Sr),Sq+Sr]
    universal_solution=solve_ivp(universal,(0,tau_max+1e-5),np.r_[fixture['q'],fixture['r'],0.0],method='DOP853',rtol=RTOL,atol=ATOL,dense_output=True)
    for sol,record in full:
        times=np.linspace(0,record['physical_time'],181)
        yy=sol.sol(times)
        p=softmax(yy[:-1],axis=0)
        P=p[:m].sum(axis=0)
        qr=np.vstack((p[:m]/P,p[m:]/(1-P)))
        predicted=universal_solution.sol(yy[-1])
        error=float(np.max(np.abs(qr-predicted[:-1])))
        clock=float(np.max(np.abs(logit(P)-logit(fixture['P0'])-predicted[-1]-record['lambda_value']*yy[-1])))
        record['universal_qr_max_residual']=error
        record['clock_max_residual']=clock
        max_universal_error=max(max_universal_error,error)
        max_clock_error=max(max_clock_error,clock)
        min_bound_slack=min(min_bound_slack,record['minimum_bound_slack'])
        max_target_residual=max(max_target_residual,record['target_residual'])
        assert record['tau']<=record['tau_upper']+2e-9
        assert record['minimum_bound_slack']>=-2e-9
    by_method={method:[record for _,record in full if record['method']==method] for method in ['DrGRPO','MaxRL']}
    for method, rows in by_method.items():
        for key,sign in [('tau',1),('q_max',1),('S_q',1),('distinct8',-1)]:
            diffs=sign*np.diff([row[key] for row in rows])
            violation=max(0.0,float(diffs.max()))
            max_monotonicity_violation=max(max_monotonicity_violation,violation)
            assert violation<2e-9,(fixture['name'],method,key,diffs)
    for dr,mx in zip(by_method['DrGRPO'],by_method['MaxRL']):
        residual=max(abs(dr['tau']-mx['tau']),max(abs(np.array(dr['q'])-np.array(mx['q']))),max(abs(np.array(dr['r'])-np.array(mx['r']))))
        max_objective_endpoint_error=max(max_objective_endpoint_error,float(residual))
    records.append(dict(fixture=fixture,runs=[record for _,record in full]))

# Directly integrate all replicated parameters, rather than merely applying
# the predicted logit kernel: z=alpha*sum_j theta_j, rate kappa=r*alpha^2.
replication=[]
fixture=FIXTURES[0]
m=len(fixture['q'])
z0=np.log(np.r_[fixture['P0']*np.array(fixture['q']),(1-fixture['P0'])*np.array(fixture['r'])])
N=len(z0)
T=2.0
for method in ['DrGRPO','MaxRL']:
    for copies in [1,2,5,11]:
        for normalization,alpha in [('sum',1.0),('sqrt',1/np.sqrt(copies)),('mean',1/copies)]:
            rate=copies*alpha**2
            theta0=np.tile(z0/(copies*alpha),(copies,1)).ravel()
            def parameter_flow(t,theta):
                z=alpha*theta.reshape(copies,N).sum(axis=0)
                P,g=logit_gradient(z,m)
                return np.tile(alpha*c_value(P,method)*g,(copies,1)).ravel()
            par=solve_ivp(parameter_flow,(0,T),theta0,method='DOP853',rtol=RTOL,atol=ATOL,dense_output=True)
            def ordinary_flow(t,z):
                P,g=logit_gradient(z,m)
                return c_value(P,method)*g
            base=solve_ivp(ordinary_flow,(0,rate*T),z0,method='DOP853',rtol=RTOL,atol=ATOL,dense_output=True)
            times=np.linspace(0,T,151)
            predicted=base.sol(rate*times)
            observed=alpha*par.sol(times).reshape(copies,N,-1).sum(axis=0)
            error=float(np.max(np.abs(observed-predicted)))
            initial_error=float(np.max(np.abs(observed[:,0]-z0)))
            replication.append(dict(method=method,copies=copies,normalization=normalization,alpha=alpha,predicted_rate=rate,max_logit_residual=error,initial_logit_residual=initial_error))
            assert error<2e-8

summary=dict(
    integration_method='DOP853',rtol=RTOL,atol=ATOL,
    full_logit_integrations=sum(len(r['runs']) for r in records),
    replication_integrations=len(replication),
    max_universal_qr_residual=max_universal_error,
    max_clock_residual=max_clock_error,
    max_DrGRPO_MaxRL_endpoint_residual=max_objective_endpoint_error,
    max_target_probability_residual=max_target_residual,
    max_monotonicity_violation=max_monotonicity_violation,
    minimum_componentwise_lower_bound_slack=min_bound_slack,
    max_replication_logit_residual=max(x['max_logit_residual'] for x in replication),
    note='Numerical consistency checks, not a proof or empirical model-size law.'
)
with OUTPUT_PATH.open('w') as f:
    json.dump(dict(summary=summary,shared_correctness=records,replication=replication),f,indent=2)
print(json.dumps(summary,indent=2))
print('\nRepresentative DrGRPO case: four correct / three incorrect, P0=.20, P*=.83')
print('lambda   tau          max_q        S_q          distinct@8')
for row in records[0]['runs']:
    if row['method']=='DrGRPO':
        print(f"{row['lambda_value']:6.2f}   {row['tau']:.8f}   {row['q_max']:.8f}   {row['S_q']:.8f}   {row['distinct8']:.8f}")
print(f'\nSaved {OUTPUT_PATH}')
