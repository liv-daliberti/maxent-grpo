from pathlib import Path
import itertools,json,math
import numpy as np
from scipy.integrate import solve_ivp
root=Path('/n/fs/similarity/maxent-grpo');a=root/'paper/audits/theory_examples_20260905'
p=np.array([.30,.15,.05,.50]);q=p[:3]/sum(p[:3]);P=sum(p[:3]);G=4
v=p*(np.array([1,1,1,0])-P)
expected={k:np.zeros(4) for k in ['drgrpo','maxrl']}
for group in itertools.product(range(4),repeat=G):
 prob=math.prod(p[i] for i in group);rewards=np.array([float(i<3) for i in group]);M=sum(rewards)
 scores=np.eye(4)[list(group)]-p
 adv={'drgrpo':rewards-rewards.mean(),'maxrl':G*rewards/M-1 if M else np.zeros(G)}
 for k in expected:expected[k]+=prob*np.mean(adv[k][:,None]*scores,axis=0)
assert np.allclose(expected['drgrpo'],.75*v,atol=1e-15)
assert np.allclose(expected['maxrl'],1.75*v,atol=1e-15)
qprime=q*(q-np.dot(q,q));assert np.allclose(qprime,[.084,-.048,-.036],atol=1e-15)
records=[]
for lam in [0.,4.]:
 L=math.log(9.)
 def reduced(s,qq):return qq*(qq-np.dot(qq,qq))/(lam+1+np.dot(qq,qq))
 sol=solve_ivp(reduced,[0,L],q,rtol=1e-12,atol=1e-14);qq=sol.y[:,-1]
 def full(t,z):
  pp=np.exp(z-max(z));pp/=sum(pp);PP=sum(pp[:3]);grad=pp*(np.array([1,1,1,0])-PP);return .75*(grad+lam*np.array([1,1,1,0])*sum(grad[:3]))
 def event(t,z):
  pp=np.exp(z-max(z));pp/=sum(pp);return sum(pp[:3])-.9
 event.terminal=True;event.direction=1
 sol2=solve_ivp(full,[0,1000],np.log(p),events=event,rtol=1e-12,atol=1e-14)
 assert len(sol2.t_events[0])==1
 pp=np.exp(sol2.y[:,-1]-max(sol2.y[:,-1]));pp/=sum(pp)
 error=float(np.max(abs(qq-pp[:3]/sum(pp[:3]))));assert error<1e-10,error
 record={'lambda':lam,'q_at_P09':qq.tolist(),'rarest_q_floor':.1*math.exp(-L/(lam+1/3+1)),'distinct8_at_P09':float(sum(1-(1-.9*qq)**8)),'pass8_at_P09':1-1e-8,'asymptotic_exponent':1/(lam+2),'independent_full_ODE_q_error':error}
 records.append(record)
r={'status':'PASS','scope':'Illustrative finite categorical calculations only, not measurements of training runs. Full group enumeration and independent full-logit/reduced-ODE agreement corroborate displayed values.','group_count':4**4,'p':p.tolist(),'q':q.tolist(),'correctness_gradient':v.tolist(),'expected_group_gradients':{k:x.tolist() for k,x in expected.items()},'conditional_q_prime':qprime.tolist(),'parameterization_examples':records,'parameter_replication_h4_speed_factors':{'gamma0':4,'gammaHalf':1,'gamma1':.25}}
(a/'verify_early_examples.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
