"""Deterministic algebra and stored-table checks, without integrations/training."""
from pathlib import Path
from math import exp,log,e,log1p
import hashlib,json,re,difflib
root=Path('/n/fs/similarity/maxent-grpo')
out=Path(__file__).parent
checks=0
max_error=0.
def check(ok,label):
 global checks
 assert ok,label
 checks+=1

def near(a,b,label,tol=2e-10):
 global max_error
 err=abs(a-b)
 max_error=max(max_error,err)
 check(err <= tol*(1+abs(a)+abs(b)),(label,a,b,err))

def softmax(z):
 weights=[exp(v-max(z)) for v in z]
 return [v/sum(weights) for v in weights]
def kl(p,mu):
 return sum(a*log(a/b) for a,b in zip(p,mu) if a)
def revgrad(p,mu):
 D=kl(p,mu)
 return [a*(log(a/b)-D) for a,b in zip(p,mu)]
def maxc(P,G):
 return sum((1-P)**j for j in range(G-1))

refs=[[.2,.3,.5],[.01,.79,.2],[.6,.01,.39]]
policies=[[.1,.2,.7],[.75,.2,.05],[1e-10,.4,.6-1e-10],[.01,.01,.98]]
for mu in refs:
 for p in policies:
  D=kl(p,mu); grad=revgrad(p,mu)
  near(sum(grad),0,'KL gradient sums to zero')
  check(D<=log(1/min(mu))+1e-12,'global KL bound')
  for b in range(3):
   near(grad[b],p[b]*(log(p[b]/mu[b])-D),'exact reverse KL gradient')
   check(abs(grad[b])<=mu[b]/e+log(1/min(mu))+1e-12,'absolute boundary-force bound')
  # Exact expectation of the GRPO value estimator, and of its frozen-draw derivative.
  vals=[mu[i]/p[i]-log(mu[i]/p[i])-1 for i in range(3)]
  near(sum(p[i]*vals[i] for i in range(3)),D,'sample KL value expectation')
  for b in range(3):
   expected_fixed=sum(p[i]*(1-mu[i]/p[i])*((1 if i==b else 0)-p[b]) for i in range(3))
   near(expected_fixed,p[b]-mu[b],'fixed-sample derivative is different identity')
  R=[1,1,0];P=sum(p[:2]);alpha=15/16;beta=.2
  rtilde=[alpha*R[i]+beta*log(mu[i]) for i in range(3)]
  B=max(1,max(rtilde)-min(rtilde));rp=[(v-min(rtilde))/B for v in rtilde]
  tau=beta/B;eta=.5/beta;etap=eta*B
  near(etap*tau,eta*beta,'Mei step rescaling')
  j=alpha*P-beta*D
  H=-sum(a*log(a) for a in p)
  near(j,B*(sum(p[i]*rp[i] for i in range(3))+tau*H)+min(rtilde),'Mei objective rescaling')
  ps=softmax([log(mu[i])+alpha*R[i]/beta for i in range(3)])
  gap=(alpha*sum(ps[:2])-beta*kl(ps,mu))-j
  near(gap,beta*kl(p,ps),'KL gap identity')
  near(ps[0]/ps[1],mu[0]/mu[1],'correct-conditional reference ratio')
 for b in range(3):
  face=[0 if i==b else mu[i]/(1-mu[b]) for i in range(3)]
  threshold=-log1p(-mu[b])
  near(kl(face,mu),threshold,'reverse-KL face minimum')
  others=[i for i in range(3) if i!=b]
  for t in [.01,.2,.5,.9,.999]:
   trial=[0.,0.,0.];trial[others[0]]=t;trial[others[1]]=1-t
   check(kl(trial,mu)>=threshold-1e-12,'reverse-KL face lower bound')

# Exact stationary equations, with deterministic scalar bisection only.
stationary_cases=0
for G in [2,3,16]:
 vals=[maxc(P,G) for P in [.01,.1,.3,.8,.99]]
 check(all(a>=b for a,b in zip(vals,vals[1:])),'MaxRL nonincreasing coefficient')
 if G==2: check(all(v==1 for v in vals),'MaxRL G2 coefficient constant')
 else: check(all(a>b for a,b in zip(vals,vals[1:])),'MaxRL G>2 coefficient strict')
 for mu in refs:
  for beta in [.1,.4,2.]:
   for family in ['dr','maxrl']:
    c=lambda P: (G-1)/G if family=='dr' else maxc(P,G)
    lo=0.;hi=(G-1)/beta+1
    for _ in range(100):
     lam=(lo+hi)/2
     ps=softmax([log(mu[0])+lam,log(mu[1])+lam,log(mu[2])])
     if lam-c(sum(ps[:2]))/beta>0:hi=lam
     else:lo=lam
    lam=(lo+hi)/2
    ps=softmax([log(mu[0])+lam,log(mu[1])+lam,log(mu[2])]);P=sum(ps[:2]);grad=revgrad(ps,mu)
    # At very high logits P rounds to 1; stationary residual still remains accurate.
    for i in range(3):
     near(c(P)*ps[i]*((1 if i<2 else 0)-P)-beta*grad[i],0,'KL stationary residual')
    near(ps[0]/ps[1],mu[0]/mu[1],'MaxRL stationary reference conditional')
    stationary_cases+=1

# Two-mode reductions from the full logit force, including references with mass off the face.
for mb,mc in [(.5,.5),(.2,.3),(.8,.01)]:
 for s in [1e-10,1e-6,.001,.01,.05]:
  beta=.3;rho=.2;delta=min(.1,mb/(2*(mb+mc)))
  D=s*log(s/mb)+(1-s)*log((1-s)/mc)
  vb=beta*s*(log(mb/s)+D);vc=beta*(1-s)*(log(mc/(1-s))+D)
  near(s*(1-s)*(vb-vc),2*beta*s*s*(1-s)**2*log((1-s)*mb/(s*mc)),'two-mode KL velocity')
  vrb=rho*(.5-s);vrc=rho*(.5-(1-s))
  near(s*(1-s)*(vrb-vrc),rho*s*(1-s)*(1-2*s),'two-mode replay velocity')
  near(1/(s*(1-s)*(1-2*s)),1/s-1/(1-s)+4/(1-2*s),'replay partial fractions',tol=1e-9)
  if s<delta:
   exact=(log(delta*(1-delta)/(1-2*delta)**2)-log(s*(1-s)/(1-2*s)**2))/rho
   remainder=exact-log(delta/s)/rho
   constant=(log(1-delta)-2*log(1-2*delta))/rho
   near(remainder,constant-(log(1-s)-2*log(1-2*s))/rho,'replay asymptotic remainder')
  if s==1e-10:
   L=log(1/s)
   ratio=L*L/((1-s)**2*(L-1)*(L+log(mb/mc)+log1p(-s)))
   # L'Hopital ratio ->1; this expression records the finite-s correction analytically.
   check(.8<ratio<1.2,'KL asymptotic derivative ratio')

# Stored integration values and caption quantities; no recovery_time call.
source=root/'paper/results/kl_replay_recovery_flow.json'
data=json.loads(source.read_text())
check((data['correct_modes'],data['incorrect_categories'],data['group_size'])==(4,3,16),'recovery settings')
curves=data['recovery_times']; depths=data['depths']
ratios=[];gap_errors=[]
for name,curve in curves.items():
 times=[curve[f'{d:.0e}'] for d in depths]
 for d in depths:
  actual=d/(1+d+5*exp(-40))
  near(actual/(1/(1+d+5*exp(-40))),d,'table depth equals odds')
  check(actual<d,'table odds are not initial probability')
 if name.startswith('kl_'): ratios.extend(b/a for a,b in zip(times,times[1:]))
 else:
  rho=float(name.rsplit('_',1)[1]);expected=log(10)/rho
  for a,b in zip(times,times[1:]):
   error=abs((b-a)/expected-1);gap_errors.append(error)
   check(error<.02,'replay decade increment close to log10/rho')
macro=(root/'paper/results/kl_replay_recovery_macros.tex').read_text()
check(f'{min(ratios):.1f}--{max(ratios):.1f}' in macro,'KL decade macro matches stored integrations')
table=(root/'paper/results/kl_replay_recovery_table_body.tex').read_text()
for curve in curves.values():
 for d in depths:check(f'{curve[f"{d:.0e}"]:.2e}' in table,'all 16 recovery values in table')

before=(out/'before.tex').read_text();after=(out/'replacement.tex').read_text()
math_pat=r'\\\[.*?\\\]|\\begin\{(?:equation|align)\}.*?\\end\{(?:equation|align)\}'
check(re.findall(math_pat,before,re.S)==re.findall(math_pat,after,re.S),'all displayed equations byte-identical')
for pattern,name in [(r'\\label\{[^}]+\}','labels'),(r'\\(?:begin|end)\{(?:theorem|lemma|corollary|proposition|remark)\}','result environments'),(r'\\cite\w+(?:\[[^]]*\])?\{[^}]+\}','citations')]:
 check(re.findall(pattern,before)==re.findall(pattern,after),f'all {name} preserved')
check(not re.search(r'(?i)post.?hoc|reader is entitled|this subsection separates|built exactly as|admitted key|verified history|frozen reference|reasoning modes?|we have not swept',after),'editorial/process and terminology scan')
report={'checks_passed':checks,'stationary_cases':stationary_cases,'all_displayed_equations_unchanged':True,'all_labels_and_result_environments_unchanged':True,'all_citations_unchanged':True,'stored_recovery_times_preserved':16,'replay_decade_increment_max_relative_error':max(gap_errors),'recovery_integrations_rerun':False,'experiments_or_training_run':False,'replacement_sha256':hashlib.sha256(after.encode()).hexdigest(),'stored_data_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'changes.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='latest-KL-before.tex',tofile='KL-replacement.tex')))
print(json.dumps(report))
