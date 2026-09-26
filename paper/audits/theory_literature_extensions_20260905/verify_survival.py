"""Independent standard-library checks for sharp survival certificates."""
from __future__ import annotations
import itertools, json, math, pathlib, random


def entropy(p):
    return -sum(x*math.log(x) for x in p if x)


def kl(p,q):
    return sum(x*math.log(x/y) for x,y in zip(p,q) if x)


def binary_kl(a,b):
    return kl([a,1-a],[b,1-b])


def binary_lower_log(a,D):
    if D == 0: return math.log(a)
    if a == 1: return -D
    left=-(D+entropy([a,1-a]))/a
    right=math.log(a)
    for _ in range(180):
        mid=(left+right)/2
        value=a*(math.log(a)-mid)+(1-a)*(math.log1p(-a)-math.log1p(-math.exp(mid)))
        if value>D: left=mid
        else: right=mid
    return (left+right)/2


def entropy_lower(m,h):
    if h <= math.log(m-1): return 0.0
    if h >= math.log(m): return 1/m
    lo,hi=0.,1/m
    for _ in range(180):
        mid=(lo+hi)/2
        value=entropy([mid,1-mid])+(1-mid)*math.log(m-1)
        if value<h: lo=mid
        else: hi=mid
    return (lo+hi)/2


def binomial(n,p,x):
    return math.comb(n,x)*p**x*(1-p)**(n-x)


def cp_one_sided(n,x,alpha,lower):
    if lower and x==0: return 0.0
    if not lower and x==n: return 1.0
    lo,hi=0.,1.
    for _ in range(100):
        mid=(lo+hi)/2
        if lower:
            tail=sum(binomial(n,mid,k) for k in range(x,n+1))
            if tail<alpha: lo=mid
            else: hi=mid
        else:
            tail=sum(binomial(n,mid,k) for k in range(x+1))
            if tail>alpha: lo=mid
            else: hi=mid
    return (lo+hi)/2


def main():
    tight=[]
    for weights,D in itertools.product([[.5,.5],[.7,.2,.1],[1/16]*16],[0.,.001,.01,.1,1.,10.]):
        for b,a in enumerate(weights):
            logp=binary_lower_log(a,D); r=math.exp(logp)
            q=[r if j==b else (1-r)*w/(1-a) for j,w in enumerate(weights)]
            error=abs(kl(weights,q)-D)
            assert error<1e-11
            C=entropy(weights)+D
            assert logp >= -C/a-1e-12
            assert r >= max(0,a-math.sqrt(D/2))-1e-12
            tight.append({'weights':weights,'D':D,'b':b,'lower':r,'log_lower':logp,'tight_KL_error':error})
    rng=random.Random(20260905)
    random_checks=0
    for _ in range(200):
        m=rng.choice([2,3,5,16]); w=[rng.random()+.05 for _ in range(m)]; w=[x/sum(w) for x in w]
        p=[rng.random()+.005 for _ in range(m+1)]; p=[x/sum(p) for x in p]
        R=-sum(x*math.log(y) for x,y in zip(w,p)); D=R-entropy(w)
        assert abs(D-kl(w+[0.],p))<1e-12
        assert sum(p[:m])>=math.exp(-D)-1e-12
        for b,a in enumerate(w):
            assert p[b]>=math.exp(binary_lower_log(a,D))-1e-12
        random_checks+=1
    entropy_cases=[]
    for m in [2,3,16,100]:
        threshold=math.log(m/(m-1))
        for fraction in [0.,.01,.5,.99,1.,2.]:
            eps=threshold*fraction
            h=math.log(m-1) if fraction==1. else math.log(m)-eps
            r=entropy_lower(m,h)
            q=[r]+[(1-r)/(m-1)]*(m-1)
            assert entropy(q)>=h-1e-12
            if fraction<1:
                assert r>0 and abs(entropy(q)-h)<1e-12
                assert abs(binary_kl(r,1/m)-eps)<1e-12
            else:
                assert r==0.
            entropy_cases.append({'m':m,'entropy_floor':h,'KL_budget':eps,'extinction_threshold':threshold,'lower':r,'attaining_distribution_entropy':entropy(q)})
    coverage=[]
    for n in [8,32,128]:
        lows=[cp_one_sided(n,x,.05,True) for x in range(n+1)]
        ups=[cp_one_sided(n,x,.05,False) for x in range(n+1)]
        assert abs(ups[0]-(1-.05**(1/n)))<1e-14
        for p in [.001,.01,.05,.2,.5,.95,.999]:
            low_fail=sum(binomial(n,p,x) for x in range(n+1) if p<lows[x]-1e-14)
            up_fail=sum(binomial(n,p,x) for x in range(n+1) if p>ups[x]+1e-14)
            assert low_fail<=.05+1e-12 and up_fail<=.05+1e-12
            coverage.append({'n':n,'p':p,'lower_error':low_fail,'upper_error':up_fail})
    examples=[]
    for D in [0.,.001,.01,.1,1.,160.]:
        log_floor=binary_lower_log(1/16,D)
        examples.append({'bank_size':16,'excess_cross_entropy_D':D,'crude_log_floor':-16*(math.log(16)+D),'sharp_log_floor':log_floor,'sharp_floor':math.exp(log_floor)if log_floor>-745 else 'below float range; see log_floor'})
    zero=[{'N':n,'single_mode_upper_95':1-.05**(1/n),'simultaneous16_mode_upper_95':1-(.05/16)**(1/n)} for n in [8,32,128,299,574]]
    result={'status':'PASS','scope':'Sharp deterministic KL/entropy certificates and fixed-binomial coverage checks. No new data or training claims.','sharp_reverse_KL_cases':len(tight),'max_tight_KL_error':max(t['tight_KL_error']for t in tight),'random_cross_entropy_checks':random_checks,'entropy_threshold_cases':entropy_cases,'one_sided_coverage_checks':coverage,'certificate_examples':examples,'zero_count_examples':zero,'observed_count_examples_N32':[{'count':x,'lower_95':cp_one_sided(32,x,.05,True),'upper_95':cp_one_sided(32,x,.05,False)}for x in [0,1,2,4,8]],'tight_constructions':tight}
    path=pathlib.Path(__file__).with_suffix('.json');path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ['tight_constructions','one_sided_coverage_checks','entropy_threshold_cases']},indent=2))

if __name__=='__main__':main()
