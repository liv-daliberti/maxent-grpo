"""Check every new numerical example and preservation of formal blocks."""
from pathlib import Path
import hashlib
import json
import math
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def close(actual, expected, tol=1e-12):
    assert abs(actual - expected) <= tol, (actual, expected)

def check_round(actual, shown, digits):
    assert f'{actual:.{digits}f}' == shown, (actual, shown, digits)

p = [0.30, 0.15, 0.05, 0.50]
correct = [1, 1, 1, 0]
P = sum(p[:3])
q = [x / P for x in p[:3]]
w = [1/3, 1/3, 1/3, 0]
a = 3/4
natural = {}
for rho in (0, 0.1):
    derivative = [a*y + rho*z/x for x,y,z in zip(p,correct,w)]
    average = sum(x*y for x,y in zip(p,derivative))
    pdot = [x*(y-average) for x,y in zip(p,derivative)]
    Pdot = sum(pdot[:3])
    qdot = [(v*P-x*Pdot)/P**2 for x,v in zip(p[:3],pdot[:3])]
    close(sum(pdot), 0)
    close(Pdot, (1-P)*(a*P+rho))
    for value, target, initial in zip(qdot,w[:3],q):
        close(value, rho/P*(target-initial))
    natural[str(rho)] = {'Pdot':Pdot,'qdot':qdot,'pdot':pdot}
close(natural['0']['Pdot'], .1875)
for value in natural['0']['qdot']: close(value, 0)
close(natural['0.1']['Pdot'], .2375)
for value, shown in zip(natural['0.1']['qdot'], ['-0.05333','0.00667','0.04667']):
    check_round(value, shown, 5)
floor = P*min(q[2],w[2]); close(floor, .05)

beta = .25
weights = [math.exp(a/beta),math.exp(a/beta),1]
optimum = [x/sum(weights) for x in weights]
stationary_derivatives = [a*good-beta*(math.log(x)+1) for good,x in zip([1,1,0],optimum)]
assert max(stationary_derivatives)-min(stationary_derivatives) < 1e-12
for value, shown in zip(optimum,['0.48786','0.48786','0.02429']): check_round(value,shown,5)
check_round(sum(optimum[:2]),'0.97571',5)
check_round(100*optimum[2],'2.43',2)

H = -sum(x*math.log(x) for x in q)
forward = sum(x*math.log(3*x) for x in q)
reverse = sum(math.log((1/3)/x)/3 for x in q)
R = -sum(math.log(x)/3 for x in p[:3])
close(H, math.log(3)-forward)
close(R, math.log(3)-math.log(P)+reverse)
Pnew=.9
Rnew = -sum(math.log(Pnew*x)/3 for x in q)
drop=R-Rnew
close(drop, math.log(Pnew/P))
for value,shown in [(H,'0.89795'),(forward,'0.20067'),(reverse,'0.24052'),(R,'2.03228'),(drop,'0.58779')]: check_round(value,shown,5)

p_rare=[.600,.389,.001,.010]
close(sum(p_rare),1)
P_rare=sum(p_rare[:3]);close(P_rare,.99)
G=16
h=sum(math.comb(G,k)*P_rare**k*(1-P_rare)**(G-k) for k in range(1,G))
close(h, 1-P_rare**G-(1-P_rare)**G)
absence=(1-p_rare[2])**G
replay=.1*(1/3-p_rare[2])
check_round(h,'0.14854',5);check_round(100*h,'14.85',2)
check_round(absence,'0.98412',5);check_round(replay,'0.03323',5)

source=(ROOT/'paper/main.tex').read_text()
manifest=json.loads((HERE/'entropy_staging_manifest.json').read_text())
for name,entry in manifest.items():
    staged=(HERE/(name+'.tex')).read_text()
    position=source.index('\\label{'+entry['label']+'}')
    start=source.rfind('\\subsection{',0,position)
    end=source.find('\\subsection{',position)
    current=source[start:end]
    for environment in ('lemma','theorem','corollary','proof'):
        pattern=r'\\begin\{'+environment+r'\}.*?\\end\{'+environment+r'\}'
        assert re.findall(pattern,current,re.S)==re.findall(pattern,staged,re.S),(name,environment)
    cites=r'\\cite\w*(?:\[[^\]]*\])*\{[^}]*\}'
    assert re.findall(cites,current)==re.findall(cites,staged),name
    entry['staged_sha256']=hashlib.sha256(staged.encode()).hexdigest()
    entry['staged_bytes']=len(staged.encode())
    entry['formal_blocks_unchanged']=True
    entry['citations_unchanged']=True
(HERE/'entropy_staging_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
result={'status':'PASS','natural_gradient':natural,'third_mode_floor':floor,
        'exact_entropy':{'optimum':optimum,'Pstar':sum(optimum[:2]),'incorrect_percent':100*optimum[2]},
        'entropy_comparison':{'H':H,'KL_q_u':forward,'KL_u_q':reverse,'R':R,'R_at_P_09':Rnew,'loss_drop':drop},
        'gradient_availability':{'mixed_probability':h,'rare_key_absence':absence,'replay_coordinate':replay},
        'formal_blocks_and_citations_unchanged':True,
        'subsection_hashes':{name:entry['staged_sha256'] for name,entry in manifest.items()}}
(HERE/'entropy_example_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
