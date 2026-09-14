#!/usr/bin/env python3
"""Fill remaining eligible E122 GPU/memory slots after the first finite batch."""
from pathlib import Path
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'ops/exp_scaling'))
p=ROOT/'ops/exp_scaling/expand_e122_finite_20260912.py'
assert hashlib.sha256(p.read_bytes()).hexdigest() == 'db0740a5c34286c4b64b58c6a2cafce6c1f1e121d562cedae507e770ea2f68ee', 'Reviewed base helper changed'
import expand_e122_finite_20260912 as m
m.SOURCE=Path(__file__).resolve()
m.ART=ROOT/'var/artifacts/e122_finite_expansion_remaining_20260912'
m.PLAN=m.ART/'plan.json'
m.CANDIDATES=['31158702','31158703','31158718','31158719']
original_new=m.new

def bound_new(path,value):
    if path==m.PLAN:
        value['selection']='Graph seed44 Dr.GRPO pair, then MathIR seed43 Dr.GRPO pair; truncate only to measured remaining GPU/CPU/memory capacity.'
        value['source_pins'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return original_new(path,value)
m.new=bound_new
if __name__=='__main__':m.main()
