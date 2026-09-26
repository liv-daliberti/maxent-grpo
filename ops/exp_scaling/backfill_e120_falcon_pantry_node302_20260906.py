#!/usr/bin/env python3
"""Prepare bounded E120 Falcon Pantry backfill; scheduler mutations are separate."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import subprocess
import campaign_stats as campaign
import backfill_e120_node105_20260905 as prior
from prioritize_e118_capacity_20260905 import command, exports, field, show, submit_tokens, atomic

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'var/artifacts/e120_falcon_node302_backfill_20260906'
PLAN=ART/'plan.json'
TARGETS=(31033700,31033701)
AMEND=ART/'scheduler_amendment.md'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capacity():
    s=command(['scontrol','show','node','-o','node302']).stdout
    configured=dict(x.split('=',1) for x in field(s,'CfgTRES').split(','))
    allocated=dict(x.split('=',1) for x in field(s,'AllocTRES').split(','))
    result={'record':s,'free_memory_mib':int(field(s,'RealMemory'))-int(field(s,'AllocMem')),
      'free_cpus':int(field(s,'CPUEfctv'))-int(field(s,'CPUAlloc')),
      'free_gpus':int(configured['gres/gpu'])-int(allocated.get('gres/gpu',0)),
      'protected_128g_slots':3,'joint_memory_mib':(3*128+64)*1024,'joint_cpus':3*16+8,'joint_gpus':4}
    assert not any(z in field(s,'State') for z in ('DRAIN','DOWN','FAIL'))
    assert 'gpu:a100:' in field(s,'Gres') and 'mltheory' in field(s,'Partitions').split(',')
    assert result['free_memory_mib']>=64*1024 and result['free_cpus']>=8 and result['free_gpus']>=1
    assert int(field(s,'RealMemory'))>=result['joint_memory_mib']
    assert int(field(s,'CPUEfctv'))>=result['joint_cpus'] and int(configured['gres/gpu'])>=4
    return result


def placed(original, old, dependency=None):
    updates={'--partition':'mltheory','--account':'mltheory','--nodelist':'node302',
      '--gres':'gpu:a100:1','--exclude':prior.EXCLUSION,
      '--comment':f'e120-falcon-node302-20260906-old{old}'}
    # SubmitLine retains a satisfied September3 afterok dependency and old node
    # choices. Use the effective current record, never revive stale dependencies.
    removed=set(updates)|{'--dependency','--hold'}
    result=[t for t in original[:-1] if t.split('=',1)[0] not in removed]
    result += [f'{k}={v}' for k,v in updates.items()]
    if dependency: result.append(f'--dependency={dependency}')
    result += ['--hold',original[-1]]
    assert [t for t in result if t.startswith('--export=')]==[t for t in original if t.startswith('--export=')]
    assert 'pvl' not in ' '.join(result).lower()
    return result


def held_audit(s, item, dependency=None):
    expected={'JobState':'PENDING','Reason':'JobHeldUser','Account':'mltheory','Partition':'mltheory',
      'ReqNodeList':'node302','MinMemoryNode':'64G','NumCPUs':'8','Nice':'100',
      'TimeLimit':'3-00:00:00','Requeue':'1','ExcNodeList':prior.EXCLUSION,'WorkDir':str(ROOT)}
    for k,v in expected.items(): assert field(s,k)==v,(k,field(s,k),v)
    dep=field(s,'Dependency')
    assert dep.startswith(dependency) if dependency else dep=='(null)'
    tr=dict(v.split('=',1) for v in field(s,'ReqTRES').split(','))
    assert tr.get('gres/gpu')=='1' and tr.get('gres/gpu:a100')=='1'
    observed=submit_tokens(s)
    assert exports(observed)==exports(item['original_command']) and observed[-1]==item['original_command'][-1]
    assert field(s,'StdOut')==str(ROOT/f"slurm-{field(s,'JobId')}.out")
    assert 'pvl' not in s.lower()


def protected_holds(main, mapping):
    rows=[]
    for r in main['runs']:
        if r['model_key']!='qwen3b':continue
        jid=mapping.get(r['job_id'],r['job_id']);s=show(jid)
        assert field(s,'JobState')=='PENDING' and field(s,'Priority')=='0',jid
        rows.append({'job_id':jid,'state':field(s,'JobState'),'priority':field(s,'Priority'),'reason':field(s,'Reason')})
    assert len(rows)==10
    return rows


def prepare():
    assert not PLAN.exists(),'prepared plan already exists; inspect before overwriting'
    ART.mkdir(exist_ok=True);prior.patch_valid();main=json.loads(campaign.E120_LEDGER.read_text())
    mapping=campaign.e120_continuation_jobs(campaign.E120_LEDGER);assert len(mapping)==7
    byid={r['job_id']:r for r in main['runs']};rows=[]
    cap=capacity()
    for jid in TARGETS:
        assert jid not in mapping,'target already has registered successor'
        r=byid[jid];assert(r['model_key'],r['domain'],r['seed'])==('falcon1b','pantry_plan',55+TARGETS.index(jid))
        s=show(jid);assert field(s,'JobState')=='PENDING' and int(field(s,'Priority'))>0
        assert field(s,'Dependency')=='(null)' and 'pvl' not in s.lower()
        original=submit_tokens(s);env=exports(original)
        assert env['SAVE_PATH']==r['run_dir'] and env['RUN_STAMP']==r['run_stamp']
        assert env['OAT_ZERO_ONLINE_CANONICAL_REPLAY_KEY_WEIGHTING']=='fresh_frequency' and env['OAT_ZERO_AUTO_RESUME']=='1'
        assert field(s,'MinMemoryNode')=='64G' and field(s,'NumCPUs')=='8'
        assert field(s,'TimeLimit')=='3-00:00:00' and field(s,'Nice')=='100'
        assert not list(Path(r['run_dir']).glob('debug_job*/train_metrics.jsonl'))
        assert not (Path(r['run_dir'])/'TRAINING_COMPLETE.json').exists()
        assert not list(Path(r['run_dir']).glob('debug_job*/checkpoints/step_*'))
        cmd=placed(original,jid)
        rows.append({'original_job_id':jid,'old_job_id':jid,'run':r,'before_record':s,
           'original_command':original,'command_without_chain':cmd,'observed_step':0,
           'effective_dependency':None,'retired_submit_dependency':[v for v in original if v.startswith('--dependency=')],
           'wrapper_sha256':sha(original[-1]),'chain_after_previous':len(rows)>0})
    AMEND.write_text('''# E120 Falcon Pantry node302 backfill, 2026-09-06

Prepared operational amendment for the existing Falcon Pantry seeds55 and56. The primary45-cell scientific ledger remains immutable. Existing zero-step pending jobs31033700 and31033701 are replaced only after new held allocations pass identity and resource audits and are registered in the authoritative continuation ledger. All exported scientific/runtime variables, run directories, frozen resource-fenced training wrapper, fresh-frequency replay treatment,64GiB memory,8CPUs,72-hour walltime, Nice100 and requeue eligibility remain unchanged. Placement changes from A6000/allcs/cs to A100/mltheory/node302, with the established explicit exclusion node list.

The first replacement may use one otherwise idle64GiB slot. Seed56 depends afterany on the seed55 replacement, so these two cells never occupy more than one64GiB slot together. Configured memory515108MiB supports three128GiB allocations plus one64GiB allocation (458752MiB total), with56CPUs and4GPUs. A live headroom check is required before execution. The ten Qwen3B holds and their ledger records remain untouched. No running job is stopped.

Current effective dependencies are empty. Their original SubmitLine strings retain an already-satisfied afterok:31033669 dependency, which is deliberately not carried into new submissions. Successful same-treatment Falcon Pantry seed57 completed on64GiB/node105 A5000 (31075737). A100 placement changes scheduler hardware, not the treatment. New seed55/56 startup still requires verification.

Preparation does not hold, cancel, release or submit a real allocation. Execution waits for root confirmation that E119 memory-repair coordination is complete.
''')
    testcmd=[v for v in rows[0]['command_without_chain'] if v!='--hold'];testcmd.insert(1,'--test-only')
    test=subprocess.run(testcmd,capture_output=True,text=True)
    if test.returncode:raise RuntimeError(test.stderr or test.stdout)
    plan={'schema':'e120-falcon-pantry-node302-backfill-v1','created_at_utc':prior.now(),'status':'prepared_only',
      'main_ledger':str(campaign.E120_LEDGER),'main_ledger_sha256':sha(campaign.E120_LEDGER),
      'continuation_ledger':str(campaign.E120_CONTINUATIONS),'continuation_ledger_sha256':sha(campaign.E120_CONTINUATIONS),
      'controller_sha256':sha(__file__),'amendment':str(AMEND),'amendment_sha256':sha(AMEND),
      'runtime_amendment':str(prior.PATCH),'runtime_amendment_sha256':sha(prior.PATCH),'capacity':cap,
      'protected_qwen3b_holds':protected_holds(main,mapping),'rows':rows,'test_only':{'command':testcmd,'stdout':test.stdout,'stderr':test.stderr,'returncode':test.returncode},
      'scientific_exports_byte_identical':True,'qwen3b_holds_changed':False,'primary_ledger_immutable':True,'outcomes_inspected':False}
    atomic(PLAN,plan);(ART/'continuations.before.json').write_bytes(campaign.E120_CONTINUATIONS.read_bytes())
    print(json.dumps({'status':plan['status'],'targets':list(TARGETS),'free_memory_gib':cap['free_memory_mib']/1024,'joint_memory_gib':448,'test_only_stdout':test.stdout,'test_only_stderr':test.stderr}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true',required=True);parser.parse_args();prepare()
