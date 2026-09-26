#!/usr/bin/env python3
"""Add reviewed A5000 node202 to the two already-released E122 Graph40 cells."""
from pathlib import Path
import fcntl
import release_e122_two_graph_cells_20260910 as original
import prioritize_e118_capacity_20260905 as base
import amend_e122_countdown_peers_node208_20260910 as prior

ART=original.ART/'node202_placement_20260911'
PROOF=original.ROOT/'var/artifacts/capacity_audit_20260910/e122_graph_second_slot_node202_fresh.json'
PROOF_SHA='2c9f6c29fadcd5267aa51589c2dcf4d65df41d15557111dac768774ae9af2e3a'
NODES=original.NEW_NODES|{'node202'}

def nodes(record):
    result=original.command(['scontrol','show','hostnames',base.field(record,'ReqNodeList')])
    original.require(result.returncode==0,'Cannot resolve placement pool')
    return set(result.stdout.split())

def run():
    original.verify_plan();pin=original.digest(original.PLAN)
    original.require(original.digest(PROOF)==PROOF_SHA,'Reviewed node202 proof changed')
    numeric=original.read(original.ART/'numeric_memory_reconciliation_20260911/complete.json')
    original.require(numeric['original_plan_sha256']==pin and numeric['memory_mib']==40960,'Graph40 resource qualification differs')
    ART.mkdir(parents=True,exist_ok=True)
    with (ART/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original.atomic_new(ART/'intent.json',{'at':original.now(),'source':str(Path(__file__).resolve()),'source_sha256':original.digest(__file__),'original_plan_sha256':pin,'proof_path':str(PROOF),'proof_sha256':PROOF_SHA,'new_nodes':sorted(NODES),'memory_unchanged':'40G','no_new_writer_or_submission':True})
        for jid in original.TARGETS:
            before=base.show(int(jid))
            if base.field(before,'JobState')!='PENDING':
                original.atomic_new(ART/f'{jid}.preserved_started.json',{'record':before,'scheduler_mutation':False});continue
            original.require(base.field(before,'MinMemoryNode')=='40G' and base.field(before,'Priority')!='0' and nodes(before)==original.NEW_NODES,'Expected releasedpending40GiB pool differs')
            node=original.command(['scontrol','show','node','-o','node202'])
            original.require(node.returncode==0 and 'gpu:a5000:' in base.field(node.stdout,'Gres'),'Reviewed node202 GPU differs')
            original.require(not any(x in base.field(node.stdout,'State') for x in ('DRAIN','DOWN','FAIL','UNKNOWN')),'node202 no longer healthy')
            argv=['scontrol','update',f'JobId={jid}','ReqNodeList='+','.join(sorted(NODES))]
            original.atomic_new(ART/f'{jid}.intent.json',{'at':original.now(),'before':before,'command':argv,'original_plan_sha256':pin})
            current=base.show(int(jid))
            if base.field(current,'JobState')!='PENDING':
                original.atomic_new(ART/f'{jid}.preserved_started.json',{'record':current,'scheduler_mutation':False});continue
            original.require(base.submit_tokens(current)==base.submit_tokens(before) and all(base.field(current,k)==base.field(before,k) for k in prior.PRESERVE) and nodes(current)==original.NEW_NODES,'Pending request changed during final placement audit')
            result=original.command(argv,timeout=5)
            original.atomic_new(ART/f'{jid}.result.json',{'at':original.now(),'command':argv,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            after=base.show(int(jid))
            original.require(base.submit_tokens(before)==base.submit_tokens(after) and all(base.field(before,k)==base.field(after,k) for k in prior.PRESERVE),'Frozen resource or SubmitLine changed')
            if result.returncode!=0 and base.field(after,'JobState') in ('RUNNING','CONFIGURING'):
                original.require(nodes(after)==original.NEW_NODES,'Rejected start-race changed route')
                original.atomic_new(ART/f'{jid}.preserved_started.json',{'record':after,'scheduler_rejected_race':True});continue
            original.require(result.returncode==0 and nodes(after)==NODES,'Additive202 placement not acknowledged')
            original.atomic_new(ART/f'{jid}.readback.json',{'at':original.now(),'record':after})
        original.atomic_new(ART/'complete.json',{'at':original.now(),'job_ids':list(original.TARGETS),'placement_only':True,'memory_unchanged':'40G','original_plan_sha256':pin})
        original.emit({'event':'graph40_node202_placement_complete','job_ids':list(original.TARGETS)})

if __name__=='__main__':run()
