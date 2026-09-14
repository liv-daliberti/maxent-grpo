#!/usr/bin/env python3
"""Complete the approved E122 Graph40 route using this Slurm client's MiB syntax."""
from pathlib import Path
import fcntl,json,time
import release_e122_two_graph_cells_20260910 as original

ART=original.ART/'numeric_memory_reconciliation_20260911'
SOURCE=Path(__file__).resolve()

def run():
    plan=original.verify_plan();pin=original.digest(original.PLAN)
    packet=original.read(original.ART/'worker_request.json')
    original.require(packet['plan_sha256']==pin and packet['campaign']['binding']==plan['binding'],'Successful release packet differs')
    for jid in original.TARGETS:
        intent=original.JOURNAL/'jobs'/f'{jid}.intent.json'
        result=original.read(intent.with_name(f'{jid}.result.json'))
        original.require(original.read(intent)['operational_amendment_sha256']==pin,'Unowned original release intent')
        original.require(result['returncode']==0 and not result.get('error') and result['intent_sha256']==original.digest(intent) and result['command']==['scontrol','release',jid],'Original release not positively acknowledged')
    failed=original.read(original.ART/'31158698.route.result.json')
    original.require(failed['returncode']==1 and failed['stderr']=='scontrol: error: Invalid MinMemoryNode value: 40G\n','Unexpected prior resource outcome')
    # Both original IDs are already released and fully budgeted. This40GiB
    # resource encoding/placement repair adds no writer or checkpoint reserve.
    ART.mkdir(parents=True,exist_ok=True)
    with (ART/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        original.atomic_new(ART/'intent.json',{'at':original.now(),'source':str(SOURCE),'source_sha256':original.digest(SOURCE),'original_plan_sha256':pin,'job_ids':list(original.TARGETS),'original_route_failure':failed,'scope':'Resource-only40GiB and same approved pool; no release command','memory_mib':40960,'memory_gib':40})
        command=original.command
        def numeric_command(argv,timeout=3):
            if argv[:2]==['scontrol','update']:
                original.require(len(argv)==5 and argv[2] in [f'JobId={j}' for j in original.TARGETS] and argv[3]=='ReqNodeList='+original.POOL and argv[4]=='MinMemoryNode=40G','Resource operation exceeds approved scope')
                actual=[*argv[:-1],'MinMemoryNode=40960'];jid=argv[2].split('=',1)[1]
                original.atomic_new(ART/f'{jid}.numeric_wire_command.json',{'at':original.now(),'requested_command':argv,'actual_command':actual,'equivalence':'40960MiB equals40GiB','original_plan_sha256':pin})
                return command(actual,timeout=timeout)
            original.require(argv[:2]!=['scontrol','release'],'Never repeat release')
            return command(argv,timeout=timeout)
        original.ART=ART;original.command=numeric_command
        original.route_pair(packet['campaign'])
        original.atomic_new(ART/'complete.json',{'at':original.now(),'original_plan_sha256':pin,'job_ids':list(original.TARGETS),'resource_only':True,'memory_mib':40960,'no_training_submission':True,'no_release_repeated':True})
        original.emit({'event':'numeric_graph40_resource_reconciliation_complete','job_ids':list(original.TARGETS),'memory_mib':40960})

if __name__=='__main__':run()
