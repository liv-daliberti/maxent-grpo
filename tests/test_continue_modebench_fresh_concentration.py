"""Execution recovery remains serial and cannot publish an unaudited panel."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from test_launch_modebench_fresh_concentration import (
    amended_campaign, campaign, complete_task, scheduler_reply)
import launch_modebench_fresh_concentration as launcher
import continue_modebench_fresh_concentration as m


def test_controller_only_excuses_explicit_superseded_attempts(amended_campaign):
    base,_,_,_=amended_campaign
    with patch.object(m,'BASE',base),patch.object(m.subprocess,'run',side_effect=scheduler_reply),patch.object(m,'submit') as submit:
        with pytest.raises(ValueError,match=r'without completion.*\[2\]'):
            m.cycle()
        submit.assert_not_called()


def test_controller_first_recovers_exact_falcon_interfaces(amended_campaign):
    base,plan,_,_=amended_campaign
    complete_task(base,plan,2)
    with patch.object(m,'BASE',base),patch.object(m.subprocess,'run',side_effect=scheduler_reply),patch.object(m,'submit',return_value={'job_id':'100'}) as submit:
        result=m.cycle()
    submit.assert_called_once_with([50,51],'falcon_interface_validation')
    assert result['newly_submitted']==2


def test_controller_later_selects_one_gpu_family(amended_campaign):
    base,plan,_,_=amended_campaign
    for i in (2,50,51):complete_task(base,plan,i,'NVIDIA A100 80GB PCIe' if i>=50 else 'NVIDIA RTX A5000')
    replacement={'job_id':'100','task_indices':[50,51],'intent_id':'replacement',
                 'plan_sha256':launcher.digest(base/'plan.json'),
                 'execution_amendment':{'path':str(base/'execution_amendments/falcon_a100.json'),
                                        'sha256':launcher.digest(base/'execution_amendments/falcon_a100.json')}}
    (base/'slurm/submissions/100.json').write_text(json.dumps(replacement))
    with patch.object(m,'BASE',base),patch.object(m.subprocess,'run',side_effect=scheduler_reply),patch.object(m,'submit',return_value={'job_id':'101'}) as submit:
        result=m.cycle()
    submit.assert_called_once_with([52,53],'full_panel')
    assert result['submitted']==3


def test_controller_never_launches_over_an_active_wave(amended_campaign):
    base,_,_,_=amended_campaign
    active=SimpleNamespace(stdout='99_2 RUNNING\n',stderr='')
    with patch.object(m,'BASE',base),patch.object(m.subprocess,'run',return_value=active),patch.object(m,'submit') as submit:
        result=m.cycle()
    submit.assert_not_called()
    assert result['active_scheduler_tasks']==1


def test_completing_and_configuring_jobs_count_toward_gpu_cap(amended_campaign):
    base,_,_,_=amended_campaign
    active=SimpleNamespace(stdout=''.join(f'99_{i} RUNNING\n' for i in range(7))+'99_8 COMPLETING\n99_9 CONFIGURING\n',stderr='')
    with patch.object(m,'BASE',base),patch.object(m.subprocess,'run',return_value=active),patch.object(m,'submit') as submit:
        with pytest.raises(ValueError,match='concurrency limit'):
            m.cycle()
        submit.assert_not_called()


def test_final_hardware_audit_precedes_analysis_publication(amended_campaign):
    base,plan,_,_=amended_campaign
    for i in range(len(plan['tasks'])):
        complete_task(base,plan,i,'NVIDIA A100 80GB PCIe' if i in (50,51,52) else 'NVIDIA RTX A5000')
    with patch.object(m,'BASE',base),patch.object(m.subprocess,'run',side_effect=scheduler_reply),patch.object(m,'build_report') as build,patch.object(m,'write_artifacts') as publish:
        with pytest.raises(ValueError,match='unexpected runtime hardware.*task-53'):
            m.cycle()
        build.assert_not_called()
        publish.assert_not_called()
    assert not (base/'collection_and_analysis_complete.json').exists()
