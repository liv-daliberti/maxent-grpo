from collections import Counter
from types import SimpleNamespace
import pytest
import torch
import train_real_domains_pilot_20260921 as old
import train_real_domains_pilot_20260921_v2 as corrected
import audit_real_domains_gradient_scaling_20260921 as literal


class WidthSensitivePolicy(literal.TinyPolicy):
    def __init__(self):
        super().__init__();self.calls=[]
    def forward(self,input_ids,attention_mask):
        self.calls.append((torch.is_grad_enabled(),tuple(input_ids[0].tolist()),tuple(attention_mask[0].tolist())))
        result=super().forward(input_ids,attention_mask)
        # Emulate numerical shape dependence deterministically, so this test
        # fails if behavior scoring silently returns to the padded rectangle.
        basis=torch.arange(7,dtype=result['logits'].dtype)[None,None,:]
        result['logits']=result['logits']+.03*input_ids.shape[1]*basis
        return result


def setup(module,micro=1):
    model=WidthSensitivePolicy()
    config={**module.DEFAULTS,'group_size':4,'train_microbatch_size':micro,'max_new_tokens':8}
    learner=module.build_learner(model,SimpleNamespace(pad_token_id=6,eos_token_id=6,vocab_size=7),config,'maxrl',torch.optim.SGD(model.parameters(),lr=0.))
    rows=[]
    for i,tokens in enumerate(([2,6],[3,4,6],[4,4,5,6],[1,2,3,5,6])):
        rows.append({'request_id':str(i),'token_ids':tokens,'text':str(tokens),'verdict':{'accepted':i<2,'canonical_key':str(i) if i<2 else None,'hard_violations':[],'receipt':{}}})
    return model,learner,rows


def test_behavior_and_live_rectangles_match_without_losing_eos_or_prompt_masks():
    model,learner,rows=setup(corrected)
    info=corrected.policy_update(learner,(0,1),rows,[])
    behavior=Counter((ids,mask) for grad,ids,mask in model.calls if not grad)
    live=Counter((ids,mask) for grad,ids,mask in model.calls if grad)
    expected=Counter((tuple([0,1]+row['token_ids']),tuple([1]*(2+len(row['token_ids'])))) for row in rows)
    assert behavior==live==expected
    assert info['fresh_tokens']==sum(len(row['token_ids']) for row in rows)==14
    assert info['logprobs_diff_max']==info['logprobs_diff_min']==0
    assert info['pg_clipfrac']==0 and learner.strategy.updates==1


def test_regression_would_detect_original_padded_behavior_scoring():
    model,learner,rows=setup(old)
    info=old.policy_update(learner,(0,1),rows,[])
    behavior=Counter((ids,mask) for grad,ids,mask in model.calls if not grad)
    live=Counter((ids,mask) for grad,ids,mask in model.calls if grad)
    assert behavior!=live
    assert max(abs(info['logprobs_diff_min']),abs(info['logprobs_diff_max']))>0


@pytest.mark.parametrize('micro',[2,4])
def test_rejects_unqualified_microbatch_in_config_and_update(micro):
    raw={'model':'/cache/'+'a'*40,'model_revision':'a'*40,'adapter_module':'unused','adapter_config':{},'train_ids':['a'],'eval_ids':['b'],'train_microbatch_size':micro}
    with pytest.raises(ValueError,match='microbatch size one'):corrected.resolve_config(raw)
    model,learner,rows=setup(corrected,micro)
    with pytest.raises(ValueError,match='microbatch size one'):corrected.policy_update(learner,(0,1),rows,[])
    assert not model.calls


@pytest.mark.parametrize('arm,successes,modes', [('maxrl',3,16),('remax',3,16),('remax',0,3),('remax',16,1)])
def test_literal_production_gradient_and_adam_moments_remain_unchanged(monkeypatch,arm,successes,modes):
    monkeypatch.setattr(literal,'pilot',corrected)
    result=literal.run_case(micro=1,arm=arm,successes=successes,modes=modes,optimizer='adamw',steps=3,max_norm=.003)
    assert result['optimizer_updates']==3


def test_corrected_checkpoint_load_rejects_old_trainer_schema(tmp_path):
    class SerializablePolicy(literal.TinyPolicy):
        def save_pretrained(self,path,safe_serialization):
            path.mkdir();torch.save(self.state_dict(),path/'adapter.pt')
    model=SerializablePolicy()
    config={**corrected.DEFAULTS,'group_size':4,'train_microbatch_size':1}
    learner=corrected.build_learner(model,SimpleNamespace(pad_token_id=6,eos_token_id=6,vocab_size=7),config,'remax')
    for module in (old,corrected):
        path=tmp_path/('old' if module is old else 'corrected')
        module.save_checkpoint(path,model,learner,module.ReplayBank(),0,'same-config-hash','remax')
        if module is old:
            assert old.read_checkpoint(path,'same-config-hash','remax')[0]['completed_updates']==0
            with pytest.raises(ValueError,match='trainer schema'):
                corrected.read_checkpoint(path,'same-config-hash','remax')
        else:
            assert corrected.read_checkpoint(path,'same-config-hash','remax')[0]['completed_updates']==0
