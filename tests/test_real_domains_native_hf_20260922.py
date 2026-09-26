"""Focused contract tests: sharding RNG, immutable counts, seals, native sampler."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import evaluate_real_domains_native_hf_20260922 as native

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'var/artifacts/real_domains_pilot_20260921'


def config():
    training = json.loads((ART / 'diagnosis_code_corrected8_maxrl/training/identity.json').read_text())['config']
    order = training['train_ids'] + training['eval_ids']
    return dict(schema=native.SCHEMA, training_config=training, arm='base', checkpoint_step=0, checkpoint_path=None,
                validation_only=True, global_task_order=order, task_ids=[order[4]], samples_per_task_by_id={order[4]: 4},
                cohort_ids={'train': training['train_ids'], 'development': training['eval_ids']}, eval_seed=119411,
                expected_initial_trainable_parameters_sha256='cea8dd35500aa5774322c2c95429411119e7c98d2c22e1676945e1a0a5c46b48', max_gpu_hours=.25,
                training_provenance={'terminal_updates':8, 'runs': {}})


class NativeContracts(unittest.TestCase):
    def test_global_seed_does_not_depend_on_shard_position(self):
        c = native.resolve_config(config())
        raw = dict(task_id=c['task_ids'][0], sample_index=5, batch_offset=1, batch_seed=119411+40000+4,
                   text='yes', token_ids=[8,9], prompt_token_ids=[1,2,3])
        row = native.enrich_row(raw, c, {'eos_token_ids':[9]})
        self.assertEqual((row['global_task_index'],row['batch_start_index'],row['request_seed']), (4,4,159415))
        self.assertEqual(row['loss_mask'], [0,0,1,1])
        self.assertEqual(row['attention_mask'], [1]*5)
        bad = dict(raw, batch_seed=119415)
        with self.assertRaisesRegex(ValueError,'global shard-independent'):
            native.enrich_row(bad,c,{'eos_token_ids':[9]})

    def test_counts_and_order_contract(self):
        for change in ({'samples_per_task_by_id':{'988_A':3}}, {'samples_per_task_by_id':{'988_A':6}}, {'task_ids':['988_A','1454_A'],'samples_per_task_by_id':{'988_A':4,'1454_A':4}}, {'global_task_order':['988_A']}, {'eval_seed':123}):
            c=config();c.update(change)
            with self.assertRaises(ValueError): native.resolve_config(c)

    def test_validation_cannot_become_primary(self):
        c=config();c['samples_per_task_by_id']['988_A']=512
        with self.assertRaisesRegex(ValueError,'at most32'): native.resolve_config(c)
        c=config();c.update(arm='remax',checkpoint_step=32,checkpoint_path='/irrelevant')
        with self.assertRaisesRegex(ValueError,'corrected8'): native.resolve_config(c)

    def test_production_budgets_and_prefixes(self):
        c=config();plan=json.loads((ART/'code_corrected128_plan_unfrozen_unsubmitted/plan.json').read_text())
        request=json.loads((ART/'code_corrected128_plan_unfrozen_unsubmitted/train_maxrl_request.json').read_text())
        c.update(validation_only=False,training_config=request['config'],global_task_order=plan['native_hf_evaluation']['global_task_order'],
                 cohort_ids={'train':plan['dataset']['train_ids'],'development':plan['dataset']['untrained_development_ids']},
                 plan_path=str(ART/'code_corrected128_plan_unfrozen_unsubmitted/plan.json'),plan_sha256=native.trainer.digest(ART/'code_corrected128_plan_unfrozen_unsubmitted/plan.json'))
        c['training_provenance']['terminal_updates']=128;c['samples_per_task_by_id']['988_A']=512
        native.resolve_config(c)
        c['samples_per_task_by_id']['988_A']=128
        with self.assertRaisesRegex(ValueError,'predeclared'): native.resolve_config(c)
        c.update(arm='maxrl',checkpoint_step=32,checkpoint_path='/checkpoint')
        native.resolve_config(c)

    def test_summarize_retains_ineligible_zero_success_task(self):
        rows=[]
        for task_id,accepted in [('a',32),('b',0)]:
            sample=[dict(accepted=i<accepted,canonical_key=('mode'+str(i%2)) if i<accepted else None,hard_violations=[]) for i in range(32)]
            rows.append(dict(task_id=task_id,**native.metrics.mode_metrics(sample)))
        summary=native.summarize_rows(rows)
        self.assertEqual(summary['pcmd_eligible_tasks'],1)
        self.assertEqual(summary['pcmd_total_tasks'],2)
        self.assertEqual(summary['samples'],64)
        self.assertEqual(summary['macro_accuracy'],.5)
        self.assertIsNone(rows[1]['pcmd'])

    def test_model_manifest_rejects_content_and_file_set_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/('a'*40);model.mkdir()
            (model/'config.json').write_text('{}');(model/'tokenizer_config.json').write_text('{}');(model/'model.safetensors').write_text('data')
            manifest={'schema':'native-hf-model-files-20260922-v1','model_root':str(model),'model_revision':'a'*40,
                      'files':[{'relative_path':p.name,'path':str(p),'sha256':native.trainer.digest(p),'size_bytes':p.stat().st_size} for p in sorted(model.iterdir())]}
            path=root/'manifest.json';path.write_text(json.dumps(manifest))
            c={'training_config':{'model':str(model),'model_revision':'a'*40},'model_manifest_path':str(path),'model_manifest_sha256':native.trainer.digest(path)}
            self.assertEqual(native.validate_model_manifest(c)['files'],3)
            (model/'config.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'checksum drift'):native.validate_model_manifest(c)
            (model/'config.json').write_text('{}');(model/'extra').write_text('x')
            with self.assertRaisesRegex(ValueError,'complete snapshot'):native.validate_model_manifest(c)

    def test_model_manifest_rejects_unbound_changed_and_unsafe_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/('a'*40);root.mkdir()
            for name in ['config.json','tokenizer_config.json','model.safetensors']:
                (root/name).write_text(name)
            manifest={'schema':'native-hf-model-files-20260922-v1','model_root':str(root),'model_revision':'a'*40,
                      'files':[{'relative_path':p.name,'path':str(p),'sha256':native.trainer.digest(p),'size_bytes':p.stat().st_size} for p in sorted(root.iterdir())]}
            path=Path(temporary)/'manifest.json';path.write_text(json.dumps(manifest))
            c={'training_config':{'model':str(root),'model_revision':'a'*40},'model_manifest_path':str(path),'model_manifest_sha256':native.trainer.digest(path)}
            self.assertEqual(native.validate_model_manifest(c)['files'],3)
            (root/'extra.json').write_text('extra')
            with self.assertRaisesRegex(ValueError,'complete snapshot'): native.validate_model_manifest(c)
            (root/'extra.json').unlink();(root/'config.json').write_text('changeddata')
            with self.assertRaisesRegex(ValueError,'file size|checksum'): native.validate_model_manifest(c)
            (root/'config.json').write_text('config.json')
            manifest['files'][0]['relative_path']='../unsafe'
            path.write_text(json.dumps(manifest));c['model_manifest_sha256']=native.trainer.digest(path)
            with self.assertRaisesRegex(ValueError,'unsafe'): native.validate_model_manifest(c)

    def test_trainable_hash_detects_mutation(self):
        model=torch.nn.Linear(3,2)
        first=native.trainable_hash(model)
        with torch.no_grad():model.weight[0,0]+=1
        self.assertNotEqual(first,native.trainable_hash(model))

    def test_seal_drift_and_cross_arm_rejected(self):
        c=config();runs={}
        for arm in ['maxrl','remax']:
            root=ART/f'diagnosis_code_corrected8_{arm}'
            runs[arm]={}
            for key,path in [('identity',root/'training/identity.json'),('result',root/'training/result.json'),('frozen_identity',root/'identity.json')]:
                runs[arm][key+'_path']=str(path);runs[arm][key+'_sha256']=native.trainer.digest(path)
        c['training_provenance']['runs']=runs
        c.update(arm='maxrl',checkpoint_step=8,checkpoint_path=str(ART/'diagnosis_code_corrected8_maxrl/training/checkpoint-8'))
        receipt=native.validate_training_provenance(native.resolve_config(c))
        self.assertEqual(receipt['checkpoint']['seal']['completed_updates'],8)
        c['arm']='remax'
        with self.assertRaisesRegex(ValueError,'arm, step'):native.validate_training_provenance(c)
        c['arm']='maxrl';c['training_provenance']['runs']['maxrl']['identity_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'checksum drift'):native.validate_training_provenance(c)

    def test_production_sampler_preserves_eos_and_batch_seeds(self):
        class Model:
            config=type('Config',(),{'vocab_size':20})()
            generation_config=type('Generation',(),{'eos_token_id':[9,10]})()
            parameter=torch.nn.Parameter(torch.zeros(1))
            seen=[]
            def eval(self):return self
            def parameters(self):yield self.parameter
            def generate(self,input_ids,attention_mask,generation_config):
                self.seen.append((torch.initial_seed(),len(input_ids),generation_config.to_dict()))
                return torch.tensor([list(row)+[7,9,9] for row in input_ids.tolist()])
        class Tokenizer:
            eos_token_id=9;pad_token_id=9;bos_token_id=1
            def __len__(self):return 19
            def decode(self,tokens,**kwargs):return 'answer'
        class Task:
            task_id='988_A';prompt='prompt'
            def verify(self,text):return dict(accepted=False,canonical_key=None,hard_violations=[],receipt={})
        c=native.resolve_config(config());model=Model()
        with tempfile.TemporaryFile(mode='w+') as handle, patch('torch.cuda.synchronize'):
            writer=native.ReceiptWriter(handle,c,{'eos_token_ids':[9,10]},float('inf'))
            samples,_=native.trainer.generate_samples(model,Tokenizer(),Task(),[1,2],8,c['training_config'],'native_hf:base:train',0,writer,seed_base=159411)
            handle.seek(0);saved=[json.loads(line) for line in handle]
        self.assertEqual([row[0] for row in model.seen],[159411,159415])
        self.assertTrue(all(row['token_ids']==[7,9] and row['finish_reason']=='eos' for row in samples))
        self.assertEqual([row['sample_index'] for row in saved],list(range(8)))
        generation=model.seen[0][2]
        self.assertEqual((generation['temperature'],generation['top_k'],generation['top_p']),(1.,0,1.))
        self.assertEqual(generation['suppress_tokens'],[19]);self.assertTrue(generation['use_cache'])
        self.assertEqual(generation['eos_token_id'],[9,10])


if __name__=='__main__':unittest.main()
