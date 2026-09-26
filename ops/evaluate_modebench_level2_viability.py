#!/usr/bin/env python3
"""Paired frozen-model Level-1/Level-2 ModeBench pass@8 evaluation."""
from __future__ import annotations
import argparse,hashlib,itertools,json,os,re,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def sha(value:Any)->str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def file_sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()
def atomic(path:Path,payload:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f: json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')
        os.replace(tmp,path)
    except BaseException:
        os.unlink(tmp); raise

def args():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--domain',required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--model-label',required=True,choices=('qwen-0.5b','falcon-1b'))
    ap.add_argument('--level2-root',type=Path,default=ROOT/'var/data/modebench_harder_v2_matched')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--temperature',type=float,default=1.0)
    ap.add_argument('--top-p',type=float,default=1.0)
    ap.add_argument('--max-tokens',type=int,default=192)
    ap.add_argument('--max-model-len',type=int,default=1024)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--dtype',choices=('float16','bfloat16'),default='float16')
    ap.add_argument('--prompt-profile',choices=('boxed_direct_v1','deliberate_domain_v2','structured_solver_v3','hybrid_solver_v4','countdown_fewshot_v5','countdown_shallow_v6'),default='boxed_direct_v1')
    ap.add_argument('--row-limit',type=int,default=0,help='development-only calibration limit; zero means all 128 rows')
    ap.add_argument('--syntax-profile',choices=('none','pantry_legal_v1','domain_legal_v1','domain_legal_v2','countdown_legal_v3'),default='none')
    return ap.parse_args()

def prompt_messages(domain:str,problem:str,profile:str)->list[dict[str,str]]:
    if profile=='countdown_shallow_v6':
        if domain!='countdown':
            raise ValueError('countdown_shallow_v6 is Countdown-only')
        system=('Use every supplied number exactly once and hit the target exactly. '
                'First test permutations of paired products a*b+c*d or a*b-c*d, '
                'paired sums (a+b)*(c+d) or (a+b)*(c-d), and one product '
                'a*b followed by adding or subtracting c and d. '
                'Return only one fully parenthesized expression inside \\boxed{}.')
        return [{'role':'system','content':system},{'role':'user','content':problem}]
    if profile=='countdown_fewshot_v5':
        if domain!='countdown':
            raise ValueError('countdown_fewshot_v5 is Countdown-only')
        system=('Solve the arithmetic target exactly. Use every supplied number exactly once. '
                'Return only one fully parenthesized expression inside \\boxed{}.')
        return [
          {'role':'system','content':system},
          {'role':'user','content':'Using the numbers [2, 3, 4], create an arithmetic expression that equals 14. Use each given number exactly once.'},
          {'role':'assistant','content':'\\boxed{(2 + (3 * 4))}'},
          {'role':'user','content':'Using the numbers [2, 3, 4, 5], create an arithmetic expression that equals 26. Use each given number exactly once.'},
          {'role':'assistant','content':'\\boxed{((2 * 3) + (4 * 5))}'},
          {'role':'user','content':problem},
        ]
    if profile=='boxed_direct_v1':
        system='Return only the final answer inside \\boxed{}. Do not explain.'
        return [{'role':'system','content':system},{'role':'user','content':problem}]
    deliberate={
      'countdown':'Systematically combine every supplied number exactly once using +, -, *, /, and parentheses. Check the exact target before answering.',
      'python_factors':'Construct one allowed lambda expression. Test small divisors with nested conditional expressions, for example 2 if n % 2 == 0 else 3 if n % 3 == 0 else 5, but adapt the tests to every listed case.',
      'mathir':'Execute candidate menu actions exactly on both sides, simplify after each action, and check that the final state isolates x. Return action IDs, not x.',
      'pantry':'Silently enumerate allowed stepped quantities for 2 to 4 non-forbidden ingredients, total every nutrient exactly, and check all bounds.',
      'graph_coloring':'Check every edge after assigning the hidden vertices.',
    }
    structured={
      'countdown':'Search systematically over pairwise combinations until every number is used exactly once. Verify the arithmetic and output exactly the boxed expression.',
      'python_factors':'Output exactly a boxed lambda. A reliable form is d1 if n == c1 else d2 if n == c2 else d3 if n == c3 else d4, choosing each di as a proper divisor of ci.',
      'mathir':'Use algebraic isolation: move the right-side x term left, remove the left constant, then divide by the combined coefficient. Match those operations to the shuffled menu IDs.',
      'pantry':'Prefer allowed high-energy/protein, very-low-sodium ingredients, especially seeds or oats. Choose stepped amounts, check every bound, and output 2 to 4 ingredient_id=grams pairs.',
      'graph_coloring':'Check every edge after assigning the hidden vertices.',
    }
    if profile=='deliberate_domain_v2':
        guidance=deliberate[domain]
    elif profile=='structured_solver_v3':
        guidance=structured[domain]
    else:
        guidance={
          'countdown':deliberate['countdown']+' Output exactly the boxed expression.',
          'python_factors':deliberate['python_factors']+' You may instead dispatch on each listed value. Output exactly the boxed lambda.',
          'mathir':structured['mathir'],
          'pantry':structured['pantry'],
          'graph_coloring':deliberate['graph_coloring'],
        }[domain]
    system=('Solve the executable constraint problem carefully. You may reason briefly, but end with exactly one final answer inside \\boxed{}. '+guidance)
    return [{'role':'system','content':system},{'role':'user','content':problem}]

def countdown_legal_choices(row:dict)->list[str]:
    """Enumerate syntax-legal expressions without consulting the target or verifier."""
    numbers=tuple(str(int(x)) for x in json.loads(str(row['answer']))['numbers'])
    operators=('+','-','*','/')
    choices=set()
    def trees(values,ops):
        if len(values)==1:
            return (values[0],)
        out=[]
        for split in range(1,len(values)):
            for left in trees(values[:split],ops[:split-1]):
                for right in trees(values[split:],ops[split:]):
                    out.append(f'({left} {ops[split-1]} {right})')
        return tuple(out)
    for values in set(itertools.permutations(numbers)):
        for ops in itertools.product(operators,repeat=len(values)-1):
            choices.update(f'\\boxed{{{expression}}}' for expression in trees(values,ops))
    return sorted(choices)

def countdown_legal_regex(row:dict)->str:
    """Compact target-blind regex for every operand permutation and tree shape."""
    numbers=tuple(str(int(x)) for x in json.loads(str(row['answer']))['numbers'])
    operator=r'[+*/-]'
    def trees(values):
        if len(values)==1:
            return (re.escape(values[0]),)
        out=[]
        for split in range(1,len(values)):
            for left in trees(values[:split]):
                for right in trees(values[split:]):
                    out.append(r'\('+left+' '+operator+' '+right+r'\)')
        return tuple(out)
    expressions=set()
    for values in set(itertools.permutations(numbers)):
        expressions.update(trees(values))
    return r'\\boxed\{(?:'+'|'.join(sorted(expressions))+r')\}'

def sampling_params(a,domain:str,row:dict):
    import vllm
    guided=None
    if a.syntax_profile=='countdown_legal_v3' and domain=='countdown':
        from vllm.sampling_params import GuidedDecodingParams
        guided=GuidedDecodingParams(regex=countdown_legal_regex(row))
    elif a.syntax_profile in ('pantry_legal_v1','domain_legal_v1','domain_legal_v2') and domain=='pantry':
        from vllm.sampling_params import GuidedDecodingParams
        spec=json.loads(str(row['answer']))
        alternatives=[]
        for ingredient in spec['ingredients']:
            ident=str(ingredient['id'])
            minimum=int(ingredient['min_if_used_g'])
            available=int(ingredient['available_g'])
            step=int(ingredient['step_g'])
            for grams in range(minimum,available+1,step):
                alternatives.append(f'{ident}={grams}')
        atom='(?:'+'|'.join(re.escape(x) for x in alternatives)+')'
        regex=r'\\boxed\{'+atom+'(?:;'+atom+r'){1,3}\}'
        guided=GuidedDecodingParams(regex=regex)
    elif a.syntax_profile in ('domain_legal_v1','domain_legal_v2'):
        from vllm.sampling_params import GuidedDecodingParams
        boxed={
          'countdown':r'\\boxed\{[0-9 +*/().-]+\}',
          'python_factors':r'\\boxed\{lambda n: [A-Za-z0-9 _%<>=!+*/().-]+\}',
          'mathir':r'\\boxed\{[A-F](?:;[A-F]){0,3}\}',
        }
        optional={
          'countdown':r'(?:\\boxed\{[0-9 +*/().-]+\}|[0-9][0-9 +*/().-]*)',
          'python_factors':boxed['python_factors'],
          'mathir':r'(?:\\boxed\{[A-F](?:;[A-F]){0,3}\}|[A-F](?:;[A-F]){0,3})',
        }
        regex=(boxed if a.syntax_profile=='domain_legal_v1' else optional).get(domain)
        if regex is not None:
            guided=GuidedDecodingParams(regex=regex)
    return vllm.SamplingParams(n=8,temperature=a.temperature,top_p=a.top_p,max_tokens=a.max_tokens,seed=a.seed,guided_decoding=guided)

def main():
    a=args()
    if a.output.exists(): raise FileExistsError(f'fresh receipt required: {a.output}')
    import vllm
    from datasets import load_from_disk
    from oat_drgrpo.math_grader import validated_modebench_outcome_key
    identity=json.loads((a.level2_root/'identity.json').read_text())
    if identity.get('decision')!='structurally_admitted_pending_frozen_base_model_viability':
        raise RuntimeError('Level-2 structural admission is not frozen and passing')
    if a.domain not in identity['domains']: raise ValueError(a.domain)
    level1=ROOT/identity['level1_reference'][a.domain]['dev']
    rows_by_level={
      'level1':[dict(x) for x in load_from_disk(str(level1))['multi_answer']],
      'level2':[dict(x) for x in load_from_disk(str(a.level2_root/a.domain/'dev'))['multi_answer']],
    }
    if a.row_limit < 0: raise ValueError('row-limit must be nonnegative')
    if a.row_limit: rows_by_level={level:rows[:a.row_limit] for level,rows in rows_by_level.items()}
    llm=vllm.LLM(model=str(a.model.resolve()),dtype=a.dtype,max_model_len=a.max_model_len,gpu_memory_utilization=.82,swap_space=16.0,enable_prefix_caching=True)
    tokenizer=llm.get_tokenizer()
    system='Return only the final answer inside \\boxed{}. Do not explain.'
    results={}
    for level,rows in rows_by_level.items():
        prompts=[tokenizer.apply_chat_template(prompt_messages(a.domain,str(row['problem']),a.prompt_profile),tokenize=False,add_generation_prompt=True) for row in rows]
        outputs=[]
        for start in range(0,len(prompts),a.batch_size):
            batch_rows=rows[start:start+a.batch_size]
            batch_params=[sampling_params(a,a.domain,row) for row in batch_rows]
            outputs.extend(llm.generate(prompts[start:start+a.batch_size],batch_params))
        if len(outputs)!=len(rows): raise RuntimeError('vLLM output count mismatch')
        prompt_results=[]
        for index,(row,out) in enumerate(zip(rows,outputs)):
            if len(out.outputs)!=8: raise RuntimeError(f'{level}/{index}: expected 8 samples')
            attempts=[]
            for sample in out.outputs:
                text=str(sample.text); key=validated_modebench_outcome_key(text,row['answer'])
                attempts.append({'text':text,'verified':key is not None,'canonical_key':key,'token_count':len(sample.token_ids)})
            prompt_results.append({'row_index':index,'passed':any(x['verified'] for x in attempts),'verified_count':sum(x['verified'] for x in attempts),'attempts':attempts})
        success=sum(x['passed'] for x in prompt_results)
        results[level]={'rows':len(rows),'success_prompts':success,'pass_at_8':success/len(rows),'rows_sha256':sha(rows),'prompt_results':prompt_results}
    l1=results['level1']['pass_at_8']; l2=results['level2']['pass_at_8']
    admitted=.10<=l2<=.90 and l2<l1
    payload={'schema':'modebench-level2-paired-base-viability-v1','generated_at':datetime.now(timezone.utc).isoformat(),'status':'pass' if admitted else 'fail','decision':'admit_domain_for_treatment_training' if admitted else 'reject_or_revise_domain_before_training','domain':a.domain,'model_label':a.model_label,'model':str(a.model.resolve()),'model_config_sha256':file_sha(a.model/'config.json'),'level2_identity_sha256':file_sha(a.level2_root/'identity.json'),'sampling':{'sample_count':8,'temperature':a.temperature,'top_p':a.top_p,'max_tokens':a.max_tokens,'max_model_len':a.max_model_len,'dtype':a.dtype,'seed':a.seed,'prompt_template':'model_native_chat_template','prompt_profile':a.prompt_profile,'row_limit':a.row_limit,'syntax_profile':a.syntax_profile},'criteria':{'minimum_level2_pass_at_8':.10,'maximum_level2_pass_at_8':.90,'level2_strictly_lower_than_level1':True},'results':results,'checks':{'level2_not_effectively_zero':l2>=.10,'level2_not_too_easy':l2<=.90,'level2_harder_than_level1':l2<l1},'information_boundary':{'development_only':True,'calibration_only':bool(a.row_limit),'evaluation_prompts_loaded':False,'treatment_training_started':False}}
    atomic(a.output,payload)
    print(json.dumps({'status':payload['status'],'domain':a.domain,'model':a.model_label,'level1_pass_at_8':l1,'level2_pass_at_8':l2,'output':str(a.output)},sort_keys=True))
if __name__=='__main__': main()
