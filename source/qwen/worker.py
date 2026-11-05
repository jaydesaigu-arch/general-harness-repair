"""Frozen, question-only Qwen full-suite pair; pilot contains no benchmark items."""
import hashlib,json,os,sys,time,traceback
from pathlib import Path
R=Path('/workspace/qwen38-27b')

def main():
    sys.stdout=sys.stderr=open(R/'generation.log','a',buffering=1)
    os.environ.update(VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from adapter import policy,views,qualify as adapter_controls
    from budget_policy import qualify as budget_controls
    from harness import qualify as channel_controls
    from grade_gpqa import qualify as choice_controls
    assert all(r['passed'] for f in [adapter_controls,budget_controls,channel_controls,choice_controls] for r in f())
    protocol=json.loads((R/'protocol.json').read_text());settings=protocol['generation']
    for n,d in protocol['solver_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==d
    limits=json.loads((R/'limits.json').read_text());deadline=limits['deadline_unix']
    tok=AutoTokenizer.from_pretrained(R/'model',trust_remote_code=False)
    native=json.loads((R/'model/generation_config.json').read_text())
    stop=native['eos_token_id'];stop=[stop] if isinstance(stop,int) else stop
    stop=sorted(set(stop+[tok.eos_token_id]));assert stop==[248044,248046]
    forced=tok.encode('</think>\n',add_special_tokens=False)
    assert tok.decode(forced,skip_special_tokens=False,clean_up_tokenization_spaces=False)=='</think>\n'
    def prompt(text):
        messages=[{'role':'user','content':text}]
        kwargs={'add_generation_prompt':True,'enable_thinking':True,'reasoning_effort':'xhigh'}
        ids=tok.apply_chat_template(messages,tokenize=True,return_dict=False,**kwargs)
        rendered=tok.apply_chat_template(messages,tokenize=False,**kwargs)
        assert ids==tok.encode(rendered,add_special_tokens=False)
        opened=rendered.endswith('<think>\n');assert opened
        return {'ids':ids,'opened_in_prompt':opened,'sha256':hashlib.sha256(rendered.encode()).hexdigest()}
    rows=json.loads((R/'questions.json').read_text());assert len(rows)==len({r['id'] for r in rows})==862
    assert all(set(r)=={'id','benchmark','prompt_before','prompt_after','output_budget'} for r in rows)
    prompts={(c,r['id']):prompt(r[field]) for c,field in [('baseline','prompt_before'),('combined_repaired','prompt_after')] for r in rows}
    assert all(len(prompts[c,r['id']]['ids'])+r['output_budget']<=12288 for c in ['baseline','combined_repaired'] for r in rows)
    model_started=time.time()
    llm=LLM(model=str(R/'model'),tokenizer=str(R/'model'),trust_remote_code=False,dtype='bfloat16',
        max_model_len=12288,max_num_seqs=48,max_num_batched_tokens=4096,gpu_memory_utilization=.92,
        enforce_eager=True,enable_prefix_caching=False,generation_config='vllm',seed=settings['seed'],
        language_model_only=True,disable_log_stats=True)
    model_seconds=time.time()-model_started
    # Synthetic fixed-length throughput check, deliberately separate from scored data.
    synthetic=[prompt('Write Python functions that calculate running totals, means, minima and maxima. Explain their invariants. Use the variable name item_'+str(i)+'.') for i in range(48)]
    timing=[]
    for batch in range(2):
        started=time.time()
        outputs=llm.generate([{'prompt_token_ids':p['ids']} for p in synthetic],
            SamplingParams(temperature=0,top_p=1,max_tokens=512,min_tokens=512,ignore_eos=True,seed=settings['seed']),use_tqdm=False)
        tokens=sum(len(r.outputs[0].token_ids) for r in outputs)
        timing.append({'batch':batch,'seconds':time.time()-started,'tokens':tokens})
    check={'ready':True,'model_load_seconds':model_seconds,'synthetic_batches':timing,
        'warm_tokens_per_second':timing[-1]['tokens']/timing[-1]['seconds'],'stop_ids':stop,'forced_ids':forced,
        'opened_reasoning_marker_is_in_input_template':True,'all_synthetic_controls_passed':True}
    (R/'pilot-ready.json').write_text(json.dumps(check,indent=2));print('PILOT_READY',json.dumps(check),flush=True)
    wait_started=time.time()
    while not (R/'GO.json').exists():
        if (R/'STOP.json').exists():raise RuntimeError('Controller declined full inference after cost check')
        if time.time()-wait_started>240:raise TimeoutError('No full-run decision within four minutes')
        time.sleep(2)
    params={}
    def sampling(n):
        if n not in params:params[n]=SamplingParams(temperature=0,top_p=1,max_tokens=n,seed=settings['seed'],stop_token_ids=stop,detokenize=False)
        return params[n]
    def trim(ids):
        ids=list(ids);end=next((i for i,x in enumerate(ids) if x in stop),len(ids));return ids[:end]
    folder=R/'generations';folder.mkdir(exist_ok=True)
    for c in ['baseline','combined_repaired']:
        started=time.time();done=[]
        assert not any((folder/(c+'-'+b+'.jsonl')).exists() for b in ['humaneval_plus','math500','gpqa_diamond'])
        for offset in range(0,len(rows),48):
            if time.time()>deadline:raise TimeoutError('Declared inference deadline reached')
            batch=rows[offset:offset+48]
            outputs=llm.generate([{'prompt_token_ids':prompts[c,r['id']]['ids']} for r in batch],
                [sampling(r['output_budget']-(1024 if c=='combined_repaired' else 0)) for r in batch],use_tqdm=False)
            saved={};continuations=[];continuation_params=[];continuation_ids=[]
            for row,response in zip(batch,outputs):
                s=response.outputs[0];ids=trim(s.token_ids);p=prompts[c,row['id']]
                if c=='combined_repaired' and s.finish_reason=='length':
                    plan=policy(tok,ids,row['output_budget'],p['opened_in_prompt']);extra=plan['forced_ids'] if plan else []
                    saved[row['id']]=(ids,extra)
                    continuations.append({'prompt_token_ids':p['ids']+ids+extra})
                    continuation_params.append(sampling(row['output_budget']-len(ids)-len(extra)))
                    continuation_ids.append(row['id'])
                else:saved[row['id']]=([],[],ids,s)
            if continuations:
                tails=llm.generate(continuations,continuation_params,use_tqdm=False)
                for k,response in zip(continuation_ids,tails):
                    prefix,extra=saved[k];s=response.outputs[0];saved[k]=(prefix,extra,trim(s.token_ids),s)
            for row in batch:
                prefix,extra,tail,s=saved[row['id']];ids=prefix+extra+tail
                assert len(ids)<=row['output_budget']
                p=prompts[c,row['id']];raw,logical,final,method=views(tok,ids,p['opened_in_prompt'])
                record={'id':row['id'],'task_id':row['id'],'benchmark':row['benchmark'],'condition':c,
                    'raw_generated_text':raw,'completion':logical,'final_completion':final,'final_channel_method':method,
                    'prompt_reasoning_open':p['opened_in_prompt'],'completion_token_ids':ids,'sampled_prefix_ids':prefix,
                    'forced_token_ids':extra,'sampled_continuation_ids':tail,'new_tokens':len(ids),
                    'sampled_token_count':len(prefix)+len(tail),'policy_applied':bool(extra),'truncated':s.finish_reason=='length',
                    'finish_reason':s.finish_reason,'stop_reason':s.stop_reason,'rendered_prompt_sha256':p['sha256'],
                    'total_output_budget':row['output_budget']}
                with (folder/(c+'-'+row['benchmark']+'.jsonl')).open('a') as f:f.write(json.dumps(record)+'\n')
                done.append(record)
            print('PROGRESS',c,len(done),862,'seconds',round(time.time()-started,1),flush=True)
            (R/'progress.json').write_text(json.dumps({'condition':c,'completed':len(done),'seconds':time.time()-started,'tokens':sum(r['new_tokens'] for r in done)}))
        assert len(done)==862 and {r['id'] for r in done}=={r['id'] for r in rows}
        meta={'condition':c,'n':862,'seconds':time.time()-started,'new_tokens':sum(r['new_tokens'] for r in done)}
        (folder/(c+'-complete.json')).write_text(json.dumps(meta,indent=2));print('COMPLETE',json.dumps(meta),flush=True)
    (R/'all-complete.json').write_text(json.dumps({'conditions':['baseline','combined_repaired'],'n_per_condition':862}))

if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'worker-error.txt').write_text(traceback.format_exc());raise
