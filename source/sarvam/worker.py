"""Two fresh full862 arms using an immutable, combined post-development harness."""
import hashlib,json,os,sys,time,traceback
from pathlib import Path
R=Path('/workspace/sarvam30b')

def main():
    sys.stdout=sys.stderr=open(R/'generation.log','a',buffering=1)
    os.environ.update(VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from budget_policy import plan,qualify,RESERVE
    from harness import final_channel,qualify as channels
    from grade_gpqa import qualify as choices
    assert all(x['passed'] for f in (qualify,channels,choices) for x in f())
    protocol=json.loads((R/'protocol.json').read_text())
    for filename,digest in protocol['solver_files'].items():
        assert hashlib.sha256((R/filename).read_bytes()).hexdigest()==digest
    rows=json.loads((R/'questions.json').read_text());assert len(rows)==862
    assert all(set(r)=={'id','benchmark','prompt_before','prompt_after','output_budget'} for r in rows)
    assert len({r['id'] for r in rows})==862
    lookup={r['id']:r for r in rows};deadline=json.loads((R/'limits.json').read_text())['deadline_unix']
    tok=AutoTokenizer.from_pretrained(R/'model',trust_remote_code=False)
    prompts={}
    for row in rows:
        for condition,field in [('baseline','prompt_before'),('combined_repaired','prompt_after')]:
            msg=[{'role':'user','content':row[field]}]
            ids=tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=True)
            text=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=True)
            assert ids==tok.encode(text,add_special_tokens=False) and ids.count(tok.bos_token_id)==1
            assert len(ids)+row['output_budget']<=12288
            prompts[condition,row['id']]={'ids':ids,'sha256':hashlib.sha256(text.encode()).hexdigest()}
    settings=protocol['generation']
    llm=LLM(model=str(R/'model'),tokenizer=str(R/'model'),trust_remote_code=True,dtype='bfloat16',
        max_model_len=12288,max_num_seqs=48,max_num_batched_tokens=4096,gpu_memory_utilization=.92,
        enforce_eager=True,enable_prefix_caching=False,generation_config='vllm',seed=settings['seed'],disable_log_stats=True)
    engine=llm.llm_engine;params={};folder=R/'generations';folder.mkdir(exist_ok=True)
    def sampling(n):
        if n not in params:params[n]=SamplingParams(temperature=0,top_p=1,max_tokens=n,seed=settings['seed'],stop_token_ids=[1,26],detokenize=False)
        return params[n]
    (R/'runtime.json').write_text(json.dumps(settings,indent=2))
    for condition in ['baseline','combined_repaired']:
        started=time.time();done=[];stage={}
        assert not any((folder/(condition+'-'+b+'.jsonl')).exists() for b in ['humaneval_plus','math500','gpqa_diamond'])
        for row in rows:
            limit=row['output_budget']-(RESERVE if condition=='combined_repaired' else 0)
            engine.add_request('prefix:'+row['id'],{'prompt_token_ids':prompts[condition,row['id']]['ids']},sampling(limit))
        while engine.has_unfinished_requests():
            if time.time()>deadline:raise TimeoutError('Declared inference deadline reached')
            for response in engine.step():
                if not response.finished:continue
                mode,k=response.request_id.split(':',1);sample=response.outputs[0];row=lookup[k]
                ids=list(sample.token_ids);end=next((i for i,x in enumerate(ids) if x in [1,26]),len(ids));ids=ids[:end]
                if condition=='combined_repaired' and mode=='prefix' and sample.finish_reason=='length':
                    p=plan(tok,ids,row['output_budget']);forced=p['forced_ids'] if p else [];stage[k]=(ids,forced)
                    remaining=row['output_budget']-len(ids)-len(forced);assert remaining>0
                    engine.add_request('final:'+k,{'prompt_token_ids':prompts[condition,k]['ids']+ids+forced},sampling(remaining));continue
                prefix,forced=stage[k] if mode=='final' else ([],[])
                total=prefix+forced+ids;assert len(total)<=row['output_budget']
                raw=tok.decode(total,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                final,method=final_channel(raw)
                record={'id':k,'task_id':k,'benchmark':row['benchmark'],'condition':condition,'completion':raw,
                    'final_completion':final,'final_channel_method':method,'completion_token_ids':total,
                    'sampled_prefix_ids':prefix,'forced_token_ids':forced,'sampled_continuation_ids':ids,
                    'new_tokens':len(total),'sampled_token_count':len(prefix)+len(ids),'policy_applied':bool(forced),
                    'truncated':sample.finish_reason=='length','finish_reason':sample.finish_reason,'stop_reason':sample.stop_reason,
                    'rendered_prompt_sha256':prompts[condition,k]['sha256'],'total_output_budget':row['output_budget']}
                with (folder/(condition+'-'+row['benchmark']+'.jsonl')).open('a') as f:f.write(json.dumps(record)+'\n')
                done.append(record)
                if len(done)%12==0:print('PROGRESS',condition,len(done),862,'seconds',round(time.time()-started,1),flush=True)
        assert len(done)==862 and {r['id'] for r in done}==set(lookup)
        meta={'condition':condition,'n':862,'seconds':time.time()-started,'new_tokens':sum(r['new_tokens'] for r in done),
            'by_benchmark':{b:{'n':sum(r['benchmark']==b for r in done),'truncated':sum(r['truncated'] for r in done if r['benchmark']==b),
                'policy_applied':sum(r['policy_applied'] for r in done if r['benchmark']==b)} for b in ['humaneval_plus','math500','gpqa_diamond']}}
        (folder/(condition+'-complete.json')).write_text(json.dumps(meta,indent=2));print('COMPLETE',json.dumps(meta),flush=True)
    (R/'all-complete.json').write_text(json.dumps({'conditions':['baseline','combined_repaired'],'n_per_condition':862}))

if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'worker-error.txt').write_text(traceback.format_exc());raise
