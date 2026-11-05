"""Failure-enriched Qwen development probe; no reference answers on solver."""
import hashlib,json,os,sys,time,traceback
from pathlib import Path
R=Path('/workspace/qwen38-27b')
def main():
    sys.stdout=sys.stderr=open(R/'generation.log','a',buffering=1)
    os.environ.update(VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from adapter import policy,views,qualify as aq
    from budget_policy import qualify as bq
    from harness import qualify as hq
    from grade_gpqa import qualify as gq
    from native_policy import effort,native_plan,qualify
    assert qualify() and all(r['passed'] for f in [aq,bq,hq,gq] for r in f())
    p=json.loads((R/'protocol.json').read_text())
    for n,d in p['solver_files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==d
    rows=json.loads((R/'questions.json').read_text());assert len(rows)==96
    assert all(set(r)=={'id','benchmark','prompt_before','prompt_after','output_budget'} for r in rows)
    deadline=json.loads((R/'limits.json').read_text())['deadline_unix']
    tok=AutoTokenizer.from_pretrained(R/'model',trust_remote_code=False)
    stop=sorted(set(json.loads((R/'model/generation_config.json').read_text())['eos_token_id']+[tok.eos_token_id]));assert stop==[248044,248046]
    arms=['qwen_native_v3'];prompts={}
    for c in arms:
        for r in rows:
            e='xhigh' if c=='legacy_transfer' else effort(r['output_budget'])
            kw=dict(add_generation_prompt=True,enable_thinking=True,reasoning_effort=e)
            m=[{'role':'user','content':r['prompt_after']}]
            ids=tok.apply_chat_template(m,tokenize=True,return_dict=False,**kw)
            text=tok.apply_chat_template(m,tokenize=False,**kw)
            assert ids==tok.encode(text,add_special_tokens=False) and text.endswith('<think>\n')
            assert len(ids)+r['output_budget']<=12288
            prompts[c,r['id']]={'ids':ids,'reasoning_effort':e,'sha256':hashlib.sha256(text.encode()).hexdigest()}
    llm=LLM(model=str(R/'model'),tokenizer=str(R/'model'),trust_remote_code=False,dtype='bfloat16',
        max_model_len=12288,max_num_seqs=48,max_num_batched_tokens=4096,gpu_memory_utilization=.92,
        enforce_eager=True,enable_prefix_caching=False,generation_config='vllm',seed=20261108,
        language_model_only=True,disable_log_stats=True)
    params={}
    def sampling(n):
        if n not in params:params[n]=SamplingParams(temperature=0,top_p=1,max_tokens=n,seed=20261108,stop_token_ids=stop,detokenize=False)
        return params[n]
    def trim(ids):
        ids=list(ids);end=next((i for i,x in enumerate(ids) if x in stop),len(ids));return ids[:end]
    folder=R/'generations';folder.mkdir(exist_ok=True)
    for c in arms:
        assert not any(folder.glob(c+'-*.jsonl'));start=time.time();done=[]
        for offset in range(0,len(rows),48):
            if time.time()>deadline:raise TimeoutError('Development batch-start deadline')
            batch=rows[offset:offset+48]
            responses=llm.generate([{'prompt_token_ids':prompts[c,r['id']]['ids']} for r in batch],[sampling(r['output_budget']-1024) for r in batch],use_tqdm=False)
            saved={};requests=[];tailparams=[];keys=[]
            for r,response in zip(batch,responses):
                s=response.outputs[0];ids=trim(s.token_ids)
                if s.finish_reason=='length':
                    decision=(policy if c=='legacy_transfer' else native_plan)(tok,ids,r['output_budget'],True)
                    extra=decision['forced_ids'] if decision else []
                    saved[r['id']]=(ids,extra);keys.append(r['id'])
                    requests.append({'prompt_token_ids':prompts[c,r['id']]['ids']+ids+extra})
                    tailparams.append(sampling(r['output_budget']-len(ids)-len(extra)))
                else:saved[r['id']]=([],[],ids,s)
            if requests:
                for k,response in zip(keys,llm.generate(requests,tailparams,use_tqdm=False)):
                    prefix,extra=saved[k];s=response.outputs[0];saved[k]=(prefix,extra,trim(s.token_ids),s)
            for r in batch:
                prefix,extra,tail,s=saved[r['id']];ids=prefix+extra+tail;assert len(ids)<=r['output_budget']
                raw,logical,final,method=views(tok,ids,True)
                out=dict(id=r['id'],task_id=r['id'],benchmark=r['benchmark'],condition=c,raw_generated_text=raw,completion=logical,final_completion=final,final_channel_method=method,prompt_reasoning_open=True,
                    completion_token_ids=ids,sampled_prefix_ids=prefix,forced_token_ids=extra,sampled_continuation_ids=tail,new_tokens=len(ids),sampled_token_count=len(prefix)+len(tail),policy_applied=bool(extra),
                    truncated=s.finish_reason=='length',finish_reason=s.finish_reason,stop_reason=s.stop_reason,rendered_prompt_sha256=prompts[c,r['id']]['sha256'],reasoning_effort=prompts[c,r['id']]['reasoning_effort'],total_output_budget=r['output_budget'])
                with (folder/(c+'-'+r['benchmark']+'.jsonl')).open('a') as f:f.write(json.dumps(out)+'\n')
                done.append(out)
            progress={'condition':c,'completed':len(done),'total':96,'seconds':time.time()-start}
            (R/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
        (folder/(c+'-complete.json')).write_text(json.dumps({'condition':c,'n':96,'seconds':time.time()-start}))
    (R/'all-complete.json').write_text(json.dumps({'complete':True,'n_per_arm':96,'arms':arms}))
if __name__=='__main__':
    try:main()
    except BaseException:(R/'worker-error.txt').write_text(traceback.format_exc());raise
