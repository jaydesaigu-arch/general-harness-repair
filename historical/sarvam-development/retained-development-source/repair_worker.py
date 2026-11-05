"""Equal-output-budget forced-final policy and unforced paired controls."""
import hashlib,json,os,sys,time,traceback
from pathlib import Path
R=Path('/workspace/sarvam30b')

def main():
    sys.stdout=sys.stderr=open(R/'repair-generation.log','a',buffering=1)
    os.environ['VLLM_NO_USAGE_STATS']='1';os.environ['DO_NOT_TRACK']='1';os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    sys.path.insert(0,str(R))
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from budget_policy import plan,qualify,RESERVE
    from harness import final_channel
    from gpu_worker import MATH,CODE
    assert all(r['passed'] for r in qualify())
    tok=AutoTokenizer.from_pretrained(R/'model',trust_remote_code=False)
    questions=json.loads((R/'questions.json').read_text());assert len(questions)==664
    base={}
    for b,n in [('humaneval_plus',164),('math500',500)]:
        rows=[json.loads(x) for x in (R/'generations'/('baseline-'+b+'.jsonl')).read_text().splitlines()]
        assert len(rows)==n
        base.update({r['id']:r for r in rows})
    assert len(base)==664
    prompts={};plans={}
    for row in questions:
        k=row['id'];q=(CODE if row['benchmark']=='humaneval_plus' else MATH).format(Question=row['question'].strip())
        msgs=[{'role':'user','content':q}]
        text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,enable_thinking=True)
        assert hashlib.sha256(text.encode()).hexdigest()==base[k]['rendered_prompt_sha256']
        assert ids==tok.encode(text,add_special_tokens=False) and ids.count(tok.bos_token_id)==1
        budget=4096 if row['benchmark']=='humaneval_plus' else 8192
        prompts[k]={'ids':ids,'budget':budget,'benchmark':row['benchmark']}
        plans[k]=plan(tok,base[k]['generated_token_ids'],budget)
    eligibility={'policy_sha256':hashlib.sha256((R/'budget_policy.py').read_bytes()).hexdigest(),
       'n':664,'eligible':[k for k,p in plans.items() if p is not None],
       'by_benchmark':{b:sum(plans[k] is not None and v['benchmark']==b for k,v in prompts.items()) for b in ['humaneval_plus','math500']},
       'selection':'Structural prefix test at fixed reasoning cutoff, not future length/accuracy/test feedback.'}
    (R/'budget-eligibility.json').write_text(json.dumps(eligibility,indent=2));print('ELIGIBLE',json.dumps({k:v for k,v in eligibility.items() if k!='eligible'}),flush=True)
    llm=LLM(model=str(R/'model'),tokenizer=str(R/'model'),trust_remote_code=True,dtype='bfloat16',
       max_model_len=12288,max_num_seqs=48,max_num_batched_tokens=4096,gpu_memory_utilization=.92,
       enforce_eager=True,enable_prefix_caching=False,generation_config='vllm',seed=20261107,
       disable_log_stats=True,worker_extension_cls='runtime_audit.RouterAudit')
    engine=llm.llm_engine
    try:
        audit=llm.collective_rpc('audit_routers')
        (R/'router-fidelity-audit.json').write_text(json.dumps(audit,indent=2))
    except Exception as e:(R/'router-fidelity-audit-unavailable.txt').write_text(str(e))
    folder=R/'generations';stop=[1,26];params={}
    def sampling(n):
        if n not in params:params[n]=SamplingParams(temperature=0,top_p=1,max_tokens=n,seed=20261107,stop_token_ids=stop,detokenize=False)
        return params[n]
    def write(name,row):
        p=folder/(name+'-'+row['benchmark']+'.jsonl')
        with open(p,'a') as f:f.write(json.dumps(row)+'\n')
    names=['budget_forced','sham_control'];done={n:[] for n in names};started=time.time()
    assert not any((folder/(n+'-'+b+'.jsonl')).exists() for n in names for b in ['humaneval_plus','math500'])
    for row in questions:
        k=row['id'];p=plans[k]
        for name in names:
            if p is None:
                result=dict(base[k]);result.update(condition=name,policy_applied=False,source='unchanged saved baseline under structural policy')
                write(name,result);done[name].append(result);continue
            forced=p['forced_ids'] if name=='budget_forced' else []
            limit=prompts[k]['budget']-len(p['prefix_ids'])-len(forced)
            ids=prompts[k]['ids']+p['prefix_ids']+forced
            assert len(ids)+limit<=12288
            engine.add_request(name+':'+k,{'prompt_token_ids':ids},sampling(limit))
    while engine.has_unfinished_requests():
        for response in engine.step():
            if not response.finished:continue
            name,k=response.request_id.split(':',1);p=plans[k];o=response.outputs[0];tail=list(o.token_ids)
            end=next((i for i,x in enumerate(tail) if x in stop),len(tail));sampled_tail=tail[:end]
            forced=p['forced_ids'] if name=='budget_forced' else []
            output=p['prefix_ids']+forced+sampled_tail
            assert len(output)<=prompts[k]['budget']
            text=tok.decode(output,skip_special_tokens=True,clean_up_tokenization_spaces=False);final,method=final_channel(text)
            result={key:value for key,value in base[k].items() if key!='generated_token_ids'}
            result.update(condition=name,completion=text,final_completion=final,final_channel_method=method,
                policy_applied=name=='budget_forced',eligible_for_policy=True,completion_token_ids=output,
                sampled_prefix_ids=p['prefix_ids'],forced_token_ids=forced,sampled_continuation_ids=sampled_tail,
                new_tokens=len(output),sampled_token_count=len(p['prefix_ids'])+len(sampled_tail),
                additional_research_decode_tokens=end,truncated=o.finish_reason=='length',finish_reason=o.finish_reason,
                stop_reason=o.stop_reason,baseline_completion_sha256=hashlib.sha256(base[k]['completion'].encode()).hexdigest(),
                fixed_prefix_sha256=p['prefix_sha256'],total_output_budget=prompts[k]['budget'])
            write(name,result);done[name].append(result)
            if sum(len(v) for v in done.values())%16==0:print('PAIRED_PROGRESS',sum(len(v) for v in done.values()),1328,'seconds',round(time.time()-started,1),flush=True)
    for name in names:
        assert len(done[name])==664 and len({r['id'] for r in done[name]})==664
        meta={'condition':name,'n':664,'seconds_for_both_arms':time.time()-started,'eligible':len(eligibility['eligible']),
            'tokens':sum(r['new_tokens'] for r in done[name]),'additional_research_decode_tokens':sum(r.get('additional_research_decode_tokens',0) for r in done[name]),
            'truncated':sum(r['truncated'] for r in done[name]),'fixed_total_output_budgets':{'code':4096,'math':8192},
            'forced_marker_counted_in_budget':True,'native_runtime_same_as_baseline':True}
        (folder/(name+'-complete.json')).write_text(json.dumps(meta,indent=2));print('PAIR_COMPLETE',name,json.dumps(meta),flush=True)
    print('WAITING_FOR_FRESH_REPEAT_DECISION',flush=True)
    # Fresh repeat is a complete end-to-end execution of the same structural
    # policy; it never selects cases using scores or answers.
    commands=R/'repair-commands';commands.mkdir(exist_ok=True)
    while True:
        for path in sorted(commands.glob('*.json')):
            flag=path.with_suffix('.done')
            if flag.exists():continue
            cmd=json.loads(path.read_text())
            if cmd=={'action':'stop'}:flag.write_text('stopped');return
            assert cmd=={'action':'fresh_repaired'}
            name='fresh_repaired';assert not any((folder/(name+'-'+b+'.jsonl')).exists() for b in ['humaneval_plus','math500'])
            t=time.time();out=[];stage={}
            for row in questions:
                k=row['id'];engine.add_request('prefix:'+k,{'prompt_token_ids':prompts[k]['ids']},sampling(prompts[k]['budget']-RESERVE))
            while engine.has_unfinished_requests():
                for response in engine.step():
                    if not response.finished:continue
                    mode,k=response.request_id.split(':',1);o=response.outputs[0];ids=list(o.token_ids)
                    end=next((i for i,x in enumerate(ids) if x in stop),len(ids));ids=ids[:end]
                    if mode=='prefix' and o.finish_reason=='length':
                        p=plan(tok,ids,prompts[k]['budget']);forced=p['forced_ids'] if p else []
                        stage[k]={'prefix':ids,'forced':forced}
                        limit=prompts[k]['budget']-len(ids)-len(forced)
                        engine.add_request('final:'+k,{'prompt_token_ids':prompts[k]['ids']+ids+forced},sampling(limit));continue
                    prefix=stage[k]['prefix'] if mode=='final' else [];forced=stage[k]['forced'] if mode=='final' else []
                    total=prefix+forced+ids;assert len(total)<=prompts[k]['budget']
                    text=tok.decode(total,skip_special_tokens=True,clean_up_tokenization_spaces=False);final,method=final_channel(text)
                    result={'id':k,'task_id':k,'benchmark':prompts[k]['benchmark'],'condition':name,'completion':text,
                        'final_completion':final,'final_channel_method':method,'completion_token_ids':total,
                        'sampled_prefix_ids':prefix,'forced_token_ids':forced,'sampled_continuation_ids':ids,
                        'policy_applied':bool(forced),'new_tokens':len(total),'sampled_token_count':len(prefix)+len(ids),
                        'truncated':o.finish_reason=='length','finish_reason':o.finish_reason,'stop_reason':o.stop_reason,
                        'rendered_prompt_sha256':base[k]['rendered_prompt_sha256'],'total_output_budget':prompts[k]['budget']}
                    write(name,result);out.append(result)
                    if len(out)%8==0:print('FRESH_PROGRESS',len(out),664,'seconds',round(time.time()-t,1),flush=True)
            assert len(out)==664 and len({r['id'] for r in out})==664
            meta={'condition':name,'n':664,'seconds':time.time()-t,'tokens':sum(r['new_tokens'] for r in out),
                  'policy_applied':sum(r['policy_applied'] for r in out),'truncated':sum(r['truncated'] for r in out)}
            (folder/(name+'-complete.json')).write_text(json.dumps(meta,indent=2));flag.write_text('completed');print('FRESH_COMPLETE',json.dumps(meta),flush=True)
        time.sleep(1)

if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'repair-worker-error.txt').write_text(traceback.format_exc());raise
