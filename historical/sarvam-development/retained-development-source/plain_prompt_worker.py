"""Full500 development candidate: remove a confusing literal placeholder.

Same checkpoint, runtime and output budget as the completed paired experiment.
Two full runs committed before observing this candidate's accuracy.
"""
import hashlib,json,os,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
R=Path('/workspace/sarvam30b')
TEMPLATE='Solve the following math problem step by step. End your response with a line that starts with Answer: followed by your final result.\n\n{Question}'
def main():
    sys.stdout=sys.stderr=open(R/'plain-prompt-generation.log','a',buffering=1)
    os.environ['VLLM_NO_USAGE_STATS']='1';os.environ['DO_NOT_TRACK']='1';os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    sys.path.insert(0,str(R))
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from budget_policy import plan,RESERVE,qualify
    from harness import final_channel
    assert all(r['passed'] for r in qualify())
    tok=AutoTokenizer.from_pretrained(R/'model',trust_remote_code=False)
    questions=[r for r in json.loads((R/'questions.json').read_text()) if r['benchmark']=='math500']
    assert len(questions)==500 and len({r['id'] for r in questions})==500
    prompts={}
    for row in questions:
        assert set(row)=={'id','question','benchmark'}
        messages=[{'role':'user','content':TEMPLATE.format(Question=row['question'].strip())}]
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=True)
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        assert ids==tok.encode(text,add_special_tokens=False) and ids.count(tok.bos_token_id)==1
        assert len(ids)+8192<=12288
        prompts[row['id']]={'ids':ids,'sha':hashlib.sha256(text.encode()).hexdigest()}
    llm=LLM(model=str(R/'model'),tokenizer=str(R/'model'),trust_remote_code=True,dtype='bfloat16',
       max_model_len=12288,max_num_seqs=48,max_num_batched_tokens=4096,gpu_memory_utilization=.92,
       enforce_eager=True,enable_prefix_caching=False,generation_config='vllm',seed=20261107,disable_log_stats=True)
    engine=llm.llm_engine;parameters={}
    def sampling(n):
        if n not in parameters:parameters[n]=SamplingParams(temperature=0,top_p=1,max_tokens=n,seed=20261107,stop_token_ids=[1,26],detokenize=False)
        return parameters[n]
    folder=R/'generations'
    for name in ['plain_prompt','plain_prompt_repeat']:
        # Retain the original job deadline. Do not begin another long full run
        # with insufficient margin for completion and artifact download.
        now=datetime.now(timezone.utc)
        if (now.hour*60+now.minute)>17*60+15:
            (folder/(name+'-not-started.json')).write_text(json.dumps({'reason':'Insufficient margin before original17:52UTC GPU deadline','time':now.isoformat()}));break
        path=folder/(name+'-math500.jsonl');assert not path.exists()
        started=time.time();stage={};done=[]
        for row in questions:engine.add_request('prefix:'+row['id'],{'prompt_token_ids':prompts[row['id']]['ids']},sampling(8192-RESERVE))
        while engine.has_unfinished_requests():
            for response in engine.step():
                if not response.finished:continue
                mode,k=response.request_id.split(':',1);o=response.outputs[0];ids=list(o.token_ids)
                end=next((i for i,x in enumerate(ids) if x in [1,26]),len(ids));ids=ids[:end]
                if mode=='prefix' and o.finish_reason=='length':
                    p=plan(tok,ids,8192);forced=p['forced_ids'] if p else []
                    stage[k]={'prefix':ids,'forced':forced}
                    engine.add_request('final:'+k,{'prompt_token_ids':prompts[k]['ids']+ids+forced},sampling(8192-len(ids)-len(forced)));continue
                prefix=stage[k]['prefix'] if mode=='final' else [];forced=stage[k]['forced'] if mode=='final' else []
                total=prefix+forced+ids;assert len(total)<=8192
                text=tok.decode(total,skip_special_tokens=True,clean_up_tokenization_spaces=False);final,method=final_channel(text)
                result={'id':k,'task_id':k,'benchmark':'math500','condition':name,'completion':text,
                    'final_completion':final,'final_channel_method':method,'completion_token_ids':total,
                    'sampled_prefix_ids':prefix,'forced_token_ids':forced,'sampled_continuation_ids':ids,
                    'policy_applied':bool(forced),'new_tokens':len(total),'sampled_token_count':len(prefix)+len(ids),
                    'truncated':o.finish_reason=='length','finish_reason':o.finish_reason,'stop_reason':o.stop_reason,
                    'rendered_prompt_sha256':prompts[k]['sha'],'total_output_budget':8192}
                with open(path,'a') as f:f.write(json.dumps(result)+'\n')
                done.append(result)
                if len(done)%8==0:print('PROGRESS',name,len(done),500,'seconds',round(time.time()-started,1),flush=True)
        assert len(done)==500 and {r['id'] for r in done}==set(prompts)
        meta={'condition':name,'n':500,'seconds':time.time()-started,'tokens':sum(r['new_tokens'] for r in done),
              'truncated':sum(r['truncated'] for r in done),'policy_applied':sum(r['policy_applied'] for r in done),
              'template':TEMPLATE,'same_fixed_checkpoint':True,'budget':8192}
        (folder/(name+'-complete.json')).write_text(json.dumps(meta,indent=2));print('COMPLETE',name,json.dumps(meta),flush=True)
    print('ALL_PLAIN_PROMPT_WORK_DONE',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'plain-prompt-error.txt').write_text(traceback.format_exc());raise
