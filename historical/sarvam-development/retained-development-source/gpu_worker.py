"""Pinned, fixed-checkpoint inference. No reference data or training code."""
import hashlib,json,os,sys,time,traceback
from pathlib import Path
R=Path('/workspace/sarvam30b')
MATH=r'''Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \boxed command.'''
CODE='Complete the following Python function. Return the complete Python code, including necessary imports, in one Python code block. Preserve the function name and signature.\n\n```python\n{Question}\n```'

def router_probe(model):
    """Synthetic fidelity audit, not a benchmark-dependent modification."""
    import torch
    rows=[]
    for name,m in model.named_modules():
        w=getattr(m,'weight',None)
        if not name.endswith('.gate') or w is None or tuple(w.shape)!=(128,4096):continue
        generator=torch.Generator(device=w.device).manual_seed(20261107)
        x=torch.randn((128,4096),device=w.device,dtype=torch.bfloat16,generator=generator)
        logits=torch.nn.functional.linear(x.float(),w.float())
        bias=getattr(m,'expert_bias',None)
        if bias is None:bias=getattr(m,'e_score_correction_bias',None)
        if bias is None:continue
        ref=logits.sigmoid()+bias.float()
        rounded=logits.to(torch.bfloat16).float().sigmoid()+bias.float()
        a=ref.topk(6,dim=-1).indices.sort(dim=-1).values
        b=rounded.topk(6,dim=-1).indices.sort(dim=-1).values
        rows.append({'name':name,'tokens':128,'changed_top6_sets':int((a!=b).any(dim=-1).sum()),
                     'router_weight_dtype':str(w.dtype),'router_weight_norm':float(w.float().norm())})
    return rows

def main():
    sys.stdout=sys.stderr=open(R/'generation.log','a',buffering=1)
    os.environ['VLLM_NO_USAGE_STATS']='1';os.environ['DO_NOT_TRACK']='1';os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    import torch,transformers,vllm
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from harness import final_channel,qualify
    assert all(r['passed'] for r in qualify())
    tok=AutoTokenizer.from_pretrained(R/'model',trust_remote_code=False)
    config=json.loads((R/'model/config.json').read_text());gc=json.loads((R/'model/generation_config.json').read_text())
    stop=set([tok.eos_token_id])
    for obj in [config,gc]:
        eos=obj.get('eos_token_id',[]);stop.update(eos if isinstance(eos,list) else [eos])
    stop=sorted(x for x in stop if x is not None);assert stop==[1,26]
    rows=json.loads((R/'questions.json').read_text());assert len(rows)==664
    prompts={}
    for row in rows:
        assert set(row)==({'id','benchmark','question','entry_point'} if row['benchmark']=='humaneval_plus' else {'id','benchmark','question'})
        template=CODE if row['benchmark']=='humaneval_plus' else MATH
        content=template.format(Question=row['question'].strip())
        msgs=[{'role':'user','content':content}]
        ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,enable_thinking=True)
        rendered=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=True)
        assert ids==tok.encode(rendered,add_special_tokens=False) and ids.count(tok.bos_token_id)==1
        limit=4096 if row['benchmark']=='humaneval_plus' else 8192
        assert len(ids)+limit<=12288
        prompts[row['id']]={'prompt_token_ids':ids,'rendered_prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest(),'limit':limit}
    settings={'model_revision':'071ae95e933605ca1104a6b4524a36a98488efa4','vllm':vllm.__version__,'torch':torch.__version__,
       'transformers':transformers.__version__,'dtype':'bfloat16','router':'author backend default','max_model_len':12288,
       'max_num_seqs':48,'max_num_batched_tokens':4096,'gpu_memory_utilization':.92,'enforce_eager':True,
       'enable_prefix_caching':False,'temperature':0.,'top_p':1.,'seed':20261107,'stop_token_ids':stop,
       'thinking':True,'code_max_new_tokens':4096,'math_max_new_tokens':8192,'literal_decode':True}
    (R/'tokenizer-preflight.json').write_text(json.dumps({'bos':tok.bos_token_id,'tokenizer_eos':tok.eos_token_id,
        'model_eos':config['eos_token_id'],'generation_eos':gc['eos_token_id'],'stop_union':stop,
        'native_cleanup':tok.clean_up_tokenization_spaces,'all664_chat_roundtrips_pass':True,
        'max_prompt_tokens':max(len(p['prompt_token_ids']) for p in prompts.values()),'channel_controls':qualify()},indent=2))
    print('LOADING_MODEL',flush=True)
    llm=LLM(model=str(R/'model'),tokenizer=str(R/'model'),trust_remote_code=True,dtype='bfloat16',
        max_model_len=12288,max_num_seqs=48,max_num_batched_tokens=4096,gpu_memory_utilization=.92,
        enforce_eager=True,enable_prefix_caching=False,generation_config='vllm',seed=20261107,disable_log_stats=True)
    (R/'runtime.json').write_text(json.dumps(settings,indent=2));engine=llm.llm_engine
    try:
        probe=llm.apply_model(router_probe)
        (R/'synthetic-router-probe.json').write_text(json.dumps(probe,indent=2))
    except Exception as e:(R/'synthetic-router-probe-unavailable.txt').write_text(str(e))
    params={n:SamplingParams(temperature=0,top_p=1,max_tokens=n,seed=20261107,stop_token_ids=stop,detokenize=False) for n in [4096,8192]}
    folder=R/'generations';folder.mkdir(exist_ok=True)
    def generate(name):
        assert name in ['baseline','retest'];paths={b:folder/(name+'-'+b+'.jsonl') for b in ['humaneval_plus','math500']}
        assert not any(p.exists() for p in paths.values()),'Never overwrite/reselect completed samples.'
        t=time.time();done=[]
        for row in rows:
            engine.add_request(name+':'+row['id'],{'prompt_token_ids':prompts[row['id']]['prompt_token_ids']},params[prompts[row['id']]['limit']])
        lookup={row['id']:row for row in rows}
        while engine.has_unfinished_requests():
            for response in engine.step():
                if not response.finished:continue
                k=response.request_id.split(':',1)[1];row=lookup[k];o=response.outputs[0];ids=list(o.token_ids)
                end=next((i for i,x in enumerate(ids) if x in stop),len(ids))
                raw=tok.decode(ids[:end],skip_special_tokens=True,clean_up_tokenization_spaces=False)
                final,method=final_channel(raw)
                result={'id':k,'task_id':k,'benchmark':row['benchmark'],'condition':name,'completion':raw,
                    'final_completion':final,'final_channel_method':method,'generated_token_ids':ids,'new_tokens':end,
                    'truncated':o.finish_reason=='length','finish_reason':o.finish_reason,'stop_reason':o.stop_reason,
                    'rendered_prompt_sha256':prompts[k]['rendered_prompt_sha256']}
                with open(paths[row['benchmark']],'a') as f:f.write(json.dumps(result)+'\n')
                done.append(result)
                if len(done)%8==0:print('PROGRESS',name,len(done),664,'seconds',round(time.time()-t,1),'tokens',sum(r['new_tokens'] for r in done),flush=True)
        assert len(done)==664 and len({r['id'] for r in done})==664
        metadata={'condition':name,'n':664,'seconds':time.time()-t,'tokens':sum(r['new_tokens'] for r in done),
            'truncated':sum(r['truncated'] for r in done),'settings':settings,
            'by_benchmark':{b:{'n':sum(r['benchmark']==b for r in done),'tokens':sum(r['new_tokens'] for r in done if r['benchmark']==b),
                 'truncated':sum(r['truncated'] for r in done if r['benchmark']==b)} for b in paths}}
        (folder/(name+'-complete.json')).write_text(json.dumps(metadata,indent=2));print('COMPLETE',name,json.dumps(metadata),flush=True)
    generate('baseline')
    command=R/'commands';command.mkdir(exist_ok=True);print('BASELINE_COMPLETE_WAITING',flush=True)
    while True:
        for p in sorted(command.glob('*.json')):
            flag=p.with_suffix('.done')
            if flag.exists():continue
            value=json.loads(p.read_text())
            if value=={'action':'retest'}:generate('retest');flag.write_text('completed')
            elif value=={'action':'stop'}:flag.write_text('stopped');return
            else:raise ValueError('Unrecognized command; no selected task retries permitted.')
        time.sleep(1)

if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'worker-error.txt').write_text(traceback.format_exc());raise
