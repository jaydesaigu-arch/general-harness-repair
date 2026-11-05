"""EvalPlus orchestration in a disposable VM with no forwarded HF credential."""
import argparse,ast,concurrent.futures,copy,hashlib,json,pickle,random,sys,time
from pathlib import Path
from common import SEED,jsonl,simple_extract
import evalplus.sanitize as _sanitize
from sanitize_dependencies_v2 import extract_target_code_or_empty as _preserving_extract
_sanitize.extract_target_code_or_empty=_preserving_extract
ROOT=Path('/workspace/he164')

def evaluate_one(args):
    task,solution,oracle,stage=args
    from evalplus.eval import untrusted_check
    started=time.time()
    # HF fixes the stack hard limit at 10 MiB. EvalPlus 0.3.1 tries to raise it
    # to its 4 GiB memory cap, which fails before executing any candidate.
    # Preserve the stricter host hard limit; leave AS/DATA caps in place.
    import resource
    original_setrlimit=resource.setrlimit
    def bounded_setrlimit(which,limits):
        hard=resource.getrlimit(which)[1]
        soft,newhard=limits
        if hard!=resource.RLIM_INFINITY:
            newhard=hard if newhard==resource.RLIM_INFINITY else min(newhard,hard)
            soft=newhard if soft==resource.RLIM_INFINITY else min(soft,newhard)
        return original_setrlimit(which,(soft,newhard))
    resource.setrlimit=bounded_setrlimit
    try:
        status,details=untrusted_check('humaneval',solution,copy.deepcopy(task[stage+'_input']),task['entry_point'],
            expected=oracle[stage],atol=task['atol'],ref_time=oracle[stage+'_time'],fast_check=True,
            min_time_limit=0.2,gt_time_limit_factor=4.0)
    finally:resource.setrlimit=original_setrlimit
    failed=None if status=='pass' or not details else len(details)-1
    result={'task_id':task['task_id'],'status':status,'seconds':time.time()-started}
    result['first_failed_input_index']=failed
    if failed is not None:
        result['first_failed_input_repr']=repr(task[stage+'_input'][failed])[:5000]
        result['reference_output_repr']=repr(oracle[stage][failed])[:5000]
    if stage=='base':
        if failed is None:
            feedback='The program failed to execute or exceeded the evaluator time limit.'
        elif task['entry_point']=='find_zero':
            feedback='For input arguments '+repr(task['base_input'][failed])[:1600]+', the returned root must make the polynomial approximately zero.'
        else:
            feedback='For input arguments '+repr(task['base_input'][failed])[:1600]+', the expected output is '+repr(oracle['base'][failed])[:1600]+'.'
        result['feedback']=feedback
    return result

def run_checks(tasks,samples,oracle,stage):
    jobs=[(tasks[r['task_id']],r['solution'],oracle[r['task_id']],stage) for r in samples]
    result=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(evaluate_one,jobs):
            result.append(r)
            if len(result)%8==0:print('SCORED',stage,len(result),len(jobs),flush=True)
    return result

def prepare():
    from evalplus.data import get_human_eval_plus,get_human_eval_plus_hash
    from evalplus.gen.util import trusted_exec
    import evalplus
    tasks=get_human_eval_plus(mini=False,noextreme=False)
    assert set(tasks)=={f'HumanEval/{i}' for i in range(164)}
    prompts=[{k:t[k] for k in ['task_id','prompt','entry_point']} for t in tasks.values()]
    (ROOT/'prompts.json').write_text(json.dumps(prompts))
    (ROOT/'splits.json').write_text(json.dumps({'all':list(tasks)}))
    (ROOT/'dataset-manifest.json').write_text(json.dumps({'version':'HumanEvalPlus-v0.1.10','md5':get_human_eval_plus_hash(),'tasks':len(tasks),'base_test_inputs':sum(len(t['base_input']) for t in tasks.values()),'plus_test_inputs':sum(len(t['plus_input']) for t in tasks.values())},indent=2))
    print('PROMPTS_PREPARED',flush=True)
    oracle={}
    if (ROOT/'oracle.pkl').exists():
        with open(ROOT/'oracle.pkl','rb') as f:oracle=pickle.load(f)
    else:
        for i,(task_id,t) in enumerate(tasks.items()):
            code=t['prompt']+t['canonical_solution'];o={}
            for stage in ['base','plus']:
                o[stage],o[stage+'_time']=trusted_exec(code,t[stage+'_input'],t['entry_point'],record_time=True)
            oracle[task_id]=o
            if (i+1)%8==0:print('ORACLE',i+1,flush=True)
    with open(ROOT/'oracle.pkl','wb') as f:pickle.dump(oracle,f)
    samples=[{'task_id':i,'solution':t['prompt']+t['canonical_solution']} for i,t in tasks.items()]
    controls={s:run_checks(tasks,samples,oracle,s) for s in ['base','plus']}
    controls['negative']=run_checks(tasks,[{'task_id':i,'solution':f'def {t["entry_point"]}(*args, **kwargs):\n    raise RuntimeError("negative control")\n'} for i,t in tasks.items()],oracle,'base')
    (ROOT/'evaluator-controls.json').write_text(json.dumps(controls,indent=2))
    # The release's Newton-based root reference does not converge on seven
    # expanded inputs. Keep the property oracle and all inputs unchanged.
    assert all(r['status']=='pass' for r in controls['base'])
    assert all(r['status']=='pass' for r in controls['plus'] if r['task_id']!='HumanEval/32')
    root_control='''import math
from decimal import Decimal, localcontext
def find_zero(xs):
    def residual(x):
        return abs(sum(c*math.pow(x,i) for i,c in enumerate(xs)))
    with localcontext() as ctx:
        ctx.prec=100
        coeff=[Decimal(str(x)) for x in xs]
        def f(x):
            y=Decimal(0)
            for c in reversed(coeff):y=y*x+c
            return y
        a,b=Decimal(-1),Decimal(1)
        while f(a)*f(b)>0:a*=2;b*=2
        fa=f(a)
        for _ in range(400):
            m=(a+b)/2;fm=f(m)
            if fm==0:a=b=m;break
            if fa*fm<=0:b=m
            else:a=m;fa=fm
        x=float((a+b)/2)
    candidates=[x];lo=hi=x
    for _ in range(8):
        lo=math.nextafter(lo,-math.inf);hi=math.nextafter(hi,math.inf)
        candidates.extend([lo,hi])
    return min(candidates,key=residual)
'''
    root_control=(ROOT/'root_positive.py').read_text().split("if __name__=='__main__':")[0]
    import numpy.polynomial.polynomial
    controls['independent_root_positive']={s:run_checks(tasks,[{'task_id':'HumanEval/32','solution':root_control}],oracle,s) for s in ['base','plus']}
    (ROOT/'evaluator-controls.json').write_text(json.dumps(controls,indent=2))
    assert all(r['status']=='pass' for r in controls['independent_root_positive']['base'])
    assert all(r['status']!='pass' for r in controls['negative'])
    (ROOT/'dataset-audit.json').write_text(json.dumps({'task':'HumanEval/32',
       'base_reference_passes':True,'plus_reference_passes':False,
       'leading_zero_coefficient_plus_inputs':sum(x[0][-1]==0 for x in tasks['HumanEval/32']['plus_input']),
       'note':'Root-task expanded inputs include contract violations and numerical conditioning issues. Keep all 164 tasks and all input cases in the primary score; separately report a 163-task sensitivity analysis excluding only this audited task. Do not treat the canonical reference failure as a model defect.'},indent=2))
    print('PREPARE_COMPLETE: base positive 164/164, expanded reference 163/164, negative 0/164. Root-task limitation recorded.',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','score'])
    p.add_argument('--samples');p.add_argument('--name');p.add_argument('--stage',choices=['base','plus'],default='base')
    p.add_argument('--extract',choices=['simple','official'],default='official');a=p.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':return prepare()
    from evalplus.data import get_human_eval_plus
    from evalplus.sanitize import sanitize
    tasks=get_human_eval_plus()
    with open(ROOT/'oracle.pkl','rb') as f:oracle=pickle.load(f)
    samples=jsonl(a.samples)
    assert len(samples)==164 and {r['task_id'] for r in samples}==set(tasks)
    folder=ROOT/'scores'/a.name;folder.mkdir(parents=True,exist_ok=True)
    sys.stdout=sys.stderr=open(folder/(a.stage+'-eval.log'),'a',buffering=1)
    processed=[];audit=[]
    for r in samples:
        raw=r['completion'];simple=simple_extract(raw)
        solution=sanitize(raw,entrypoint=tasks[r['task_id']]['entry_point']) if a.extract=='official' else simple
        syntax_error=None
        try:ast.parse(solution)
        except SyntaxError as ex:syntax_error=str(ex)
        processed.append({'task_id':r['task_id'],'solution':solution})
        audit.append({'task_id':r['task_id'],'extraction_changed':solution!=simple,'syntax_error':syntax_error,
            'truncated':r.get('truncated',False),'empty':not solution.strip()})
    (folder/'solutions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in processed))
    (folder/'extraction-audit.json').write_text(json.dumps(audit,indent=2))
    result=run_checks(tasks,processed,oracle,a.stage)
    (folder/(a.stage+'.json')).write_text(json.dumps(result,indent=2))
    splits=json.loads((ROOT/'splits.json').read_text())
    summary={'stage':a.stage,'condition':a.name,'n':164,'passed':sum(r['status']=='pass' for r in result),
        'splits':{s:{'n':len(ids),'passed':sum(r['status']=='pass' for r in result if r['task_id'] in ids)} for s,ids in splits.items()}}
    (folder/(a.stage+'-summary.json')).write_text(json.dumps(summary,indent=2))
    print('SCORING_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
