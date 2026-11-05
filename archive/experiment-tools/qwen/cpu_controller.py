"""Bounded cheap CPU evaluator, independent of the GPU, with shutdown/harvest."""
import dataclasses,hashlib,json,math,time,zipfile
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,Sandbox
import huggingface_hub._sandbox as mod
H=Path(__file__).resolve().parent;R=H.parents[1];O=R/'outputs/qwen38-27b';P=H/'cpu-session.json';E=H/'evaluator'
assert not P.exists(),'Never duplicate evaluator allocation.'
protocol=json.loads((O/'protocol.json').read_text())
for n,d in protocol['evaluator_files'].items():assert hashlib.sha256((E/n).read_bytes()).hexdigest()==d
api=HfApi();hw=next(x for x in api.list_jobs_hardware() if x.name=='cpu-basic')
rate=hw.unit_cost_usd*(60 if hw.unit_label=='minute' else 3600);assert rate<=.01003
mod.SANDBOX_MAX_LIFETIME='8h';original=HfApi.run_job;box=None;state={'phase':'allocating','completed':[]}
def capture(self,**kwargs):
    j=original(self,**kwargs);P.write_text(json.dumps({'id':j.id,'namespace':j.owner.name,'url':j.url,'created_at_utc':str(j.created_at),
        'hardware':'cpu-basic','hardware_usd_per_hour':rate,'port_usd_per_hour':.01,'timeout_seconds':28800,'forward_hf_token':False},indent=2))
    print('CPU_SUBMITTED',j.id,flush=True);return j
HfApi.run_job=capture
def save():(H/'cpu-controller-state.json').write_text(json.dumps(state,indent=2))
def exists(path):return box.run(['test','-f',path],check=False).exit_code==0
def harvest():
    script="import zipfile;from pathlib import Path;z=zipfile.ZipFile('/workspace/cpu-evidence.zip','w',zipfile.ZIP_DEFLATED);[z.write(p,str(p.relative_to('/workspace'))) for root in ['/workspace/he164','/workspace/math500','/workspace/sarvam30b'] for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts];z.close()"
    rr=box.run(['python','-c',script],timeout=120,check=False);assert rr.exit_code==0
    target=H/'cpu-private-evidence.zip';box.files.download('/workspace/cpu-evidence.zip',str(target))
    dest=H/'private-cpu';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert all((dest/n).resolve().is_relative_to(dest.resolve()) for n in z.namelist());z.extractall(dest)
    state['evidence_sha256']=hashlib.sha256(target.read_bytes()).hexdigest();save()
try:
    box=Sandbox.create(image='python:3.12-slim',flavor='cpu-basic',idle_timeout='15m',forward_hf_token=False,start_timeout=600,
        labels={'experiment':'qwen38-27b-full862-evaluator','hard-lifetime':'8h'})
    s=json.loads(P.read_text());created=datetime.fromisoformat(s['created_at_utc']).timestamp();state.update(job=s['id'],phase='bootstrap');save()
    box.run(['mkdir','-p','/workspace/he164','/workspace/math500/scores','/workspace/sarvam30b'],check=True)
    for n in ['cpu_eval.py','common.py','root_positive.py','patch_evalplus.py','sanitize_dependencies_v2.py']:
        box.files.upload(str(E/n),'/workspace/he164/'+n)
    for n in ['math_grade.py','math_grade_v2.py','math_grade_v3.py','audit_gold_parse.py']:
        box.files.upload(str(E/n),'/workspace/math500/'+n)
    for n in ['private-gold.json','private-reference-solutions.json']:
        box.files.upload(str(R/'work/llama8b-math500'/n),'/workspace/math500/'+n)
    for n in ['cpu_bootstrap.py','score_cpu.py','harness.py']:
        box.files.upload(str(H/n),'/workspace/sarvam30b/'+n)
    limits=H/'cpu-limits.json';limits.write_text(json.dumps({'deadline_unix':created+28400}));box.files.upload(str(limits),'/workspace/sarvam30b/limits.json')
    proc=box.run(['python','/workspace/sarvam30b/cpu_bootstrap.py'],background=True,check=False)
    while not exists('/workspace/sarvam30b/cpu-ready.json'):
        procs={x.pid:x for x in box.processes()}
        if proc.pid in procs and not procs[proc.pid].running:raise RuntimeError('Evaluator qualification failed')
        if time.time()>created+2100:raise TimeoutError('Evaluator bootstrap exceeded35min')
        save();time.sleep(15)
    state['phase']='waiting_for_generations';save();print('CPU_QUALIFIED',flush=True)
    for condition in ['baseline','combined_repaired']:
        marker=H/'private-remote/generations'/(condition+'-complete.json')
        while not marker.exists():
            gpu_state=json.loads((H/'gpu-controller-state.json').read_text())
            if gpu_state.get('phase')=='error':raise RuntimeError('Solver failed; evaluator stopped without tuning')
            if time.time()>created+27900:raise TimeoutError('No complete frozen generation before CPU deadline')
            time.sleep(15)
        for benchmark in ['humaneval_plus','math500']:
            name=condition+'-'+benchmark+'.jsonl';box.files.upload(str(marker.parent/name),'/workspace/sarvam30b/'+name)
        state.update(phase='grading',condition=condition);save()
        scoring=box.run(['python','/workspace/sarvam30b/score_cpu.py',condition],background=True,check=False)
        while not exists('/workspace/sarvam30b/'+condition+'-scores-complete.json'):
            procs={x.pid:x for x in box.processes()}
            if scoring.pid in procs and not procs[scoring.pid].running:raise RuntimeError('Scoring stopped without completion marker')
            if time.time()>created+28200:raise TimeoutError('Scoring reached CPU deadline')
            time.sleep(15)
        state['completed'].append(condition);harvest();print('CPU_SCORED',condition,flush=True)
    state['phase']='complete';save()
except BaseException as error:
    state.update(phase='error',error=repr(error));save()
    if box is not None:
        try:harvest()
        except Exception as extra:state['harvest_error']=repr(extra);save()
    raise
finally:
    if P.exists():
        s=json.loads(P.read_text())
        try:
            Sandbox.kill(s['id'],namespace=s['namespace']);now=datetime.now(timezone.utc)
            s.update(stopped=True,stopped_at_utc=now.isoformat(),final_stage=api.inspect_job(job_id=s['id'],namespace=s['namespace']).status.stage)
            mins=math.ceil((now-datetime.fromisoformat(s['created_at_utc'])).total_seconds()/60)
            s['conservative_cpu_and_port_usd']=math.ceil(mins/60*(rate+.01)*10000)/10000
            P.write_text(json.dumps(s,indent=2));(O/'CPU-COST.json').write_text(json.dumps(s,indent=2));state['cpu_stopped']=True;save();print('CPU_STOPPED',flush=True)
        finally:
            if box is not None:box.close()
