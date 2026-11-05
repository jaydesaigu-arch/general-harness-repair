"""One bounded GPU allocation; frozen source; automatic harvest and shutdown."""
import dataclasses,hashlib,json,math,sys,time,zipfile
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,Sandbox
import huggingface_hub._sandbox as sandbox_module
H=Path(__file__).resolve().parent;O=H.parents[1]/'outputs/sarvam30b-frozen-replication';P=H/'gpu-session.json'
assert not P.exists(),'Never duplicate a billable replication job.'
protocol=json.loads((O/'protocol.json').read_text())
for n,d in protocol['solver_files'].items():assert hashlib.sha256((H/n).read_bytes()).hexdigest()==d
api=HfApi();hardware=next(x for x in api.list_jobs_hardware() if x.name=='a100-large')
rate=hardware.unit_cost_usd*(60 if hardware.unit_label=='minute' else 3600);assert rate<=2.50003
sandbox_module.SANDBOX_MAX_LIFETIME='3h';original=HfApi.run_job
state={'phase':'allocating','complete':False};session=None;box=None
def save():
    (H/'gpu-controller-state.json').write_text(json.dumps(state,indent=2))
def capture(self,**kwargs):
    job=original(self,**kwargs)
    P.write_text(json.dumps({'id':job.id,'namespace':job.owner.name,'url':job.url,'created_at_utc':str(job.created_at),
        'hardware':'a100-large','hardware_usd_per_hour':rate,'port_usd_per_hour':.01,'timeout_seconds':10800,
        'maximum_gpu_and_port_usd':3*(rate+.01),'forward_hf_token':False},indent=2))
    print('SUBMITTED',job.id,flush=True);return job
HfApi.run_job=capture
def exists(path):return box.run(['test','-f',path],check=False).exit_code==0
def harvest(label):
    script="import zipfile;from pathlib import Path;r=Path('/workspace/sarvam30b');z=zipfile.ZipFile('/workspace/frozen-evidence.zip','w',zipfile.ZIP_DEFLATED);[z.write(p,str(p.relative_to(r))) for p in r.rglob('*') if p.is_file() and 'model' not in p.relative_to(r).parts and '__pycache__' not in p.parts];z.close()"
    result=box.run(['python','-c',script],timeout=120,check=False);assert result.exit_code==0
    target=H/(label+'-private-evidence.zip');box.files.download('/workspace/frozen-evidence.zip',str(target))
    destination=H/'private-remote';destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert all((destination/n).resolve().is_relative_to(destination.resolve()) for n in z.namelist())
        z.extractall(destination)
    state[label+'_archive_sha256']=hashlib.sha256(target.read_bytes()).hexdigest();save()

try:
    save();box=Sandbox.create(image='pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime',flavor='a100-large',
        idle_timeout='15m',forward_hf_token=False,start_timeout=600,
        labels={'experiment':'sarvam30b-frozen-full862','phase':'independent-combined-bundle','hard-lifetime':'3h'})
    session=json.loads(P.read_text());state.update(job=session['id'],phase='bootstrap');save()
    created=datetime.fromisoformat(session['created_at_utc']).timestamp()
    box.run(['mkdir','-p','/workspace/sarvam30b'],check=True)
    for name in protocol['solver_files']:
        box.files.upload(str(H/name),'/workspace/sarvam30b/'+name)
    box.files.upload(str(H/'protocol.json'),'/workspace/sarvam30b/protocol.json')
    (H/'limits.json').write_text(json.dumps({'deadline_unix':created+10500}))
    box.files.upload(str(H/'limits.json'),'/workspace/sarvam30b/limits.json')
    bootstrap=box.run(['python','/workspace/sarvam30b/bootstrap_gpu.py'],background=True,check=False)
    session['bootstrap_pid']=bootstrap.pid;P.write_text(json.dumps(session,indent=2));start=time.time()
    while not exists('/workspace/sarvam30b/environment.json'):
        procs={p.pid:p for p in box.processes()}
        if bootstrap.pid in procs and not procs[bootstrap.pid].running:raise RuntimeError('GPU bootstrap failed; see preserved log')
        if time.time()-start>1200:raise TimeoutError('GPU bootstrap exceeded20 minutes')
        save();time.sleep(15)
    worker=box.run(['python','/workspace/sarvam30b/worker.py'],background=True,check=False)
    state.update(phase='inference',worker_pid=worker.pid);save();print('INFERENCE_STARTED',worker.pid,flush=True)
    baseline_saved=False
    while True:
        if exists('/workspace/sarvam30b/all-complete.json'):
            state.update(phase='harvest',complete=True);harvest('complete');break
        if not baseline_saved and exists('/workspace/sarvam30b/generations/baseline-complete.json'):
            harvest('baseline');baseline_saved=True;print('BASELINE_HARVESTED',flush=True)
        if exists('/workspace/sarvam30b/worker-error.txt'):raise RuntimeError('Frozen worker failed; code will not be tuned after observing scores')
        procs={p.pid:p for p in box.processes()}
        if worker.pid in procs and not procs[worker.pid].running:raise RuntimeError('Worker exited without complete marker')
        if time.time()>created+10545:raise TimeoutError('Declared inference lifetime reached')
        # Only counts/timing are read during the run, never correctness.
        progress=box.run(['tail','-n','2','/workspace/sarvam30b/generation.log'],check=False)
        state['last_progress']=progress.stdout.strip();save();time.sleep(15)
except BaseException as e:
    state.update(phase='error',error=repr(e));save()
    if box is not None:
        try:harvest('error')
        except Exception as extra:state['harvest_error']=repr(extra);save()
    raise
finally:
    if P.exists():
        session=json.loads(P.read_text())
        try:
            info=api.inspect_job(job_id=session['id'],namespace=session['namespace'])
            session['before_stop']={'created_at':str(info.created_at),'started_at':str(info.started_at),'durations':dataclasses.asdict(info.durations) if info.durations else None}
            Sandbox.kill(session['id'],namespace=session['namespace'])
            now=datetime.now(timezone.utc);session.update(stopped=True,stopped_at_utc=now.isoformat(),final_stage=api.inspect_job(job_id=session['id'],namespace=session['namespace']).status.stage)
            minutes=math.ceil((now-datetime.fromisoformat(session['created_at_utc'])).total_seconds()/60)
            session['conservative_gpu_and_port_usd']=math.ceil(minutes/60*(rate+.01)*10000)/10000
            P.write_text(json.dumps(session,indent=2));(O/'GPU-COST.json').write_text(json.dumps(session,indent=2))
            state['gpu_stopped']=True;save();print('GPU_STOPPED',session['final_stage'],flush=True)
        finally:
            if box is not None:box.close()
