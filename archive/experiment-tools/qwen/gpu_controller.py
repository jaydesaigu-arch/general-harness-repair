"""One bounded GPU allocation; frozen source; automatic harvest and shutdown."""
import dataclasses,hashlib,json,math,sys,time,zipfile
from pathlib import Path
from datetime import datetime,timezone
from huggingface_hub import HfApi,Sandbox
import huggingface_hub._sandbox as sandbox_module
H=Path(__file__).resolve().parent;O=H.parents[1]/'outputs/qwen38-27b';P=H/'gpu-session.json'
assert not P.exists(),'Never duplicate a billable replication job.'
protocol=json.loads((O/'protocol.json').read_text())
for n,d in protocol['solver_files'].items():assert hashlib.sha256((H/n).read_bytes()).hexdigest()==d
api=HfApi();hardware=next(x for x in api.list_jobs_hardware() if x.name=='a100-large')
rate=hardware.unit_cost_usd*(60 if hardware.unit_label=='minute' else 3600);assert rate<=2.50003
sandbox_module.SANDBOX_MAX_LIFETIME='8h';original=HfApi.run_job
state={'phase':'allocating','complete':False};session=None;box=None
def save():
    (H/'gpu-controller-state.json').write_text(json.dumps(state,indent=2))
def capture(self,**kwargs):
    job=original(self,**kwargs)
    P.write_text(json.dumps({'id':job.id,'namespace':job.owner.name,'url':job.url,'created_at_utc':str(job.created_at),
        'hardware':'a100-large','hardware_usd_per_hour':rate,'port_usd_per_hour':.01,'timeout_seconds':28800,
        'maximum_gpu_and_port_usd':8*(rate+.01),'forward_hf_token':False},indent=2))
    print('SUBMITTED',job.id,flush=True);return job
HfApi.run_job=capture
def exists(path):return box.run(['test','-f',path],check=False).exit_code==0
def harvest(label):
    script="import zipfile;from pathlib import Path;r=Path('/workspace/qwen38-27b');z=zipfile.ZipFile('/workspace/frozen-evidence.zip','w',zipfile.ZIP_DEFLATED);[z.write(p,str(p.relative_to(r))) for p in r.rglob('*') if p.is_file() and 'model' not in p.relative_to(r).parts and '__pycache__' not in p.parts];z.close()"
    result=box.run([state.get('python_executable','python3'),'-c',script],timeout=120,check=False);assert result.exit_code==0
    target=H/(label+'-private-evidence.zip');box.files.download('/workspace/frozen-evidence.zip',str(target))
    destination=H/'private-remote';destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert all((destination/n).resolve().is_relative_to(destination.resolve()) for n in z.namelist())
        z.extractall(destination)
    state[label+'_archive_sha256']=hashlib.sha256(target.read_bytes()).hexdigest();save()

try:
    save();box=Sandbox.create(image='vllm/vllm-openai:v0.26.0',flavor='a100-large',
        idle_timeout='15m',forward_hf_token=False,start_timeout=600,
        labels={'experiment':'qwen38-27b-full862','phase':'frozen-transfer','hard-lifetime':'8h'})
    session=json.loads(P.read_text());state.update(job=session['id'],phase='bootstrap');save()
    python_path=box.run(['/bin/sh','-c','command -v python3 || command -v python'],check=True).stdout.strip().splitlines()[0]
    state['python_executable']=python_path;save()
    created=datetime.fromisoformat(session['created_at_utc']).timestamp()
    box.run(['mkdir','-p','/workspace/qwen38-27b'],check=True)
    for name in protocol['solver_files']:
        box.files.upload(str(H/name),'/workspace/qwen38-27b/'+name)
    box.files.upload(str(H/'protocol.json'),'/workspace/qwen38-27b/protocol.json')
    (H/'limits.json').write_text(json.dumps({'deadline_unix':created+28500}))
    box.files.upload(str(H/'limits.json'),'/workspace/qwen38-27b/limits.json')
    bootstrap=box.run([state.get('python_executable','python3'),'/workspace/qwen38-27b/bootstrap_gpu.py'],background=True,check=False)
    session['bootstrap_pid']=bootstrap.pid;P.write_text(json.dumps(session,indent=2));start=time.time()
    while not exists('/workspace/qwen38-27b/environment.json'):
        procs={p.pid:p for p in box.processes()}
        if bootstrap.pid in procs and not procs[bootstrap.pid].running:raise RuntimeError('GPU bootstrap failed; see preserved log')
        if time.time()-start>1200:raise TimeoutError('GPU bootstrap exceeded20 minutes')
        save();time.sleep(15)
    worker=box.run([state.get('python_executable','python3'),'/workspace/qwen38-27b/worker.py'],background=True,check=False)
    state.update(phase='inference',worker_pid=worker.pid);save();print('INFERENCE_STARTED',worker.pid,flush=True)
    while not exists('/workspace/qwen38-27b/pilot-ready.json'):
        if exists('/workspace/qwen38-27b/worker-error.txt'):raise RuntimeError('Qwen compatibility/pilot failed; preserve evidence before retry')
        if time.time()>created+2400:raise TimeoutError('Qwen compatibility and pilot exceeded40min')
        save();time.sleep(10)
    pilot_path=H/'pilot-results.json';box.files.download('/workspace/qwen38-27b/pilot-ready.json',str(pilot_path))
    pilot=json.loads(pilot_path.read_text());tps=pilot['warm_tokens_per_second']
    maximum_tokens=2*sum(r['output_budget'] for r in json.loads((H/'questions.json').read_text()))
    forecast=maximum_tokens/tps*1.10
    pilot['full_pair_max_token_projection_seconds']=forecast
    pilot['remaining_gpu_lifetime_seconds']=created+28500-time.time()
    pilot_path.write_text(json.dumps(pilot,indent=2));(O/'pilot-results.json').write_text(json.dumps(pilot,indent=2))
    print('PILOT_COST_CHECK',json.dumps({'tokens_per_second':tps,'max_output_projection_hours':forecast/3600,'remaining_hours':pilot['remaining_gpu_lifetime_seconds']/3600}),flush=True)
    if forecast>pilot['remaining_gpu_lifetime_seconds']*.97:
        state.update(phase='cost_review_required',pilot=pilot);save()
        stop=H/'STOP.json';stop.write_text(json.dumps({'reason':'Worst-case full pair may exceed first bounded allocation; review measured throughput before spending more.'}));box.files.upload(str(stop),'/workspace/qwen38-27b/STOP.json')
        harvest('pilot');raise RuntimeError('Full run not started: conservative throughput projection requires a revised runtime/funding plan')
    go=H/'GO.json';go.write_text(json.dumps({'full_pair_authorized_under_frozen_protocol':True,'basis':'Measured synthetic throughput fits declared bounded lifetime, even at maximum output lengths.'}));box.files.upload(str(go),'/workspace/qwen38-27b/GO.json')
    state['phase']='scored_inference';save();print('FULL_PAIR_STARTED',flush=True)
    baseline_saved=False
    while True:
        if exists('/workspace/qwen38-27b/all-complete.json'):
            state.update(phase='harvest',complete=True);harvest('complete');break
        if not baseline_saved and exists('/workspace/qwen38-27b/generations/baseline-complete.json'):
            harvest('baseline');baseline_saved=True;print('BASELINE_HARVESTED',flush=True)
        if exists('/workspace/qwen38-27b/worker-error.txt'):raise RuntimeError('Frozen worker failed; code will not be tuned after observing scores')
        procs={p.pid:p for p in box.processes()}
        if worker.pid in procs and not procs[worker.pid].running:raise RuntimeError('Worker exited without complete marker')
        if time.time()>created+28545:raise TimeoutError('Declared inference lifetime reached')
        # Only counts/timing are read during the run, never correctness.
        progress=box.run(['tail','-n','2','/workspace/qwen38-27b/generation.log'],check=False)
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
