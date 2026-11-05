"""Download the pinned original Qwen checkpoint inside the owned GPU job."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/workspace/qwen38-27b');sys.stdout=sys.stderr=open(R/'bootstrap.log','a',buffering=1)
try:
    started=time.time()
    os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_TELEMETRY='1')
    # Image already contains pinned vLLM; only align the tokenizer/config dependency.
    subprocess.run([sys.executable,'-m','pip','install','-q','--disable-pip-version-check','transformers==5.8.1'],check=True)
    from huggingface_hub import snapshot_download
    p=json.loads((R/'protocol.json').read_text())
    snapshot_download(p['model'],revision=p['revision'],local_dir=str(R/'model'),
        allow_patterns=['*.json','*.jinja','*.safetensors','README.md','LICENSE'],max_workers=6)
    import vllm,torch,transformers
    env={'vllm':vllm.__version__,'torch':torch.__version__,'transformers':transformers.__version__,
        'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'bootstrap_seconds':time.time()-started,
        'weight_shards':len(list((R/'model').glob('*.safetensors')))}
    assert env['vllm']=='0.26.0' and env['weight_shards']==18
    (R/'environment.json').write_text(json.dumps(env,indent=2));print('BOOTSTRAP_COMPLETE',json.dumps(env),flush=True)
except BaseException:
    (R/'bootstrap-error.txt').write_text(traceback.format_exc());raise
