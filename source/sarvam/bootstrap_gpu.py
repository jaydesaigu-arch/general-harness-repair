"""Install pinned author integration and fetch original checkpoint in the cloud."""
import hashlib,importlib.metadata,json,os,shutil,subprocess,sys,time
from pathlib import Path
R=Path('/workspace/sarvam30b');R.mkdir(parents=True,exist_ok=True)
sys.stdout=sys.stderr=open(R/'bootstrap.log','a',buffering=1)
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
t=time.time()
subprocess.run(['apt-get','update','-qq'],check=True)
subprocess.run(['apt-get','install','-y','-qq','build-essential'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--disable-pip-version-check','vllm==0.15.0','transformers==4.57.6'],check=True)
from huggingface_hub import snapshot_download
import vllm
directory=Path(vllm.__file__).parent/'model_executor/models'
registry=directory/'registry.py';text=registry.read_text();needle='_TEXT_GENERATION_MODELS = {'
assert needle in text
if '"SarvamMoEForCausalLM"' not in text:
    text=text.replace(needle,needle+'\n    "SarvamMoEForCausalLM": ("sarvam", "SarvamMoEForCausalLM"),',1);registry.write_text(text)
shutil.copy2(R/'sarvam.py',directory/'sarvam.py')
snapshot_download('sarvamai/sarvam-30b',revision='071ae95e933605ca1104a6b4524a36a98488efa4',local_dir=str(R/'model'),
    allow_patterns=['*.json','*.jinja','*.py','model-*.safetensors','README.md'],max_workers=4)
import torch,transformers
environment={'vllm':vllm.__version__,'torch':torch.__version__,'transformers':transformers.__version__,
 'gpu':torch.cuda.get_device_name(),'bootstrap_seconds':time.time()-t,
 'author_executor_sha256':hashlib.sha256((R/'sarvam.py').read_bytes()).hexdigest(),
 'weights_downloaded':len(list((R/'model').glob('model-*.safetensors')))}
assert environment['weights_downloaded']==26
(R/'environment.json').write_text(json.dumps(environment,indent=2));print('BOOTSTRAP_COMPLETE',json.dumps(environment),flush=True)
