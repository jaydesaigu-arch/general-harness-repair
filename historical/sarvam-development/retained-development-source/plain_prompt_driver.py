"""Release original fresh engine before the next declared candidate."""
import os,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/workspace/sarvam30b')
try:
    with open(R/'plain-prompt-driver.log','a',buffering=1) as f:
        while not (R/'generations/fresh_repaired-complete.json').exists():time.sleep(3)
        (R/'repair-commands/999-stop.json').write_text('{"action":"stop"}')
        def used():
            p=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            return int(p.stdout.strip().splitlines()[0])
        limit=time.time()+120
        while used()>2048 and time.time()<limit:time.sleep(2)
        if used()>2048:
            p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            for v in p.stdout.splitlines():
                if v.strip().isdigit():
                    try:os.kill(int(v),signal.SIGTERM)
                    except ProcessLookupError:pass
            time.sleep(3)
        assert used()<2048,'Do not allocate duplicate model'
        f.write('PREVIOUS_ENGINE_RELEASED\n');f.flush()
        p=subprocess.run([sys.executable,str(R/'plain_prompt_worker.py')],stdout=f,stderr=f)
        f.write('EXIT '+str(p.returncode)+'\n')
except BaseException:
    (R/'plain-prompt-driver-error.txt').write_text(traceback.format_exc());raise
