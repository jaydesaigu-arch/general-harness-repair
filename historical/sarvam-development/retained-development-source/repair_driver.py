"""Wait for immutable baseline, release its engine, then run controlled repair."""
import os,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path('/workspace/sarvam30b')
try:
    with open(R/'repair-driver.log','a',buffering=1) as f:
        while not (R/'generations/baseline-complete.json').exists():time.sleep(3)
        (R/'commands').mkdir(exist_ok=True)
        (R/'commands/000-stop.json').write_text('{"action":"stop"}')
        def used():
            p=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            return int(p.stdout.strip().splitlines()[0])
        deadline=time.time()+120
        while used()>2048 and time.time()<deadline:time.sleep(2)
        if used()>2048:
            # This isolated GPU sandbox contains only our Sarvam engine.
            p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            for value in p.stdout.splitlines():
                if value.strip().isdigit():
                    try:os.kill(int(value),signal.SIGTERM)
                    except ProcessLookupError:pass
            time.sleep(3)
        assert used()<2048,'Old engine has not released GPU memory; do not allocate a duplicate model.'
        f.write('BASELINE_ENGINE_RELEASED\n');f.flush()
        p=subprocess.run([sys.executable,str(R/'repair_worker.py')],stdout=f,stderr=f)
        f.write('EXIT '+str(p.returncode)+'\n');raise SystemExit(p.returncode)
except BaseException:
    (R/'repair-driver-error.txt').write_text(traceback.format_exc());raise
