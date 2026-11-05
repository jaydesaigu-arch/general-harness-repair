"""Same frozen code/math graders on both views and arms; no solver feedback."""
import hashlib,json,subprocess,sys
from pathlib import Path
R=Path('/workspace/sarvam30b');sys.path.insert(0,str(R))
from harness import final_channel
condition=sys.argv[1];assert condition in ['baseline','combined_repaired']
sys.stdout=sys.stderr=open(R/(condition+'-grading.log'),'a',buffering=1)
for benchmark,n,folder in [('humaneval_plus',164,Path('/workspace/he164')),('math500',500,Path('/workspace/math500'))]:
    rows=[json.loads(x) for x in (R/(condition+'-'+benchmark+'.jsonl')).read_text().splitlines()]
    assert len(rows)==n and len({r['id'] for r in rows})==n
    for view in ['raw','final']:
        name=condition+'-'+view;processed=[]
        for row in rows:
            final,method=final_channel(row['completion']);assert final==row['final_completion'] and method==row['final_channel_method']
            out=dict(row)
            if view=='final':out['completion']=final
            processed.append(out)
        path=folder/(name+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in processed))
        if benchmark=='humaneval_plus':
            for stage in ['base','plus']:
                subprocess.run([sys.executable,str(folder/'cpu_eval.py'),'score','--samples',str(path),'--name',name,'--stage',stage,'--extract','official'],check=True)
        else:subprocess.run([sys.executable,str(folder/'math_grade_v3.py'),'score','--name',name],check=True)
(R/(condition+'-scores-complete.json')).write_text(json.dumps({'complete':True,'condition':condition,'code_n':164,'math_n':500}))
print('SCORES_COMPLETE',condition,flush=True)
