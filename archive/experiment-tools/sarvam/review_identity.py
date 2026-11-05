"""Carry prior Codex judgments only for exact same question AND response/code.

No automatic verdict, reference answer, or old numerical score chooses a label.
Unmatched/conflicting cases remain pending for fresh case-by-case inspection.
"""
import csv,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1];D=H/'private-review';D.mkdir(exist_ok=True)
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
registries={'math500':{},'humaneval_plus':{},'gpqa_diamond':{}}
def register(benchmark,question,digest,label,source,notes):
    key=(question,digest);record={'label':label,'source':source,'notes':notes}
    registries[benchmark].setdefault(key,[]).append(record)
for benchmark,filename,hashfield,labelfield,prefix in [
    ('math500','math500-per-response.csv','completion_sha256','manual_correct','math500/'),
    ('humaneval_plus','code-per-program.csv','solution_sha256','manual_review','HumanEval/')]:
    path=R/'outputs/manual-regrade'/filename
    with path.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f):
            if row['condition'] not in ['SB','SR']:continue
            q=prefix+(f"{int(row['id']):03}" if benchmark=='math500' else str(int(row['id'])))
            register(benchmark,q,row[hashfield],row[labelfield],str(path.relative_to(R))+':'+row['condition'],row['notes'])
gp=R/'work/sarvam30b-gpqa/private-manual-frozen.json'
for condition,rows in json.loads(gp.read_text()).items():
    for key,row in rows.items():register('gpqa_diamond',key,row['completion_sha256'],row['choice'],str(gp.relative_to(R))+':'+condition,row['notes'])
records=[];summary={};pending=[]
for condition in ['baseline','combined_repaired']:
    for benchmark,n in [('humaneval_plus',164),('math500',500),('gpqa_diamond',198)]:
        source='partial-review-only' if '--partial' in sys.argv else 'private-remote'
        path=H/source/'generations'/(condition+'-'+benchmark+'.jsonl')
        if not path.exists():continue
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        assert len({r['id'] for r in rows})==len(rows)
        solutions={}
        if benchmark=='humaneval_plus':
            codepath=H/'private-cpu/he164/scores'/(condition+'-raw')/'solutions.jsonl'
            if '--partial' in sys.argv:codepath=H/'partial-review-only/code'/(condition+'-solutions.jsonl')
            if codepath.exists():solutions={r['task_id']:r['solution'] for r in [json.loads(s) for s in codepath.read_text().splitlines()]}
        for row in rows:
            q=row['id'];text=solutions.get(q) if benchmark=='humaneval_plus' else row['completion']
            digest=sha(text) if text is not None else None
            previous=registries[benchmark].get((q,digest),[])
            labels={json.dumps(r['label'],sort_keys=True) for r in previous}
            matched=len(labels)==1
            out={'id':q,'condition':condition,'benchmark':benchmark,'completion_sha256':sha(row['completion']),
                 'reviewed_view_sha256':digest,'matched_prior_judgment':matched,
                 'reviewer':'Codex, not an independent human panel',
                 'status':'reused_exact_same_question_and_response' if matched else 'pending_fresh_review',
                 'label':previous[0]['label'] if matched else None,'previous_judgments':previous}
            records.append(out)
            key=condition+':'+benchmark;summary.setdefault(key,{'total_available':0,'exact_prior_judgments':0,'pending':0})
            summary[key]['total_available']+=1;summary[key]['exact_prior_judgments']+=int(matched);summary[key]['pending']+=int(not matched)
            if not matched:pending.append({'id':q,'condition':condition,'benchmark':benchmark,'completion_sha256':out['completion_sha256']})
(D/'identity-reused-judgments.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
(D/'pending.json').write_text(json.dumps(pending,indent=2),encoding='utf-8')
(D/'coverage.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
