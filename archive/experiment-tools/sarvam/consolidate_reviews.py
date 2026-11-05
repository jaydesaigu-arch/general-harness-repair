"""Verify review identities against final harvested evidence; never open GPQA key."""
import argparse, hashlib, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

H=Path(__file__).resolve().parent
D=H/'private-review';O=H.parents[1]/'outputs/sarvam30b-frozen-replication'
p=argparse.ArgumentParser();p.add_argument('--condition',choices=['baseline','combined_repaired','both'],default='both');a=p.parse_args()
conditions=['baseline','combined_repaired'] if a.condition=='both' else [a.condition]
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
fresh=json.loads((D/'fresh-judgments.json').read_text())
reused=json.loads((D/'identity-reused-judgments.json').read_text())
records=[];summary={}
for condition in conditions:
    summary[condition]={}
    for benchmark,n in [('humaneval_plus',164),('math500',500),('gpqa_diamond',198)]:
        source=H/'private-remote/generations'/(condition+'-'+benchmark+'.jsonl')
        rows=[json.loads(s) for s in source.read_text().splitlines()]
        assert len(rows)==len({r['id'] for r in rows})==n
        programs={}
        if benchmark=='humaneval_plus':
            code=H/'private-cpu/he164/scores'/(condition+'-raw')/'solutions.jsonl'
            programs={r['task_id']:r['solution'] for r in map(json.loads,code.read_text().splitlines())}
            assert len(programs)==n
        current=[]
        for row in rows:
            key=row['id'];digest=sha(row['completion'])
            labels=[r for r in fresh if r['condition']==condition and r['benchmark']==benchmark and r['id']==key and r['completion_sha256']==digest]
            if labels:
                assert len(labels)==1
                review=dict(labels[0])
                if benchmark=='humaneval_plus':assert review['solution_sha256']==sha(programs[key])
            else:
                labels=[r for r in reused if r['condition']==condition and r['benchmark']==benchmark and r['id']==key and r['completion_sha256']==digest and r['matched_prior_judgment']]
                assert len(labels)==1,('Missing or duplicate review',condition,key)
                review=dict(labels[0])
                if benchmark=='humaneval_plus':assert review['reviewed_view_sha256']==sha(programs[key])
            current.append(review);records.append(review)
        counts=Counter(str(r['label']) for r in current)
        item={'n':n,'fresh_case_reviews':sum(r['status']=='fresh_case_review' for r in current),'exact_prior_reuses':sum(r['status'].startswith('reused_exact') for r in current)}
        if benchmark=='humaneval_plus':item.update(no_defect_found=counts['1'],defect_or_missing=counts['0'],unresolved=counts['?'],metric='Static program assessment; not benchmark pass@1')
        elif benchmark=='math500':item.update(correct_resolved_value=counts['1'],incorrect_or_unresolved=counts['0'],metric='Semantic resolved-value review; not full proof audit')
        else:item.update(resolved=sum(r['label'] is not None for r in current),unresolved=sum(r['label'] is None for r in current),accuracy='Not scored here; expert key remains isolated until all396 choices frozen')
        summary[condition][benchmark]=item
    target=D/(condition+'-review-frozen.json')
    encoded=json.dumps([r for r in records if r['condition']==condition],indent=2)
    if target.exists():assert target.read_text()==encoded,'A frozen review may not be silently rewritten'
    else:target.write_text(encoded,encoding='utf-8')
report={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'Codex, not independent human graders','conditions':summary,'private_review_sha256':{c:hashlib.sha256((D/(c+'-review-frozen.json')).read_bytes()).hexdigest() for c in conditions}}
(O/('MANUAL-COVERAGE-'+a.condition+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
