"""Append explicit Codex case judgments, validating source identity."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('decisions');p.add_argument('--partial',action='store_true');a=p.parse_args()
D=H/'private-review';D.mkdir(exist_ok=True);target=D/'fresh-judgments.json'
ledger=json.loads(target.read_text()) if target.exists() else []
items=json.loads(Path(a.decisions).read_text())
lookup={(r['id'],r['condition'],r['completion_sha256']):r for r in ledger}
record_cache={};program_cache={}
for case in items:
    benchmark=case['benchmark'];condition=case['condition'];q=case['id']
    source=H/('partial-review-only' if a.partial else 'private-remote')/'generations'/(condition+'-'+benchmark+'.jsonl')
    if source not in record_cache:
        record_cache[source]={r['id']:r for r in [json.loads(s) for s in source.read_text().splitlines()]}
    records=record_cache[source]
    row=records[q];digest=hashlib.sha256(row['completion'].encode()).hexdigest();key=(q,condition,digest)
    assert key not in lookup,'Record corrections explicitly; do not overwrite earlier judgments.'
    if benchmark=='gpqa_diamond':assert case['label'] in ['A','B','C','D',None]
    elif benchmark=='math500':assert case['label'] in [0,1]
    else:assert case['label'] in ['0','1','?']
    if benchmark=='humaneval_plus':
        codepath=H/'partial-review-only/code'/(condition+'-solutions.jsonl') if a.partial else H/'private-cpu/he164/scores'/(condition+'-raw')/'solutions.jsonl'
        if codepath not in program_cache:
            program_cache[codepath]={r['task_id']:r['solution'] for r in [json.loads(s) for s in codepath.read_text().splitlines()]}
        programs=program_cache[codepath]
        assert case['solution_sha256']==hashlib.sha256(programs[q].encode()).hexdigest()
    record={**case,'completion_sha256':digest,'reviewer':'Codex, not an independent human panel',
        'review_time_utc':datetime.now(timezone.utc).isoformat(),'status':'fresh_case_review',
        'scope':'Semantic resolved-choice extraction, not adjudication of scientific truth' if benchmark=='gpqa_diamond' else 'Semantic final-value review, not full proof audit' if benchmark=='math500' else 'Static program assessment, not benchmark pass@1'}
    ledger.append(record);lookup[key]=record
target.write_text(json.dumps(ledger,indent=2),encoding='utf-8')
print(json.dumps({'new_judgments':len(items),'total_fresh_judgments':len(ledger)}))
