"""Freeze complete reviewed choices before opening the key for fresh scoring."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
H=Path(__file__).resolve().parent;O=H.parents[1]/'outputs/qwen38-27b';D=H/'private-review'
target=D/'gpqa-choices-frozen.json';assert not target.exists()
reused=json.loads((D/'identity-reused-judgments.json').read_text())
fresh=json.loads((D/'fresh-judgments.json').read_text())
choices={}
for condition in ['baseline','combined_repaired']:
    path=H/'private-remote/generations'/(condition+'-gpqa_diamond.jsonl')
    rows=[json.loads(s) for s in path.read_text().splitlines()];assert len(rows)==198
    choices[condition]={}
    for row in rows:
        q=row['id'];digest=hashlib.sha256(row['completion'].encode()).hexdigest()
        labels=[r for r in fresh if r['benchmark']=='gpqa_diamond' and r['condition']==condition and r['id']==q and r['completion_sha256']==digest]
        if not labels:labels=[r for r in reused if r['benchmark']=='gpqa_diamond' and r['condition']==condition and r['id']==q and r['completion_sha256']==digest and r['matched_prior_judgment']]
        assert len(labels)==1,('Missing or duplicate current choice review',condition,q)
        label=labels[0];assert label['label'] in ['A','B','C','D',None]
        choices[condition][q]={'choice':label['label'],'completion_sha256':digest,'notes':label.get('notes',label.get('rationale',label.get('previous_judgments')))}
target.write_text(json.dumps(choices,indent=2),encoding='utf-8')
record={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'conditions':2,'n_per_condition':198,
        'private_choice_ledger_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'scope':'Resolved-choice review frozen before fresh expert-key scoring; Codex review, not independent human science adjudication.'}
(O/'manual-choice-freeze.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
