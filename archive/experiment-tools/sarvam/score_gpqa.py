"""Score frozen GPQA generation/review consistently; never publish examples."""
import hashlib,json,math
from pathlib import Path
from grade_gpqa import official,repaired
H=Path(__file__).resolve().parent;R=H.parents[1];O=R/'outputs/sarvam30b-frozen-replication'
freeze=json.loads((O/'manual-choice-freeze.json').read_text());path=H/'private-review/gpqa-choices-frozen.json'
assert hashlib.sha256(path.read_bytes()).hexdigest()==freeze['private_choice_ledger_sha256']
choices=json.loads(path.read_text());gold={r['id']:r['answer'] for r in json.loads((R/'work/sarvam30b-gpqa/private-gold.json').read_text())}
assert len(gold)==198
details=[];summary={'conditions':{}}
for condition in ['baseline','combined_repaired']:
    rows=[json.loads(s) for s in (H/'private-remote/generations'/(condition+'-gpqa_diamond.jsonl')).read_text().splitlines()]
    assert len(rows)==198 and {r['id'] for r in rows}==set(gold)==choices[condition].keys()
    current=[]
    for row in rows:
        k=row['id'];review=choices[condition][k];assert hashlib.sha256(row['completion'].encode()).hexdigest()==review['completion_sha256']
        pred={'official_raw':official(row['completion']),'official_final':official(row['final_completion']),
              'qualified_final':repaired(row['completion']),'codex_resolved_choice':review['choice']}
        detail={'id':k,'condition':condition,'predictions':pred,'gold':gold[k],'correct':{m:v==gold[k] for m,v in pred.items()},
            'completion_sha256':review['completion_sha256']};current.append(detail);details.append(detail)
    summary['conditions'][condition]={'n':198,'metrics':{m:{'correct':sum(d['correct'][m] for d in current),'unresolved':sum(d['predictions'][m] is None for d in current)} for m in pred}}
by={(d['condition'],d['id']):d for d in details};summary['paired']={}
for metric in ['official_raw','official_final','qualified_final','codex_resolved_choice']:
    gains=sum(not by['baseline',k]['correct'][metric] and by['combined_repaired',k]['correct'][metric] for k in gold)
    losses=sum(by['baseline',k]['correct'][metric] and not by['combined_repaired',k]['correct'][metric] for k in gold)
    n=gains+losses;p=min(1,2*sum(math.comb(n,i) for i in range(min(gains,losses)+1))/2**n) if n else 1
    summary['paired'][metric]={'gains':gains,'losses':losses,'net':gains-losses,'descriptive_exact_two_sided_mcnemar_p':p}
(H/'private-review/gpqa-per-problem-scores.json').write_text(json.dumps(details,indent=2))
(O/'GPQA-RESULTS.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
