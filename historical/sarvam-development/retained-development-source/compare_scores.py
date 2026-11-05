"""Paired outcomes with immutable automatic grades; no adjudication overrides."""
import json,math
from pathlib import Path
O=Path(__file__).resolve().parents[2]/'outputs/sarvam30b'
names=['baseline','budget_forced','sham_control','fresh_repaired']
grades={};summary={}
for name in names:
    grades[name]={};summary[name]={}
    for arm in ['raw','final']:
        p=O/'raw/scores'/f'{name}-{arm}'
        base={r['task_id']:r['status']=='pass' for r in json.loads((p/'base.json').read_text())}
        plus={r['task_id']:r['status']=='pass' for r in json.loads((p/'plus.json').read_text())}
        code={k:base[k] and plus[k] for k in base}
        m={r['id']:r['correct'] for r in json.loads((p/f'{name}-{arm}-v2.json').read_text())}
        m3={r['id']:r['correct'] for r in json.loads((p/f'{name}-{arm}-v3.json').read_text())}
        assert len(code)==164 and len(m)==500
        grades[name][arm]={'humaneval_plus':code,'math500':m,'math500_formatting_v3':m3}
        summary[name][arm]={'humaneval_base':sum(base.values()),'humaneval_plus':sum(code.values()),'math500':sum(m.values()),'math500_formatting_v3':sum(m3.values())}
comparisons={}
for before,after,before_arm,after_arm in [
    (x,y,a,a) for x,y in [('baseline','budget_forced'),('sham_control','budget_forced'),('baseline','sham_control'),('baseline','fresh_repaired')] for a in ['raw','final']
]+[('baseline','budget_forced','raw','final'),('baseline','fresh_repaired','raw','final'),('baseline','baseline','raw','final')]:
    key=before+'-'+before_arm+'__'+after+'-'+after_arm;comparisons[key]={}
    for b in ['humaneval_plus','math500','math500_formatting_v3']:
        a=grades[before][before_arm][b];z=grades[after][after_arm][b];assert a.keys()==z.keys()
        gains=sorted(k for k in a if not a[k] and z[k]);losses=sorted(k for k in a if a[k] and not z[k])
        n=len(gains)+len(losses)
        p=min(1.,2*sum(math.comb(n,i) for i in range(min(len(gains),len(losses))+1))/2**n) if n else 1.
        comparisons[key][b]={'before':sum(a.values()),'after':sum(z.values()),'n':len(a),'net':len(gains)-len(losses),
            'gains':gains,'losses':losses,'exact_paired_two_sided_p':p,
            'p_value_scope':'Exploratory paired sign/McNemar exact test; no multiple-comparison correction, no population-wide claim.'}
result={'automatic_scores':summary,'comparisons':comparisons,
        'primary_before':'baseline-raw: original literal completion, transferred frozen grader',
        'primary_after':'budget_forced-final: fixed-prefix inference policy plus final-channel extraction',
        'fresh_after':'fresh_repaired-final: full end-to-end repeated policy, generator variation disclosed',
        'manual_review_changes_official_scores':False}
(O/'RESULTS.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
for k,v in comparisons.items():print(k,json.dumps({b:{x:y for x,y in z.items() if x not in ['gains','losses','p_value_scope']} for b,z in v.items()}))
