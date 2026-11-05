"""Controller-only audit of all paired records; no model inference or grading."""
import hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
O=H.parents[1]/'outputs/sarvam30b'
def read(name):
    rows=[]
    for b,n in [('humaneval_plus',164),('math500',500)]:
        part=[json.loads(x) for x in (O/'raw'/f'{name}-{b}.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(part)==n and len({x['id'] for x in part})==n
        rows+=part
    assert len({r['id'] for r in rows})==664
    return {r['id']:r for r in rows}
base=read('baseline')
elig=json.loads((O/'raw/budget-eligibility.json').read_text())
eligible=set(elig['eligible']);assert len(eligible)==len(elig['eligible'])
report={'n':664,'eligible':len(eligible),'arms':{},'checks':[
    'Every original problem retained exactly once in each arm',
    'Every eligible case included without score-based selection',
    'Saved baseline prefix exactly preserved in both paired arms',
    'Forced delimiters explicitly separated from sampled tokens',
    'Total output including forced tokens within unchanged original budget',
    'Noneligible responses byte-identical to baseline',
    'Rendered prompt hashes unchanged',
]}
names=sys.argv[1:] or ['budget_forced','sham_control','fresh_repaired']
assert set(names)<=set(['budget_forced','sham_control','fresh_repaired'])
for name in names:
    rows=read(name);assert rows.keys()==base.keys()
    applied=0;truncated=0;tokens=0;forced_sequences=set()
    for k,r in rows.items():
        b=base[k];budget=4096 if b['benchmark']=='humaneval_plus' else 8192
        assert r['rendered_prompt_sha256']==b['rendered_prompt_sha256']
        ids=r.get('completion_token_ids',r.get('generated_token_ids'))
        assert ids is not None
        # Original vLLM records retain a terminal EOS in token_ids; new_tokens
        # and literal completion explicitly exclude it. Continuation records
        # already strip that terminal token. Keep this distinction auditable.
        if 'completion_token_ids' not in r:
            end=next((i for i,t in enumerate(ids) if t in [1,26]),len(ids))
            assert end==len(ids) or end==len(ids)-1
            ids=ids[:end]
        assert len(ids)==r['new_tokens'] and len(ids)<=budget
        tokens+=len(ids);truncated+=r['truncated']
        if name!='fresh_repaired' and k not in eligible:
            assert r['completion']==b['completion'] and r['generated_token_ids']==b['generated_token_ids']
            assert not r['policy_applied']
            continue
        prefix=r['sampled_prefix_ids'];forced=r['forced_token_ids'];tail=r['sampled_continuation_ids']
        assert ids==prefix+forced+tail
        assert r['sampled_token_count']==len(prefix)+len(tail)
        assert r['total_output_budget']==budget
        if name!='fresh_repaired':
            assert k in eligible and prefix==b['generated_token_ids'][:budget-1024]
            assert len(prefix)==budget-1024
            assert r['baseline_completion_sha256']==hashlib.sha256(b['completion'].encode()).hexdigest()
            assert r['fixed_prefix_sha256']==hashlib.sha256(json.dumps(prefix).encode()).hexdigest()
            assert bool(forced)==(name=='budget_forced')
        assert r['policy_applied']==bool(forced)
        if forced:
            assert len(prefix)==budget-1024 and len(forced)<1024
            forced_sequences.add(tuple(forced));applied+=1
    assert len(forced_sequences)<=1
    report['arms'][name]={'n':len(rows),'policy_applied':applied,'truncated':truncated,
                          'total_output_tokens':tokens,'forced_token_sequences':[list(t) for t in forced_sequences]}
report['passed']=True
(O/('lineage-audit.json' if 'fresh_repaired' in names else 'paired-lineage-audit.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
