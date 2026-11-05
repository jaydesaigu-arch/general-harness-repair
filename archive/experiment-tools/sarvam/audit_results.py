"""Audit immutable fresh full-test artifacts; do not infer missing results."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;O=H.parents[1]/'outputs/sarvam30b-frozen-replication'
protocol=json.loads((O/'protocol.json').read_text());G=H/'private-remote/generations'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,d in protocol['solver_files'].items():assert sha(H/n)==d
for n,d in protocol['evaluator_files'].items():assert sha(H/'evaluator'/n)==d
for path,digest in json.loads((H/'reference-identities.json').read_text())['sha256'].items():assert sha(H.parents[1]/path)==digest
questions={r['id']:r for r in json.loads((H/'questions.json').read_text())}
assert len(questions)==862
from harness import final_channel
audit={'status':'PASS','protocol_sha256':sha(O/'protocol.json'),'conditions':{},'all_pairs':{}}
allrows={}
for condition in ['baseline','combined_repaired']:
    allrows[condition]={};current={}
    for benchmark,n in [('humaneval_plus',164),('math500',500),('gpqa_diamond',198)]:
        p=G/(condition+'-'+benchmark+'.jsonl');rows=[json.loads(s) for s in p.read_text().splitlines()]
        ids={r['id'] for r in rows};expected={k for k,q in questions.items() if q['benchmark']==benchmark}
        assert len(rows)==len(ids)==n and ids==expected
        for row in rows:
            q=questions[row['id']];budget=q['output_budget'];total=row['completion_token_ids']
            assert row['condition']==condition and row['benchmark']==benchmark
            assert len(total)==row['new_tokens']<=budget and row['total_output_budget']==budget
            assert row['sampled_prefix_ids']+row['forced_token_ids']+row['sampled_continuation_ids']==total
            assert row['forced_token_ids'] in ([],[24,107])
            if condition=='baseline':assert not row['forced_token_ids'] and not row['sampled_prefix_ids']
            if row['forced_token_ids']:assert len(row['sampled_prefix_ids'])==budget-1024 and len(row['sampled_continuation_ids'])<=1022
            assert final_channel(row['completion'])==(row['final_completion'],row['final_channel_method'])
            allrows[condition][row['id']]=row
        current[benchmark]={'n':n,'generation_sha256':sha(p),'new_tokens':sum(r['new_tokens'] for r in rows),
            'truncated':sum(r['truncated'] for r in rows),'policy_applied':sum(r['policy_applied'] for r in rows)}
    assert len(allrows[condition])==862;audit['conditions'][condition]=current
for benchmark,n in [('humaneval_plus',164),('math500',500),('gpqa_diamond',198)]:
    ids=[k for k,q in questions.items() if q['benchmark']==benchmark]
    same=sum(allrows['baseline'][k]['rendered_prompt_sha256']==allrows['combined_repaired'][k]['rendered_prompt_sha256'] for k in ids)
    if benchmark!='math500':assert same==n
    else:assert same==0
    audit['all_pairs'][benchmark]={'n':n,'same_prompt_hash_count':same,'same_completion_count':sum(allrows['baseline'][k]['completion']==allrows['combined_repaired'][k]['completion'] for k in ids)}
(O/'generation-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
