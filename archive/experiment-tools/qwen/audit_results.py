"""Verify the completed frozen Qwen pair, including prompt-opened channel accounting."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1];O=R/'outputs/qwen38-27b'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((H/'protocol.json').read_text())
for n,d in p['solver_files'].items():assert sha(H/n)==d
for n,d in p['evaluator_files'].items():assert sha(H/'evaluator'/n)==d
execution=json.loads((H/'execution-freeze.json').read_text())
for n,d in execution['sha256'].items():assert sha(H/n)==d
# Check harvested deployed sources as well as the local frozen bundle.
for n,d in p['solver_files'].items():assert sha(H/'private-remote'/n)==d
declared={**p['evaluator_files'],**p['solver_files'],**execution['sha256']}
deployed_cpu=[]
for path in (H/'private-cpu').rglob('*.py'):
    if path.name in declared:
        assert sha(path)==declared[path.name],('Deployed CPU source changed',str(path))
        deployed_cpu.append(str(path.relative_to(H/'private-cpu')))
assert 'sarvam30b\\score_cpu.py' in deployed_cpu or 'sarvam30b/score_cpu.py' in deployed_cpu
assert sha(H/'questions.json')==sha(R/'work/sarvam30b-frozen-replication/questions.json')
from harness import final_channel
questions={r['id']:r for r in json.loads((H/'questions.json').read_text())}
rows={};audit={'status':'PASS','protocol_sha256':sha(H/'protocol.json'),'conditions':{},'pairs':{},
    'deployed_gpu_source_files_verified':len(p['solver_files']),
    'deployed_cpu_sources_verified':deployed_cpu}
for condition in ['baseline','combined_repaired']:
    rows[condition]={};report={}
    for benchmark,n in [('humaneval_plus',164),('math500',500),('gpqa_diamond',198)]:
        path=H/'private-remote/generations'/(condition+'-'+benchmark+'.jsonl')
        items=[json.loads(x) for x in path.read_text().splitlines()]
        assert len(items)==len({r['id'] for r in items})==n
        assert {r['id'] for r in items}=={k for k,q in questions.items() if q['benchmark']==benchmark}
        for r in items:
            assert r['condition']==condition and r['benchmark']==benchmark
            assert r['prompt_reasoning_open'] is True
            assert r['completion']=='<think>\n'+r['raw_generated_text']
            assert final_channel(r['completion'])==(r['final_completion'],r['final_channel_method'])
            assert r['sampled_prefix_ids']+r['forced_token_ids']+r['sampled_continuation_ids']==r['completion_token_ids']
            budget=questions[r['id']]['output_budget']
            assert len(r['completion_token_ids'])==r['new_tokens']<=budget==r['total_output_budget']
            assert r['forced_token_ids'] in [[],[248069,198]]
            if condition=='baseline':assert not r['sampled_prefix_ids'] and not r['forced_token_ids']
            if r['forced_token_ids']:
                assert len(r['sampled_prefix_ids'])==budget-1024 and len(r['sampled_continuation_ids'])<=1022
            assert r['sampled_token_count']==len(r['sampled_prefix_ids'])+len(r['sampled_continuation_ids'])
            rows[condition][r['id']]=r
        report[benchmark]={'n':n,'sha256':sha(path),'tokens':sum(r['new_tokens'] for r in items),
            'truncated':sum(r['truncated'] for r in items),'policy_applied':sum(r['policy_applied'] for r in items)}
    assert len(rows[condition])==862;audit['conditions'][condition]=report
for benchmark,n in [('humaneval_plus',164),('math500',500),('gpqa_diamond',198)]:
    ids=[k for k,q in questions.items() if q['benchmark']==benchmark]
    same=sum(rows['baseline'][k]['rendered_prompt_sha256']==rows['combined_repaired'][k]['rendered_prompt_sha256'] for k in ids)
    assert same==(0 if benchmark=='math500' else n)
    audit['pairs'][benchmark]={'n':n,'same_prompt_hash_count':same}
(O/'generation-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
