"""Describe paired token-prefix agreement without scoring or changing any answer."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
H=Path(__file__).resolve().parent;O=H.parents[1]/'outputs/qwen38-27b'
report={'at_utc':datetime.now(timezone.utc).isoformat(),'benchmarks':{},'sources':[],
        'interpretation':['These are descriptive output diagnostics, not additional model calls or grading changes.',
        'A matching sampled prefix through the forced transition provides a direct within-response comparison up to that transition.',
        'When prefixes differ before a forced transition, later differences cannot be attributed solely to that transition.',
        'Mathematics also changes a disclosed generic instruction. Batched BF16 generation with a fixed seed is not a guarantee of bitwise-identical unforced trajectories; this study has no repeated-run variance estimate.']}
for benchmark,n in [('humaneval_plus',164),('math500',500),('gpqa_diamond',198)]:
    arms={}
    for c in ['baseline','combined_repaired']:
        p=H/'private-remote/generations'/(c+'-'+benchmark+'.jsonl')
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        assert len(rows)==len({r['id'] for r in rows})==n
        arms[c]={r['id']:r for r in rows}
        report['sources'].append({'path':str(p.relative_to(H)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    assert arms['baseline'].keys()==arms['combined_repaired'].keys()
    common=[];forced=matched=exact=sameprompt=0
    for q,a in arms['baseline'].items():
        b=arms['combined_repaired'][q];x=a['completion_token_ids'];y=b['completion_token_ids']
        k=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)))
        common.append(k);exact+=x==y;sameprompt+=a['rendered_prompt_sha256']==b['rendered_prompt_sha256']
        if b['forced_token_ids']:
            forced+=1;prefix=b['sampled_prefix_ids'];matched+=x[:len(prefix)]==prefix
    common.sort()
    report['benchmarks'][benchmark]={'n':n,'same_rendered_prompt_count':sameprompt,
        'exact_full_output_token_sequence_count':exact,'forced_transition_cases':forced,
        'forced_transition_with_identical_pretransition_sampled_prefix':matched,
        'common_output_prefix_tokens':{'minimum':common[0],'median':common[len(common)//2],'maximum':common[-1]}}
(O/'PAIRWISE-PREFIX-DIAGNOSTICS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report['benchmarks'],indent=2))
