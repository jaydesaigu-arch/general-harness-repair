"""Validate released artifacts without model calls, networking or generated-program execution."""
import gzip,hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=read(R/'MANIFEST.json');checked=0
for name,d in manifest['files'].items():
 p=R/name;assert p.resolve().is_relative_to(R.resolve()) and p.is_file(),name
 assert sha(p.read_bytes())==d,name;checked+=1
safe={'completion_sha256','rendered_prompt_sha256','tokens','truncated','policy_applied','automatic_primary','automatic_official_final','automatic_qualified_final','codex_correct','codex_resolved'}
text_records=metadata_records=0
for model in ['sarvam','qwen']:
 protocol=read(R/'protocols'/(model+'.json'))
 for n,d in protocol['solver_files'].items():
  if n!='questions.json':assert sha((R/'source'/model/n).read_bytes())==d
 for n,d in protocol['evaluator_files'].items():assert sha((R/'source'/model/'evaluator'/n).read_bytes())==d
 for b,n in [('humaneval_plus',164),('math500',500),('gpqa_diamond',198)]:
  paired=[json.loads(s) for s in (R/'answers'/model/'paired'/(b+'.jsonl')).read_text(encoding='utf-8').splitlines()]
  assert len(paired)==len({r['id'] for r in paired})==n
  for c in ['baseline','combined_repaired']:
   if b=='gpqa_diamond':
    for row in paired:
     a=row[c];assert set(a)==safe
     assert re.fullmatch('[0-9a-f]{64}',a['completion_sha256']) and re.fullmatch('[0-9a-f]{64}',a['rendered_prompt_sha256'])
     assert isinstance(a['tokens'],int) and a['tokens']<=8192
     assert all(type(v) is bool for k,v in a.items() if k not in ['completion_sha256','rendered_prompt_sha256','tokens'])
    metadata_records+=n;continue
   blob=gzip.decompress((R/'answers'/model/'token-records'/(c+'-'+b+'.jsonl.gz')).read_bytes())
   records={v['id']:v for v in map(json.loads,blob.decode('utf-8').splitlines())};assert len(records)==n
   for pair in paired:
    a=pair[c];v=records[pair['id']]
    assert sha(v['completion'].encode())==a['completion_sha256']
    assert v['completion']==a['raw_completion'] and v['final_completion']==a['delivered_final']
    assert v['sampled_prefix_ids']+v['forced_token_ids']+v['sampled_continuation_ids']==v['completion_token_ids']
    assert len(v['completion_token_ids'])==v['new_tokens']<=v['total_output_budget']==(4096 if b=='humaneval_plus' else 8192)
    assert v['condition']==c and v['benchmark']==b
   text_records+=n
assert text_records==2656 and metadata_records==792
print(json.dumps({'status':'PASS','manifest_files':checked,'primary_full_text_responses':text_records,'primary_gpqa_metadata_records':metadata_records,'source_hashes_verified':True,'no_generation_or_program_execution':True},indent=2))
