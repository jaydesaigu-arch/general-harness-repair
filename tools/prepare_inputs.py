"""Locally reconstruct the exact solver projection from an authorized GPQA CSV. Never upload it."""
import argparse,csv,hashlib,io,json,random
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--gpqa-csv',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--gold-output',type=Path)
a=p.parse_args();assert not a.output.exists(),'Never overwrite an existing private projection'
if a.gold_output:assert not a.gold_output.exists() and a.gold_output.resolve()!=a.output.resolve()
spec=json.loads((R/'inputs/gpqa-reconstruction.json').read_text(encoding='utf-8'));data=a.gpqa_csv.read_bytes()
assert hashlib.sha256(data).hexdigest()==spec['csv_sha256'],'CSV differs from evaluated artifact; obtain the pinned authorized version'
rows=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))));assert len(rows)==198
lookup={r['id']:r for r in json.loads((R/'inputs/code-math-projection.json').read_text(encoding='utf-8'))}
rng=random.Random(0);gold=[]
for i,row in enumerate(rows):
 choices=[row['Correct Answer']]+[row[f'Incorrect Answer {j}'] for j in [1,2,3]]
 ordered=[choices[j] for j in rng.sample(range(4),4)];k=f'gpqa_diamond/{i:03d}'
 prompt=spec['template'].format(Question=row['Question'],**dict(zip('ABCD',ordered)))
 lookup[k]={'id':k,'benchmark':'gpqa_diamond','prompt_before':prompt,'prompt_after':prompt,'output_budget':8192}
 gold.append({'id':k,'answer':'ABCD'[ordered.index(row['Correct Answer'])]})
order=json.loads((R/'inputs/order.json').read_text(encoding='utf-8'));projection=[lookup[k] for k in order]
assert len(projection)==len({r['id'] for r in projection})==862
blob=json.dumps(projection,ensure_ascii=False).encode('utf-8');digest=hashlib.sha256(blob).hexdigest()
assert digest=='732f96bca7bafc621e65092e3ea343a6b20fd3cf817308aefa1a5e36a6039bba',digest
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(blob)
if a.gold_output:
 a.gold_output.parent.mkdir(parents=True,exist_ok=True);a.gold_output.write_text(json.dumps(gold,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'private_projection_written':True,'n':862,'sha256':digest,'separate_cpu_gold_written':bool(a.gold_output)}))
