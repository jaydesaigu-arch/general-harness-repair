"""Aggregate complete frozen evidence. No solver calls or scorer modifications."""
import csv,hashlib,json,math
from pathlib import Path
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;R=H.parents[1];O=R/'outputs/sarvam30b-frozen-replication'
read=lambda p:json.loads(p.read_text())
assert read(O/'generation-audit.json')['status']=='PASS'
assert read(H/'gpu-controller-state.json').get('gpu_stopped')
assert read(H/'cpu-controller-state.json').get('cpu_stopped')
assert set(read(H/'cpu-controller-state.json')['completed'])=={'baseline','combined_repaired'}
gpqa=read(O/'GPQA-RESULTS.json')
conditions=['baseline','combined_repaired'];metrics={};sources=[]
def load(p):
 sources.append({'path':str(p.relative_to(R)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 return read(p)
def add(name,b,c,d):
 expected={'humaneval_plus':164,'math500':500,'gpqa_diamond':198}[b]
 assert len(d)==expected
 metrics.setdefault(name,{'benchmark':b,'n':expected,'conditions':{}})['conditions'][c]=d
for c in conditions:
 for view in ['raw','final']:
  folder=H/'private-cpu/he164/scores'/(c+'-'+view)
  base={r['task_id']:r['status']=='pass' for r in load(folder/'base.json')}
  plus={r['task_id']:r['status']=='pass' for r in load(folder/'plus.json')}
  assert base.keys()==plus.keys()
  add('humaneval_plus_'+view,'humaneval_plus',c,{k:base[k] and plus[k] for k in base})
  add('humaneval_base_'+view,'humaneval_plus',c,base)
  rows=load(H/'private-cpu/math500/scores'/(c+'-'+view+'-v3.json'))
  add('math500_'+view+'_v3','math500',c,{r['id']:bool(r['correct']) for r in rows})
 reviews=load(H/'private-review'/(c+'-review-frozen.json'))
 for benchmark,metric in [('math500','math500_codex_resolved_value'),('humaneval_plus','code_codex_no_defect')]:
  items=[r for r in reviews if r['benchmark']==benchmark]
  add(metric,benchmark,c,{r['id']:str(r['label'])=='1' for r in items})
details=load(H/'private-review/gpqa-per-problem-scores.json')
for c in conditions:
 for m in ['official_raw','official_final','qualified_final','codex_resolved_choice']:
  add('gpqa_'+m,'gpqa_diamond',c,{r['id']:r['correct'][m] for r in details if r['condition']==c})
summary={'status':'COMPLETE','completed_at_utc':datetime.now(timezone.utc).isoformat(),'metrics':{},'source_artifacts':sources,
 'notes':['Fresh generation after combined harness freeze on the same public benchmarks, not an unseen-task holdout.',
          'Weights unchanged. Single completion per problem and arm. Code ceiling4096; math/GPQA ceiling8192.',
          'Primary automatic: raw HumanEval base AND expanded, raw Math v3, GPQA official raw.',
          'Identical frozen graders in both arms. Plain generic math instruction belongs to the combined repaired bundle.',
          'Codex reviews are neither independent human grades nor a blinded panel. Code no-defect labels are static assessments, not execution pass@1.',
          'Math semantic labels assess resolved final values, not complete mathematical proof correctness.',
          'GPQA Codex labels extract chosen options without an expert key, frozen before key scoring; not expert scientific adjudication.',
          'Manual comparisons do not override automatic benchmark results; all regressions retained.']}
def pair(a,b):
 assert a.keys()==b.keys()
 gains=sum(not a[k] and b[k] for k in a);losses=sum(a[k] and not b[k] for k in a)
 n=gains+losses
 p=min(1,2*sum(math.comb(n,i) for i in range(min(gains,losses)+1))/2**n) if n else 1
 return {'gains':gains,'losses':losses,'net':gains-losses,'descriptive_exact_two_sided_mcnemar_p':p}
for name,m in metrics.items():
 a,b=(m['conditions'][c] for c in conditions)
 summary['metrics'][name]={'benchmark':m['benchmark'],'n':m['n'],
  'baseline_correct':sum(a.values()),'combined_repaired_correct':sum(b.values()),
  'baseline_percent':100*sum(a.values())/m['n'],'combined_repaired_percent':100*sum(b.values())/m['n'],
  'percentage_point_change':100*(sum(b.values())-sum(a.values()))/m['n'],**pair(a,b)}
gpu=load(O/'GPU-COST.json');cpu=load(O/'CPU-COST.json')
summary['compute_cost']={'gpu_and_port_conservative_usd':gpu['conservative_gpu_and_port_usd'],
 'cpu_and_port_conservative_usd':cpu['conservative_cpu_and_port_usd'],
 'total_conservative_usd':round(gpu['conservative_gpu_and_port_usd']+cpu['conservative_cpu_and_port_usd'],4),
 'basis':'Elapsed allocation including startup and both full arms, rounded up by minute; estimate, not a Hugging Face invoice.',
 'weight_training_usd':0,'gpu_stopped':gpu['stopped'],'cpu_stopped':cpu['stopped']}
(O/'RESULTS.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
fields=['metric','benchmark','n','baseline_correct','combined_repaired_correct','baseline_percent','combined_repaired_percent','percentage_point_change','gains','losses','net','descriptive_exact_two_sided_mcnemar_p']
with (O/'RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({'metric':k,**v} for k,v in summary['metrics'].items())
print(json.dumps({'metrics':summary['metrics'],'compute_cost':summary['compute_cost']},indent=2))
