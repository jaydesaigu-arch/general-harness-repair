"""Recompute the six reported primary counts from retained per-item verdicts. No model calls."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
expected=json.loads((R/'results/primary-counts-recomputed.json').read_text(encoding='utf-8'))
for model,benchmarks in expected.items():
 for benchmark,want in benchmarks.items():
  rows=[json.loads(s) for s in (R/'answers'/model/'paired'/(benchmark+'.jsonl')).read_text(encoding='utf-8').splitlines()]
  assert len(rows)==len({r['id'] for r in rows})==want['n']
  a=sum(r['baseline']['automatic_primary'] for r in rows);b=sum(r['combined_repaired']['automatic_primary'] for r in rows)
  g=sum(not r['baseline']['automatic_primary'] and r['combined_repaired']['automatic_primary'] for r in rows)
  l=sum(r['baseline']['automatic_primary'] and not r['combined_repaired']['automatic_primary'] for r in rows)
  assert (a,b,g,l)==(want['baseline'],want['repaired'],want['gains'],want['losses'])
  print(f'{model:7} {benchmark:15} {a}/{len(rows)} -> {b}/{len(rows)}; gains={g}, losses={l}')
print('PASS: saved verdicts reproduce every reported primary count; no independent rescoring or generation performed.')
