"""Evaluator-private cross-check of full-solution CSV targets vs final answers."""
import json,sys
from pathlib import Path
sys.path.insert(0,'/workspace/math500')
from math_grade import equal,parsed
ROOT=Path('/workspace/math500')
rows=json.loads((ROOT/'private-reference-solutions.json').read_text())
out=[]
for r in rows:
    ok=equal(r['openai_answer'],r['hf_answer'])
    out.append({'id':r['id'],'equivalent':ok,'csv_parsed':repr(parsed(r['openai_answer'])),
                'final_parsed':repr(parsed(r['hf_answer']))})
(ROOT/'gold-parse-audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'n':len(out),'equivalent':sum(r['equivalent'] for r in out),
                  'mismatched_ids':[r['id'] for r in out if not r['equivalent']]}),flush=True)
