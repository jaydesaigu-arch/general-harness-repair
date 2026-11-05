"""Qualify the frozen isolated evaluator; never calls the generator."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path('/workspace/sarvam30b');sys.stdout=sys.stderr=open(R/'bootstrap-cpu.log','a',buffering=1)
subprocess.run([sys.executable,'-m','pip','install','-q','--disable-pip-version-check',
 'numpy==1.26.4','psutil==6.1.1','appdirs==1.4.4','tempdir==0.7.1','wget==3.2','tqdm==4.67.1','termcolor==2.5.0',
 'fire==0.7.0','rich==13.9.4','multipledispatch==1.0.0','tree-sitter==0.23.2','tree-sitter-python==0.23.6',
 'datasets==4.4.1','pyarrow==21.0.0','math-verify[antlr4_13_2]==0.9.0'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','evalplus==0.3.1'],check=True)
for script,args in [('/workspace/he164/patch_evalplus.py',[]),('/workspace/math500/math_grade.py',['qualify']),
 ('/workspace/math500/math_grade_v2.py',['qualify']),('/workspace/math500/math_grade_v3.py',['qualify']),
 ('/workspace/math500/audit_gold_parse.py',[]),('/workspace/he164/cpu_eval.py',['prepare'])]:
    subprocess.run([sys.executable,script]+args,check=True)
(R/'cpu-ready.json').write_text(json.dumps({'ready':True,'time':time.time()}));print('CPU_READY',flush=True)
deadline=json.loads((R/'limits.json').read_text())['deadline_unix']
while time.time()<deadline:time.sleep(20)
