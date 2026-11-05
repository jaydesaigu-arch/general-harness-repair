"""Backport the exact upstream find_zero progress bookkeeping fix."""
from pathlib import Path
import hashlib,json
import evalplus.eval
path=Path(evalplus.eval.__file__)
text=path.read_text()
before=hashlib.sha256(text.encode()).hexdigest()
old='assert abs(_poly(*inp, out)) <= atol\n                            continue'
new='assert abs(_poly(*inp, out)) <= atol\n                            details[i] = True\n                            progress.value += 1\n                            continue'
if old in text:text=text.replace(old,new);path.write_text(text)
else:assert new in text
Path('/workspace/he164/evalplus-patch.json').write_text(json.dumps({'source':str(path),
 'before_sha256':before,'after_sha256':hashlib.sha256(text.encode()).hexdigest(),
 'upstream':'https://github.com/evalplus/evalplus/blob/master/evalplus/eval/__init__.py',
 'change':'Mark successful find_zero input as passed and increment progress before continue. No comparison tolerance or test input changed.'},indent=2))
print('Applied upstream find_zero bookkeeping fix')
