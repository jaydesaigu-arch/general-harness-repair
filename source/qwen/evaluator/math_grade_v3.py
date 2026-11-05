"""General answer-field formatting compatibility; v1/v2 scores stay immutable.

Developed after baseline inspection. These are evaluator repairs, not evidence
of improved model reasoning. Never select an answer by comparison with gold.
"""
import argparse,hashlib,json,re
from pathlib import Path
from math_grade import ROOT,parsed,verify,boxed,extract
from math_grade_v2 import notation,qualify as unused_v2_qualify

def clean(value):
    if value is None:return None
    value=value.strip()
    # Closing emphasis immediately after a formatted label: **Answer:** 13.
    value=re.sub(r'^\*{1,2}\s*','',value)
    # Sentence punctuation and a closing quotation around the answer field.
    # An apostrophe is deliberately preserved (it may denote a derivative).
    value=re.sub(r'[.\u3002]\s*$','',value).strip()
    if len(value)>=2 and value[0]==value[-1] and value[0] in ['"','`']:
        value=value[1:-1].strip()
    elif value.endswith('"'):value=value[:-1].rstrip()
    if value.endswith('**'):value=value[:-2].rstrip()
    return notation(value)

def extract_v3(text):
    # Last occurrence, including a nested label on the same line. Do not search
    # earlier alternatives for a value that agrees with the reference.
    labels=list(re.finditer(r'(?i)(?<![A-Za-z])Answer\s*:',text))
    if labels:
        for label in reversed(labels):
            tail=text[label.end():].split('\n',1)[0]
            value=clean(tail)
            # Preserve v1/v2's nonempty-field rule. A truncated empty label
            # supplies no replacement value; an explicit nonempty revision,
            # including an admission of failure, always replaces the old one.
            if value:return value,'last_nonempty_answer_label_formatting_v3'
        return None,'missing'
    old,method=extract(text,'final_or_boxed')
    return clean(old),method

def equal_v3(gold,pred):
    if pred is None:return False
    value=clean(pred)
    if not value:return False
    if verify(parsed(gold),parsed(value),float_rounding=6,numeric_precision=15,strict=True,timeout_seconds=5):return True
    # Plain multi-letter names are text, not a product of individual variables.
    # This conversion is prediction-only and never substitutes reference text.
    if re.fullmatch(r'[A-Za-z][A-Za-z ]{1,79}',value):
        return bool(verify(parsed(gold),parsed(r'\text{'+value+'}'),float_rounding=6,numeric_precision=15,strict=True,timeout_seconds=5))
    return False

def qualify():
    original=json.loads((ROOT/'qualification.json').read_text())
    rows=[]
    for r in original['fixtures']:
        actual=equal_v3(r['gold'],r['prediction'])
        rows.append({'name':'original_'+r['name'],'passed':actual==r['expected']})
    fixtures=[
      ('markdown_label','13','**Answer:** $13','13',True),
      ('markdown_wrong','13','**Answer:** $14','14',False),
      ('sentence_period',r'\frac{5}{8}','Answer: 0.625.','0.625',True),
      ('sentence_wrong',r'\frac{5}{8}','Answer: 0.626.','0.626',False),
      ('quoted_radical',r'3\sqrt{5}',r'Answer: "3$\sqrt{5}$". ',r'3\sqrt{5}',True),
      ('quoted_wrong_radical',r'3\sqrt{5}',r'Answer: "3$\sqrt{6}$". ',r'3\sqrt{6}',False),
      ('nested_label','23','Final answer: Answer: 23','23',True),
      ('nested_wrong','23','Final answer: Answer: 24','24',False),
      ('last_revision_wrong','23','Answer: 23\nAnswer: 24','24',False),
      ('last_revision_correct','23','Answer: 24\nAnswer: 23','23',True),
      ('same_line_last_wrong','23','Answer: 23; revised Answer: 24','24',False),
      ('truncated_empty_label','23','Answer: 23\nDraft final line: Answer:','23',True),
      ('empty_label_no_gold_search','23','Answer: 24\nAnswer:','24',False),
      ('only_empty_label','23','Answer:',None,False),
      ('name',r'\text{Nadia}','Answer: Nadia.','Nadia',True),
      ('wrong_name',r'\text{Nadia}','Answer: Nadir.','Nadir',False),
      ('name_embedded',r'\text{Nadia}','Answer: Perhaps Nadia or Nadir.','Perhaps Nadia or Nadir',False),
      ('no_name_search',r'\text{Nadia}','Nadia is possible but no answer supplied.',None,False),
      ('no_earlier_fallback','7','Answer: 7\nAnswer: I cannot solve it','I cannot solve it',False),
      ('bold_answer','17','Answer: **17**','17',True),
      ('bold_wrong','17','Answer: **18**','18',False),
      ('inline_code','19','Answer: `19`','19',True),
      ('inline_wrong','19','Answer: `20`','20',False),
      ('algebra_prime',r"f'(x)",r"Answer: f'(x)",r"f'(x)",True),
      ('tuple_order','(2,9)','Answer: (9,2).','(9,2)',False),
      ('interval_closed','(2,9)','Answer: [2,9].','[2,9]',False),
      ('wrong_sign','-7','Answer: 7.','7',False),
      ('no_rounding_relaxation',r'\frac{2}{3}','Answer: 0.67.','0.67',False),
      ('decimal_point','2.75','Answer: 2.75','2.75',True),
      ('decimal_integer','2','Answer: 2.','2',True),
      ('wrong_units',r'\frac{\pi}{2}','Answer: 90.','90',False),
    ]
    for name,gold,text,expected,p in fixtures:
        value,method=extract_v3(text);actual=equal_v3(gold,value)
        rows.append({'name':name,'gold':gold,'response':text,'expected_answer':expected,'answer':value,
                     'expected_correct':p,'actual_correct':actual,'passed':value==expected and actual==p})
    result={'qualification_passed':all(r['passed'] for r in rows),'n':len(rows),'controls':rows,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'qualification-v3.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    assert result['qualification_passed']

def score(name):
    q=json.loads((ROOT/'qualification-v3.json').read_text());assert q['qualification_passed']
    assert q['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    gold={r['id']:r['Answer'] for r in json.loads((ROOT/'private-gold.json').read_text())}
    rows=[json.loads(x) for x in (ROOT/(name+'.jsonl')).read_text().splitlines()]
    assert len(rows)==500 and {r['id'] for r in rows}==gold.keys()
    out=[]
    for row in rows:
        answer,method=extract_v3(row['completion'])
        out.append({'id':row['id'],'answer':answer,'extraction':method,'correct':equal_v3(gold[row['id']],answer)})
    (ROOT/'scores'/(name+'-v3.json')).write_text(json.dumps(out,indent=2))
    result={'name':name,'n':500,'formatting_v3':sum(r['correct'] for r in out)}
    (ROOT/'scores'/(name+'-v3-summary.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['qualify','score']);p.add_argument('--name',default='baseline-raw');a=p.parse_args()
    qualify() if a.mode=='qualify' else score(a.name)
