"""General notation compatibility, keeping v1 results and semantics intact."""
import argparse,hashlib,json,re,time
from pathlib import Path
from math_grade import ROOT,parsed,extract,verify,equal

def notation(text):
    # Inline math fences may surround only part of an answer, e.g. 9$\\sqrt{3}$.
    # Preserve escaped monetary dollars, which the math parser already handles.
    text=re.sub(r'(?<!\\)\$', '',text).strip()
    def roots(s):
        out=[];i=0
        while i<len(s):
            if s[i]!='√':out.append(s[i]);i+=1;continue
            j=i+1
            while j<len(s) and s[j].isspace():j+=1
            if j<len(s) and s[j] in '({':
                opening=s[j];closing=')' if opening=='(' else '}';depth=1;k=j+1
                while k<len(s) and depth:
                    if s[k]==opening:depth+=1
                    elif s[k]==closing:depth-=1
                    k+=1
                if depth==0:
                    out.append(r'\sqrt{'+roots(s[j+1:k-1])+'}');i=k;continue
            atom=re.match(r'(?:[0-9]+(?:\.[0-9]+)?|[A-Za-z])',s[j:])
            if atom:
                out.append(r'\sqrt{'+atom.group(0)+'}');i=j+len(atom.group(0));continue
            out.append(s[i]);i+=1
        return ''.join(out)
    return roots(text)

def equal_v2(gold,pred):
    if pred is None:return False
    return bool(verify(parsed(gold),parsed(notation(pred)),float_rounding=6,numeric_precision=15,strict=True,timeout_seconds=5))

def qualify():
    original=json.loads((ROOT/'qualification.json').read_text())
    rows=[]
    for r in original['fixtures']:
        actual=equal_v2(r['gold'],r['prediction'])
        rows.append({'name':'original_'+r['name'],'expected':r['expected'],'actual':actual,'passed':actual==r['expected']})
    fixtures=[
        ('fragmented_math',r'9\sqrt{3}',r'9$\sqrt{3}$',True),
        ('fragmented_wrong_coefficient',r'9\sqrt{3}',r'8$\sqrt{3}$',False),
        ('separate_fences',r'7\sqrt{6}',r'$7$$\sqrt{6}$',True),
        ('fenced_fraction',r'\frac{8}{15}',r'$\frac{8}{15}$',True),
        ('unicode_integer_root','7','√49',True),
        ('unicode_wrong_root','49','√49',False),
        ('unicode_multidigit_root','10','√100',True),
        ('unicode_no_digit_split','2','√12',False),
        ('unicode_product',r'6\sqrt{3}','3√12',True),
        ('unicode_sum_root','3','√(2+7)',True),
        ('unicode_braced_root','3','√{2+7}',True),
        ('unicode_nested_root','2','√(1+√9)',True),
        ('unicode_negative_root','-2','-√4',True),
        ('unicode_not_negative','2','-√4',False),
        ('unicode_wrong_scope','5','√(9+16)',True),
        ('unicode_scope_negative','7','√(9+16)',False),
        ('single_symbol',r'\sqrt{x}','√x',True),
        ('incorrect_tuple_stays_wrong','(3,8)','(8,3)',False),
        ('wrong_interval_stays_wrong','(1,4]','(1,4)',False),
        ('approximation_not_relaxed',r'\frac{2}{3}','0.67',False)]
    for name,gold,pred,expected in fixtures:
        actual=equal_v2(gold,pred)
        rows.append({'name':name,'gold':gold,'prediction':pred,'normalized_prediction':notation(pred),
                     'expected':expected,'actual':actual,'passed':actual==expected})
    refs=json.loads((ROOT/'private-reference-solutions.json').read_text())
    identity=[{'id':r['id'],'passed':equal_v2(r['openai_answer'],r['hf_answer'])} for r in refs]
    result={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'fixtures':rows,'reference_crosscheck':identity,
            'qualification_passed':all(r['passed'] for r in rows+identity),
            'synthetic_passed':sum(r['passed'] for r in rows),'synthetic_total':len(rows)}
    (ROOT/'qualification-v2.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['fixtures','reference_crosscheck']}),flush=True)

def score(name):
    q=json.loads((ROOT/'qualification-v2.json').read_text())
    assert q['qualification_passed'] and q['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    old=json.loads((ROOT/'scores'/(name+'.json')).read_text())
    gold={r['id']:r['Answer'] for r in json.loads((ROOT/'private-gold.json').read_text())}
    rows=[json.loads(x) for x in (ROOT/(name+'.jsonl')).read_text().splitlines()]
    assert len(rows)==len(old)==500 and len({r['id'] for r in rows})==500
    out=[]
    for row in rows:
        answer,method=extract(row['completion'],'final_or_boxed')
        out.append({'id':row['id'],'answer':answer,'normalized_answer':notation(answer) if answer else None,
                    'extraction':method,'correct':equal_v2(gold[row['id']],answer)})
    (ROOT/'scores'/(name+'-v2.json')).write_text(json.dumps(out,indent=2))
    summary={'name':name,'n':500,'normalized_final':sum(r['correct'] for r in out)}
    (ROOT/'scores'/(name+'-v2-summary.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['qualify','score']);p.add_argument('--name',default='baseline');a=p.parse_args()
    if a.mode=='qualify':qualify()
    else:score(a.name)
