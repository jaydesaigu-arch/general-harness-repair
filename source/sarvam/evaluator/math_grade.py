"""Offline mathematical-equivalence checks. Never sends answers to a model."""
import argparse,hashlib,importlib.metadata,json,re,sys,time
from pathlib import Path
from math_verify import parse,verify,LatexExtractionConfig
ROOT=Path('/workspace/math500')
PATTERN=r'(?i)Answer\s*:\s*([^\n]+)'

def parsed(text):
    text=text.strip()
    if text.startswith('$') and text.endswith('$'):text=text[1:-1]
    return parse('$'+text+'$',extraction_config=[LatexExtractionConfig()],fallback_mode='first_match',
        extraction_mode='first_match',parsing_timeout=5)

def equal(gold,pred):
    if pred is None:return False
    return bool(verify(parsed(gold),parsed(pred),float_rounding=6,numeric_precision=15,strict=True,timeout_seconds=5))

def boxed(text):
    results=[]
    for match in re.finditer(r'\\boxed\s*\{',text):
        start=match.end();depth=1;end=start
        while end<len(text) and depth:
            if text[end]=='{':depth+=1
            elif text[end]=='}':depth-=1
            end+=1
        if depth==0:results.append((match.start(),text[start:end-1]))
    return results

def extract(text,mode):
    matches=list(re.finditer(PATTERN,text))
    if matches:
        match=matches[0] if mode=='reference_first' else matches[-1]
        return match.group(1).strip(),'answer_label'
    if mode=='final_or_boxed':
        candidates=boxed(text)
        if candidates and candidates[-1][0]>=max(0,len(text)-1200):return candidates[-1][1].strip(),'final_boxed_fallback'
    return None,'missing'

def qualify():
    fixtures=[
      ('fraction',r'\frac{7}{12}',r'\frac{14}{24}',True),
      ('radical',r'\sqrt{72}',r'6\sqrt{2}',True),
      ('decimal',r'\frac{1}{8}','0.125',True),
      ('sign_parenthesis',r'(-3)^2','9',True),
      ('pi_fraction',r'\frac{5\pi}{6}',r'5\pi/6',True),
      ('ordered_tuple','(2,5)','(2,5)',True),
      ('unordered_set',r'\{2,5\}',r'\{5,2\}',True),
      ('polynomial',r'x^2+2x+1',r'(x+1)^2',True),
      ('power','10^3','1000',True),
      ('negative_fraction',r'-\frac{3}{7}',r'\frac{-3}{7}',True),
      ('text',r'\text{Nadia}',r'\text{Nadia}',True),
      ('zero','0','0.0',True),
      ('wrong_fraction',r'\frac{7}{12}',r'\frac{5}{12}',False),
      ('wrong_radical',r'\sqrt{2}','2',False),
      ('wrong_sign',r'(-3)^2','-9',False),
      ('wrong_decimal','2.5','2.05',False),
      ('strict_inequality','x<3',r'x\leq3',False),
      ('different_set',r'\{2,5\}',r'\{2,6\}',False),
      ('tuple_order','(2,5)','(5,2)',False),
      ('different_function',r'\frac{1}{x-1}',r'\frac{1}{x}',False),
      ('units_not_implicit',r'\frac{\pi}{2}','90',False),
      ('different_trig',r'\sin(x)',r'\cos(x)',False),
      ('wrong_approximation',r'\frac{1}{3}','0.3330',False),
      ('wrong_zero','0','1',False)]
    rows=[]
    for name,a,b,expected in fixtures:
        actual=equal(a,b);rows.append({'name':name,'gold':a,'prediction':b,'expected':expected,'actual':actual,'passed':actual==expected})
    parser_fixtures=[
      ('first_label','Answer: 7\nRevision\nAnswer: 9','reference_first','7'),
      ('last_label','Answer: 7\nRevision\nAnswer: 9','final_or_boxed','9'),
      ('boxed',r'Therefore $\boxed{\frac{7}{12}}$.','final_or_boxed',r'\frac{7}{12}'),
      ('no_guess','Compute 7 and 12. No conclusion.','final_or_boxed',None)]
    parser_rows=[{'name':n,'expected':e,'actual':extract(t,m)[0],'passed':extract(t,m)[0]==e} for n,t,m,e in parser_fixtures]
    gold=json.loads((ROOT/'private-gold.json').read_text())
    identities=[]
    for row in gold:
        identities.append({'id':row['id'],'passed':equal(row['Answer'],row['Answer']),'parsed_repr':repr(parsed(row['Answer']))})
    report={'math_verify':importlib.metadata.version('math-verify'),'fixtures':rows,'parser_fixtures':parser_rows,
        'gold_identity':identities,'synthetic_passed':sum(r['passed'] for r in rows),
        'synthetic_total':len(rows),'gold_identity_passed':sum(r['passed'] for r in identities),
        'qualification_passed':all(r['passed'] for r in rows+parser_rows+identities)}
    (ROOT/'qualification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['fixtures','gold_identity','parser_fixtures']}),flush=True)

def score(path,name):
    qualification=json.loads((ROOT/'qualification.json').read_text())
    assert qualification['qualification_passed'],'Qualify all mathematical and extraction controls before scoring.'
    gold={r['id']:r for r in json.loads((ROOT/'private-gold.json').read_text())}
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    assert len(rows)==500 and {r['id'] for r in rows}==set(gold)
    result=[]
    for row in rows:
        record={'id':row['id'],'truncated':row['truncated'],'new_tokens':row['new_tokens']}
        for mode in ['reference_first','final_or_boxed']:
            answer,method=extract(row['completion'],mode)
            record[mode]={'answer':answer,'extraction':method,'correct':equal(gold[row['id']]['Answer'],answer)}
        result.append(record)
        if len(result)%25==0:print('SCORED',name,len(result),500,flush=True)
    (ROOT/'scores').mkdir(exist_ok=True)
    (ROOT/'scores'/(name+'.json')).write_text(json.dumps(result,indent=2))
    summary={'name':name,'n':500,'truncated':sum(r['truncated'] for r in rows),
        'scores':{mode:sum(r[mode]['correct'] for r in result) for mode in ['reference_first','final_or_boxed']}}
    (ROOT/'scores'/(name+'-summary.json')).write_text(json.dumps(summary,indent=2))
    print('SCORING_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['qualify','score']);p.add_argument('--input');p.add_argument('--name',default='baseline');a=p.parse_args()
    if a.mode=='qualify':qualify()
    else:score(a.input,a.name)
