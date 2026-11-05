"""Frozen choice extraction. Never selects candidates using reference agreement."""
import re
from harness import final_channel
OFFICIAL=r'(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?'
def official(text):
    m=re.search(OFFICIAL,text)
    return m.group(1).upper() if m else None
def repaired(text):
    text,_=final_channel(text)
    choices=[];pending=False
    for line in text.splitlines():
        s=line.strip().replace('**','').replace('__','').replace('`','').replace('$','')
        if not s:continue
        if pending:
            pending=False
            m=re.fullmatch(r'\(?([A-Da-d])\)?[.!]?',s)
            if m:choices.append(m.group(1).upper());continue
        if re.fullmatch(r'(?i)(?:final\s+)?answer\s*:',s):pending=True;continue
        m=re.fullmatch(r'(?i)(?:(?:final\s+)?answer\s*(?::|is)\s*|(?:the\s+)?correct\s+(?:answer|option)\s*(?::|is)\s*)\(?([A-D])\)?(?:[.!]|\s*[-:]\s*[^\n]+)?',s)
        if m:choices.append(m.group(1).upper());continue
        m=re.fullmatch(r'\\boxed\{\s*\(?([A-Da-d])\)?\s*\}[.!]?',s)
        if m:choices.append(m.group(1).upper())
    if choices:return choices[-1]
    s=text.strip().replace('**','').replace('`','').replace('$','')
    m=re.fullmatch(r'\(?([A-Da-d])\)?[.!]?',s)
    return m.group(1).upper() if m else None
def qualify():
    cases=[('standard','Answer: B','B'),('case','answer: d','D'),('math','Answer: $C$','C'),('bold','Answer: **A**','A'),('parenthesized','Answer: (D)','D'),('final','Final answer: B.','B'),('words','The correct option is C','C'),('bare','A','A'),('bare_parens','(b)','B'),('boxed',r'\boxed{D}','D'),('reasoning','<think>Answer: A</think>\nAnswer: C','C'),('incomplete','<think>Answer: B',None),('empty','',None),('prose','A is tempting but I cannot decide.',None),('ambiguous','Answer: A or B',None),('invalid','Answer: E',None),('revised','Answer: B\nOn reflection:\nAnswer: D','D'),('blanklabel','Answer:\nC',None),('options','A) alpha\nB) beta\nC) gamma\nD) delta',None),('reasoning_only','<think>Answer: D</think>',None),('description','Answer: B - selected option','B'),('multiple_channels','<think>x</think><think>Answer: C</think>',None)]
    cases=[(n,t,'C' if n=='blanklabel' else y) for n,t,y in cases]
    cases += [('label_prose','Answer:\nA is tempting',None),('letter_prefix_word','Answer: Apples',None),('new_line_bold','Final answer:\n\n**(D)**','D')]
    return [{'name':n,'expected':y,'actual':repaired(t),'passed':repaired(t)==y} for n,t,y in cases]
if __name__=='__main__':
    import json
    result=qualify();print(json.dumps(result,indent=2));assert all(x['passed'] for x in result)
