"""Reserve final-answer space by ending an unfinished delimited reasoning span."""
import hashlib,json
RESERVE=1024

def open_reasoning(prefix_text):
    value=prefix_text.lstrip()
    return value.startswith('<think>') and '</think>' not in value

def plan(tokenizer,generated_ids,total_budget,reserve=RESERVE):
    assert total_budget>reserve>0
    cut=total_budget-reserve
    if len(generated_ids)<cut:return None
    prefix=list(generated_ids[:cut])
    text=tokenizer.decode(prefix,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    if not open_reasoning(text):return None
    forced=tokenizer.encode('</think>\n',add_special_tokens=False)
    assert 0<len(forced)<reserve
    return {'prefix_ids':prefix,'forced_ids':forced,'max_final_tokens':reserve-len(forced),
            'total_budget':total_budget,'prefix_sha256':hashlib.sha256(json.dumps(prefix).encode()).hexdigest()}

def qualify():
    fixtures=[('open','<think>working',True),('closed','<think>working</think>Answer: 9',False),
      ('direct','Answer: 9',False),('literal','```python\nprint("<think>")\n```',False),
      ('blank','',False),('leading_space',' \n<think>working',True),
      ('already_final','<think></think>\n```python\ndef f():return 9\n```',False)]
    rows=[{'name':n,'passed':open_reasoning(t)==expected} for n,t,expected in fixtures]
    class Tokenizer:
        def decode(self,ids,**kwargs):return ''.join(chr(x) for x in ids)
        def encode(self,text,**kwargs):return [ord(x) for x in text]
    tok=Tokenizer();prefix='<think>'+('x'*4000);ids=tok.encode(prefix)
    p=plan(tok,ids,4096)
    rows.append({'name':'same_total_budget','passed':len(p['prefix_ids'])+len(p['forced_ids'])+p['max_final_tokens']==4096})
    rows.append({'name':'not_based_on_later_truncation','passed':plan(tok,tok.encode(prefix+'</think>done'),4096)==p})
    rows.append({'name':'short_finished_response_unchanged','passed':plan(tok,tok.encode('<think>a</think>done'),4096) is None})
    rows.append({'name':'already_closed_at_cut_unchanged','passed':plan(tok,tok.encode('<think>a</think>'+('x'*4000)),4096) is None})
    rows.append({'name':'both_budgets_accounted','passed':all((b-RESERVE)+len(tok.encode('</think>\n'))+(RESERVE-len(tok.encode('</think>\n')))==b for b in [4096,8192])})
    return rows
