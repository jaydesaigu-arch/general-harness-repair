"""Adapt unchanged structural policy to a template-opened native Qwen channel."""
from budget_policy import plan
from harness import final_channel

class ContextTokenizer:
    def __init__(self,tokenizer,opened_in_prompt):
        self.tokenizer=tokenizer;self.opened_in_prompt=opened_in_prompt
    def decode(self,ids,**kwargs):
        # Qwen marks thinking delimiters as special tokens; retain them literally.
        text=self.tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        return ('<think>\n' if self.opened_in_prompt else '')+text
    def encode(self,text,**kwargs):
        return self.tokenizer.encode(text,add_special_tokens=False)

def policy(tokenizer,ids,budget,opened_in_prompt):
    return plan(ContextTokenizer(tokenizer,opened_in_prompt),ids,budget)

def views(tokenizer,ids,opened_in_prompt):
    sampled=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    logical=('<think>\n' if opened_in_prompt else '')+sampled
    final,method=final_channel(logical)
    return sampled,logical,final,method

def qualify():
    class T:
        def decode(self,ids,**kwargs):return ''.join(map(chr,ids))
        def encode(self,text,**kwargs):return list(map(ord,text))
    t=T();cut=4096-1024
    cases=[('prompt_open_no_closer','x'*cut,True,True),
           ('prompt_open_closed','work</think>Answer: 7'+'x'*cut,True,False),
           ('generated_open','<think>'+'x'*cut,False,True),
           ('direct','Answer: 7'+'x'*cut,False,False),
           ('code_literal','```python\nprint("<think>")\n```'+'x'*cut,False,False)]
    out=[{'name':n,'passed':bool(policy(t,t.encode(s)[:cut],4096,opened))==expected} for n,s,opened,expected in cases]
    out += [{'name':'open_prompt_final_view','passed':views(t,t.encode('work</think>\nAnswer: 7'),True)[2]=='Answer: 7'},
            {'name':'unclosed_prompt_empty_final','passed':views(t,t.encode('work'),True)[2]==''},
            {'name':'not_double_counted_output_tokens','passed':len(t.encode('work'))==4 and views(t,t.encode('work'),True)[0]=='work'}]
    return out
