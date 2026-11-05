"""Qwen-native budget calibration, with no question or correctness inputs."""
from adapter import ContextTokenizer
from budget_policy import plan

def effort(output_budget):
    return 'xhigh'


def native_plan(tokenizer,ids,total_budget,opened_in_prompt):
    result=plan(ContextTokenizer(tokenizer,opened_in_prompt),ids,total_budget)
    if result is None:return None
    forced=tokenizer.encode('</think>\n\n',add_special_tokens=False)
    assert 0<len(forced)<1024
    return {**result,'forced_ids':forced,'max_final_tokens':1024-len(forced)}

def qualify():
    assert [effort(b) for b in [2048,4096,4097,8192,8193]]==['xhigh']*5
    class T:
        def encode(self,text,**kwargs):return list(map(ord,text))
        def decode(self,ids,**kwargs):return ''.join(map(chr,ids))
    tok=T()
    for b in [4096,8192]:
        q=native_plan(tok,tok.encode('x'*(b-1024)),b,True)
        assert len(q['prefix_ids'])+len(q['forced_ids'])+q['max_final_tokens']==b
        assert tok.decode(q['forced_ids'])=='</think>\n\n'
        assert native_plan(tok,tok.encode('work</think>done'+'x'*b)[:b-1024],b,True) is None
    return True
