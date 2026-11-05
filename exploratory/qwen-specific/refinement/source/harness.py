"""Reasoning-channel extraction independent of models, tasks and references."""
def final_channel(text):
    stripped=text.lstrip()
    if not stripped.startswith('<think>'):return text,'no_leading_reasoning_channel'
    end=stripped.find('</think>',len('<think>'))
    if end<0:return '', 'incomplete_reasoning_no_final'
    final=stripped[end+len('</think>'):].lstrip()
    if final.startswith('<think>'):return '', 'ambiguous_multiple_reasoning_channels'
    return final,'after_complete_reasoning_channel'

def qualify():
    fixtures=[
      ('direct','Answer: 23','Answer: 23'),
      ('reasoning','<think>Answer: 17</think>\nAnswer: 23','Answer: 23'),
      ('empty_reasoning','<think></think>\nAnswer: 23','Answer: 23'),
      ('incomplete','<think>Answer: 23',''),
      ('blank_final','<think>Answer: 23</think>',''),
      ('code_literal','```python\ndef f():\n    return "<think>literal</think>"\n```','```python\ndef f():\n    return "<think>literal</think>"\n```'),
      ('code_after_reasoning','<think>```python\ndef f(): return 0\n```</think>\n```python\ndef f(): return 1\n```','```python\ndef f(): return 1\n```'),
      ('whitespace',' \n<think>draft</think>\n Final value: 23','Final value: 23'),
      ('ambiguous','<think>a</think><think>b</think>Answer: 23','')]
    return [{'name':n,'passed':final_channel(t)[0]==expected} for n,t,expected in fixtures]
