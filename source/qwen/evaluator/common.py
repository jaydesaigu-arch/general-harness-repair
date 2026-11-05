import json,re
SEED=20261105
ROOT='/workspace/he164'
EOS=[128001,128008,128009]

def content(prompt):
    return 'Complete the following Python function. Return the complete Python code, including necessary imports, in one Python code block. Preserve the function name and signature.\n\n```python\n'+prompt.strip()+'\n```'

def messages(row,feedback=None):
    first={'role':'user','content':content(row['prompt'])}
    if feedback is None:return [first]
    detail=feedback['feedback']
    return [first,{'role':'assistant','content':feedback['completion']},
        {'role':'user','content':'The solution failed the original HumanEval tests. '+detail+
         '\nRepair the implementation. Check the full specification and boundary cases. Return the complete corrected Python code in one code block, with no explanation.'}]

def simple_extract(text):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S)
    return (blocks[0] if blocks else text).strip()

def jsonl(path):
    return [json.loads(x) for x in open(path) if x.strip()]
