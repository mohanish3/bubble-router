"""Offline, bounded Laya inference in a short-lived two-thread CPU process."""
import json
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')

def main():
    root = Path(sys.argv[1])
    task = json.load(sys.stdin)
    kind, state = task['kind'], task['state']
    if kind not in {'weak_answers','failed_generation'} or not isinstance(state,str) or len(state)>1200:
        raise ValueError('Invalid decision input')
    paths = json.loads((root/'config/model-paths.json').read_text())
    key = 'laya-answer-quality' if kind=='weak_answers' else 'laya-failure'
    path = paths.get(key,paths['convaiinnovations/laya'])['path']
    import laya
    import torch
    torch.set_num_threads(2)
    agent = laya.load(path,device='cpu')
    if len(agent.tok.encode(state,add_special_tokens=False))>300:
        print(json.dumps({'available':False})); return
    if kind=='weak_answers':
        criteria = {'keep':'The answer reasonably addresses the request; brevity, style and uncertainty are acceptable.',
            'escalate':'The answer is clearly irrelevant, contradictory, repetitive or misses the main requested task. A stronger model should try.',
            'review':'Not enough evidence, ambiguous weakness, policy refusal, or missing context. Keep the answer.'}
    else:
        criteria = {'retry':'Temporary connection or timeout failure; a manual retry could work.',
            'simplify':'Too many requested actions or an overloaded generation prompt; simplify first.',
            'settings':'Insufficient memory, invalid dimensions or unsupported settings; adjust settings first.',
            'review':'Unknown failure or uncertain diagnosis; inspect details manually.'}
    result = agent.predict(state,{'decision':{'type':'choice','instructions':'Choose the next action conservatively from the evidence.','criteria':criteria}})
    usage = result.get('usage',{})
    scores = result.get('answers',{}).get('decision',{}).get('probabilities',{})
    available = not usage.get('truncated') and not usage.get('truncated_questions') and set(scores)==set(criteria)
    print(json.dumps({'available':available,'scores':scores}))

if __name__=='__main__': main()
