"""Sparse, advisory CPU decisions. Never execute tools or retry media jobs."""
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import time


class SoftDecisions:
    def __init__(self, root, policy=None):
        self.root = Path(root)
        self.policy = policy or {}
        self.last = {}
        self.lock = asyncio.Lock()

    def selected(self, kind, key):
        rule = self.policy.get(kind, {})
        if not rule.get('enabled', False):
            return False
        rate = max(0., min(1., float(rule.get('sample_rate', .05))))
        fraction = int(hashlib.sha256((kind+key).encode()).hexdigest()[:8], 16)/2**32
        now = time.monotonic()
        if fraction >= rate or now-self.last.get(kind, float('-inf')) < max(0, float(rule.get('cooldown_seconds',600))):
            return False
        if self.lock.locked():
            return False
        self.last[kind] = now
        return True

    def _predict(self, kind, state):
        worker = Path(__file__).with_name('soft_decision_worker.py')
        interpreter = self.root/'repos/bubble-router/.venv/bin/python'
        done = subprocess.run([str(interpreter),str(worker),str(self.root)],
            input=json.dumps({'kind':kind,'state':state}),capture_output=True,text=True,timeout=15,check=True)
        result = json.loads(done.stdout)
        scores = result.get('scores', {})
        allowed = {'keep','escalate','review'} if kind=='weak_answers' else {'retry','simplify','settings','review'}
        if result.get('available') is not True or set(scores)!=allowed:
            return {'action':'review','available':False}
        import math
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):
            return {'action':'review','available':False}
        order = sorted(scores,key=scores.get,reverse=True)
        threshold = max(.95, float(self.policy.get(kind,{}).get('threshold',.97)))
        action = order[0] if scores[order[0]]>=threshold and scores[order[0]]-scores[order[1]]>=.5 else 'review'
        return {'action':action,'available':True,'scores':scores,'confidence_calibrated':False}

    async def decide(self, kind, state):
        if self.lock.locked():
            return {'action':'review','available':False}
        async with self.lock:
            try:
                return await asyncio.to_thread(self._predict,kind,state)
            except (OSError,ValueError,KeyError,subprocess.SubprocessError):
                return {'action':'review','available':False}

    async def weak_answer(self, user, result):
        try:
            message = result['choices'][0]['message']
            answer = message.get('content')
            if message.get('tool_calls') or not isinstance(answer,str) or not 20<=len(answer)<=12000:
                return {'action':'keep','available':False}
            # A policy refusal is not evidence of answer weakness.
            if any(s in answer.lower() for s in ("i can't help",'i cannot help','not able to assist','unable to assist')):
                return {'action':'keep','available':False}
            state = 'User request: '+user[:300]+'\nAnswer: '+answer[:650]
            return await self.decide('weak_answers',state)
        except (KeyError,IndexError,TypeError):
            return {'action':'review','available':False}

    async def failed_generation(self, job, exc):
        if job.resource not in {'qwen-image','minimax-h3','qwen-audio','ace-music'}:
            return None
        if not self.selected('failed_generation',job.id):
            return None
        # Pass a generic symptom, never raw backend exceptions or prompts.
        text = str(exc).lower()
        symptom = ('memory admission or out of memory' if any(s in text for s in ('memory','allocation','oom')) else
                   'timeout or temporary connection failure' if any(s in text for s in ('timeout','timed out','connection')) else
                   'invalid input or unsupported settings' if isinstance(exc,ValueError) else 'unknown backend failure')
        decision = await self.decide('failed_generation',f'Resource: {job.resource}. Failure: {symptom}.')
        explanations = {'retry':'A temporary failure may be worth one manual retry.',
            'simplify':'Try simplifying the generation prompt before retrying.',
            'settings':'Check generation settings and available memory before retrying.',
            'review':'Check the failure details before retrying.'}
        return {**decision,'message':explanations[decision['action']],'automatic_retry':False}
