"""Conservative, independently fine-tuned eligibility for draft-only planning."""
from __future__ import annotations
import json
import math
import re
import threading
from pathlib import Path

PLAN_RE=re.compile(r'\b(?:plan(?:ning)?|agenda|checklist|routine|outline|sequence|session|study schedule|draft.{0,20}schedule)\b',re.I)
RISK_RE=re.compile(r'\b(?:code|coding|rust|python|typescript|javascript|database|deployment|deploy|server|backend|'
    r'infrastructure|kubernetes|production|rollback|api|sql|architecture|migration|algorithm|authentication|'
    r'optimi[sz]e|optimal|trade.?offs?|conflicting|competing constraints|uncertain demand|uncertain outcomes|'
    r'legal|lawsuit|visa|tax|invest(?:ment|ing)?|retirement|mortgage|loan|debt|stock|trading|'
    r'treatment|diagnos\w*|therap\w*|dosage|insulin|medication adjustment|quit my job|job offers?|'
    r'private results?|previous tool|earlier tool|attached|attachment|image|audio|video|screenshots?)\b',re.I)
ACTION_RE=re.compile(r'\b(?:send|book|purchase|buy|execute|run|deploy|install|delete|remove|cancel|'
    r'retry|resume|search|browse|fetch|look up|update|save|write to|create.{0,24}(?:calendar|reminder)|'
    r'schedule.{0,24}(?:meeting|event|reminder)|set.{0,12}reminder)\b',re.I)
NEGATED_ACTION_RE=re.compile(r'\b(?:do not|don[’\x27]t|never|without)\s+(?:actually\s+)?'
    r'(?:send|book|purchase|buy|execute|run|schedule|save|act|use tools|take action)(?:\s+(?:or|and)\s+'
    r'(?:send|book|purchase|buy|execute|run|schedule|save|act))*\b',re.I)

def planning_candidate(text):return bool(PLAN_RE.search(text or ''))

def planning_issue(messages,text,visual=False):
    """A model confidence score can never override these scope boundaries."""
    if visual:return 'visual'
    if not text or not planning_candidate(text):return 'not-planning'
    if re.search(r'\b(?:explain|describe|define|meaning).{0,70}\b(?:quote|agenda|plan|checklist)\b'
                 r'|\b(?:do not|don[’\x27]t)\s+(?:write|make|create)\s+one\b',text,re.I):return 'discussion-only'
    if len(text)>1000:return 'long-request'
    if not messages or messages[-1].get('role')!='user':return 'continuation'
    if any(m.get('role')=='tool' or m.get('tool_calls') for m in messages[-8:]):return 'recent-tool-history'
    if '```' in text:return 'code'
    if RISK_RE.search(text):return 'complex-or-consequential'
    clean=NEGATED_ACTION_RE.sub('',text)
    clean=re.sub(r'\bbook\s+(?:swap|club|discussion|shelf)\b','reading activity',clean,flags=re.I)
    if ACTION_RE.search(clean):return 'action'
    return None

class PlanningClassifier:
    """CPU Laya checkpoint separate from route-v3 and security-v4."""
    def __init__(self,path,*,loader=None):
        self.path=Path(path);self._loader=loader;self._agent=None;self._lock=threading.Lock()
        policy=json.loads((self.path/'planning_policy.json').read_text())
        self.question=policy['question'];self.threshold=float(policy['confidence_threshold'])
        if not math.isfinite(self.threshold) or not .75<=self.threshold<=1:raise ValueError('Invalid planning threshold')
        if set(self.question)!={'simple_planning'}:raise ValueError('Invalid planning question')

    def predict(self,text):
        if not isinstance(text,str) or not text or len(text)>1000:return {'eligible':False,'reason':'input-scope'}
        with self._lock:
            if self._agent is None:
                if self._loader is None:
                    from laya import load
                    self._agent=load(str(self.path),device='cpu')
                else:self._agent=self._loader(str(self.path))
            result=self._agent.predict({'source':'user message','content':text},self.question)
        usage=result.get('usage',{})
        if usage.get('truncated') or usage.get('truncated_questions'):raise ValueError('Incomplete planning prediction')
        answer=result['answers']['simple_planning'];probs=answer['probabilities']
        a,b=float(probs['A']),float(probs['B'])
        if not all(math.isfinite(p) and 0<=p<=1 for p in (a,b)) or abs(a+b-1)>.02:
            raise ValueError('Invalid planning scores')
        return {'eligible':a>=self.threshold and a>b,'score':round(a,6),'threshold':self.threshold}
