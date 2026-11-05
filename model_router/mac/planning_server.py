"""Authenticated, single-request text-only MLX service for draft planning.

No tool declarations, tool continuations, attachments, or client template
overrides are accepted. The Router buffers this endpoint and can retry Swift.
"""
from __future__ import annotations
import asyncio
import json
import math
import os
from pathlib import Path
import secrets
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

MAX_BODY=2*1024*1024
MAX_TOKENS=2048
CONTEXT=32768

def validate_payload(payload):
    if not isinstance(payload,dict):raise ValueError('JSON object required')
    messages=payload.get('messages')
    if not isinstance(messages,list) or not messages or len(messages)>128:raise ValueError('Invalid messages')
    if payload.get('tools') or payload.get('tool_choice') not in (None,'none'):
        raise ValueError('Planning backend accepts draft text only')
    clean=[]
    for m in messages:
        if not isinstance(m,dict) or m.get('role') not in ('system','developer','user','assistant') or m.get('tool_calls'):
            raise ValueError('Tool continuations are outside planning scope')
        content=m.get('content')
        if isinstance(content,list):
            if any(not isinstance(p,dict) or p.get('type')!='text' or not isinstance(p.get('text'),str) for p in content):
                raise ValueError('Planning backend is text only')
            content='\n'.join(p['text'] for p in content)
        if not isinstance(content,str):raise ValueError('Text content required')
        clean.append({'role':m['role'],'content':content})
    if clean[-1]['role']!='user':raise ValueError('Fresh user request required')
    cap=payload.get('max_tokens',payload.get('max_completion_tokens',MAX_TOKENS))
    if isinstance(cap,bool) or not isinstance(cap,int) or cap<1:raise ValueError('Invalid output token limit')
    return clean,min(cap,MAX_TOKENS)

def final_text(raw,tokenizer):
    start=getattr(tokenizer,'think_start',None) or '<|channel>thought\n'
    end=getattr(tokenizer,'think_end',None) or '<channel|>'
    if end in raw:raw=raw.split(end,1)[1]
    elif start in raw:raise ValueError('No final answer')
    if any(marker in raw for marker in ('<|tool_call>','<tool_call>','<|tool_response>','<|channel>')):
        raise ValueError('Unexpected native control output')
    for token in ('<turn|>','<eos>'):raw=raw.replace(token,'')
    if not raw.strip():raise ValueError('Empty final answer')
    return raw.strip()

def generate(model,tokenizer,messages,cap):
    import psutil
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                                         enable_thinking=False,preserve_thinking=False)
    tokens=tokenizer.encode(prompt,add_special_tokens=False)
    if len(tokens)+cap>CONTEXT:raise ValueError('Context limit exceeded; no truncation')
    chunks=[];last=None;started=time.monotonic()
    def guard(*args):
        if psutil.virtual_memory().available/2**30<5:raise RuntimeError('Preserved memory reserve reached')
        if time.monotonic()-started>120:raise RuntimeError('Planning generation time limit')
    guard();mx.reset_peak_memory()
    try:
        for chunk in stream_generate(model,tokenizer,tokens,max_tokens=cap,sampler=make_sampler(temp=0),
                prefill_step_size=256,prompt_progress_callback=guard):
            guard();chunks.append(chunk.text);last=chunk
        if last is None:raise ValueError('No generation output')
        if last.finish_reason=='length':raise ValueError('Planning output exhausted token budget')
        return {'id':'chatcmpl-planning-'+secrets.token_hex(8),'object':'chat.completion','created':int(time.time()),
            'model':'thinkingcap','choices':[{'index':0,'message':{'role':'assistant','content':final_text(''.join(chunks),tokenizer)},
                'finish_reason':'stop'}],
            'usage':{'prompt_tokens':last.prompt_tokens,'completion_tokens':last.generation_tokens,
                     'total_tokens':last.prompt_tokens+last.generation_tokens}}
    finally:mx.clear_cache()

def create_app(*,loader=None,generator=None,api_key=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.key=api_key or os.environ.get('BUBBLE_PLANNING_KEY','')
        if not app.state.key:raise RuntimeError('Planning backend credential is required')
        app.state.lock=asyncio.Lock()
        if loader is None:
            from mlx_lm import load
            path=Path(os.environ['BUBBLE_PLANNING_MODEL'])
            app.state.model,app.state.tokenizer=await asyncio.to_thread(load,str(path),trust_remote_code=False)
        else:app.state.model,app.state.tokenizer=loader()
        yield
    app=FastAPI(title='Private planning backend',lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)
    @app.middleware('http')
    async def auth(request,call_next):
        value=request.headers.get('authorization','')
        if not secrets.compare_digest(value,'Bearer '+app.state.key):return JSONResponse({'error':'authentication required'},status_code=401)
        return await call_next(request)
    @app.get('/health')
    async def health():return {'ok':True,'model':'thinkingcap','scope':'draft-planning','thinking':False}
    @app.post('/v1/chat/completions')
    async def chat(request:Request):
        raw=bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw)>MAX_BODY:raise HTTPException(413,'Planning request too large')
        try:messages,cap=validate_payload(json.loads(raw))
        except (ValueError,TypeError):raise HTTPException(422,'Invalid draft-planning payload') from None
        async with app.state.lock:
            # Shield the owned synchronous generation so a client disconnect
            # cannot release the single-model lock while Metal is still active.
            task=asyncio.create_task(asyncio.to_thread(generator or generate,app.state.model,app.state.tokenizer,messages,cap))
            try:return await asyncio.shield(task)
            except asyncio.CancelledError:
                try:await task
                finally:raise
            except ValueError:raise HTTPException(502,'Planning output invalid; use stronger model') from None
            except Exception:raise HTTPException(503,'Planning backend unavailable') from None
    return app

app=create_app()
