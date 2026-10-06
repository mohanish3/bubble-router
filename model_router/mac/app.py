"""Loopback Mac control plane. Debian entry point remains model_router.main."""
from __future__ import annotations
import asyncio
import copy
import dataclasses
import json
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse, Response, FileResponse

from .resources import BackendConfig, ResourceManager
from .scheduler import Scheduler, QueueFullError, SchedulerDrainingError
from .security import SecurityScanner, SecurityPolicy, SecurityBlocked, SecurityUnavailable, SecurityInputTooLarge
from .media import MediaBackend

ROOT = Path(os.environ.get('AI_STACK_ROOT', str(Path.home() / 'AI')))

class SecurityRejected(Exception):
    def __init__(self, report):
        super().__init__('request blocked by security policy')
        self.report = report


ROUTE_FOLLOW_UP_CHARS = 80


def latest_user_text(messages):
    """Latest user text; a short follow-up is routed with the user turn before it."""
    texts = []
    for message in reversed(messages or []):
        if message.get('role') != 'user': continue
        content = message.get('content')
        if isinstance(content, list):
            content = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        if not isinstance(content, str) or not content.strip(): continue
        texts.append(content)
        if len(texts) == 2 or len(content) >= ROUTE_FOLLOW_UP_CHARS: break
    return '\n'.join(reversed(texts))


def plain(value):
    if hasattr(value, 'to_dict'): return value.to_dict()
    if dataclasses.is_dataclass(value): return dataclasses.asdict(value)
    if hasattr(value, 'model_dump'): return value.model_dump()
    return value

def safe_asset(value: str, root: Path = ROOT) -> Path:
    path = Path(value.removeprefix('file://')).expanduser().resolve(strict=True)
    if not any(path.is_relative_to((root / part).resolve()) for part in ('test-assets', 'outputs', 'state/uploads')):
        raise ValueError('Media input must be in approved AI asset directories')
    if not path.is_file() or path.stat().st_size > 100 * 1024**2:
        raise ValueError('Invalid or oversized media asset')
    return path

def prepare_chat(payload, root=ROOT):
    allowed = {'model','messages','max_tokens','max_completion_tokens','stream','stream_options','temperature','top_p','seed','tools','tool_choice','parallel_tool_calls','response_format','stop','frequency_penalty','presence_penalty','context_length'}
    result = {key:copy.deepcopy(value) for key,value in payload.items() if key in allowed}
    result['model'] = 'qwen'
    if not isinstance(result.get('messages'), list) or not result['messages']:
        raise ValueError('messages must be a nonempty list')
    max_tokens = int(result.get('max_tokens', result.pop('max_completion_tokens', 2048)))
    if not 1 <= max_tokens <= 8192: raise ValueError('max_tokens must be 1..8192')
    result['max_tokens'] = max_tokens
    # A conservative text admission bound; actual KV grows with each request.
    context = int(result.pop('context_length', 32768))
    if not 1024 <= context <= 131072: raise ValueError('context_length must be 1024..131072')
    text_chars = 0
    for message in result['messages']:
        content=message.get('content','')
        text_chars += len(content) if isinstance(content,str) else sum(len(item.get('text','')) for item in content or [] if isinstance(item,dict))
    if text_chars > context * 3:
        raise ValueError('Text exceeds requested context budget')
    visual = False
    for message in result['messages']:
        message.pop('source', None); message.pop('provenance', None)
        if isinstance(message.get('content'), list):
            for item in message['content']:
                kind = item.get('type')
                if kind in ('image_url', 'video_url', 'input_video'):
                    visual = True
                    key = 'image_url' if kind == 'image_url' else 'video_url'
                    value = item.get(key)
                    url = value.get('url') if isinstance(value, dict) else value
                    if not isinstance(url, str): raise ValueError('Invalid media URL')
                    if url.startswith('data:image/') and kind == 'image_url':
                        if len(url) > 20 * 1024**2: raise ValueError('Image is too large')
                    else:
                        url = str(safe_asset(url, root))
                        if isinstance(value, dict): value['url'] = url
                        else: item[key] = url
    return result, visual

def chat_incremental_gib(payload):
    # The pinned Qwen config has 16 full-attention layers, 4 KV heads and
    # 256-dimensional BF16 keys/values: 65536 bytes per cached token.
    # UTF-8 byte count bounds text token growth without reserving 128K on
    # short requests. Visual inputs receive a separate conservative allowance.
    messages = payload.get('messages', [])
    text = json.dumps(messages, ensure_ascii=False) + json.dumps(payload.get('tools', []), ensure_ascii=False)
    tokens = len(text.encode('utf-8')) + int(payload.get('max_tokens', 2048))
    visuals = sum(1 for message in messages if isinstance(message.get('content'), list)
                  for item in message['content'] if item.get('type') in ('image_url','video_url','input_video'))
    tokens = min(131072, tokens + visuals * 16384)
    return 4.0 + tokens * 65536 / 1024**3

def create_app(root: Path = ROOT, *, injected=None):
    @asynccontextmanager
    async def lifespan(app):
        if injected:
            app.state.config, app.state.manager, app.state.scheduler, app.state.scanner, app.state.media = injected
        else:
            config = json.loads((root/'config/control-plane.json').read_text())
            backends = {name: BackendConfig(**profile) for name, profile in config['backends'].items()}
            app.state.config = config
            app.state.manager = ResourceManager(backends, reserve_gib=config.get('reserve_gib', 12), ownership_path=root/'state/owned-processes.json')
            app.state.scheduler = Scheduler(root/'state/jobs.sqlite3', app.state.manager)
            paths = json.loads((root/'config/model-paths.json').read_text())
            app.state.scanner = SecurityScanner(paths['convaiinnovations/laya']['path'], config.get('security_policy', {'max_input_chars':393216}))
            app.state.media = MediaBackend(app.state.manager, paths, root)
        app.state.metrics = {'requests': 0, 'backend_requests': 0, 'security_blocks': 0, 'security_active':0, 'started': time.time()}
        app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=5), trust_env=False)
        await app.state.scheduler.start()
        if hasattr(app.state.manager,'recover_owned_orphans'): await app.state.manager.recover_owned_orphans()
        if app.state.config.get('start_drained'): await app.state.scheduler.drain()
        if hasattr(app.state.scanner, 'load'): await asyncio.to_thread(app.state.scanner.load)
        async def warm_core():
            try:
                await app.state.manager.restore_core()
                app.state.metrics['core_warm']='ready'
            except Exception as exc:
                app.state.metrics['core_warm']='failed:'+type(exc).__name__
        warm = asyncio.create_task(warm_core()) if not injected and app.state.config.get('warm_core',True) else None
        yield
        if warm and not warm.done():
            warm.cancel()
            try: await warm
            except asyncio.CancelledError: pass
        await app.state.scheduler.close()
        for name in app.state.config['backends']:
            await app.state.manager.stop(name)
        await app.state.http.aclose()

    app = FastAPI(title='BubbleRouter Mac control plane', lifespan=lifespan)

    @app.exception_handler(SecurityRejected)
    async def security_rejected(request: Request, exc: SecurityRejected):
        return JSONResponse({'error': {'message': 'Request blocked by the local security policy (Laya).',
                                       'type': 'invalid_request_error', 'code': 'content_policy_violation'},
                             'security': exc.report}, status_code=400)

    @app.exception_handler(QueueFullError)
    async def queue_full(request, exc): return JSONResponse({'error':'queue is full'},status_code=429)

    @app.exception_handler(SchedulerDrainingError)
    async def queue_draining(request, exc): return JSONResponse({'error':'scheduler is draining'},status_code=503)

    @app.middleware('http')
    async def guard(request, call_next):
        if request.url.path != '/health':
            presented = request.headers.get('authorization', '')
            principal = None
            for field, role in (('api_key','inference'),('provider_key','openclaw'),('admin_key','admin')):
                token = app.state.config.get(field)
                if token and secrets.compare_digest(presented, 'Bearer '+token): principal=role
            if principal is None:
                return JSONResponse({'error': 'authentication required'}, status_code=401)
            request.state.principal=principal
            if request.url.path.startswith('/router/control') and principal!='admin':
                return JSONResponse({'error':'administrator credential required'},status_code=403)
            if request.url.path.startswith('/router/control') and request.client and request.client.host not in ('127.0.0.1', '::1', 'testclient'):
                return JSONResponse({'error': 'control API is loopback only'}, status_code=403)
        try: return await call_next(request)
        except ValueError as exc: return JSONResponse({'error': str(exc)}, status_code=422)

    def metadata(request):
        return {'authenticated':True,'principal':request.state.principal,
                'trusted_system':request.state.principal=='openclaw',
                **{key: request.headers.get('x-'+key, '')[:120] for key in ('agent-id', 'session-id', 'source')}}

    async def scan(messages, meta):
        app.state.metrics['requests'] += 1
        app.state.metrics['security_active']+=1
        def audit(report):
            # Scores and provenance are auditable; request bodies never enter logs.
            entry = {'time': time.time(), 'metadata': meta, 'security': plain(report)}
            path = root/'state/security-audit.jsonl'
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, 'a') as handle: handle.write(json.dumps(entry)+'\n')
        try:
            result = await app.state.scanner.scan_messages(messages, meta)
            audit(result[1])
            return result
        except SecurityBlocked as exc:
            audit(exc.report)
            app.state.metrics['security_blocks'] += 1
            # 400 + content_policy_violation, not 403: OpenAI-style clients such as
            # OpenClaw read 401/403 as bad credentials and disable the key.
            raise SecurityRejected(plain(exc.report))
        except SecurityInputTooLarge: raise HTTPException(413,'Security scanner input limit exceeded')
        except SecurityUnavailable: raise HTTPException(503,'Security scanner unavailable; inference rejected')
        finally: app.state.metrics['security_active']-=1

    def job_json(job):
        if hasattr(job, 'to_dict'): return job.to_dict()
        if hasattr(job, 'snapshot'): return {**job.snapshot(), 'result':plain(job.result)}
        return {key: plain(getattr(job, key, None)) for key in ('id','resource','priority','metadata','state','result','error')}

    async def wait_job(request, resource, priority, operation):
        job = await app.state.scheduler.submit(resource, priority, operation, metadata(request))
        try: return await asyncio.shield(job.future)
        except asyncio.CancelledError:
            await app.state.scheduler.cancel(job.id); raise
        except Exception:
            raise HTTPException(503, detail={'job_id': job.id, 'error': 'backend execution failed; inspect job status'})

    @app.get('/health')
    async def health():
        security = plain(app.state.scanner.status())
        return {'ok': bool(security.get('ready', security.get('healthy', False))), 'security': security, 'service': 'bubble-router-mac'}

    @app.get('/v1/models')
    async def models():
        return {'object':'list','data':[{'id': name,'object':'model','owned_by':'local'} for name in ('auto','qwen','ornith','qwen-asr','qwen-tts','qwen-image','minimax-h3')]}

    @app.get('/router/status')
    async def status():
        return {'resources': await app.state.manager.status(), 'queues': await app.state.scheduler.queues(), 'security': plain(app.state.scanner.status()),
                'queue_paused':not app.state.scheduler._accepting,
                'pinned_protection':{'router':'in-process','laya':'in-process','openclaw':'observed only; never managed','tailscale':'observed only; never managed'},
                'future_profiles':app.state.config.get('future_profiles',{})}

    @app.get('/router/resources')
    async def resources(): return await app.state.manager.status()
    @app.get('/router/queues')
    async def queues(): return {'paused':not app.state.scheduler._accepting,'resources':await app.state.scheduler.queues()}
    @app.get('/router/security')
    async def security(): return plain(app.state.scanner.status())
    @app.get('/metrics')
    async def metrics(): return {**app.state.metrics, 'security': plain(app.state.scanner.status()), 'queues': await app.state.scheduler.queues(),
                                 'jobs':await app.state.scheduler.metrics(), 'resources':await app.state.manager.status()}

    def chat_targets():
        config = app.state.config
        targets = {'qwen': {'resource': 'qwen-vlm', 'url': 'http://127.0.0.1:8080', 'model': config['qwen_model']}}
        targets.update(config.get('chat_models', {}))
        return targets

    async def choose_chat_model(requested, messages, visual):
        """Explicit qwen/ornith win; auto and legacy aliases are routed by Laya:
        coding and reasoning go to Qwen, general and agentic work to Ornith."""
        targets = chat_targets()
        requested = str(requested or 'auto').split('/')[-1]
        if requested in targets: return requested, 'requested'
        if 'ornith' not in targets: return 'qwen', 'only-model'
        if visual: return 'qwen', 'vision'
        text = latest_user_text(messages)
        if not text or not hasattr(app.state.scanner, 'route_request'): return 'ornith', 'default'
        decision = await asyncio.to_thread(app.state.scanner.route_request, text)
        app.state.metrics[f"route_{decision['route']}"] = app.state.metrics.get(f"route_{decision['route']}", 0) + 1
        name = 'qwen' if decision['route'] == 'reasoning' else 'ornith'
        return name, f"laya:{decision['route']}:{decision['reasoning_score']}"

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        original = await request.json()
        original['messages'], report = await scan(original.get('messages', []), metadata(request))
        payload, visual = prepare_chat(original, root)
        name, reason = await choose_chat_model(original.get('model'), original['messages'], visual)
        target = chat_targets()[name]
        payload['model']=target['model']
        ready = asyncio.get_running_loop().create_future()
        consumed = asyncio.Event()
        async def operation(job):
            response = None
            try:
                async with app.state.manager.lease(target['resource'], incremental_gib=chat_incremental_gib(payload)):
                    app.state.metrics['backend_requests'] += 1
                    headers = {'authorization':'Bearer '+app.state.config.get('backend_key','')}
                    if payload.get('stream'):
                        response = await app.state.http.send(app.state.http.build_request('POST',target['url']+'/v1/chat/completions',json=payload,headers=headers), stream=True)
                        response.raise_for_status()
                        ready.set_result(response)
                        await consumed.wait()
                        return {'stream':'consumed'}
                    response = await app.state.http.post(target['url']+'/v1/chat/completions',json=payload,headers=headers)
                    response.raise_for_status()
                    result = response.json(); result['model']=name; ready.set_result(result); return {'completed':True}
            except BaseException as exc:
                if not ready.done(): ready.set_exception(exc)
                raise
            finally:
                if response is not None: await response.aclose()
        job = await app.state.scheduler.submit(target['resource'], 2 if visual else 1, operation, metadata(request))
        # Consume exceptions on the durable job future; the endpoint awaits ready.
        job.future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        try: result = await asyncio.shield(ready)
        except asyncio.CancelledError:
            await app.state.scheduler.cancel(job.id); raise
        except Exception: raise HTTPException(503,detail={'job_id':job.id,'error':f'{name} backend unavailable'})
        headers = {'X-Bubble-Job-ID':job.id, 'X-Bubble-Security': str(plain(report).get('action','allow')),
                   'X-Bubble-Route': f'{name};{reason}'}
        if not payload.get('stream'): return JSONResponse(result,headers=headers)
        async def chunks():
            try:
                async for chunk in result.aiter_bytes(): yield chunk
            finally: consumed.set()
        return StreamingResponse(chunks(),media_type='text/event-stream',headers=headers)

    @app.post('/v1/audio/speech')
    async def speech(request: Request):
        payload = await request.json()
        filtered, _ = await scan([{'role':'user','content':payload.get('input','')}],metadata(request))
        if len(payload.get('input','')) > 5000: raise ValueError('Speech text exceeds 5000 characters')
        if payload.get('ref_audio'): payload['ref_audio'] = str(safe_asset(payload['ref_audio'], root))
        payload['model'] = app.state.config['audio_models']['tts']
        payload['input'] = filtered[0]['content']; payload['verbose'] = False
        payload.setdefault('voice','Ryan'); payload.setdefault('lang_code','English')
        payload['stream'] = False
        if payload.get('response_format','wav') not in ('wav','mp3','flac'): raise ValueError('Unsupported speech format')
        payload.setdefault('response_format','wav')
        async def operation(job):
            async with app.state.manager.lease('qwen-audio'):
                app.state.metrics['backend_requests'] += 1
                response = await app.state.http.post('http://127.0.0.1:8765/v1/audio/speech',json=payload)
                response.raise_for_status()
                output = root/'outputs/audio'/f'{job.id}.{payload["response_format"]}'
                output.write_bytes(response.content)
                return {'artifact':str(output),'content_type':response.headers.get('content-type','audio/wav')}
        result = await wait_job(request,'qwen-audio',3 if request.headers.get('x-job-background')=='true' else 1,operation)
        return FileResponse(result['artifact'],media_type=result['content_type'])

    @app.post('/v1/audio/transcriptions')
    async def transcription(request:Request,file:UploadFile=File(...),model:str=Form('qwen-asr'),language:str=Form('English')):
        data = await file.read(100*1024**2+1)
        if len(data)>100*1024**2: raise ValueError('Audio exceeds 100 MiB')
        async def operation(job):
            async with app.state.manager.lease('qwen-audio'):
                app.state.metrics['backend_requests'] += 1
                response=await app.state.http.post('http://127.0.0.1:8765/v1/audio/transcriptions',files={'file':('input.wav',data,file.content_type or 'audio/wav')},data={'model':app.state.config['audio_models']['asr'],'language':language,'response_format':'json'})
                response.raise_for_status(); return response.json()
        return await wait_job(request,'qwen-audio',3 if request.headers.get('x-job-background')=='true' else 1,operation)

    @app.post('/v1/audio/transcriptions/batch')
    async def transcription_batch(request:Request,files:list[UploadFile]=File(...),language:str=Form('English')):
        if not 1 <= len(files) <= 8: raise ValueError('Batch must contain 1..8 audio files')
        data=[];total=0
        for file in files:
            audio=await file.read(100*1024**2+1)
            total+=len(audio)
            if total>100*1024**2: raise ValueError('Combined audio exceeds 100 MiB')
            data.append((audio,file.content_type or 'audio/wav'))
        async def operation(job):
            # Ordered batch submission uses one native ASR worker and lease.
            # It deliberately does not create concurrent model executions.
            async with app.state.manager.lease('qwen-audio'):
                results=[]
                for index,(audio,mime) in enumerate(data):
                    app.state.metrics['backend_requests']+=1
                    response=await app.state.http.post('http://127.0.0.1:8765/v1/audio/transcriptions',files={'file':('input.wav',audio,mime)},data={'model':app.state.config['audio_models']['asr'],'language':language,'response_format':'json'})
                    response.raise_for_status();results.append({'index':index,**response.json()})
                return {'results':results}
        return await wait_job(request,'qwen-audio',3 if request.headers.get('x-job-background')=='true' else 1,operation)

    async def submit_media(request, resource, method):
        paths=json.loads((root/'config/model-paths.json').read_text())
        required='abenzerps/Qwen-Image-2.1-Uncensored-GGUF' if resource=='qwen-image' else 'MiniMaxAI/MiniMax-H3'
        if required not in paths: raise HTTPException(503,'Selected model download is not complete')
        payload = await request.json()
        messages=[{'role':'user','content':payload.get('prompt','')}]
        if payload.get('negative_prompt'): messages.append({'role':'user','content':payload['negative_prompt']})
        filtered, _ = await scan(messages, metadata(request))
        payload['prompt'] = filtered[0]['content']
        if len(filtered)>1: payload['negative_prompt']=filtered[1]['content']
        async def operation(job):
            async with app.state.manager.lease(resource):
                app.state.metrics['backend_requests'] += 1
                return await method(payload,job.id)
        job = await app.state.scheduler.submit(resource,4 if resource=='qwen-image' else 5,operation,metadata(request))
        job.future.add_done_callback(lambda f:f.exception() if not f.cancelled() else None)
        return JSONResponse({'id':job.id,'status':'queued','url':f'/v1/jobs/{job.id}'},status_code=202)
    @app.post('/v1/images/generations')
    async def image(request:Request): return await submit_media(request,'qwen-image',app.state.media.generate_image)
    @app.post('/v1/images/edits')
    async def edit(request:Request): return await submit_media(request,'qwen-image',app.state.media.edit_image)
    @app.post('/v1/videos/generations')
    async def video(request:Request): return await submit_media(request,'minimax-h3',app.state.media.generate_video)
    @app.get('/v1/jobs/{job_id}')
    async def get_job(job_id:str):
        job=await app.state.scheduler.get(job_id)
        if job is None: raise HTTPException(404,'Unknown job')
        return job_json(job)
    @app.post('/v1/jobs/{job_id}/cancel')
    async def cancel(job_id:str): return {'cancelled':await app.state.scheduler.cancel(job_id)}
    @app.get('/v1/jobs/{job_id}/artifact')
    async def artifact(job_id:str):
        job=await app.state.scheduler.get(job_id)
        if job is None or not isinstance(job.result,dict) or not job.result.get('artifact'): raise HTTPException(404,'No artifact')
        path=safe_asset(job.result['artifact'],root)
        if not path.is_relative_to((root/'outputs').resolve()): raise HTTPException(403,'Invalid artifact')
        return FileResponse(path,media_type=job.result.get('content_type'))
    @app.post('/router/control/{action}')
    async def control(action:str,request:Request):
        payload=await request.json()
        if action=='security':
            # Validate before replacing either the live policy or its saved copy.
            policy = SecurityPolicy.from_value(payload)
            updated = {**app.state.config, 'security_policy':dataclasses.asdict(policy)}
            path = root/'config/control-plane.json'
            temporary = path.with_suffix('.json.tmp')
            temporary.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd,'w') as handle: json.dump(updated,handle,indent=2)
            os.replace(temporary,path)
            app.state.scanner.update_policy(policy)
            app.state.config=updated
            return {'policy':dataclasses.asdict(policy)}
        if action=='drain': return await app.state.scheduler.drain()
        if action=='resume': return await app.state.scheduler.resume()
        if action=='restore':
            async def restore(job):
                await app.state.manager.restore_core()
                return {'core':'restored'}
            return await wait_job(request,'control',0,restore)
        name=payload.get('resource')
        if name not in app.state.config['backends']: raise ValueError('Unknown resource')
        if action not in ('load','unload'): raise ValueError('Unknown control action')
        async def operation(job):
            if action=='load': await app.state.manager.ensure(name)
            else: await app.state.manager.stop(name)
            return {'resource':name,'action':action}
        return await wait_job(request,name,0,operation)
    return app

app = create_app()
