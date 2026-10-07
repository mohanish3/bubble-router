"""Loopback Mac control plane. Debian entry point remains model_router.main."""
from __future__ import annotations
import asyncio
import hashlib
import copy
import dataclasses
import json
import math
import os
import re
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
from .media import MediaBackend, video_frame_plan
from .embeddings import EMBED_URL, MODEL_ID as EMBED_MODEL_ID, parse_inputs as parse_embedding_inputs, prompted as embedding_prompted

ROOT = Path(os.environ.get('AI_STACK_ROOT', str(Path.home() / 'AI')))

class SecurityRejected(Exception):
    def __init__(self, report):
        super().__init__('request blocked by security policy')
        self.report = report


ROUTE_FOLLOW_UP_CHARS = 80
# OpenClaw prefixes "[Tue 2026-10-06 23:00 GMT+5:30] " and appends a
# "Runtime: agent=... | host=... | repo=..." line to user messages.
OPENCLAW_TIMESTAMP_RE = re.compile(r'^\[[A-Z][a-z]{2} \d{4}-\d{2}-\d{2} [^\]\n]{1,40}\]\s*')
OPENCLAW_RUNTIME_RE = re.compile(r'\n\s*Runtime: agent=[^\n]*(?:\n.*)?\Z', re.S)
OPENCLAW_INTERNAL_RE = re.compile(r'<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>.*?(?:<<<END_OPENCLAW_INTERNAL_CONTEXT>>>|\Z)', re.S)


def text_chars(payload):
    total = 0
    for message in payload['messages']:
        content = message.get('content', '')
        total += len(content) if isinstance(content, str) else sum(len(item.get('text', '')) for item in content or [] if isinstance(item, dict))
    return total


def fits_context(payload, context):
    """Text admission bound at ~4 chars/token. English averages ~4.5; denser text
    (code, Devanagari) may run past the working window, which the models' native
    262K context absorbs. Actual KV grows per request."""
    return text_chars(payload) <= context * 4


# A line that runs a local program: `/path/prog args`, `~/path/prog`, or a `$ ` prompt.
UNVERIFIED_MEDIA_RE = re.compile(r'(?:^|\s)MEDIA:\S+', re.M)
UNRUN_COMMAND_RE = re.compile(r'^\s*(?:\$\s+\S|(?:/|~/)[\w.-]+/[\w./-]*[\w-]+(?:\s|$))', re.M)


def printed_instead_of_calling(result):
    """Detect a first-turn command or media attachment without a tool call."""
    message = (result.get('choices') or [{}])[0].get('message') or {}
    content = message.get('content') or ''
    return not message.get('tool_calls') and bool(UNRUN_COMMAND_RE.search(content) or UNVERIFIED_MEDIA_RE.search(content))


def completion_as_sse(result):
    """Replay a finished chat completion as an OpenAI-style event stream."""
    choice = (result.get('choices') or [{}])[0]
    message = choice.get('message') or {}
    delta = {'role': 'assistant', 'content': message.get('content')}
    for key in ('reasoning_content', 'reasoning'):
        if message.get(key): delta[key] = message[key]
    if message.get('tool_calls'):
        delta['tool_calls'] = [{**call, 'index': index} for index, call in enumerate(message['tool_calls'])]
    base = {'id': result.get('id', 'chatcmpl-bubble'), 'object': 'chat.completion.chunk',
            'created': result.get('created', int(time.time())), 'model': result.get('model')}
    events = [{**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
              {**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': choice.get('finish_reason') or 'stop'}],
               **({'usage': result['usage']} if result.get('usage') else {})}]
    return ''.join(f'data: {json.dumps(event)}\n\n' for event in events) + 'data: [DONE]\n\n'


def latest_user_text(messages):
    """Latest user text; a short follow-up is routed with the user turn before it."""
    texts = []
    for message in reversed(messages or []):
        if message.get('role') != 'user': continue
        content = message.get('content')
        if isinstance(content, list):
            content = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        if not isinstance(content, str): continue
        # OpenClaw sends its own bookkeeping (exec sessions, recovery notes) in
        # user-role messages; route on what the person wrote. Routing only;
        # security scanning still sees the whole message.
        content = OPENCLAW_INTERNAL_RE.sub('', content)
        content = OPENCLAW_RUNTIME_RE.sub('', OPENCLAW_TIMESTAMP_RE.sub('', content)).strip()
        if not content: continue
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
    allowed = {'model','messages','max_tokens','max_completion_tokens','stream','stream_options','temperature','top_p','top_k','seed','tools','tool_choice','parallel_tool_calls','response_format','stop','frequency_penalty','presence_penalty','context_length','reasoning_effort','thinking_budget'}
    result = {key:copy.deepcopy(value) for key,value in payload.items() if key in allowed}
    result['model'] = 'qwen'
    if not isinstance(result.get('messages'), list) or not result['messages']:
        raise ValueError('messages must be a nonempty list')
    max_tokens = int(result.get('max_tokens', result.pop('max_completion_tokens', 2048)))
    if not 1 <= max_tokens <= 8192: raise ValueError('max_tokens must be 1..8192')
    result['max_tokens'] = max_tokens
    if 'reasoning_effort' in result:
        effort = result['reasoning_effort']
        if not isinstance(effort, str): raise ValueError('reasoning_effort must be a supported string')
        effort = effort.strip().lower()
        if effort not in {'low','medium','xhigh','none','off'}:
            raise ValueError('reasoning_effort must be low, medium, xhigh, none, or off')
        result['reasoning_effort'] = effort
    if 'thinking_budget' in result:
        budget = result['thinking_budget']
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise ValueError('thinking_budget must be an integer')
        if not 0 <= budget <= min(max_tokens, 8192):
            raise ValueError('thinking_budget must be 0..max_tokens (up to 8192)')
    # The context budget is checked per model after routing (check_context).
    if 'context_length' in result:
        context = int(result.pop('context_length'))
        if not 1024 <= context <= 131072: raise ValueError('context_length must be 1024..131072')
        result['_context_length'] = context
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

def apply_tool_reasoning_defaults(payload, target):
    """Bound AREX reasoning on tool calls unless the caller explicitly chose values."""
    if not payload.get('tools'):
        return payload
    profile = target or {}
    if profile.get('resource') != 'arex' and profile.get('name') != 'arex':
        return payload
    result = dict(payload)
    if 'reasoning_effort' not in result:
        effort = profile.get('tool_reasoning_effort', 'low')
        if not isinstance(effort, str) or effort.strip().lower() not in {'low','medium','xhigh','none','off'}:
            raise ValueError('AREX tool_reasoning_effort configuration is invalid')
        result['reasoning_effort'] = effort.strip().lower()
    if 'thinking_budget' not in result and result.get('reasoning_effort') not in {'none','off'}:
        budget = profile.get('tool_thinking_budget', 1024)
        if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= 8192:
            raise ValueError('AREX tool_thinking_budget configuration must be 0..8192')
        result['thinking_budget'] = min(budget, result.get('max_tokens', 2048))
    return result

BYTES_PER_TOKEN_ESTIMATE = 3

# mlx_vlm.server samples greedily (temperature 0) when a request omits it, and
# OpenClaw omits it. Greedy decoding replays an identical reply for an
# identical context, so agent tool loops never break. These are the
# generation_config.json values of Ornith, Swift and Qwen; a chat model's
# "sampling" config overrides them, and request values always win. The
# server also uses a fixed seed when none is sent, which makes sampled
# replies repeat for an identical context, so each request gets a fresh one.
DEFAULT_SAMPLING = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20}


def apply_sampling_defaults(payload, target):
    defaults = {**DEFAULT_SAMPLING, **target.get('sampling', {}), 'seed': secrets.randbelow(2**31)}
    return {**{key: value for key, value in defaults.items() if payload.get(key) is None}, **{
        key: value for key, value in payload.items() if value is not None or key not in defaults}}


def chat_incremental_gib(payload, kv_bytes_per_token=65536):
    # KV bytes per cached token come from each model's config: Qwen and Swift
    # have 16 full-attention layers x 4 KV heads x 256-dim BF16 (65536);
    # Ornith has 10 layers x 2 heads (20480). Chat/JSON text averages 3.5-4
    # UTF-8 bytes per token, so bytes/3 stays conservative without charging
    # one token per byte. Visual inputs receive a separate allowance.
    messages = payload.get('messages', [])
    text = json.dumps(messages, ensure_ascii=False) + json.dumps(payload.get('tools', []), ensure_ascii=False)
    tokens = len(text.encode('utf-8')) // BYTES_PER_TOKEN_ESTIMATE + int(payload.get('max_tokens', 2048))
    visuals = sum(1 for message in messages if isinstance(message.get('content'), list)
                  for item in message['content'] if item.get('type') in ('image_url','video_url','input_video'))
    tokens = min(131072, tokens + visuals * 16384)
    return 4.0 + tokens * kv_bytes_per_token / 1024**3


def h3_video_admission(config, payload):
    """Return the H3 frame plan and duration-aware memory admission values."""
    plan = video_frame_plan(payload)
    profile = config.get('backends', {}).get('minimax-h3', {})
    base_value = profile.get('estimate_gib', 40.0) if isinstance(profile, dict) else None
    long_value = config.get('h3_long_estimate_gib', 40.0)
    def finite_positive(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} must be a finite positive number')
        try: numeric = float(value)
        except (OverflowError, ValueError) as exc: raise ValueError(f'{name} must be a finite positive number') from exc
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError(f'{name} must be a finite positive number')
        return numeric
    base = finite_positive('minimax-h3 estimate_gib', base_value)
    long_peak = finite_positive('h3_long_estimate_gib', long_value)
    if long_peak < base:
        raise ValueError('h3_long_estimate_gib must be at least the H3 base estimate')
    needs_long_headroom = plan['native_frames'] > 124
    peak = long_peak if needs_long_headroom else base
    incremental = max(0.0, long_peak - base) if needs_long_headroom else 0.0
    return plan, incremental, peak

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
            # Optional fine-tuned checkpoints: laya-security screens, laya-route routes.
            security_path = paths.get('laya-security', paths['convaiinnovations/laya'])['path']
            route_path = paths.get('laya-route', {}).get('path')
            app.state.scanner = SecurityScanner(security_path, config.get('security_policy', {'max_input_chars':393216}), route_path=route_path)
            app.state.media = MediaBackend(app.state.manager, paths, root)
        app.state.route_affinity = {'target': None, 'count': 0}
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
        return {'object':'list','data':[{'id': name,'object':'model','owned_by':'local'} for name in ('auto','qwen','ornith','swift','arex','qwen-asr','qwen-tts','qwen-image','minimax-h3')]}

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
        targets = {'qwen': {'resource': 'qwen-vlm', 'url': 'http://127.0.0.1:8080', 'model': config['qwen_model'], 'vision': True}}
        targets.update(config.get('chat_models', {}))
        return targets

    def route_target(route):
        """Model for a Laya route class; config route_targets may override."""
        targets = chat_targets()
        mapping = {'general': 'ornith', 'swift': 'swift', 'reasoner': 'qwen', **app.state.config.get('route_targets', {})}
        for name in (mapping.get(route), 'qwen' if route != 'general' else 'ornith', 'qwen'):
            if name in targets: return name
        return 'qwen'

    async def choose_chat_model(requested, messages, visual):
        """Explicit model names win; auto and legacy aliases are routed by Laya:
        general and simple agentic work to Ornith, deep agentic work, simple
        coding, subtle bugs and costly plans to Swift, and long-horizon
        reasoning and complex coding to the reasoner tier."""
        targets = chat_targets()
        requested = str(requested or 'auto').split('/')[-1]
        if requested in targets: return requested, 'requested'
        if 'ornith' not in targets: return 'qwen', 'only-model'
        text = latest_user_text(messages)
        if not text or not hasattr(app.state.scanner, 'route_request'):
            return (vision_fallback(), 'vision') if visual else sticky_choice('ornith', 'default', visual)
        decision = await asyncio.to_thread(app.state.scanner.route_request, text)
        app.state.metrics[f"route_{decision['route']}"] = app.state.metrics.get(f"route_{decision['route']}", 0) + 1
        entry = {'time': time.time(), 'route': decision['route'], 'scores': decision['scores'],
                 'chars': len(text), 'sha256': hashlib.sha256(text.encode()).hexdigest()[:16]}
        if os.environ.get('BUBBLE_ROUTE_DEBUG') == '1':
            entry['preview'] = text[:160]  # debugging only; never enabled by default
        fd = os.open(root/'state/route-audit.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, 'a') as handle: handle.write(json.dumps(entry)+'\n')
        score = decision['scores'].get(decision['route'], 0.0)
        name = route_target(decision['route'])
        if visual and not targets[name].get('vision'):
            return vision_fallback(), f"laya:{decision['route']}:{score};vision"
        return sticky_choice(name, f"laya:{decision['route']}:{score}", visual)

    def sticky_choice(preferred, reason, visual):
        """Keep using the chat model that is already loaded. Switching costs a
        19-22 GB reload, so auto requests move only after route_switch_after
        consecutive requests prefer the same other model."""
        targets = chat_targets()
        is_loaded = getattr(app.state.manager, 'is_loaded', None)
        loaded = [name for name, target in targets.items() if is_loaded and is_loaded(target['resource'])]
        affinity = app.state.route_affinity
        current = loaded[0] if loaded else None
        if current is None or preferred in loaded or (visual and not targets[current].get('vision')):
            affinity.update(target=None, count=0)
            return preferred, reason
        affinity['count'] = affinity['count'] + 1 if affinity['target'] == preferred else 1
        affinity['target'] = preferred
        threshold = max(1, int(app.state.config.get('route_switch_after', 3)))
        if affinity['count'] >= threshold:
            affinity.update(target=None, count=0)
            return preferred, f'{reason};switch:after-{threshold}'
        return current, f"{reason};sticky:{current}:{affinity['count']}/{threshold}"

    def vision_fallback():
        targets = chat_targets()
        name = app.state.config.get('vision_default', 'qwen')
        return name if targets.get(name, {}).get('vision') else 'qwen'

    async def dispatch_chat(name, payload, visual, meta):
        """Queue one chat request on a model; returns (job, ready future, consumed event)."""
        target = chat_targets()[name]
        payload = apply_sampling_defaults({**payload, 'model': target['model']}, target)
        ready = asyncio.get_running_loop().create_future()
        consumed = asyncio.Event()
        async def operation(job):
            response = None
            try:
                async with app.state.manager.lease(target['resource'], incremental_gib=chat_incremental_gib(payload, target.get('kv_bytes_per_token', 65536))):
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
        job = await app.state.scheduler.submit(target['resource'], 2 if visual else 1, operation, meta)
        # Consume exceptions on the durable job future; the endpoint awaits ready.
        job.future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        return job, ready, consumed

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        original = await request.json()
        original['messages'], report = await scan(original.get('messages', []), metadata(request))
        payload, visual = prepare_chat(original, root)
        requested_context = payload.pop('_context_length', None)
        name, reason = await choose_chat_model(original.get('model'), original['messages'], visual)
        targets = chat_targets()
        context = requested_context or targets[name].get('context', 32768)
        if not fits_context(payload, context):
            # Long conversations go to the long-context tier when the routed
            # model's working window is smaller.
            longest = max(targets, key=lambda n: targets[n].get('context', 32768))
            if requested_context is None and (not visual or targets[longest].get('vision')) \
                    and fits_context(payload, targets[longest].get('context', 32768)):
                name, reason = longest, reason + ';long-context'
            else:
                raise HTTPException(422, 'Text exceeds the model context budget')
        payload = apply_tool_reasoning_defaults(payload, {'name':name, **targets[name]})
        meta = metadata(request)
        attempts = [name]
        # Ornith is the first try for general work; if it fails before
        # answering, escalate the same request to Swift.
        escalate_to = app.state.config.get('escalate', {}).get(name)
        if escalate_to in chat_targets(): attempts.append(escalate_to)
        # With tools offered, Ornith sometimes prints the command instead of
        # calling the tool. Its reply is buffered so that case can escalate too.
        # Only a reply to a fresh user turn is checked; after tool results,
        # listing paths is a normal answer.
        check_tools = len(attempts) > 1 and bool(payload.get('tools')) and \
            (payload['messages'] or [{}])[-1].get('role') == 'user'
        for index, current in enumerate(attempts):
            buffered = check_tools and index == 0
            attempt = {k: v for k, v in payload.items() if k != 'stream_options'} | {'stream': False} if buffered else payload
            job, ready, consumed = await dispatch_chat(current, attempt, visual, meta)
            try:
                result = await asyncio.shield(ready)
                if buffered and printed_instead_of_calling(result):
                    app.state.metrics['escalations'] = app.state.metrics.get('escalations', 0) + 1
                    reason += f';escalated-from-{current}:printed-command'
                    continue
                if buffered and payload.get('stream'):
                    sse = completion_as_sse(result)
                    headers = {'X-Bubble-Job-ID': job.id, 'X-Bubble-Security': str(plain(report).get('action', 'allow')),
                               'X-Bubble-Route': f'{current};{reason}'}
                    return Response(sse, media_type='text/event-stream', headers=headers)
                break
            except asyncio.CancelledError:
                await app.state.scheduler.cancel(job.id); raise
            except Exception:
                if index + 1 < len(attempts):
                    app.state.metrics['escalations'] = app.state.metrics.get('escalations', 0) + 1
                    reason += f';escalated-from-{current}'
                    continue
                raise HTTPException(503,detail={'job_id':job.id,'error':f'{current} backend unavailable'})
        name = current
        headers = {'X-Bubble-Job-ID':job.id, 'X-Bubble-Security': str(plain(report).get('action','allow')),
                   'X-Bubble-Route': f'{name};{reason}'}
        if not payload.get('stream'): return JSONResponse(result,headers=headers)
        async def chunks():
            try:
                async for chunk in result.aiter_bytes(): yield chunk
            finally: consumed.set()
        return StreamingResponse(chunks(),media_type='text/event-stream',headers=headers)

    @app.post('/v1/embeddings')
    async def embeddings(request: Request):
        # Vectors only, for OpenClaw memory search: nothing is generated or
        # executed, so the input is not routed through the Laya scan.
        payload = await request.json()
        texts = parse_embedding_inputs(payload)
        app.state.metrics['requests'] += 1
        try:
            response = await app.state.http.post(EMBED_URL, json={'input': embedding_prompted(payload, texts)})
            response.raise_for_status()
        except httpx.HTTPError:
            raise HTTPException(503, 'embedding service unavailable')
        vectors = [item['embedding'] for item in sorted(response.json()['data'], key=lambda item: item['index'])]
        return {'object': 'list', 'model': EMBED_MODEL_ID,
                'data': [{'object': 'embedding', 'index': i, 'embedding': v} for i, v in enumerate(vectors)],
                'usage': {'prompt_tokens': 0, 'total_tokens': 0}}

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
        h3_incremental = 0.0
        job_metadata = metadata(request)
        if resource == 'minimax-h3':
            _, h3_incremental, peak = h3_video_admission(app.state.config, payload)
            job_metadata['estimated_peak_gib'] = peak
        messages=[{'role':'user','content':payload.get('prompt','')}]
        if payload.get('negative_prompt'): messages.append({'role':'user','content':payload['negative_prompt']})
        filtered, _ = await scan(messages, metadata(request))
        payload['prompt'] = filtered[0]['content']
        if len(filtered)>1: payload['negative_prompt']=filtered[1]['content']
        async def operation(job):
            lease = (app.state.manager.lease(resource, incremental_gib=h3_incremental)
                     if resource == 'minimax-h3' and h3_incremental > 0
                     else app.state.manager.lease(resource))
            async with lease:
                app.state.metrics['backend_requests'] += 1
                return await method(payload,job.id)
        job = await app.state.scheduler.submit(resource,4 if resource=='qwen-image' else 5,operation,job_metadata)
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
