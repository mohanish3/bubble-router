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
from .route_ettin import EttinRouter
from .soft_decisions import SoftDecisions
from .simple_planning import PlanningClassifier, planning_candidate, planning_issue
from .media import MediaBackend, video_frame_plan, image_kind
from .music import MusicBackend
from .idol import IdolLocalBackend
from .openclaw_identity import OpenClawIdentity, channel_title
from .embeddings import EMBED_URL, MODEL_ID as EMBED_MODEL_ID, parse_inputs as parse_embedding_inputs, prompted as embedding_prompted

ROOT = Path(os.environ.get('AI_STACK_ROOT', str(Path.home() / 'AI')))

class SecurityRejected(Exception):
    def __init__(self, report):
        super().__init__('request blocked by security policy')
        self.report = report


MEDIA_UPLOAD_MAX_BYTES = 20 * 1024**2
MEDIA_INT_FIELDS = frozenset({'width','height','steps','seed','frames','upscale','num_gaussians'})
MEDIA_FLOAT_FIELDS = frozenset({'duration','seconds','cfg_strength','rescale_t'})

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



def unexecuted_tool_output(result, tools):
    """Recognize leaked calls and native execution envelopes, never execute text.

    This is used only before a fresh user turn has any tool results. Ordinary
    JSON answers are allowed; a result envelope needs its specific native keys.
    """
    message = (result.get('choices') or [{}])[0].get('message') or {}
    if message.get('tool_calls'): return None
    content = message.get('content') or ''
    if not isinstance(content, str): return None
    names = {tool['function']['name'] for tool in tools
             if isinstance(tool, dict) and isinstance(tool.get('function'), dict)
             and isinstance(tool['function'].get('name'), str)}
    if re.search(r'<tool_call>\s*<function=[\w.-]+>', content):
        return 'printed-tool-call'
    decoder = json.JSONDecoder()
    # Only standalone JSON blocks, bounded to avoid scanning prose character
    # by character. Explanations containing inline JSON are ordinary text.
    for match in list(re.finditer(r'^\s*(?=[{\[])', content, re.M))[:16]:
        try: value, _ = decoder.raw_decode(content[match.end():])
        except ValueError: continue
        if not isinstance(value, dict): continue
        function = value.get('function')
        if isinstance(function, dict) and isinstance(function.get('name'), str) and function['name'] in names and 'arguments' in function:
            return 'printed-tool-call'
        if isinstance(value.get('name'), str) and value['name'] in names and ('arguments' in value or 'parameters' in value):
            return 'printed-tool-call'
        if isinstance(value.get('tool_calls'), list) and value['tool_calls']:
            return 'printed-tool-call'
        if value.get('type') == 'process_group' and all(key in value for key in
                ('processGroupId', 'state', 'runningCount', 'count', 'results')):
            return 'printed-tool-result'
        if value.get('type') == 'timer' and all(key in value for key in
                ('name', 'triggered', 'next', 'status')):
            return 'printed-tool-result'
    return None


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


def person_text(message):
    """What the person wrote in a user message, without OpenClaw bookkeeping."""
    if message.get('role') != 'user': return ''
    content = message.get('content')
    if isinstance(content, list):
        content = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
    if not isinstance(content, str): return ''
    # OpenClaw sends its own bookkeeping (exec sessions, recovery notes) in
    # user-role messages; route on what the person wrote. Routing only;
    # security scanning still sees the whole message.
    content = OPENCLAW_INTERNAL_RE.sub('', content)
    return OPENCLAW_RUNTIME_RE.sub('', OPENCLAW_TIMESTAMP_RE.sub('', content)).strip()


def latest_user_text(messages):
    """Latest user text; a short follow-up is routed with the user turn before it."""
    texts = []
    for message in reversed(messages or []):
        content = person_text(message)
        if not content: continue
        texts.append(content)
        if len(texts) == 2 or len(content) >= ROUTE_FOLLOW_UP_CHARS: break
    return '\n'.join(reversed(texts))


CODE_PATH_RE = re.compile(
    r'\.(?:rs|py|pyi|ts|tsx|js|jsx|mjs|cjs|go|java|kt|swift|c|cc|cpp|h|hpp|cs|rb|php|scala|sql|sh|zsh|vue|svelte)\b'
    r'|\b(?:Cargo\.toml|package\.json|pyproject\.toml|Makefile|go\.mod)\b')
CODE_COMMAND_RE = re.compile(
    r'\b(?:cargo|rustc|pytest|tsc|npm|pnpm|yarn|bun|gradle|mvn|xcodebuild|make|go (?:build|test|run|vet)'
    r'|python3? -m (?:pytest|mypy|ruff)|swift (?:build|test)|git (?:diff|apply))\b')


def coding_run(messages):
    """(tool calls, code tool calls) since the person's latest message: one
    agent run. A call counts as code when its arguments name a source file
    or a build/test command."""
    calls = code = 0
    for message in reversed(messages or []):
        if person_text(message): break
        if message.get('role') != 'assistant': continue
        for call in message.get('tool_calls') or []:
            arguments = (call.get('function') or {}).get('arguments') or ''
            arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
            calls += 1
            code += bool(CODE_PATH_RE.search(arguments) or CODE_COMMAND_RE.search(arguments))
    return calls, code


# Ornith answers simple requests fast, but given work past one obvious step it
# has invented launches and printed commands, MEDIA paths or tool results
# instead of calling a tool (main 2026-10-07: video/image requests; validator
# 2026-10-15: "Validate 10 more", then its own invented process_group JSON
# copied into every follow-up). It only gets requests that pass every check.
# Build commands as a person writes them; plain "make" is ordinary English there.
TEXT_CODE_COMMAND_RE = re.compile(CODE_COMMAND_RE.pattern.replace('|make|', '|'))
SIMPLE_ROUTE_DEFAULTS = {'model': 'ornith', 'min_score': 0.6, 'max_chars': 400, 'max_run_tool_calls': 3, 'switch_after': 2}
# Chat lanes by x-source: manual requests shelve background ones. prefer_loaded
# lists the chat models a manual auto request may stay on instead of a reload.
SHELF_DEFAULTS = {'grace_seconds': 120, 'max_hold_seconds': 900, 'manual_sources': ['', 'manual'],
                  'background_sources': ['background', 'cron', 'loop'], 'prefer_loaded': []}
HEAVY_ACTION_RE = re.compile(
    r'\b(?:videos?|images?|pictures?|photos?|render\w*|songs?|music|draw\w*|paint\w*'
    r'|sub-?agents?|spawn\w*|batch\w*|in parallel|simulat\w*|validat\w*|\d+\s+(?:more|ideas|files|tasks|items|runs)'
    r'|every\s+(?:\d+\s*)?(?:min|minute|hour|half|day|week)\w*|each\s+(?:half\s+)?(?:min|minute|hour|day)\w*'
    r'|monitor\w*|watch\w*|loop\w*|cron|schedul\w*|remind\w*'
    r'|deploy\w*|install\w*|migrat\w*|delete\w*|remove\w*|refactor\w*|implement\w*)\b', re.I)


def printed_tool_history(messages, tools):
    """True when an earlier fresh-turn reply printed a tool call, command,
    MEDIA path or tool result instead of calling a tool. Models copy their own
    earlier replies, so such a conversation is not simple any more."""
    previous = None
    for message in messages or []:
        role = message.get('role')
        if role == 'assistant' and previous == 'user' and not message.get('tool_calls'):
            content = message.get('content')
            if isinstance(content, list):
                content = '\n'.join(part.get('text', '') for part in content if isinstance(part, dict))
            if isinstance(content, str) and content:
                result = {'choices': [{'message': {'content': content}}]}
                if unexecuted_tool_output(result, tools or []) or printed_instead_of_calling(result):
                    return True
        if role in ('user', 'assistant', 'tool'):
            previous = role
    return False


def simple_request_issue(messages, text, scores, tools, policy):
    """Why a Laya 'general' request is too much for the fast tier, or None."""
    if not text: return 'no-text'
    if scores.get('general', 0.0) < policy['min_score']: return 'low-score'
    if len(text) > policy['max_chars']: return 'long'
    if '```' in text or CODE_PATH_RE.search(text) or TEXT_CODE_COMMAND_RE.search(text): return 'code'
    if HEAVY_ACTION_RE.search(text): return 'action'
    calls, _ = coding_run(messages)
    if calls >= policy['max_run_tool_calls']: return f'run-{calls}-calls'
    if printed_tool_history(messages, tools): return 'printed-tool-history'
    return None


# Who sent a job, for status views. Scheduling never reads these fields: the
# chat lane still comes from x-source alone (chat_lane).
#
# OpenClaw provider headers are fixed per provider, so x-source can say
# "loop" or "background" but not which loop run or cron job. Those turns name
# themselves in the user message that starts them, which OpenClaw forwards
# unchanged, so that message is where per-run identity reaches the Router:
#  - the Taste Gate loop engine (loops/runtime.py, RunContext.prompt) opens
#    every turn prompt with this line;
LOOP_TURN_RE = re.compile(r'^TASTE GATE LOOP turn for loop "(?P<name>[^\n]*)" '
                          r'\((?P<id>[\w.:-]{1,80})\), run (?P<run>[\w:-]{1,80})\.', re.M)
#  - OpenClaw opens an isolated cron agentTurn with "[cron:<job id> <job name>]".
CRON_TURN_RE = re.compile(r'\[cron:(?P<id>[\w-]{1,64}) (?P<name>[^\]\n]{1,120})\]')
EXPERIMENT_SOURCE_RE = re.compile(
    r'(?:^|[-_])(?:poc|evals?|tests?|smoke|probes?|verification|acceptance|benchmarks?|pilot|experiments?)(?:$|[-_])')
ORIGIN_TITLES = {'loop': 'Loop', 'cron': 'Cron', 'manual': 'Manual', 'background': 'Background', 'experiment': 'Experiment'}
ORIGIN_NAME_CHARS = 80


def turn_markers(messages, search=False):
    """Loop run or cron job named by the user message that started this turn.

    Only the latest user message with person text is read and the marker must
    open it, so a person quoting a loop prompt in a chat is not relabelled.
    With search (x-source: loop, where every user message is a loop prompt)
    the marker may follow the engine's retry note, and a few earlier user
    messages are read in case OpenClaw added one of its own after the prompt."""
    if not isinstance(messages, list): return {}
    checked = 0
    for message in reversed(messages):
        if not isinstance(message, dict): continue
        text = person_text(message)[:4096]
        if not text: continue
        loop = (LOOP_TURN_RE.search if search else LOOP_TURN_RE.match)(text)
        if loop:
            return {'loop_id': loop['id'], 'loop_run_id': loop['run'],
                    'loop_name': loop['name'].strip()[:ORIGIN_NAME_CHARS] or loop['id']}
        cron = CRON_TURN_RE.match(text)
        if cron:
            return {'cron_id': cron['id'], 'cron_name': cron['name'].strip()[:ORIGIN_NAME_CHARS]}
        checked += 1
        if checked >= (8 if search else 1): break
    return {}


def job_origin(meta, markers, policy):
    """origin (loop, cron, background, manual, experiment or other), a label
    such as "Loop · Daily AI & business trends", and the loop or cron identity.

    Markers are believed only on OpenClaw's own sources (untagged, manual,
    background, cron, loop); other callers keep the source they declared."""
    source = meta.get('source') or ''
    agent = meta.get('agent_name') or meta.get('agent-id') or ''
    if source not in ('loop', 'cron', 'background') and source not in policy['manual_sources']:
        markers = {}
    if markers.get('loop_id') or source == 'loop':
        kind, detail = 'loop', markers.get('loop_name') or markers.get('loop_id')
    elif markers.get('cron_id') or source == 'cron':
        kind, detail = 'cron', markers.get('cron_name') or markers.get('cron_id')
    elif source in policy['manual_sources']:
        kind, detail = 'manual', agent
        if meta.get('channel'):  # "Telegram · Mot": where a person talked, and to whom
            return {'origin': kind, 'origin_label': ' · '.join(filter(None, (channel_title(meta['channel']), agent)))[:120]}
    elif source in policy['background_sources']:
        kind, detail = 'background', source if source != 'background' else agent
    elif EXPERIMENT_SOURCE_RE.search(source):
        kind, detail = 'experiment', source
    else:
        kind, detail = 'other', None
    if kind == 'other':
        label = ' · '.join(dict.fromkeys(part for part in (source, agent) if part)) or 'Untagged'
    else:
        label = f'{ORIGIN_TITLES[kind]} · {detail}' if detail else ORIGIN_TITLES[kind]
    keep = {'loop': ('loop_id', 'loop_run_id', 'loop_name'), 'cron': ('cron_id', 'cron_name')}.get(kind, ())
    return {'origin': kind, 'origin_label': label[:120], **{key: markers[key] for key in keep if markers.get(key)}}


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


SERVING_MODES = ('mlx', 'vllm', 'splash')
# A profile with no entry for the global mode tries these next: in splash mode,
# models Splash cannot serve (video on qwen-vlm) stay on vLLM.
MODE_FALLBACK = {'splash': ('vllm',)}

def apply_serving_mode(config):
    """Resolve config['serving_mode'] (default 'mlx'): a backend's or chat
    model's "modes" entry for that mode (or its MODE_FALLBACK) overrides its
    base (MLX) profile. Entries without one keep their MLX server in every mode."""
    mode = config.get('serving_mode', 'mlx')
    if mode not in SERVING_MODES: raise ValueError(f'serving_mode must be one of {", ".join(SERVING_MODES)}')
    resolved = copy.deepcopy(config)
    for section in ('backends', 'chat_models'):
        for profile in resolved.get(section, {}).values():
            # A profile's own "mode" pins it to that entry whatever the global
            # mode is, e.g. Swift on Splash while the rest serve on vLLM.
            pinned = profile.pop('mode', None)
            modes = profile.pop('modes', {})
            if pinned is not None and pinned not in modes: raise ValueError(f'mode {pinned} has no "modes" entry')
            if pinned is not None: profile.update(modes[pinned]); continue
            profile.update(next((modes[m] for m in (mode, *MODE_FALLBACK.get(mode, ())) if m in modes), {}))
    return resolved

def resource_slots(backends):
    """Concurrent request slots per backend: a vLLM server's --max-num-seqs.
    MLX servers and all other backends take one request at a time."""
    slots = {}
    for name, profile in backends.items():
        command = list(profile.get('command') or ())
        if profile.get('engine') == 'vllm' and '--max-num-seqs' in command:
            slots[name] = max(1, int(command[command.index('--max-num-seqs') + 1]))
    return slots

def backend_profile(profile):
    """BackendConfig fields of a resolved backend entry (engine is router-side)."""
    return {key: value for key, value in profile.items() if key != 'engine'}

def vllm_chat_payload(payload, think_by_default=False, default_effort=None):
    """Translate an MLX-server chat request to vLLM's field names."""
    result = copy.deepcopy(payload)
    # mlx_vlm leaves reasoning out of replies; keep vLLM replies the same shape.
    result.setdefault('include_reasoning', False)
    # Streamed replies end with a usage event so the job can record its token counts.
    if result.get('stream'): result['stream_options'] = {'include_usage': True, **(result.get('stream_options') or {})}
    if 'thinking_budget' in result:
        result['thinking_token_budget'] = result.pop('thinking_budget')
    # Keep each model's MLX default: mlx_vlm thinks only when reasoning_effort
    # asks; Swift's mlx_lm server thinks unless told not to (think_by_default).
    # vLLM follows the chat template, which always thinks by default.
    effort = result.pop('reasoning_effort', None)
    thinking = think_by_default if effort is None else effort not in {'none', 'off'}
    kwargs = {'enable_thinking': thinking}
    # The templates read reasoning_effort as a template variable (default
    # xhigh), as mlx_vlm passes it; vLLM's top-level field does not reach it.
    if thinking and (effort or default_effort): kwargs['reasoning_effort'] = effort or default_effort
    result['chat_template_kwargs'] = {**kwargs, **result.get('chat_template_kwargs', {})}
    # prepare_chat turns media into checked local paths; vLLM reads local
    # files only as file:// URLs under --allowed-local-media-path.
    for message in result.get('messages', []):
        if not isinstance(message.get('content'), list): continue
        for item in message['content']:
            kind = item.get('type')
            if kind not in ('image_url', 'video_url', 'input_video'): continue
            # input_video carries its URL under video_url (see prepare_chat).
            key = 'image_url' if kind == 'image_url' else 'video_url'
            item['type'] = key
            value = item.get(key)
            url = value.get('url') if isinstance(value, dict) else value
            if isinstance(url, str) and url.startswith('/'):
                url = 'file://' + url
            item[key] = {**value, 'url': url} if isinstance(value, dict) else {'url': url}
    return result

def splash_chat_payload(payload, thinking_budget=None):
    """Adjust a vllm_chat_payload request for a Splash server: it reads the
    effort as a top-level field and enforces thinking_token_budget itself."""
    result = copy.deepcopy(payload)
    # Splash has no include_reasoning switch; its replies carry reasoning_content.
    result.pop('include_reasoning', None)
    kwargs = dict(result.get('chat_template_kwargs') or {})
    if 'reasoning_effort' in kwargs: result['reasoning_effort'] = kwargs.pop('reasoning_effort')
    result['chat_template_kwargs'] = kwargs
    if kwargs.get('enable_thinking') and 'thinking_token_budget' not in result and thinking_budget is not None:
        result['thinking_token_budget'] = thinking_budget
    return result

# Output tokens a concurrent vLLM request reserves. The server allocates KV as a
# reply grows, and agent replies run well under this, so a large max_tokens
# need not keep a request from running beside others.
KV_OUTPUT_RESERVE = 4096

THINKING_CUT_NOTE = "\n\nI'm out of thinking time, so I'll answer now with what I have."

def thinking_cut_budget(payload, default=None):
    """Reasoning tokens the router allows a vLLM request before cutting its thinking, or None.

    vLLM-Metal accepts thinking_token_budget but does not enforce it, so the
    router does. A budget at or above max_tokens could never be reached first.
    """
    if not (payload.get('chat_template_kwargs') or {}).get('enable_thinking'): return None
    budget = payload.get('thinking_token_budget', default)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0: return None
    limit = payload.get('max_tokens') or payload.get('max_completion_tokens')
    return budget if limit is None or budget < limit else None

def fold_chat_stream(events):
    """One non-streamed chat completion from vLLM chat-completion stream events."""
    text, thought, calls, finish, usage = [], [], {}, None, None
    for event in events:
        usage = event.get('usage') or usage
        for choice in event.get('choices') or []:
            delta = choice.get('delta') or {}
            text.append(delta.get('content') or '')
            thought.append(delta.get('reasoning') or delta.get('reasoning_content') or '')
            for call in delta.get('tool_calls') or []:
                slot = calls.setdefault(call.get('index', 0), {'id': None, 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                slot['id'] = call.get('id') or slot['id']
                function = call.get('function') or {}
                slot['function']['name'] += function.get('name') or ''
                slot['function']['arguments'] += function.get('arguments') or ''
            finish = choice.get('finish_reason') or finish
    message = {'role': 'assistant', 'content': ''.join(text) or None}
    if ''.join(thought): message['reasoning'] = ''.join(thought)
    if calls: message['tool_calls'] = [calls[index] for index in sorted(calls)]
    first = events[0] if events else {}
    return {'id': first.get('id', 'chatcmpl-bubble'), 'object': 'chat.completion', 'created': first.get('created', int(time.time())),
            'model': first.get('model'), 'choices': [{'index': 0, 'message': message, 'finish_reason': finish or 'stop'}],
            **({'usage': usage} if usage else {})}

async def budgeted_chat(http, url, payload, headers, budget):
    """A vLLM chat completion whose thinking stops after `budget` reasoning tokens.

    The reply streams in so the router can count reasoning; within budget it is
    returned as is. Past it, the stream is dropped (vLLM aborts the request) and
    the model continues from the thinking so far, closed with </think>, to answer
    or call tools. Thinking is off for that continuation so vLLM parses the
    output as the answer instead of more reasoning.
    """
    body = {key: value for key, value in payload.items() if key != 'stream_options'} | {
        'stream': True, 'stream_options': {'include_usage': True}, 'include_reasoning': True}
    events, reasoning = [], []
    async with http.stream('POST', url + '/v1/chat/completions', json=body, headers=headers) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith('data: ') or line == 'data: [DONE]': continue
            event = json.loads(line[6:]); events.append(event)
            for choice in event.get('choices') or []:
                piece = (choice.get('delta') or {}).get('reasoning') or (choice.get('delta') or {}).get('reasoning_content')
                if piece: reasoning.append(piece)
            if len(reasoning) > budget: break
        else:
            result = fold_chat_stream(events)
            if not payload.get('include_reasoning'): result['choices'][0]['message'].pop('reasoning', None)
            return result
    thought = ''.join(reasoning).rstrip() + THINKING_CUT_NOTE
    limit = payload.get('max_tokens') or payload.get('max_completion_tokens')
    follow = {key: value for key, value in payload.items() if key not in ('stream', 'stream_options', 'max_completion_tokens')} | {
        'messages': payload['messages'] + [{'role': 'assistant', 'content': '', 'reasoning': thought}],
        'add_generation_prompt': False, 'continue_final_message': True,
        'chat_template_kwargs': {**payload.get('chat_template_kwargs', {}), 'enable_thinking': False}}
    follow.pop('thinking_token_budget', None)
    if limit: follow['max_tokens'] = max(limit - len(reasoning), 256)
    response = await http.post(url + '/v1/chat/completions', json=follow, headers=headers)
    response.raise_for_status()
    result = response.json()
    message = result['choices'][0]['message']
    if payload.get('include_reasoning'): message['reasoning'] = thought
    if result.get('usage'):
        result['usage']['completion_tokens'] = result['usage'].get('completion_tokens', 0) + len(reasoning)
        result['usage']['total_tokens'] = result['usage'].get('total_tokens', 0) + len(reasoning)
    result['bubble_thinking_cut'] = len(reasoning)
    return result

def stream_usage(tail):
    """Token usage from the last usage-bearing event at the end of a chat stream, or None."""
    for line in reversed(tail.decode('utf-8', 'ignore').splitlines()):
        if line.startswith('data: {') and '"usage"' in line:
            try: usage = json.loads(line[6:]).get('usage')
            except ValueError: continue
            if isinstance(usage, dict): return usage
    return None

def chat_kv_tokens(payload):
    """Conservative KV tokens one vLLM chat request occupies (same bytes/3
    prompt estimate as chat_incremental_gib, output capped at KV_OUTPUT_RESERVE)."""
    messages = payload.get('messages', [])
    text = json.dumps(messages, ensure_ascii=False) + json.dumps(payload.get('tools', []), ensure_ascii=False)
    output = payload.get('max_tokens') or payload.get('max_completion_tokens') or 2048
    visuals = sum(1 for message in messages if isinstance(message.get('content'), list)
                  for item in message['content'] if isinstance(item, dict) and item.get('type') in ('image_url','video_url','input_video'))
    return len(text.encode('utf-8')) // BYTES_PER_TOKEN_ESTIMATE + min(int(output), KV_OUTPUT_RESERVE) + visuals * 16384

def vllm_kv_capacity(metrics_text):
    """(KV cache tokens, per-request overhead) from a vLLM /metrics page, or None.
    Each request rounds up to whole blocks and hybrid models keep recurrent
    state in blocks too, so two blocks per request are set aside."""
    line = next((l for l in metrics_text.splitlines() if l.startswith('vllm:cache_config_info{')), '')
    size = re.search(r'kv_cache_size_tokens="(\d+)"', line)
    block = re.search(r'block_size="(\d+)"', line)
    if not size:
        return None
    return int(size.group(1)), 2 * int(block.group(1)) if block else 0

def splash_capacity(status):
    """(batch width, KV cache tokens) from a Splash /status reply, or None."""
    budget = ((status or {}).get('memory_plan') or {}).get('budget') or {}
    width, tokens = budget.get('maximum_batch_width'), budget.get('kv_capacity_tokens')
    if not isinstance(width, int) or not isinstance(tokens, int) or width < 1 or tokens < 1:
        return None
    return width, tokens

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


class DisconnectWatch:
    """Outermost ASGI wrapper: once a request body is read, watch the connection and set
    scope['bubble.disconnected'] when the client hangs up. Request.is_disconnected() cannot
    see this behind BaseHTTPMiddleware (the @app.middleware guard)."""

    def __init__(self, app): self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http': return await self.app(scope, receive, send)
        gone = scope['bubble.disconnected'] = asyncio.Event()
        watcher = None

        async def watch():
            while (await receive())['type'] != 'http.disconnect': pass
            gone.set()

        async def wrapped():
            nonlocal watcher
            if watcher is not None:
                await gone.wait(); return {'type': 'http.disconnect'}
            message = await receive()
            if message['type'] == 'http.disconnect': gone.set()
            elif not message.get('more_body'): watcher = asyncio.ensure_future(watch())
            return message

        try: await self.app(scope, wrapped, send)
        finally:
            if watcher is not None: watcher.cancel()

def create_app(root: Path = ROOT, *, injected=None):
    @asynccontextmanager
    async def lifespan(app):
        if injected:
            app.state.config, app.state.manager, app.state.scheduler, app.state.scanner, app.state.media = injected
        else:
            config = apply_serving_mode(json.loads((root/'config/control-plane.json').read_text()))
            backends = {name: BackendConfig(**backend_profile(profile)) for name, profile in config['backends'].items()}
            idol_resource = IdolLocalBackend.resource_config(root)
            if idol_resource is not None:
                backends[IdolLocalBackend.RESOURCE] = idol_resource
            app.state.config = config
            app.state.manager = ResourceManager(backends, reserve_gib=config.get('reserve_gib', 12), ownership_path=root/'state/owned-processes.json')
            shelf = {**SHELF_DEFAULTS, **config.get('shelf', {})}
            app.state.scheduler = Scheduler(root/'state/jobs.sqlite3', app.state.manager,
                                            shelf_grace_seconds=shelf['grace_seconds'],
                                            shelf_max_hold_seconds=shelf['max_hold_seconds'])
            app.state.scheduler.resource_slots = resource_slots(config['backends'])
            paths = json.loads((root/'config/model-paths.json').read_text())
            # Optional fine-tuned checkpoints: laya-security screens, laya-route routes.
            security_path = paths.get('laya-security', paths['convaiinnovations/laya'])['path']
            route_path = paths.get('laya-route', {}).get('path')
            app.state.scanner = SecurityScanner(security_path, config.get('security_policy', {'max_input_chars':393216}), route_path=route_path)
            # route-ettin reads long requests in full (Laya sees 1,600 chars); Laya stays the fallback.
            ettin = paths.get('route-ettin', {})
            app.state.route_classifier = EttinRouter(ettin['path'], ettin.get('max_tokens', 4096)) if ettin.get('path') else None
            app.state.media = MediaBackend(app.state.manager, paths, root)
        app.state.music = MusicBackend(app.state.manager, root)
        app.state.idol_local = IdolLocalBackend(root, app.state.manager)
        app.state.soft_decisions = SoftDecisions(root,app.state.config.get('soft_decisions'))
        app.state.scheduler.failure_advisor = app.state.soft_decisions.failed_generation
        app.state.route_affinity = {'target': None, 'count': 0}
        app.state.planning_classifier = None
        planning = app.state.config.get('planning_router', {})
        if planning.get('enabled'):
            try:
                app.state.planning_classifier = PlanningClassifier(planning['checkpoint'])
            except Exception:
                # A missing or invalid specialized checkpoint never weakens
                # the existing routing or security classifier.
                pass
        app.state.metrics = {'requests': 0, 'backend_requests': 0, 'security_blocks': 0, 'security_active':0, 'started': time.time()}
        app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=5), trust_env=False)
        await app.state.scheduler.start()
        if hasattr(app.state.manager,'recover_owned_orphans'): await app.state.manager.recover_owned_orphans()
        if app.state.config.get('start_drained'): await app.state.scheduler.drain()
        if hasattr(app.state.scanner, 'load'): await asyncio.to_thread(app.state.scanner.load)
        if getattr(app.state, 'route_classifier', None) is not None:
            try:
                await asyncio.to_thread(app.state.route_classifier.load)
            except Exception:
                app.state.route_classifier = None  # Laya keeps routing
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
    app.state.openclaw_identity = OpenClawIdentity()

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
            admin_action = request.url.path.startswith('/router/control') or (request.method == 'DELETE' and request.url.path.startswith('/v1/jobs/'))
            if admin_action and principal!='admin':
                return JSONResponse({'error':'administrator credential required'},status_code=403)
            if admin_action and request.client and request.client.host not in ('127.0.0.1', '::1', 'testclient'):
                return JSONResponse({'error': 'control API is loopback only'}, status_code=403)
        try: return await call_next(request)
        except ValueError as exc: return JSONResponse({'error': str(exc)}, status_code=422)

    def metadata(request):
        meta = {'authenticated':True,'principal':request.state.principal,
                'trusted_system':request.state.principal=='openclaw',
                **{key: request.headers.get('x-'+key, '')[:120] for key in ('agent-id', 'session-id', 'source')}}
        # OpenClaw names the agent and channel only through its session id (openclaw_identity).
        session = request.headers.get('x-session-affinity') or request.headers.get('session_id')
        if meta['trusted_system'] and session and not meta['agent-id']:
            meta.update(app.state.openclaw_identity.lookup(session) or {'session-id': meta['session-id'] or session[:120]})
        return meta

    def job_meta(request, markers=None):
        """Job metadata: the caller's headers plus its display origin."""
        meta = metadata(request)
        return {**meta, **job_origin(meta, markers or {}, shelf_policy())}

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

    async def until_done_or_gone(request, future, job_id, poll=1.0):
        """Await a job, cancelling it if the client hangs up first: uvicorn does not cancel a
        non-streaming handler on disconnect, so an abandoned request would otherwise still run."""
        shielded = asyncio.shield(future)
        while True:
            done, _ = await asyncio.wait({shielded}, timeout=poll)
            if done: return shielded.result()
            if request.scope['bubble.disconnected'].is_set():
                await app.state.scheduler.cancel(job_id); raise asyncio.CancelledError('client disconnected')

    async def wait_job(request, resource, priority, operation, prompt=None):
        job = await app.state.scheduler.submit(resource, priority, operation, job_meta(request), prompt=prompt)
        try: return await until_done_or_gone(request, job.future, job.id)
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
        names = dict.fromkeys(('auto', *chat_targets(), 'speridlabs/idol',
                              'qwen-asr','qwen-tts','qwen-image','minimax-h3',
                              'acestep-v15-xl-turbo','acestep-v15-turbo'))
        return {'object':'list','data':[{'id': name,'object':'model',
                                         'owned_by':'local-mps' if name == 'speridlabs/idol' else 'local'}
                                        for name in names]}

    @app.get('/router/status')
    async def status():
        return {'resources': await app.state.manager.status(), 'queues': await app.state.scheduler.queues(), 'security': plain(app.state.scanner.status()),
                'planning_router': {'enabled':bool(app.state.config.get('planning_router',{}).get('enabled')),
                    'available':app.state.planning_classifier is not None,'scope':'simple-draft-only'},
                'soft_decisions':{kind:{k:v for k,v in rule.items() if k in {'enabled','sample_rate','cooldown_seconds','threshold','targets'}}
                    for kind,rule in app.state.soft_decisions.policy.items()},
                'queue_paused':not app.state.scheduler._accepting,
                'shelf':app.state.scheduler.shelf_status(),
                'serving_mode':app.state.config.get('serving_mode','mlx'),
                'resource_slots':getattr(app.state.scheduler,'resource_slots',{}),
                'resource_kv_tokens':{name:{'tokens':tokens,'per_request':extra} for name,(tokens,extra)
                    in getattr(app.state.scheduler,'resource_kv_tokens',{}).items()},
                'pinned_protection':{'router':'in-process','laya':'in-process','openclaw':'observed only; never managed','tailscale':'observed only; never managed'},
                'future_profiles':app.state.config.get('future_profiles',{})}

    @app.get('/router/resources')
    async def resources(): return await app.state.manager.status()
    @app.get('/router/queues')
    async def queues(): return {'paused':not app.state.scheduler._accepting,'shelf':app.state.scheduler.shelf_status(),
                                'resources':await app.state.scheduler.queues()}
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

    def simple_policy():
        return {**SIMPLE_ROUTE_DEFAULTS, **app.state.config.get('simple_route', {})}

    def shelf_policy():
        return {**SHELF_DEFAULTS, **app.state.config.get('shelf', {})}

    def chat_lane(meta):
        """manual: a person is waiting; background: scheduled agent work that
        manual traffic shelves. Other tagged callers are neither."""
        policy = shelf_policy()
        source = meta.get('source', '')
        if source in policy['background_sources']: return 'background'
        if source in policy['manual_sources']: return 'manual'
        return None

    def keep_loaded(name, reason, visual):
        """A person is waiting: answer on the loaded chat model when it is a
        capable tier rather than waiting for a 19-22 GB reload."""
        targets = chat_targets()
        is_loaded = getattr(app.state.manager, 'is_loaded', None)
        if not is_loaded or is_loaded(targets[name]['resource']): return name, reason
        loaded = [n for n in shelf_policy()['prefer_loaded'] if n in targets and not targets[n].get('planning_only') and is_loaded(targets[n]['resource'])]
        if not loaded or (visual and not targets[loaded[0]].get('vision')): return name, reason
        return loaded[0], f'{reason};manual-kept-loaded:{loaded[0]}'

    def gate_simple(name, reason, messages, text, scores, tools):
        """Send a request routed to the fast tier up to Swift unless it is simple."""
        policy = simple_policy()
        if name != policy['model']: return name, reason
        issue = simple_request_issue(messages, text, scores, tools, policy)
        if issue is None: return name, reason
        app.state.metrics['not_simple'] = app.state.metrics.get('not_simple', 0) + 1
        return route_target('swift'), f'{reason};not-simple:{issue}'

    async def choose_chat_model(requested, messages, visual, tools=None):
        """Explicit model names win; auto and legacy aliases are routed by Laya:
        very simple general and tool work to Ornith (every other general request
        goes to Swift), deep agentic work, simple coding, subtle bugs and costly
        plans to Swift, and long-horizon reasoning and complex coding to the
        reasoner tier."""
        targets = chat_targets()
        requested = str(requested or 'auto').split('/')[-1]
        if requested in targets and not targets[requested].get('planning_only'): return requested, 'requested'
        if len(targets) == 1: return next(iter(targets)), 'only-model'
        # Long, complex coding goes to the reasoner tier (AREX) as soon as the
        # agent run shows it, without waiting for the sticky switch: Laya sees
        # only the person's message, which stays the same for the whole run.
        rule = {'min_tool_calls': 6, 'min_code_calls': 3, **app.state.config.get('coding_escalation', {})}
        calls, code = coding_run(messages)
        if calls >= rule['min_tool_calls'] and code >= rule['min_code_calls']:
            app.state.route_affinity.update(target=None, count=0)
            return route_target('reasoner'), f'coding-run:{code}/{calls}'
        text = latest_user_text(messages)
        ettin = getattr(app.state, 'route_classifier', None)
        if not text or (ettin is None and not hasattr(app.state.scanner, 'route_request')):
            if visual: return vision_fallback(), 'vision'
            return sticky_choice(*gate_simple(route_target('general'), 'default', messages, text, {}, tools), visual)
        router, decision = 'laya', None
        if ettin is not None:
            try:
                decision, router = await asyncio.to_thread(ettin.route_request, text), 'ettin'
            except Exception:
                app.state.metrics['route_ettin_errors'] = app.state.metrics.get('route_ettin_errors', 0) + 1
        if decision is None:
            decision = await asyncio.to_thread(app.state.scanner.route_request, text)
        app.state.metrics[f"route_{decision['route']}"] = app.state.metrics.get(f"route_{decision['route']}", 0) + 1
        entry = {'time': time.time(), 'router': router, 'route': decision['route'], 'scores': decision['scores'],
                 'chars': len(text), 'sha256': hashlib.sha256(text.encode()).hexdigest()[:16]}
        if os.environ.get('BUBBLE_ROUTE_DEBUG') == '1':
            entry['preview'] = text[:160]  # debugging only; never enabled by default
        fd = os.open(root/'state/route-audit.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, 'a') as handle: handle.write(json.dumps(entry)+'\n')
        score = decision['scores'].get(decision['route'], 0.0)
        name = route_target(decision['route'])
        planning = app.state.config.get('planning_router', {})
        explicit_planner = requested in targets and targets[requested].get('planning_only')
        if planning.get('enabled') and (planning_candidate(text) or explicit_planner):
            issue = planning_issue(messages, text, visual)
            classifier = app.state.planning_classifier
            prediction = {}
            if issue is None:
                if classifier is None: issue = 'classifier-unavailable'
                else:
                    try:
                        prediction = await asyncio.to_thread(classifier.predict, text)
                        if not prediction.get('eligible'): issue = 'classifier-rejected'
                    except Exception:
                        issue = 'classifier-error'
            planner = planning.get('model', 'thinkingcap')
            if issue is None and targets.get(planner,{}).get('planning_only'):
                app.state.route_affinity.update(target=None,count=0)
                return planner, f"planning:{prediction.get('score',0)}"
            if explicit_planner:
                raise HTTPException(422, 'ThinkingCap is restricted to simple draft-only planning')
            # Complex planning and uncertain eligibility use the stronger
            # route. Never fall back from a rejected plan to Ornith.
            name = name if decision['route'] == 'reasoner' else route_target('swift')
            app.state.route_affinity.update(target=None,count=0)
            if visual and not targets[name].get('vision'):name=vision_fallback()
            return name, f"{router}:{decision['route']}:{score};planning-fallback:{issue or 'target-unavailable'}"
        if explicit_planner:
            raise HTTPException(422, 'Simple-planning routing is unavailable')
        if visual and not targets[name].get('vision'):
            return vision_fallback(), f"{router}:{decision['route']}:{score};vision"
        name, reason = gate_simple(name, f"{router}:{decision['route']}:{score}", messages, text, decision['scores'], tools)
        return sticky_choice(name, reason, visual)

    def sticky_choice(preferred, reason, visual):
        """Keep using the chat model that is already loaded. Switching costs a
        19-22 GB reload, so auto requests move only after route_switch_after
        consecutive requests prefer the same other model."""
        targets = chat_targets()
        is_loaded = getattr(app.state.manager, 'is_loaded', None)
        loaded = [name for name, target in targets.items() if is_loaded and is_loaded(target['resource'])]
        affinity = app.state.route_affinity
        current = loaded[0] if loaded else None
        if current is not None and targets[current].get('planning_only') and current != preferred:
            affinity.update(target=None,count=0)
            return preferred, reason + ';leave-planning-only'
        if current is None or preferred in loaded or (visual and not targets[current].get('vision')):
            affinity.update(target=None, count=0)
            return preferred, reason
        affinity['count'] = affinity['count'] + 1 if affinity['target'] == preferred else 1
        affinity['target'] = preferred
        threshold = max(1, int(app.state.config.get('route_switch_after', 3)))
        if preferred == simple_policy()['model']:
            # Moving down to the fast tier is never needed for quality, so a
            # reload waits for repeated simple requests; moving up stays prompt.
            threshold = max(threshold, int(simple_policy()['switch_after']))
        if affinity['count'] >= threshold:
            affinity.update(target=None, count=0)
            return preferred, f'{reason};switch:after-{threshold}'
        return current, f"{reason};sticky:{current}:{affinity['count']}/{threshold}"

    def vision_fallback():
        targets = chat_targets()
        name = app.state.config.get('vision_default', 'qwen')
        return name if targets.get(name, {}).get('vision') else 'qwen'

    async def read_kv_capacity(resource, url, headers):
        """Learn a vLLM server's KV size once it is up; until then its requests
        run one at a time. A failed read is retried by the next request."""
        try:
            response = await app.state.http.get(url + '/metrics', headers=headers, timeout=5)
            capacity = vllm_kv_capacity(response.text) if response.status_code == 200 else None
        except httpx.HTTPError:
            capacity = None
        if capacity:
            app.state.scheduler.set_kv_capacity(resource, *capacity)

    @asynccontextmanager
    async def leased(job, resource, **kwargs):
        """Hold a model lease. The job is stamped ready_at once the model is loaded, so the
        dashboard's run time and tok/s leave out loading (and waiting to unload another model)."""
        async with app.state.manager.lease(resource, **kwargs):
            app.state.scheduler.mark_ready(job)
            yield

    async def read_splash_capacity(resource, url, headers):
        """Learn a Splash server's batch width and KV size once it is up, so its
        requests run side by side the way vLLM's do. A failed read is retried."""
        try:
            response = await app.state.http.get(url + '/status', headers=headers, timeout=5)
            capacity = splash_capacity(response.json()) if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            capacity = None
        if capacity:
            app.state.scheduler.resource_slots[resource] = capacity[0]
            app.state.scheduler.set_kv_capacity(resource, capacity[1])

    async def dispatch_chat(name, payload, visual, meta, lane=None):
        """Queue one chat request on a model; returns (job, ready future, consumed event)."""
        target = chat_targets()[name]
        payload = apply_sampling_defaults({**payload, 'model': target['model']}, target)
        if target.get('planning_only'):
            payload = {key:value for key,value in payload.items() if key not in
                       ('tools','tool_choice','parallel_tool_calls','response_format','chat_template_kwargs','reasoning_effort','thinking_budget','stream_options')}
            payload.update(stream=False, max_tokens=min(int(payload.get('max_tokens',2048)),2048))
        engine = app.state.config.get('backends', {}).get(target['resource'], {}).get('engine')
        vllm = engine == 'vllm'
        # Splash speaks vLLM's OpenAI dialect, with a few differences.
        if engine in ('vllm', 'splash'):
            payload = vllm_chat_payload(payload, target.get('think_by_default', False), target.get('default_reasoning_effort'))
        if engine == 'splash': payload = splash_chat_payload(payload, target.get('thinking_budget'))
        # vLLM and Splash reserve their KV cache and scratch space when they
        # load (estimate_gib covers it), so a request adds no memory on top.
        incremental = 0.0 if engine in ('vllm', 'splash') else chat_incremental_gib(payload, target.get('kv_bytes_per_token', 65536))
        # kv_tokens sizes side-by-side admission (scheduler._fits_alongside); the dashboard shows it as the prompt size.
        if engine in ('vllm', 'splash'): meta = {**meta, 'kv_tokens': chat_kv_tokens(payload)}
        budget = thinking_cut_budget(payload, target.get('thinking_budget')) if vllm else None
        ready = asyncio.get_running_loop().create_future()
        consumed = asyncio.Event()
        async def operation(job):
            response = None
            try:
                async with leased(job, target['resource'], incremental_gib=incremental):
                    app.state.metrics['backend_requests'] += 1
                    headers = {'authorization':'Bearer '+app.state.config.get('backend_key','')}
                    if vllm and target['resource'] not in app.state.scheduler.resource_kv_tokens:
                        await read_kv_capacity(target['resource'], target['url'], headers)
                    if engine == 'splash' and target['resource'] not in app.state.scheduler.resource_kv_tokens:
                        await read_splash_capacity(target['resource'], target['url'], headers)
                    if budget is not None:
                        result = await budgeted_chat(app.state.http, target['url'], payload, headers, budget)
                        if result.pop('bubble_thinking_cut', 0):
                            app.state.metrics['thinking_cuts'] = app.state.metrics.get('thinking_cuts', 0) + 1
                        result['model'] = name; ready.set_result(result); return {'completed': True, 'usage': result.get('usage')}
                    if payload.get('stream'):
                        response = await app.state.http.send(app.state.http.build_request('POST',target['url']+'/v1/chat/completions',json=payload,headers=headers), stream=True)
                        response.raise_for_status()
                        ready.set_result(response)
                        await consumed.wait()
                        return {'stream':'consumed', 'usage': getattr(consumed, 'usage', None)}
                    response = await app.state.http.post(target['url']+'/v1/chat/completions',json=payload,headers=headers)
                    response.raise_for_status()
                    result = response.json(); result['model']=name; ready.set_result(result); return {'completed':True, 'usage': result.get('usage')}
            except BaseException as exc:
                if not ready.done(): ready.set_exception(exc)
                raise
            finally:
                if response is not None: await response.aclose()
        job = await app.state.scheduler.submit(target['resource'], 2 if visual else 1, operation, meta, lane=lane)
        # Consume exceptions on the durable job future; the endpoint awaits ready.
        job.future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        return job, ready, consumed

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        original = await request.json()
        # As the caller sent them: the scan may rewrite text, and loop and cron
        # turns name themselves in their opening user message.
        sent = original.get('messages')
        original['messages'], report = await scan(original.get('messages', []), metadata(request))
        payload, visual = prepare_chat(original, root)
        requested_context = payload.pop('_context_length', None)
        name, reason = await choose_chat_model(original.get('model'), original['messages'], visual, payload.get('tools'))
        # The draft-only backend cannot satisfy a caller's structured-output
        # or mandatory-tool contract. Preserve it on the stronger backend.
        if chat_targets()[name].get('planning_only') and (payload.get('response_format')
                or payload.get('tool_choice') not in (None,'none','auto')):
            requested = str(original.get('model') or 'auto').split('/')[-1]
            if requested == name:raise HTTPException(422,'ThinkingCap supports draft text only')
            name,reason = route_target('swift'), reason + ';planning-fallback:output-contract'
        meta = metadata(request)
        lane = chat_lane(meta)
        meta = job_meta(request, turn_markers(sent, search=meta['source'] == 'loop'))
        # Explicit models and long coding runs keep their choice.
        if lane == 'manual' and reason not in ('requested', 'only-model') and not reason.startswith(('coding-run','planning:')) and ';planning-fallback:' not in reason:
            name, reason = keep_loaded(name, reason, visual)
        targets = chat_targets()
        requested_model = str(original.get('model') or 'auto').split('/')[-1]
        explicit_model = requested_model != 'auto' and requested_model in targets
        context = requested_context or targets[name].get('context', 32768)
        if not fits_context(payload, context):
            if explicit_model:
                raise HTTPException(422, 'Text exceeds the requested model context budget')
            # Long conversations go to the long-context tier when the routed
            # model's working window is smaller.
            longest = max((n for n in targets if not targets[n].get('planning_only')), key=lambda n: targets[n].get('context', 32768))
            if requested_context is None and (not visual or targets[longest].get('vision')) \
                    and fits_context(payload, targets[longest].get('context', 32768)):
                name, reason = longest, reason + ';long-context'
            else:
                raise HTTPException(422, 'Text exceeds the model context budget')
        payload = apply_tool_reasoning_defaults(payload, {'name':name, **targets[name]})
        attempts = [name]
        # An explicit model request is a strict user choice. Automatic
        # routing keeps the configured escalation behavior unchanged.
        if not explicit_model:
            escalate_to = app.state.config.get('escalate', {}).get(name)
            if escalate_to in chat_targets(): attempts.append(escalate_to)
        # Buffer fresh tool-enabled replies so leaked calls or invented native
        # results can retry on the configured fallback before any text is sent.
        # After tool results, result summaries and paths are normal answers.
        check_tools = not explicit_model and bool(payload.get('tools')) and \
            (payload['messages'] or [{}])[-1].get('role') == 'user'
        weak_targets = app.state.config.get('soft_decisions',{}).get('weak_answers',{}).get('targets',{'ornith':'swift'})
        weak_target = weak_targets.get(name)
        user_text = latest_user_text(original['messages'])
        weak_candidate = (str(original.get('model','auto')).split('/')[-1] not in targets
            and weak_target in targets and not visual and not payload.get('tools') and not payload.get('response_format')
            and (payload['messages'] or [{}])[-1].get('role')=='user' and len(user_text)>=30
            and app.state.soft_decisions.selected('weak_answers',meta.get('session-id','')+user_text))
        original_answer = None
        for index, current in enumerate(attempts):
            planning_reply = targets[current].get('planning_only',False)
            buffered = check_tools or weak_candidate or planning_reply
            attempt = {k: v for k, v in payload.items() if k != 'stream_options'} | {'stream': False} if buffered else payload
            job, ready, consumed = await dispatch_chat(current, attempt, visual, meta, lane)
            try:
                result = await until_done_or_gone(request, ready, job.id)
                invalid_tool_reply = (unexecuted_tool_output(result, payload['tools']) or
                    ('printed-command' if printed_instead_of_calling(result) else None)) if check_tools else None
                if planning_reply:
                    choice=(result.get('choices') or [{}])[0];message=choice.get('message') or {}
                    if (choice.get('finish_reason') not in (None,'stop') or message.get('tool_calls')
                            or not isinstance(message.get('content'),str) or not message['content'].strip()
                            or printed_instead_of_calling(result)
                            or unexecuted_tool_output(result,payload.get('tools',[]))):
                        invalid_tool_reply = 'invalid-planning-reply'
                if invalid_tool_reply:
                    if index + 1 < len(attempts):
                        app.state.metrics['escalations'] = app.state.metrics.get('escalations', 0) + 1
                        reason += f';escalated-from-{current}:{invalid_tool_reply}'
                        continue
                    app.state.metrics['invalid_tool_replies'] = app.state.metrics.get('invalid_tool_replies', 0) + 1
                    raise HTTPException(502, detail={'code':'unexecuted_tool_output', 'model':current,
                        'job_id':job.id, 'error':'Model printed a tool call or result without executing a tool; no action was performed.'})
                if weak_candidate and index==0:
                    decision = await app.state.soft_decisions.weak_answer(user_text,result)
                    app.state.metrics['weak_answer_checks'] = app.state.metrics.get('weak_answer_checks',0)+1
                    if decision['action']=='escalate' and fits_context(payload,targets[weak_target].get('context',32768)):
                        original_answer = (result,job,current)
                        if len(attempts)==1: attempts.append(weak_target)
                        app.state.metrics['escalations'] = app.state.metrics.get('escalations',0)+1
                        reason += f';weak-answer-from-{current}'
                        continue
                if buffered and payload.get('stream'):
                    sse = completion_as_sse(result)
                    headers = {'X-Bubble-Job-ID': job.id, 'X-Bubble-Security': str(plain(report).get('action', 'allow')),
                               'X-Bubble-Route': f'{current};{reason}'}
                    return Response(sse, media_type='text/event-stream', headers=headers)
                break
            except asyncio.CancelledError:
                await app.state.scheduler.cancel(job.id); raise
            except HTTPException:
                raise
            except Exception:
                if original_answer:
                    result,job,current = original_answer
                    reason += ';kept-original:escalation-unavailable'
                    headers = {'X-Bubble-Job-ID':job.id,'X-Bubble-Security':str(plain(report).get('action','allow')),
                               'X-Bubble-Route':f'{current};{reason}'}
                    return Response(completion_as_sse(result),media_type='text/event-stream',headers=headers) if payload.get('stream') else JSONResponse(result,headers=headers)
                if index + 1 < len(attempts):
                    app.state.metrics['escalations'] = app.state.metrics.get('escalations', 0) + 1
                    reason += f';escalated-from-{current}'
                    continue
                raise HTTPException(503,detail={'job_id':job.id,'error':f'{current} backend unavailable'})
        name = current
        headers = {'X-Bubble-Job-ID':job.id, 'X-Bubble-Security': str(plain(report).get('action','allow')),
                   'X-Bubble-Route': f'{name};{reason}'}
        if not payload.get('stream'): return JSONResponse(result,headers=headers)
        # A thinking-budgeted vLLM reply arrives complete; replay it as a stream.
        if isinstance(result, dict): return Response(completion_as_sse(result),media_type='text/event-stream',headers=headers)
        async def chunks():
            tail = b''
            try:
                async for chunk in result.aiter_bytes():
                    tail = (tail + chunk)[-8192:]
                    yield chunk
            finally:
                # The job records the reply's token usage from its last stream event.
                consumed.usage = stream_usage(tail); consumed.set()
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
        original_prompt = payload.get('input', '')
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
            async with leased(job, 'qwen-audio'):
                app.state.metrics['backend_requests'] += 1
                response = await app.state.http.post('http://127.0.0.1:8765/v1/audio/speech',json=payload)
                response.raise_for_status()
                output = root/'outputs/audio'/f'{job.id}.{payload["response_format"]}'
                output.write_bytes(response.content)
                return {'artifact':str(output),'content_type':response.headers.get('content-type','audio/wav')}
        result = await wait_job(request,'qwen-audio',3 if request.headers.get('x-job-background')=='true' else 1,operation,prompt=original_prompt)
        return FileResponse(result['artifact'],media_type=result['content_type'])

    @app.post('/v1/audio/transcriptions')
    async def transcription(request:Request,file:UploadFile=File(...),model:str=Form('qwen-asr'),language:str=Form('English')):
        data = await file.read(100*1024**2+1)
        if len(data)>100*1024**2: raise ValueError('Audio exceeds 100 MiB')
        async def operation(job):
            async with leased(job, 'qwen-audio'):
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
            async with leased(job, 'qwen-audio'):
                results=[]
                for index,(audio,mime) in enumerate(data):
                    app.state.metrics['backend_requests']+=1
                    response=await app.state.http.post('http://127.0.0.1:8765/v1/audio/transcriptions',files={'file':('input.wav',audio,mime)},data={'model':app.state.config['audio_models']['asr'],'language':language,'response_format':'json'})
                    response.raise_for_status();results.append({'index':index,**response.json()})
                return {'results':results}
        return await wait_job(request,'qwen-audio',3 if request.headers.get('x-job-background')=='true' else 1,operation)

    async def media_payload(request, file_fields):
        """JSON body, or a form whose image file fields are saved under state/uploads."""
        if not request.headers.get('content-type','').startswith(('multipart/form-data','application/x-www-form-urlencoded')):
            payload = await request.json()
            if not isinstance(payload, dict): raise ValueError('Request body must be a JSON object')
            return payload, []
        form = await request.form(max_files=len(file_fields), max_fields=32)
        payload, uploads = {}, []
        try:
            for key, value in form.multi_items():
                if key in payload: raise ValueError(f'{key} was given more than once')
                if isinstance(value, str):
                    if key in MEDIA_INT_FIELDS:
                        try: payload[key] = int(value)
                        except ValueError: raise ValueError(f'{key} must be an integer') from None
                    elif key in MEDIA_FLOAT_FIELDS:
                        try: payload[key] = float(value)
                        except ValueError: raise ValueError(f'{key} must be a number') from None
                    else: payload[key] = value
                    continue
                if key not in file_fields: raise ValueError(f'File upload is only accepted for: {", ".join(file_fields)}')
                data = await value.read(MEDIA_UPLOAD_MAX_BYTES + 1)
                if len(data) > MEDIA_UPLOAD_MAX_BYTES: raise ValueError(f'{key} exceeds 20 MiB')
                kind = image_kind(data[:16])
                if kind is None: raise ValueError(f'{key} must be a PNG, JPEG or WebP image')
                folder = root/'state/uploads'; folder.mkdir(parents=True, exist_ok=True)
                destination = folder/f'upload-{secrets.token_hex(16)}.{kind}'
                descriptor = os.open(destination, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
                with os.fdopen(descriptor, 'wb') as handle: handle.write(data)
                uploads.append(destination); payload[key] = str(destination)
        except BaseException:
            for upload in uploads: upload.unlink(missing_ok=True)
            raise
        finally:
            await form.close()
        return payload, uploads

    async def submit_media(request, resource, method, operation_name, file_fields=()):
        paths=json.loads((root/'config/model-paths.json').read_text())
        required=(('abenzerps/Qwen-Image-2.1-Uncensored-GGUF',) if resource=='qwen-image' else ('MiniMaxAI/MiniMax-H3',))
        if any(repository not in paths for repository in required): raise HTTPException(503,'Selected model download is not complete')
        payload, uploads = await media_payload(request, file_fields)
        try:
            return await queue_media(request, resource, method, operation_name, payload, uploads)
        except BaseException:
            for upload in uploads: upload.unlink(missing_ok=True)
            raise

    async def queue_media(request, resource, method, operation_name, payload, uploads):
        app.state.media.preflight(operation_name, payload)
        original_prompt = payload.get('prompt', '')
        h3_incremental = 0.0
        job_metadata = job_meta(request)
        if resource == 'minimax-h3':
            _, h3_incremental, peak = h3_video_admission(app.state.config, payload)
            job_metadata['estimated_peak_gib'] = peak
        messages=[{'role':'user','content':payload.get('prompt','')}]
        if payload.get('negative_prompt'): messages.append({'role':'user','content':payload['negative_prompt']})
        filtered, _ = await scan(messages, metadata(request))
        payload['prompt'] = filtered[0]['content']
        if len(filtered)>1: payload['negative_prompt']=filtered[1]['content']
        async def operation(job):
            lease = (leased(job, resource, incremental_gib=h3_incremental)
                     if resource == 'minimax-h3' and h3_incremental > 0
                     else leased(job, resource))
            try:
                async with lease:
                    app.state.metrics['backend_requests'] += 1
                    return await method(payload,job.id)
            finally:
                # Uploaded inputs are single-use; outputs live in outputs/.
                for upload in uploads: upload.unlink(missing_ok=True)
        job = await app.state.scheduler.submit(resource,4 if resource=='qwen-image' else 5,operation,job_metadata,prompt=original_prompt)
        job.future.add_done_callback(lambda f:f.exception() if not f.cancelled() else None)
        return JSONResponse({'id':job.id,'status':'queued','url':f'/v1/jobs/{job.id}'},status_code=202)
    @app.post('/v1/images/generations')
    async def image(request:Request): return await submit_media(request,'qwen-image',app.state.media.generate_image,'image')
    @app.post('/v1/images/edits')
    async def edit(request:Request): return await submit_media(request,'qwen-image',app.state.media.edit_image,'edit',('image',))
    @app.post('/v1/videos/generations')
    async def video(request:Request): return await submit_media(request,'minimax-h3',app.state.media.generate_video,'video',('first_frame','last_frame','initial_image'))
    @app.post('/v1/3d/generations')
    async def idol_generation(request: Request):
        payload, uploads = await media_payload(request, ('image',))
        accepted = False

        def cleanup_uploads():
            for upload in uploads:
                upload.unlink(missing_ok=True)

        try:
            model = payload.get('model', 'speridlabs/idol')
            if model not in ('speridlabs/idol', 'idol'):
                raise ValueError('model must be speridlabs/idol')
            image_value = payload.get('image')
            if not isinstance(image_value, str) or not image_value.strip():
                raise ValueError('image is required as a multipart upload or approved local image path')
            try:
                image_path = safe_asset(image_value, root)
            except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError):
                raise ValueError('image must be a PNG, JPEG or WebP image in an approved asset directory') from None
            with image_path.open('rb') as handle:
                image_header = handle.read(16)
            if image_kind(image_header) is None:
                raise ValueError('image must be a PNG, JPEG or WebP image')

            mode = payload.get('mode', 'synthetic')
            if mode not in ('synthetic', 'real'):
                raise ValueError("mode must be 'synthetic' or 'real'")

            def integer(name, default, minimum, maximum):
                value = payload.get(name, default)
                if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                    raise ValueError(f'{name} must be an integer between {minimum} and {maximum}')
                return value

            def number(name, default, minimum, maximum):
                value = payload.get(name, default)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f'{name} must be a finite number between {minimum} and {maximum}')
                value = float(value)
                if not math.isfinite(value) or not minimum <= value <= maximum:
                    raise ValueError(f'{name} must be a finite number between {minimum} and {maximum}')
                return value

            steps = integer('steps', 50, 10, 100)
            seed = integer('seed', 42, 0, 2**31 - 1)
            num_gaussians = integer('num_gaussians', 262144, 65536, 1048576)
            if num_gaussians not in (65536, 131072, 262144, 524288, 1048576):
                raise ValueError('num_gaussians must be one of 65536, 131072, 262144, 524288, 1048576')
            cfg_strength = number('cfg_strength', 3.0, 0.0, 10.0)
            rescale_t = number('rescale_t', 3.0, 1.0, 6.0)

            readiness = await app.state.idol_local.check_ready()
            if not readiness.get('ready'):
                missing = readiness.get('missing') or ['Local Idol is not ready.']
                raise HTTPException(503, detail={'error': 'local Idol is not ready', 'missing': missing})

            async def operation(job):
                app.state.metrics['backend_requests'] += 1
                try:
                    async with leased(job, IdolLocalBackend.RESOURCE):
                        return await app.state.idol_local.generate(
                            job.id, image_path, mode=mode, steps=steps, cfg_strength=cfg_strength,
                            rescale_t=rescale_t, seed=seed, num_gaussians=num_gaussians)
                finally:
                    cleanup_uploads()

            job = await app.state.scheduler.submit(IdolLocalBackend.RESOURCE, 2, operation, job_meta(request))
            accepted = True
            job.future.add_done_callback(lambda f: (f.exception() if not f.cancelled() else None, cleanup_uploads()))
            return JSONResponse({'id': job.id, 'status': 'queued', 'url': f'/v1/jobs/{job.id}'}, status_code=202)
        finally:
            if not accepted:
                cleanup_uploads()
    @app.post('/v1/audio/music')
    async def music(request: Request):
        body = await request.json()
        if not isinstance(body, dict): raise ValueError('JSON object required')
        if 'ace-music' not in app.state.config['backends']:
            raise HTTPException(503, 'Music model is not installed')
        payload = await asyncio.to_thread(app.state.music.validate, body)
        fields = [name for name in ('prompt','caption','lyrics','genre','mood','instruments') if payload.get(name)]
        filtered, _ = await scan([{'role':'user','content':payload[name]} for name in fields], metadata(request))
        for name, message in zip(fields, filtered): payload[name] = message['content']
        async def operation(job):
            async with leased(job, 'ace-music'):
                app.state.metrics['backend_requests'] += 1
                return await app.state.music.generate(payload, job.id)
        job = await app.state.scheduler.submit('ace-music', 3, operation, job_meta(request), prompt=body['prompt'])
        job.future.add_done_callback(lambda f:f.exception() if not f.cancelled() else None)
        return JSONResponse({'id':job.id,'status':'queued','url':f'/v1/jobs/{job.id}'},status_code=202)
    @app.get('/v1/jobs/{job_id}')
    async def get_job(job_id:str):
        job=await app.state.scheduler.get(job_id)
        if job is None: raise HTTPException(404,'Unknown job')
        return {**job_json(job), 'prompt':app.state.scheduler.get_prompt(job_id)}
    @app.delete('/v1/jobs/{job_id}')
    async def delete_job(job_id:str):
        try:
            return await app.state.scheduler.delete_media(job_id, root/'outputs')
        except KeyError:
            raise HTTPException(404,'Unknown job') from None
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from None
        except OSError:
            raise HTTPException(500,'Could not delete media file; job details retained') from None
    @app.post('/v1/jobs/{job_id}/cancel')
    async def cancel(job_id:str): return {'cancelled':await app.state.scheduler.cancel(job_id)}
    @app.get('/v1/jobs/{job_id}/original')
    async def original(job_id:str):
        return await serve_artifact(job_id, original=True)
    @app.get('/v1/jobs/{job_id}/artifact')
    async def artifact(job_id:str):
        return await serve_artifact(job_id)
    async def serve_artifact(job_id:str, original=False):
        job=await app.state.scheduler.get(job_id)
        field = 'original_artifact_path' if original else 'artifact'
        if job is None or not isinstance(job.result,dict) or not job.result.get(field): raise HTTPException(404,'No artifact')
        artifact_value=job.result[field]
        if not isinstance(artifact_value,str): raise HTTPException(403,'Invalid artifact')
        try:
            candidate=Path(artifact_value.removeprefix('file://')).expanduser().resolve(strict=False)
            output_root=(root/'outputs').resolve()
        except (OSError,RuntimeError,TypeError,ValueError):
            raise HTTPException(403,'Invalid artifact') from None
        if not candidate.is_relative_to(output_root): raise HTTPException(403,'Invalid artifact')
        try:
            path=safe_asset(artifact_value,root)
        except FileNotFoundError:
            raise HTTPException(404,'Artifact not found') from None
        except (OSError,TypeError,ValueError):
            raise HTTPException(403,'Invalid artifact') from None
        if not path.is_relative_to(output_root): raise HTTPException(403,'Invalid artifact')
        return FileResponse(path,media_type=job.result.get('original_content_type' if original else 'content_type'))
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
        if action=='shelve': return await app.state.scheduler.shelve(payload.get('seconds', shelf_policy()['grace_seconds']))
        if action=='unshelve': return await app.state.scheduler.unshelve()
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
    app.add_middleware(DisconnectWatch)
    return app

app = create_app()
