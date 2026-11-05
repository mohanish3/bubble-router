import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
import httpx
import pytest
from fastapi.testclient import TestClient
from model_router.mac.media import MediaBackend
from model_router.mac.app import apply_serving_mode, resource_slots, apply_tool_reasoning_defaults, create_app, prepare_chat, safe_asset, vllm_chat_payload, splash_chat_payload, splash_capacity, chat_kv_tokens, vllm_kv_capacity, thinking_cut_budget
from model_router.mac.app import SHELF_DEFAULTS, job_origin, turn_markers
from model_router.mac.scheduler import Scheduler
from model_router.mac.security import SecurityBlocked

class Manager:
    def __init__(self):
        self.active=0;self.calls=[];self.lease_calls=[]
        self.backends={'minimax-h3':SimpleNamespace(estimate_gib=28.0)}
    @asynccontextmanager
    async def lease(self,name,**kwargs):
        self.calls.append(name);self.lease_calls.append((name,dict(kwargs)));self.active+=1
        try:yield
        finally:self.active-=1
    async def status(self):return {'active':self.active}
    async def stop(self,name):pass

class Scanner:
    def __init__(self):self.calls=0
    def status(self):return {'ready':True,'scans':self.calls}
    async def scan_messages(self,messages,metadata):
        self.calls+=1
        assert metadata['authenticated']
        if messages[0]['content']=='attack':
            report=type('Report',(),{'to_dict':lambda self:{'action':'block'}})()
            raise SecurityBlocked(report)
        return messages,{'action':'allow'}

class LazyScheduler:
    """Create SQLite on the application's own event-loop thread."""
    def __init__(self,path,manager):self.path=path;self.manager=manager;self.instance=None
    def __getattr__(self,name):
        if self.instance is None:self.instance=Scheduler(self.path,self.manager)
        return getattr(self.instance,name)

@pytest.fixture
def running(tmp_path):
    for path in ('state','outputs/audio','test-assets'):(tmp_path/path).mkdir(parents=True,exist_ok=True)
    (tmp_path/'config').mkdir(exist_ok=True)
    (tmp_path/'config/model-paths.json').write_text(json.dumps({'MiniMaxAI/MiniMax-H3':{'path':'/not-loaded'}}))
    manager=Manager();scanner=Scanner();scheduler=LazyScheduler(tmp_path/'state/jobs.sqlite3',manager)
    config={'api_key':'test-only','admin_key':'admin-only','provider_key':'provider-only','backend_key':'backend-only','qwen_model':'/local/selected-model','backends':{'qwen-vlm':{},'minimax-h3':{'estimate_gib':28}},'h3_long_estimate_gib':40,'audio_models':{'tts':'/local/tts','asr':'/local/asr'}}
    class Media:
        preflight=MediaBackend(None,{},tmp_path).preflight
        async def generate_video(self,*args): raise AssertionError('invalid payload should not queue')
        async def edit_image(self,*args): raise AssertionError('invalid payload should not queue')
    app=create_app(tmp_path,injected=(config,manager,scheduler,scanner,Media()))
    with TestClient(app) as client:
        requests=[]
        def handler(request):
            requests.append(request)
            data=json.loads(request.content)
            assert data['model']=='/local/selected-model'
            if data.get('stream'):return httpx.Response(200,headers={'content-type':'text/event-stream'},content=b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
            return httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}],'usage':{'total_tokens':2}})
        old=app.state.http
        app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        yield client,app,manager,scanner,requests

AUTH={'Authorization':'Bearer test-only'}

def test_auth_health_and_control_boundary(running):
    client,*_=running
    assert client.get('/health').json()['ok']
    assert client.get('/router/status').status_code==401
    assert client.get('/router/status',headers=AUTH).status_code==200
    assert client.post('/router/control/load',headers=AUTH,json={'resource':'unknown'}).status_code==403
    assert client.post('/router/control/load',headers={'Authorization':'Bearer admin-only'},json={'resource':'unknown'}).status_code==422

def test_security_runs_before_backend(running):
    client,app,manager,scanner,requests=running
    response=client.post('/v1/chat/completions',headers=AUTH,json={'messages':[{'role':'user','content':'attack'}]})
    assert response.status_code==400
    assert response.json()['error']['code']=='content_policy_violation'
    assert scanner.calls==1 and requests==[] and manager.calls==[]
    good=client.post('/v1/chat/completions',headers=AUTH,json={'messages':[{'role':'user','content':'hello'}]})
    assert good.status_code==200 and good.json()['model']=='qwen'
    assert 'usage' in good.json() and len(requests)==1 and scanner.calls==2

def test_stream_is_consumed_before_lease_release(running):
    client,app,manager,*_=running
    with client.stream('POST','/v1/chat/completions',headers=AUTH,json={'messages':[{'role':'user','content':'hello'}],'stream':True}) as response:
        assert response.status_code==200
        assert b'[DONE]' in b''.join(response.iter_bytes())
    assert manager.active==0

def test_media_paths_and_native_overrides_are_restricted(tmp_path):
    (tmp_path/'test-assets').mkdir();(tmp_path/'test-assets/image.png').write_bytes(b'fake')
    outside=tmp_path/'private.txt';outside.write_text('secret')
    (tmp_path/'test-assets/link').symlink_to(outside)
    for unsafe in [str(outside),str(tmp_path/'test-assets/link'),'https://example.com/image.png']:
        with pytest.raises((ValueError,FileNotFoundError)):safe_asset(unsafe,tmp_path)
    payload={'messages':[{'role':'user','content':[{'type':'text','text':'identify'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+'A'*200000}}]}],'adapter_path':'/private/adapter','model':'remote/unapproved'}
    result,visual=prepare_chat(payload,tmp_path)
    assert visual and result['model']=='qwen' and 'adapter_path' not in result
    assert result['messages'][0]['content'][1]['image_url']['url'].endswith('A'*10)

def test_video_duration_is_validated_before_security_scan_or_queue(running):
    client,app,manager,scanner,*_=running
    invalid=client.post('/v1/videos/generations',headers=AUTH,json={'prompt':'a short scene','duration':15.01})
    assert invalid.status_code==422
    assert '15 seconds' in invalid.json()['error']
    assert scanner.calls==0 and manager.calls==[]
    conflict=client.post('/v1/videos/generations',headers=AUTH,json={'prompt':'a short scene','duration':5,'frames':120})
    assert conflict.status_code==422
    assert scanner.calls==0 and manager.calls==[]


def _submit_h3_video(client, duration):
    response=client.post('/v1/videos/generations',headers=AUTH,json={'prompt':'a short scene','duration':duration})
    assert response.status_code==202
    job_id=response.json()['id']
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        job=client.get(f'/v1/jobs/{job_id}',headers=AUTH).json()
        if job['state'] in ('succeeded','failed','cancelled'): return job
        time.sleep(.01)
    raise AssertionError('H3 injected job did not reach a terminal state')


def test_deleted_completed_artifact_is_404_and_requires_auth(running, tmp_path):
    client,app,*_=running
    missing=tmp_path/'outputs'/'deleted.mp4'
    async def complete(*args): return {'artifact':str(missing),'content_type':'video/mp4'}
    app.state.media.generate_video=complete
    job=_submit_h3_video(client,5)
    assert job['state']=='succeeded'
    assert client.get(f'/v1/jobs/{job["id"]}/artifact').status_code==401
    response=client.get(f'/v1/jobs/{job["id"]}/artifact',headers=AUTH)
    assert response.status_code==404


def test_delete_media_requires_admin_and_removes_file_job_and_prompt(running, tmp_path):
    client,app,*_=running
    output=tmp_path/'outputs'/'fixture.mp4'
    output.write_bytes(b'disposable fixture')
    async def complete(*args): return {'artifact':str(output),'content_type':'video/mp4'}
    app.state.media.generate_video=complete
    job=_submit_h3_video(client,5)
    assert client.get(f'/v1/jobs/{job["id"]}',headers=AUTH).json()['prompt']
    assert client.delete(f'/v1/jobs/{job["id"]}').status_code==401
    assert client.delete(f'/v1/jobs/{job["id"]}',headers=AUTH).status_code==403
    assert output.exists()
    response=client.delete(f'/v1/jobs/{job["id"]}',headers={'Authorization':'Bearer admin-only'})
    assert response.status_code==200
    assert response.json()['media_files_deleted']==1
    assert not output.exists()
    assert client.get(f'/v1/jobs/{job["id"]}',headers=AUTH).status_code==404
    assert client.get(f'/v1/jobs/{job["id"]}/artifact',headers=AUTH).status_code==404


def test_delete_missing_artifact_still_removes_job(running, tmp_path):
    client,app,*_=running
    async def complete(*args): return {'artifact':str(tmp_path/'outputs'/'missing.mp4'),'content_type':'video/mp4'}
    app.state.media.generate_video=complete
    job=_submit_h3_video(client,5)
    response=client.delete(f'/v1/jobs/{job["id"]}',headers={'Authorization':'Bearer admin-only'})
    assert response.status_code==200
    assert response.json()['media_files_deleted']==0


def test_completed_artifact_outside_outputs_remains_forbidden(running, tmp_path):
    client,app,*_=running
    outside=tmp_path/'test-assets'/'image.png'
    outside.write_bytes(b'fixture')
    async def complete(*args): return {'artifact':str(outside),'content_type':'image/png'}
    app.state.media.generate_video=complete
    job=_submit_h3_video(client,5)
    assert job['state']=='succeeded'
    response=client.get(f'/v1/jobs/{job["id"]}/artifact',headers=AUTH)
    assert response.status_code==403


def test_invalid_completed_artifact_remains_forbidden(running, tmp_path):
    client,app,*_=running
    invalid=tmp_path/'outputs'/'directory-not-file'
    invalid.mkdir()
    async def complete(*args): return {'artifact':str(invalid),'content_type':'video/mp4'}
    app.state.media.generate_video=complete
    job=_submit_h3_video(client,5)
    assert job['state']=='succeeded'
    response=client.get(f'/v1/jobs/{job["id"]}/artifact',headers=AUTH)
    assert response.status_code==403


def test_h3_short_video_uses_base_estimate_and_records_metadata(running):
    client,app,manager,scanner,*_=running
    async def complete(*args): return {'artifact':'fake'}
    app.state.media.generate_video=complete
    job=_submit_h3_video(client,5)
    assert job['state']=='succeeded'
    assert job['metadata']['estimated_peak_gib']==28.0
    assert manager.lease_calls==[('minimax-h3',{})]


def test_h3_long_video_requests_headroom_and_persists_long_estimate(running):
    client,app,manager,scanner,*_=running
    async def complete(*args): return {'artifact':'fake'}
    app.state.media.generate_video=complete
    job=_submit_h3_video(client,15)
    assert job['state']=='succeeded'
    assert job['metadata']['estimated_peak_gib']==40.0
    assert manager.lease_calls==[('minimax-h3',{'incremental_gib':12.0})]
    with sqlite3.connect(f'file:{app.state.scheduler.path}?mode=ro',uri=True) as db:
        persisted=db.execute('select metadata_json from jobs where id=?',(job['id'],)).fetchone()
    assert json.loads(persisted[0])['estimated_peak_gib']==40.0


@pytest.mark.parametrize(('base_estimate','long_estimate'), [
    (28, float('nan')), (float('inf'), 40), (41, 40),
])
def test_h3_admission_rejects_invalid_estimates_before_scan_or_queue(running, base_estimate, long_estimate):
    client,app,manager,scanner,*_=running
    app.state.config['backends']['minimax-h3']['estimate_gib']=base_estimate
    app.state.config['h3_long_estimate_gib']=long_estimate
    response=client.post('/v1/videos/generations',headers=AUTH,json={'prompt':'a short scene','duration':15})
    assert response.status_code==422
    assert 'estimate' in response.json()['error'] or 'finite positive number' in response.json()['error']
    assert scanner.calls==0 and manager.calls==[]
    conflict=client.post('/v1/videos/generations',headers=AUTH,json={'prompt':'a short scene','duration':5,'frames':120})
    assert conflict.status_code==422
    assert scanner.calls==0 and manager.calls==[]

def test_context_is_admitted_without_silent_truncation():
    from model_router.mac.app import fits_context
    payload,_=prepare_chat({'messages':[{'role':'user','content':'A'*5000}],'context_length':1024})
    assert payload['_context_length']==1024 and not fits_context(payload,1024)
    with pytest.raises(ValueError):prepare_chat({'messages':[{'role':'user','content':'ok'}],'context_length':999999})
    with pytest.raises(ValueError):prepare_chat({'messages':[{'role':'user','content':'ok'}],'max_tokens':100000})

@pytest.mark.parametrize('effort',['low','medium','xhigh','none','off'])
def test_reasoning_effort_is_forwarded_and_normalized(effort):
    payload,_=prepare_chat({'messages':[{'role':'user','content':'hi'}],'reasoning_effort':effort.upper()})
    assert payload['reasoning_effort']==effort

@pytest.mark.parametrize('effort',[None,True,1,{},[],'high','disabled',' '])
def test_invalid_reasoning_effort_is_rejected(effort):
    with pytest.raises(ValueError):prepare_chat({'messages':[{'role':'user','content':'hi'}],'reasoning_effort':effort})

@pytest.mark.parametrize('budget',[True,1.5,'12',{},-1,8193])
def test_invalid_thinking_budget_is_rejected(budget):
    with pytest.raises(ValueError):prepare_chat({'messages':[{'role':'user','content':'hi'}],'thinking_budget':budget})

def test_thinking_budget_is_bounded_by_completion_limit():
    payload,_=prepare_chat({'messages':[{'role':'user','content':'hi'}],'max_tokens':512,'thinking_budget':512})
    assert payload['thinking_budget']==512
    with pytest.raises(ValueError):prepare_chat({'messages':[{'role':'user','content':'hi'}],'max_tokens':512,'thinking_budget':513})

def test_arex_tool_reasoning_defaults_only_apply_to_tool_calls():
    profile={'name':'arex','resource':'arex','tool_reasoning_effort':'low','tool_thinking_budget':1024}
    plain={'messages':[{'role':'user','content':'hello'}],'max_tokens':512}
    assert apply_tool_reasoning_defaults(plain,profile)==plain
    with_tools={**plain,'tools':[{'type':'function','function':{'name':'f','parameters':{}}}]}
    result=apply_tool_reasoning_defaults(with_tools,profile)
    assert result['reasoning_effort']=='low' and result['thinking_budget']==512
    explicit={**with_tools,'reasoning_effort':'medium','thinking_budget':128}
    result=apply_tool_reasoning_defaults(explicit,profile)
    assert result['reasoning_effort']=='medium' and result['thinking_budget']==128
    disabled={**with_tools,'reasoning_effort':'off'}
    assert 'thinking_budget' not in apply_tool_reasoning_defaults(disabled,profile)
    assert 'reasoning_effort' not in apply_tool_reasoning_defaults(with_tools,{'name':'swift','resource':'swift'})

def test_arex_tool_reasoning_defaults_reach_backend(running):
    client,app,manager,scanner,_=running
    app.state.config['chat_models']={'arex':{'resource':'arex','url':'http://arex.test','model':'/local/arex',
        'vision':True,'context':131072,'tool_reasoning_effort':'low','tool_thinking_budget':1024}}
    seen=[]
    def handler(request):
        body=json.loads(request.content);seen.append(body)
        return httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}],'usage':{'total_tokens':2}})
    app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tools=[{'type':'function','function':{'name':'f','parameters':{'type':'object'}}}]
    response=client.post('/v1/chat/completions',headers=AUTH,json={'model':'arex','messages':[{'role':'user','content':'call f'}],
        'tools':tools,'max_tokens':512})
    assert response.status_code==200
    assert seen[0]['reasoning_effort']=='low' and seen[0]['thinking_budget']==512
    assert manager.calls==['arex']

def test_reasoning_values_forward_to_backend_and_invalid_values_stop_before_backend(running):
    client,app,manager,scanner,requests=running
    for effort in ('low','none'):
        response=client.post('/v1/chat/completions',headers=AUTH,json={'model':'qwen','messages':[{'role':'user','content':'hi'}],
            'reasoning_effort':effort})
        assert response.status_code==200
        assert json.loads(requests[-1].content)['reasoning_effort']==effort
    before=len(requests);leases=len(manager.calls)
    for field,value in [('reasoning_effort','high'),('reasoning_effort',{}),('thinking_budget',True),('thinking_budget',9000)]:
        response=client.post('/v1/chat/completions',headers=AUTH,json={'model':'qwen','messages':[{'role':'user','content':'hi'}],field:value})
        assert response.status_code==422
    assert len(requests)==before and len(manager.calls)==leases


@pytest.fixture
def routed(tmp_path):
    for path in ('state','outputs/audio','test-assets'):(tmp_path/path).mkdir(parents=True,exist_ok=True)
    class RoutingScanner(Scanner):
        def route_request(self,text):
            if 'quit my job' in text: return {'route':'reasoner','scores':{'general':0.05,'swift':0.05,'reasoner':0.9}}
            if 'race condition' in text: return {'route':'swift','scores':{'general':0.1,'swift':0.8,'reasoner':0.1}}
            return {'route':'general','scores':{'general':0.9,'swift':0.05,'reasoner':0.05}}
    manager=Manager();scanner=RoutingScanner();scheduler=LazyScheduler(tmp_path/'state/jobs.sqlite3',manager)
    config={'api_key':'test-only','admin_key':'admin-only','provider_key':'provider-only','backend_key':'backend-only','qwen_model':'/local/qwen',
            'chat_models':{'ornith':{'resource':'ornith','url':'http://ornith.test','model':'/local/ornith'},
                           'arex':{'resource':'arex','url':'http://arex.test','model':'/local/arex','vision':True,'context':131072},
                           'swift':{'resource':'swift','url':'http://swift.test','model':'/local/swift','vision':True}},
            'vision_default':'swift',
            'escalate':{'ornith':'swift'},
            'backends':{'qwen-vlm':{},'ornith':{}},'audio_models':{'tts':'/local/tts','asr':'/local/asr'}}
    app=create_app(tmp_path,injected=(config,manager,scheduler,scanner,object()))
    with TestClient(app) as client:
        seen=[]
        def handler(request):
            seen.append((str(request.url),json.loads(request.content)['model']))
            if 'ornith' in str(request.url) and 'BROKEN' in request.content.decode(): return httpx.Response(500,json={'error':'boom'})
            if 'ornith' in str(request.url) and 'PRINT' in request.content.decode():
                return httpx.Response(200,json={'choices':[{'message':{'content':'Running it now.\n\n/Users/me/AI/bin/openclaw-local-ai video --text "fox"'},'finish_reason':'stop'}]})
            if 'ornith' in str(request.url) and 'FAKEMEDIA' in request.content.decode():
                return httpx.Response(200,json={'choices':[{'message':{'content':'Generating now.\nMEDIA:/Users/me/AI/assets/images/invented.mp4'},'finish_reason':'stop'}]})
            if 'ornith' in str(request.url) and 'CALL' in request.content.decode():
                call={'id':'c1','type':'function','function':{'name':'exec','arguments':'{"command":"/Users/me/AI/bin/x"}'}}
                return httpx.Response(200,json={'id':'r1','choices':[{'message':{'content':'On it.','tool_calls':[call]},'finish_reason':'tool_calls'}],'usage':{'total_tokens':5}})
            return httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}],'usage':{'total_tokens':2}})
        app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        yield client,manager,seen

def ask(client,model,text):
    return client.post('/v1/chat/completions',headers=AUTH,json={'model':model,'messages':[{'role':'user','content':text}]})

@pytest.mark.parametrize('stream',[False,True])
def test_sparse_weak_answer_escalates_once_and_stream_is_valid(routed,stream):
    client,manager,seen=routed
    layer=client.app.state.soft_decisions
    layer.selected=lambda *args:True
    calls=[]
    async def weak(user,result):
        calls.append(user)
        return {'action':'escalate','available':True}
    layer.weak_answer=weak
    response=client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','stream':stream,
        'messages':[{'role':'user','content':'Explain a long ordinary question clearly please.'}]})
    assert response.status_code==200 and response.headers['X-Bubble-Route'].startswith('swift;')
    assert 'weak-answer-from-ornith' in response.headers['X-Bubble-Route']
    assert manager.calls==['ornith','swift'] and len(calls)==1
    assert '[DONE]' in response.text if stream else response.json()['model']=='swift'

def test_weak_answer_fallback_preserves_original_when_stronger_backend_fails(routed):
    client,manager,seen=routed
    layer=client.app.state.soft_decisions
    layer.selected=lambda *args:True
    async def weak(*args): return {'action':'escalate','available':True}
    layer.weak_answer=weak
    def handler(request):
        if request.url.host=='swift.test':return httpx.Response(500,json={'error':'unavailable'})
        return httpx.Response(200,json={'choices':[{'message':{'content':'Original imperfect but usable answer.'}}]})
    client.app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response=ask(client,'auto','Explain a long ordinary question clearly please.')
    assert response.status_code==200 and response.json()['choices'][0]['message']['content']=='Original imperfect but usable answer.'
    assert 'kept-original:escalation-unavailable' in response.headers['X-Bubble-Route']

def test_weak_answer_review_skips_explicit_models_and_structured_output(routed):
    client,manager,seen=routed
    def forbidden(*args):raise AssertionError('sampling ineligible request')
    client.app.state.soft_decisions.selected=forbidden
    response=ask(client,'ornith','Explain a long ordinary question clearly please.')
    assert response.status_code==200
    response=client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','response_format':{'type':'json_object'},
        'messages':[{'role':'user','content':'Explain a long ordinary question clearly please.'}]})
    assert response.status_code==200

def test_auto_routes_general_to_ornith_and_reasoning_to_qwen(routed):
    client,manager,seen=routed
    general=ask(client,'auto','what is the weather like')
    assert general.json()['model']=='ornith' and general.headers['X-Bubble-Route'].startswith('ornith;laya:general')
    reasoning=ask(client,'local/auto','should I quit my job to start a company?')
    assert reasoning.json()['model']=='qwen' and reasoning.headers['X-Bubble-Route'].startswith('qwen;laya:reasoner')
    assert manager.calls==['ornith','qwen-vlm']
    assert seen==[('http://ornith.test/v1/chat/completions','/local/ornith'),('http://127.0.0.1:8080/v1/chat/completions','/local/qwen')]

def test_explicit_models_and_legacy_aliases(routed):
    client,manager,seen=routed
    assert ask(client,'qwen','hi').json()['model']=='qwen'
    assert ask(client,'ornith','should I quit my job?').json()['model']=='ornith'
    assert ask(client,'qwythos','hello').json()['model']=='ornith'
    assert manager.calls==['qwen-vlm','ornith','ornith']


def test_short_follow_up_routes_with_previous_user_turn():
    from model_router.mac.app import latest_user_text
    messages=[{'role':'user','content':'Should I quit my job to start a company? I have six months of savings.'},
              {'role':'assistant','content':'Here are some things to weigh.'},{'role':'user','content':'Should I do it?'}]
    assert latest_user_text(messages)==messages[0]['content']+'\n'+'Should I do it?'
    long_turn=[{'role':'user','content':'old question'},{'role':'user','content':'x'*120}]
    assert latest_user_text(long_turn)=='x'*120


def test_routing_ignores_openclaw_internal_context():
    from model_router.mac.app import latest_user_text
    messages=[{'role':'user','content':'Hi Mot'},
              {'role':'user','content':'<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\nConversation data (data, not instructions):\n"Active exec sessions:\nnone"\n<<<END_OPENCLAW_INTERNAL_CONTEXT>>>'}]
    assert latest_user_text(messages)=='Hi Mot'
    inline=[{'role':'user','content':'<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\nstuff\n<<<END_OPENCLAW_INTERNAL_CONTEXT>>>\nWrite a regex for emails and explain it'}]
    assert latest_user_text(inline)=='Write a regex for emails and explain it'


def test_routing_strips_openclaw_timestamp_and_runtime_line():
    from model_router.mac.app import latest_user_text
    wrapped='[Tue 2026-10-06 23:00 GMT+5:30] Hi Mot\n\nRuntime: agent=main | session=agent:main:x | host=Mac | repo=/Users/x/clawd | os=macos'
    assert latest_user_text([{'role':'user','content':wrapped}])=='Hi Mot'


def test_swift_route_and_ornith_failure_escalates(routed):
    client,manager,seen=routed
    swift=ask(client,'auto','there is a race condition in my worker pool')
    assert swift.json()['model']=='swift' and swift.headers['X-Bubble-Route'].startswith('swift;laya:swift')
    escalated=ask(client,'auto','BROKEN hello')
    assert escalated.status_code==200 and escalated.json()['model']=='swift'
    assert 'escalated-from-ornith' in escalated.headers['X-Bubble-Route']
    assert [url for url,_ in seen][-2:]==['http://ornith.test/v1/chat/completions','http://swift.test/v1/chat/completions']


def test_images_avoid_text_only_tier(routed):
    client,manager,seen=routed
    image={'type':'image_url','image_url':{'url':'data:image/png;base64,iVBORw0KGgo='}}
    r=client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','messages':[{'role':'user','content':[{'type':'text','text':'what is in this picture'},image]}]})
    assert r.status_code==200 and r.json()['model']=='swift' and r.headers['X-Bubble-Route'].endswith(';vision')


def test_long_conversation_promotes_to_long_context_tier(routed):
    client,manager,seen=routed
    history=[{'role':'user','content':'background '*13000},{'role':'assistant','content':'noted'}]
    r=client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','messages':history+[{'role':'user','content':'what is the weather like'}]})
    assert r.status_code==200 and r.json()['model']=='arex' and r.headers['X-Bubble-Route'].endswith(';long-context')
    too_long=[{'role':'user','content':'x'*600000}]
    assert client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','messages':too_long}).status_code==422


EXEC_TOOL=[{'type':'function','function':{'name':'exec','parameters':{'type':'object','properties':{'command':{'type':'string'}}}}}]

def test_printed_command_instead_of_tool_call_escalates(routed):
    client,manager,seen=routed
    body={'model':'auto','tools':EXEC_TOOL,'stream':True,'messages':[{'role':'user','content':'PRINT list my downloads folder'}]}
    r=client.post('/v1/chat/completions',headers=AUTH,json=body)
    assert r.status_code==200 and r.headers['X-Bubble-Route'].startswith('swift;') and 'printed-command' in r.headers['X-Bubble-Route']
    assert [url for url,_ in seen][-2:]==['http://ornith.test/v1/chat/completions','http://swift.test/v1/chat/completions']
    # After tool results, paths in the answer are normal and stay on Ornith.
    after_tool=body|{'stream':False,'messages':body['messages']+[{'role':'assistant','content':'','tool_calls':[]},{'role':'tool','tool_call_id':'c1','content':'ok'}]}
    assert client.post('/v1/chat/completions',headers=AUTH,json=after_tool).json()['model']=='ornith'


def test_buffered_tool_call_replays_as_stream(routed):
    client,manager,seen=routed
    body={'model':'auto','tools':EXEC_TOOL,'stream':True,'stream_options':{'include_usage':True},'messages':[{'role':'user','content':'CALL list my downloads folder'}]}
    r=client.post('/v1/chat/completions',headers=AUTH,json=body)
    assert r.status_code==200 and r.headers['X-Bubble-Route'].startswith('ornith;') and r.headers['content-type'].startswith('text/event-stream')
    events=[json.loads(line[6:]) for line in r.text.splitlines() if line.startswith('data: {')]
    assert events[0]['choices'][0]['delta']['tool_calls'][0]['index']==0
    assert events[0]['choices'][0]['delta']['tool_calls'][0]['function']['name']=='exec'
    assert events[-1]['choices'][0]['finish_reason']=='tool_calls' and events[-1]['usage']=={'total_tokens':5}
    assert r.text.endswith('data: [DONE]\n\n')


def test_fabricated_media_in_first_reply_escalates(routed):
    client,manager,seen=routed
    body={'model':'auto','tools':EXEC_TOOL,'messages':[{'role':'user','content':'FAKEMEDIA show me the latest file'}]}
    r=client.post('/v1/chat/completions',headers=AUTH,json=body)
    assert r.status_code==200 and r.json()['model']=='swift'
    assert 'escalated-from-ornith:printed-command' in r.headers['X-Bubble-Route']
    assert manager.calls==['ornith','swift']
    assert 'MEDIA:' not in r.json()['choices'][0]['message']['content']
    # An attachment after actual tool output is allowed through unchanged.
    after=body|{'messages':body['messages']+[{'role':'assistant','content':'','tool_calls':[]},{'role':'tool','tool_call_id':'c1','content':'artifact path'}]}
    r=client.post('/v1/chat/completions',headers=AUTH,json=after)
    assert r.json()['model']=='ornith'


def post_auto(client,messages,tools=EXEC_TOOL):
    return client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','tools':tools,'messages':messages})


@pytest.mark.parametrize('text',['Validate 10 more','Generate GTA 6 video',
    'Generate an image of a fox on the moon in Renaissance style',
    'Create a 5s video of what you imagine GTA 6 gameplay is',
    'Check each half hour and tell me or report and fix if there are any errors',
    'Spawn subagents to research these', 'Remind me to call mum at 6'])
def test_actions_ornith_failed_on_go_straight_to_swift(routed,text):
    # Prompts from the transcripts where Ornith printed a command, an invented
    # MEDIA path or a fake process_group result instead of calling a tool.
    client,manager,seen=routed
    r=post_auto(client,[{'role':'user','content':text}])
    assert r.json()['model']=='swift' and r.headers['X-Bubble-Route']=='swift;laya:general:0.9;not-simple:action'
    assert manager.calls==['swift']


def test_simple_tool_requests_stay_on_ornith(routed):
    client,manager,seen=routed
    for text in ('what time is it in Tokyo','list my downloads folder','make a note that the plumber came','check my calendar for today'):
        r=post_auto(client,[{'role':'user','content':text}])
        assert r.json()['model']=='ornith' and r.headers['X-Bubble-Route']=='ornith;laya:general:0.9', text


def test_short_follow_up_inherits_the_action_it_follows(routed):
    client,manager,seen=routed
    r=post_auto(client,[{'role':'user','content':'Validate 10 more'},{'role':'assistant','content':'Starting.'},
                        {'role':'user','content':'Is it running?'}])
    assert r.json()['model']=='swift' and r.headers['X-Bubble-Route'].endswith('not-simple:action')


def test_printed_tool_output_earlier_in_session_keeps_it_off_ornith(routed):
    client,manager,seen=routed
    fake='Boss. Checking now.\n\n{"type":"process_group","processGroupId":"spawn-batch-a","state":"Running","runningCount":10,"count":10,"results":[]}'
    r=post_auto(client,[{'role':'user','content':'how are things going over there'},{'role':'assistant','content':fake},
                        {'role':'user','content':'what is the date today then, friend?'}])
    assert r.json()['model']=='swift' and r.headers['X-Bubble-Route'].endswith('not-simple:printed-tool-history')
    # Paths in a reply after real tool output are ordinary answers.
    call={'id':'c1','type':'function','function':{'name':'exec','arguments':'{"command":"ls"}'}}
    r=post_auto(client,[{'role':'user','content':'list my downloads folder'},{'role':'assistant','content':'','tool_calls':[call]},
                        {'role':'tool','tool_call_id':'c1','content':'a.txt'},{'role':'assistant','content':'/Users/me/Downloads/a.txt'},
                        {'role':'user','content':'what is the date today then, friend?'}])
    assert r.json()['model']=='ornith'


def test_multi_step_runs_long_text_code_and_unsure_scores_go_to_swift(routed):
    client,manager,seen=routed
    r=post_auto(client,agent_run('tidy my notes',[('read',{'path':f'notes{i}.md'}) for i in range(3)]))
    assert r.json()['model']=='swift' and r.headers['X-Bubble-Route'].endswith('not-simple:run-3-calls')
    assert post_auto(client,agent_run('tidy my notes',[('read',{'path':'notes.md'})]*2)).json()['model']=='ornith'
    assert post_auto(client,[{'role':'user','content':'tell me about it '*30}]).headers['X-Bubble-Route'].endswith('not-simple:long')
    assert post_auto(client,[{'role':'user','content':'what does main.py do'}]).headers['X-Bubble-Route'].endswith('not-simple:code')
    assert post_auto(client,[{'role':'user','content':'run pytest and tell me'}]).headers['X-Bubble-Route'].endswith('not-simple:code')
    client.app.state.scanner.route_request=lambda text:{'route':'general','scores':{'general':0.5,'swift':0.45,'reasoner':0.05}}
    assert post_auto(client,[{'role':'user','content':'what time is it in Tokyo'}]).headers['X-Bubble-Route']=='swift;laya:general:0.5;not-simple:low-score'
    # Requests without person text cannot be judged simple.
    internal='<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\nexec finished\n<<<END_OPENCLAW_INTERNAL_CONTEXT>>>'
    assert post_auto(client,[{'role':'user','content':internal}]).headers['X-Bubble-Route']=='swift;default;not-simple:no-text'


def test_simple_route_thresholds_are_configurable(routed):
    client,manager,seen=routed
    client.app.state.config['simple_route']={'max_chars':20}
    assert post_auto(client,[{'role':'user','content':'what time is it in Tokyo'}]).json()['model']=='swift'
    # Without Ornith as the simple model the gate does nothing.
    client.app.state.config['simple_route']={'model':'none'}
    assert post_auto(client,[{'role':'user','content':'Validate 10 more'}]).json()['model']=='ornith'


def test_moving_down_to_ornith_waits_but_moving_up_is_immediate(routed):
    client,manager,seen=routed
    client.app.state.config['route_switch_after']=1
    loaded={'swift'}
    manager.is_loaded=lambda resource: resource in loaded
    first=post_auto(client,[{'role':'user','content':'what time is it in Tokyo'}])
    assert first.json()['model']=='swift' and first.headers['X-Bubble-Route'].endswith('sticky:swift:1/2')
    second=post_auto(client,[{'role':'user','content':'what time is it in Tokyo'}])
    assert second.json()['model']=='ornith' and second.headers['X-Bubble-Route'].endswith('switch:after-2')
    loaded.clear(); loaded.add('ornith')
    assert post_auto(client,[{'role':'user','content':'Validate 10 more'}]).json()['model']=='swift'


def test_text_code_commands_ignore_plain_make():
    from model_router.mac.app import TEXT_CODE_COMMAND_RE, CODE_COMMAND_RE
    assert CODE_COMMAND_RE.search('make a list') and not TEXT_CODE_COMMAND_RE.search('make a list')
    assert TEXT_CODE_COMMAND_RE.search('then run cargo build')


def test_media_guard_preserves_real_tool_calls_and_plain_text():
    from model_router.mac.app import printed_instead_of_calling
    def response(content,**extra):
        return {'choices':[{'message':{'content':content,**extra}}]}
    assert printed_instead_of_calling(response('Boss.\nMEDIA:/missing/video.mp4'))
    assert not printed_instead_of_calling(response('Video generation takes a minute.'))
    assert not printed_instead_of_calling(response('MEDIA:/missing/video.mp4',tool_calls=[{'id':'t1'}]))

def test_embeddings_endpoint_proxies_to_local_service_without_scan(running):
    client,app,manager,scanner,requests=running
    seen=[]
    def handler(request):
        seen.append((str(request.url),json.loads(request.content)))
        body=json.loads(request.content)
        return httpx.Response(200,json={'data':[{'index':i,'embedding':[float(len(t)),0.0]} for i,t in reversed(list(enumerate(body['input'])))]})
    app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert client.post('/v1/embeddings',json={'input':'hi'}).status_code==401
    response=client.post('/v1/embeddings',headers=AUTH,json={'model':'x','input':['ab','abc'],'input_type':'query'})
    assert response.status_code==200
    body=response.json()
    assert body['model']=='embeddinggemma-2'
    q='task: search result | query: '
    assert [d['embedding'][0] for d in body['data']]==[len(q)+2.0,len(q)+3.0] and [d['index'] for d in body['data']]==[0,1]
    assert seen==[('http://127.0.0.1:8766/v1/embeddings',{'input':[q+'ab',q+'abc']})]
    client.post('/v1/embeddings',headers=AUTH,json={'input':'doc'})
    assert seen[-1][1]['input']==['title: none | text: doc']
    assert scanner.calls==0 and manager.calls==[]
    assert client.post('/v1/embeddings',headers=AUTH,json={'input':[]}).status_code==422
    assert client.post('/v1/embeddings',headers=AUTH,json={'input':['x']*257}).status_code==422
    def down(request): raise httpx.ConnectError('refused')
    app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(down))
    assert client.post('/v1/embeddings',headers=AUTH,json={'input':'x'}).status_code==503

def test_auto_stays_on_loaded_chat_model_until_repeated_demand_switches(routed):
    client,manager,seen=routed
    loaded={'ornith'}
    manager.is_loaded=lambda resource: resource in loaded
    # Swift is preferred, but Ornith is loaded: the first two stay put.
    for expected in ('1/3','2/3'):
        reply=ask(client,'auto','fix this race condition')
        assert reply.json()['model']=='ornith'
        assert reply.headers['X-Bubble-Route'].endswith(f'sticky:ornith:{expected}')
    # A request that prefers the loaded model resets the streak.
    assert ask(client,'auto','what is the weather like').headers['X-Bubble-Route']=='ornith;laya:general:0.9'
    for _ in range(2): ask(client,'auto','fix this race condition')
    third=ask(client,'auto','fix this race condition')
    assert third.json()['model']=='swift' and third.headers['X-Bubble-Route'].endswith('switch:after-3')
    loaded.clear(); loaded.add('swift')
    # Now Swift is loaded, so general requests stay on Swift.
    reply=ask(client,'auto','what is the weather like')
    assert reply.json()['model']=='swift' and 'sticky:swift:1/3' in reply.headers['X-Bubble-Route']
    # Explicit model names always switch immediately.
    assert ask(client,'ornith','what is the weather like').json()['model']=='ornith'

def test_auto_sticky_skips_loaded_model_without_vision(routed):
    client,manager,seen=routed
    manager.is_loaded=lambda resource: resource=='ornith'
    image={'type':'image_url','image_url':{'url':'data:image/png;base64,iVBORw0KGgo='}}
    reply=client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','messages':[{'role':'user','content':[{'type':'text','text':'what is in this picture'},image]}]})
    assert reply.json()['model']!='ornith'

def test_auto_sticky_threshold_is_configurable(routed):
    client,manager,seen=routed
    client.app.state.config['route_switch_after']=1
    manager.is_loaded=lambda resource: resource=='ornith'
    assert ask(client,'auto','fix this race condition').json()['model']=='swift'

def test_chat_incremental_estimate_uses_model_kv_size_and_bytes_per_token():
    from model_router.mac.app import chat_incremental_gib
    payload={'messages':[{'role':'user','content':'x'*90000}],'max_tokens':2048}
    tokens=len(json.dumps(payload['messages']).encode())//3+2048
    assert chat_incremental_gib(payload)==pytest.approx(4.0+tokens*65536/1024**3)
    assert chat_incremental_gib(payload,20480)==pytest.approx(4.0+tokens*20480/1024**3)
    assert chat_incremental_gib(payload,20480)<chat_incremental_gib(payload)<6.1


def test_sampling_defaults_fill_only_missing_values():
    from model_router.mac.app import apply_sampling_defaults
    filled = apply_sampling_defaults({'model':'m'},{})
    assert {k:v for k,v in filled.items() if k!='seed'} == {'model':'m','temperature':1.0,'top_p':0.95,'top_k':20}
    assert isinstance(filled['seed'], int) and len({apply_sampling_defaults({},{})['seed'] for _ in range(5)}) > 1
    assert apply_sampling_defaults({'model':'m','temperature':0.2,'top_p':None,'seed':7},{'sampling':{'top_k':40}}) == \
        {'model':'m','temperature':0.2,'top_p':0.95,'top_k':40,'seed':7}
    assert prepare_chat({'messages':[{'role':'user','content':'hi'}],'top_k':5})[0]['top_k'] == 5


def agent_run(user_text, calls):
    messages=[{'role':'user','content':user_text}]
    for name,args in calls:
        messages.append({'role':'assistant','content':None,'tool_calls':[{'id':'c','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]})
        messages.append({'role':'tool','tool_call_id':'c','content':'ok'})
    return messages

CODE_CALLS=[('read',{'path':'src/strategy.rs'}),('edit',{'path':'src/strategy.rs','old':'a','new':'b'}),('exec',{'command':'cargo build'}),
            ('read',{'path':'notes.md'}),('exec',{'command':'ls'}),('read',{'path':'src/main.rs'})]

def test_coding_run_counts_only_the_current_agent_run():
    from model_router.mac.app import coding_run
    assert coding_run(agent_run('fix the strategy', CODE_CALLS))==(6,4)
    later=agent_run('fix the strategy', CODE_CALLS)+[{'role':'assistant','content':'done'},{'role':'user','content':'thanks, what time is it?'}]
    assert coding_run(later)==(0,0)
    bookkeeping=agent_run('fix the strategy', CODE_CALLS)+[{'role':'user','content':'<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\nx\n<<<END_OPENCLAW_INTERNAL_CONTEXT>>>'}]
    assert coding_run(bookkeeping)==(6,4)
    assert coding_run(agent_run('tidy my notes',[('read',{'path':f'notes{i}.md'}) for i in range(8)]))==(8,0)

def test_long_coding_run_goes_to_arex_immediately(routed):
    client,manager,seen=routed
    client.app.state.config['route_targets']={'reasoner':'arex'}
    manager.is_loaded=lambda resource: resource=='ornith'
    post=lambda messages: client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','messages':messages})
    short=post(agent_run('fix the strategy', CODE_CALLS[:5]))
    assert short.json()['model']=='ornith'
    long=post(agent_run('fix the strategy', CODE_CALLS))
    assert long.json()['model']=='arex' and long.headers['X-Bubble-Route']=='arex;coding-run:4/6'
    notes=post(agent_run('tidy my notes',[('read',{'path':f'notes{i}.md'}) for i in range(8)]))
    assert notes.json()['model']=='ornith'
    assert client.post('/v1/chat/completions',headers=AUTH,json={'model':'swift','messages':agent_run('x',CODE_CALLS)}).json()['model']=='swift'


def image_fixture():
    import struct, zlib
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 512, 512, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\x00' + b'\x00\x80\x80' * 512) * 512)) + chunk(b'IEND', b''))

PNG_BYTES = image_fixture()


@pytest.mark.parametrize("field", ["first_frame", "initial_image"])
def test_video_multipart_first_frame_upload_reaches_h3_and_is_removed(running, tmp_path, field):
    client,app,*_=running
    seen={}
    async def complete(payload, job_id):
        seen.update(payload)
        seen['bytes']=Path(payload[field]).read_bytes()
        return {'artifact':'fake'}
    app.state.media.generate_video=complete
    response=client.post('/v1/videos/generations',headers=AUTH,
        data={'prompt':'the mug slowly rotates','duration':'2','steps':'4','width':'256','height':'256'},
        files={field:('start.png',PNG_BYTES,'image/png')})
    assert response.status_code==202, response.text
    job_id=response.json()['id']
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        job=client.get(f'/v1/jobs/{job_id}',headers=AUTH).json()
        if job['state'] in ('succeeded','failed'): break
        time.sleep(.01)
    assert job['state']=='succeeded', job
    assert seen['bytes']==PNG_BYTES
    assert seen['steps']==4 and seen['width']==256 and seen['duration']==2.0
    upload=Path(seen[field])
    assert upload.parent==(tmp_path/'state'/'uploads') and upload.suffix=='.png'
    assert not upload.exists()


@pytest.mark.parametrize('files,data,message',[
    ({'first_frame':('x.png',b'GIF89a not allowed','image/png')},{'prompt':'x'},'PNG, JPEG or WebP'),
    ({'image':('x.png',PNG_BYTES,'image/png')},{'prompt':'x'},'only accepted for'),
    ({'first_frame':('x.png',PNG_BYTES,'image/png')},{'prompt':'x','steps':'many'},'steps must be an integer'),
])
def test_video_multipart_rejects_bad_uploads_without_queueing_or_leaving_files(running, tmp_path, files, data, message):
    client,app,*_=running
    response=client.post('/v1/videos/generations',headers=AUTH,data=data,files=files)
    assert response.status_code==422
    assert message in response.json()['error']
    assert not any((tmp_path/'state'/'uploads').glob('*')) if (tmp_path/'state'/'uploads').exists() else True


def test_edit_multipart_image_upload_is_validated(running, tmp_path):
    client,app,*_=running
    (tmp_path/'config/model-paths.json').write_text(json.dumps({'abenzerps/Qwen-Image-2.1-Uncensored-GGUF':{'path':'/x'},'MiniMaxAI/MiniMax-H3':{'path':'/x'}}))
    seen={}
    async def complete(payload, job_id):
        seen.update(payload); return {'artifact':'fake'}
    app.state.media.edit_image=complete
    missing=client.post('/v1/images/edits',headers=AUTH,data={'prompt':'make it blue'})
    assert missing.status_code==422 and 'image is required' in missing.json()['error']
    response=client.post('/v1/images/edits',headers=AUTH,
        data={'prompt':'make it blue','width':'768','height':'512'},
        files={'image':('mug.png',PNG_BYTES,'image/png')})
    assert response.status_code==202, response.text
    deadline=time.monotonic()+2
    while not seen and time.monotonic()<deadline: time.sleep(.01)
    assert seen['width']==768 and seen['height']==512
    assert Path(seen['image']).parent==tmp_path/'state'/'uploads'


def test_text_to_image_requires_original_qwen_model_before_queueing(running):
    client,app,manager,scanner,*_=running
    async def generate(*args): raise AssertionError("missing model must not be queued")
    app.state.media.generate_image=generate
    response=client.post('/v1/images/generations',headers=AUTH,json={'prompt':'a red mug'})
    assert response.status_code==503
    assert scanner.calls==0 and manager.calls==[]


def test_original_qwen_text_to_image_accepts_repository_and_keeps_resource(running):
    client,app,manager,scanner,*_=running
    root=app.state.media.preflight.__self__.root
    (root/'config/model-paths.json').write_text(json.dumps({'abenzerps/Qwen-Image-2.1-Uncensored-GGUF':{'path':'/fixture-not-loaded'}}))
    async def generate(payload,job_id): return {'artifact':str(root/'outputs'/f'{job_id}.png'),'content_type':'image/png'}
    app.state.media.generate_image=generate
    response=client.post('/v1/images/generations',headers=AUTH,json={'prompt':'a red mug'})
    assert response.status_code==202
    job_id=response.json()['id'];deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        job=client.get(f'/v1/jobs/{job_id}',headers=AUTH).json()
        if job['state']=='succeeded':break
        time.sleep(.01)
    assert job['state']=='succeeded'
    assert manager.calls==['qwen-image'] and scanner.calls==1


def test_atomic_missing_encoder_is_rejected_before_security_or_queue(running):
    from model_router.mac.media import ATOMIC_IMAGE_REPO
    client,app,manager,scanner,*_=running
    root=app.state.media.preflight.__self__.root
    (root/'config/model-paths.json').write_text(json.dumps({ATOMIC_IMAGE_REPO:{'path':'/fixture-not-loaded'}}))
    async def generate(*args): raise AssertionError('missing encoder must not be queued')
    app.state.media.generate_image=generate
    response=client.post('/v1/images/generations',headers=AUTH,json={'prompt':'a red mug'})
    assert response.status_code==503 and scanner.calls==0 and manager.calls==[]


def test_swift_first_auto_without_ornith_keeps_specialist_routes(routed):
    client, manager, seen = routed
    config = client.app.state.config
    config['chat_models'].pop('ornith')
    config['backends'].pop('ornith')
    config['route_targets'] = {'general': 'swift', 'swift': 'swift', 'reasoner': 'arex'}
    config['vision_default'] = 'arex'
    config['route_switch_after'] = 1
    manager.is_loaded = lambda resource: resource == 'swift'
    assert ask(client, 'auto', 'hello').json()['model'] == 'swift'
    assert ask(client, 'auto', 'fix this race condition').json()['model'] == 'swift'
    assert ask(client, 'auto', 'should I quit my job?').json()['model'] == 'arex'
    assert ask(client, 'auto', '').json()['model'] == 'swift'
    assert ask(client, 'ornith', 'hello').json()['model'] == 'swift'
    ids = {m['id'] for m in client.get('/v1/models', headers=AUTH).json()['data']}
    assert 'ornith' not in ids and {'auto', 'swift', 'arex', 'qwen'} <= ids
    assert 'ornith' not in manager.calls

def test_serving_mode_swaps_only_backends_with_a_variant():
    base={'backends':{'ornith':{'command':['mlx_vlm'],'estimate_gib':22,'modes':{'vllm':{'command':['vllm'],'estimate_gib':24,'engine':'vllm'}}},
                      'swift':{'command':['swift-serve'],'estimate_gib':19}}}
    mlx=apply_serving_mode(base)
    assert mlx['backends']['ornith']=={'command':['mlx_vlm'],'estimate_gib':22}
    vllm=apply_serving_mode({**base,'serving_mode':'vllm'})
    assert vllm['backends']['ornith']=={'command':['vllm'],'estimate_gib':24,'engine':'vllm'}
    assert vllm['backends']['swift']==base['backends']['swift']
    assert 'modes' in base['backends']['ornith']  # the file's config is left untouched
    with pytest.raises(ValueError): apply_serving_mode({**base,'serving_mode':'llama'})
    models={'backends':{},'chat_models':{'swift':{'model':'/old','modes':{'vllm':{'model':'/new','think_by_default':True}}}}}
    assert apply_serving_mode(models)['chat_models']['swift']=={'model':'/old'}
    assert apply_serving_mode({**models,'serving_mode':'vllm'})['chat_models']['swift']=={'model':'/new','think_by_default':True}

def test_profile_mode_pins_one_backend_to_its_own_variant():
    base={'serving_mode':'vllm','backends':{'ornith':{'command':['mlx'],'modes':{'vllm':{'command':['vllm'],'engine':'vllm'}}},
          'swift':{'command':['mlx'],'mode':'splash','modes':{'vllm':{'command':['vllm']},'splash':{'command':['splash'],'engine':'splash'}}}}}
    resolved=apply_serving_mode(base)
    assert resolved['backends']['swift']=={'command':['splash'],'engine':'splash'}
    assert resolved['backends']['ornith']['engine']=='vllm'
    assert apply_serving_mode({**base,'serving_mode':'mlx'})['backends']['swift']['engine']=='splash'
    with pytest.raises(ValueError): apply_serving_mode({'backends':{'swift':{'mode':'splash','modes':{}}}})

def test_splash_payload_moves_effort_top_level_and_sets_thinking_budget():
    thinking=splash_chat_payload(vllm_chat_payload({'messages':[],'max_tokens':4096},True,'xhigh'),2048)
    assert 'include_reasoning' not in thinking and thinking['reasoning_effort']=='xhigh'
    assert thinking['chat_template_kwargs']=={'enable_thinking':True} and thinking['thinking_token_budget']==2048
    # A request's own budget wins; no budget is set when thinking is off.
    own=splash_chat_payload(vllm_chat_payload({'messages':[],'thinking_budget':64},True),2048)
    assert own['thinking_token_budget']==64
    off=splash_chat_payload(vllm_chat_payload({'messages':[],'reasoning_effort':'none'},True,'xhigh'),2048)
    assert off['chat_template_kwargs']=={'enable_thinking':False} and 'thinking_token_budget' not in off and 'reasoning_effort' not in off

def test_splash_backend_gets_native_budget_and_no_router_cut(running):
    client,app,manager,scanner,_=running
    app.state.config['chat_models']={'swift':{'resource':'swift','url':'http://swift.test','model':'/m','context':65536,
        'think_by_default':True,'default_reasoning_effort':'xhigh','thinking_budget':2048}}
    app.state.config['backends']['swift']={'engine':'splash'}
    seen=[]
    def handler(request):
        assert request.url.path!='/metrics'  # no vLLM /metrics read
        if request.url.path=='/status':
            return httpx.Response(200,json={'memory_plan':{'budget':{'maximum_batch_width':4,'kv_capacity_tokens':368640}}})
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':'ok','reasoning_content':'r'}}],'usage':{'total_tokens':2}})
    app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response=client.post('/v1/chat/completions',headers=AUTH,json={'model':'swift','messages':[{'role':'user','content':'hi'}],'max_tokens':4096})
    assert response.status_code==200 and response.json()['choices'][0]['message']['content']=='ok'
    assert seen[0]['thinking_token_budget']==2048 and seen[0]['reasoning_effort']=='xhigh' and 'include_reasoning' not in seen[0]
    assert manager.lease_calls[-1]==('swift',{'incremental_gib':0.0})
    assert app.state.metrics.get('thinking_cuts',0)==0
    # ready_at marks when the model was loaded, so tok/s can leave out loading.
    row=sqlite3.connect(app.state.scheduler.path).execute('select started_at,finished_at,metadata_json from jobs').fetchone()
    assert row[0] <= json.loads(row[2])['ready_at'] <= row[1]
    # The router learns Splash's batch width and KV size, so requests run side by side.
    assert app.state.scheduler.resource_slots['swift']==4 and app.state.scheduler.resource_kv_tokens['swift']==(368640,0)

def test_splash_capacity_reads_the_memory_plan():
    assert splash_capacity({'memory_plan':{'budget':{'maximum_batch_width':4,'kv_capacity_tokens':368640}}})==(4,368640)
    assert splash_capacity({'memory_plan':{'budget':{'maximum_batch_width':0,'kv_capacity_tokens':5}}}) is None
    assert splash_capacity({}) is None and splash_capacity(None) is None

def test_vllm_payload_uses_vllm_field_names_and_file_urls():
    payload={'model':'/m','max_tokens':512,'thinking_budget':256,'messages':[{'role':'user','content':[
        {'type':'image_url','image_url':{'url':'/ai/test-assets/a.png'}},{'type':'image_url','image_url':'data:image/png;base64,xx'},
        {'type':'input_video','video_url':'/ai/outputs/v.mp4'},{'type':'text','text':'hi'}]}]}
    result=vllm_chat_payload(payload)
    assert result['include_reasoning'] is False and result['thinking_token_budget']==256 and 'thinking_budget' not in result
    content=result['messages'][0]['content']
    assert content[0]['image_url']=={'url':'file:///ai/test-assets/a.png'}
    assert content[1]['image_url']=={'url':'data:image/png;base64,xx'}
    assert content[2]=={'type':'video_url','video_url':{'url':'file:///ai/outputs/v.mp4'}}
    assert payload['messages'][0]['content'][0]['image_url']=={'url':'/ai/test-assets/a.png'}
    # Thinking follows mlx_vlm: off unless reasoning_effort asks for it.
    assert result['chat_template_kwargs']=={'enable_thinking':False}
    off=vllm_chat_payload({'messages':[],'reasoning_effort':'off'})
    assert 'reasoning_effort' not in off and off['chat_template_kwargs']=={'enable_thinking':False}
    low=vllm_chat_payload({'messages':[],'reasoning_effort':'low'})
    assert 'reasoning_effort' not in low and low['chat_template_kwargs']=={'enable_thinking':True,'reasoning_effort':'low'}
    # Swift's mlx_lm server thinks unless asked not to.
    assert vllm_chat_payload({'messages':[]},think_by_default=True)['chat_template_kwargs']=={'enable_thinking':True}
    # A model's default effort applies only when the request names none.
    assert vllm_chat_payload({'messages':[]},True,'low')['chat_template_kwargs']=={'enable_thinking':True,'reasoning_effort':'low'}
    assert vllm_chat_payload({'messages':[],'reasoning_effort':'xhigh'},True,'low')['chat_template_kwargs']['reasoning_effort']=='xhigh'
    assert vllm_chat_payload({'messages':[]},False,'low')['chat_template_kwargs']=={'enable_thinking':False}
    assert vllm_chat_payload({'messages':[],'reasoning_effort':'off'},think_by_default=True)['chat_template_kwargs']=={'enable_thinking':False}

def test_vllm_backend_gets_translated_request_and_no_per_request_memory(running):
    client,app,manager,scanner,_=running
    app.state.config['chat_models']={'arex':{'resource':'arex','url':'http://arex.test','model':'/local/arex','context':65536}}
    app.state.config['backends']['arex']={'engine':'vllm'}
    seen=[]
    def handler(request):
        if request.url.path=='/metrics':
            return httpx.Response(200,text='vllm:cache_config_info{block_size="784",kv_cache_size_tokens="139358"} 1.0\n')
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'choices':[{'message':{'content':'ok'}}],'usage':{'total_tokens':2}})
    app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tools=[{'type':'function','function':{'name':'f','parameters':{'type':'object'}}}]
    response=client.post('/v1/chat/completions',headers=AUTH,json={'model':'arex','messages':[{'role':'user','content':'call f'}],
        'tools':tools,'max_tokens':512})
    assert response.status_code==200
    assert seen[0]['thinking_token_budget']==512 and 'thinking_budget' not in seen[0] and seen[0]['include_reasoning'] is False
    assert seen[0]['chat_template_kwargs']=={'enable_thinking':True,'reasoning_effort':'low'}  # arex tool calls get low effort
    assert manager.lease_calls[-1]==('arex',{'incremental_gib':0.0})
    # The router learns the server's KV size and sizes each request against it.
    assert app.state.scheduler.resource_kv_tokens['arex']==(139358,1568)
    assert client.get('/router/status',headers=AUTH).json()['resource_kv_tokens']['arex']=={'tokens':139358,'per_request':1568}
    app.state.config['backends']['arex']={}
    client.post('/v1/chat/completions',headers=AUTH,json={'model':'arex','messages':[{'role':'user','content':'hi'}]})
    assert 'include_reasoning' not in seen[-1] and manager.lease_calls[-1][1]['incremental_gib']>4

def test_thinking_cut_budget_needs_thinking_and_room_below_max_tokens():
    thinking={'messages':[],'chat_template_kwargs':{'enable_thinking':True}}
    assert thinking_cut_budget(thinking,2048)==2048
    assert thinking_cut_budget({**thinking,'thinking_token_budget':300},2048)==300
    assert thinking_cut_budget({**thinking,'max_tokens':2048},2048) is None
    assert thinking_cut_budget({**thinking,'max_tokens':8192},2048)==2048
    assert thinking_cut_budget({'messages':[],'chat_template_kwargs':{'enable_thinking':False}},2048) is None
    assert thinking_cut_budget(thinking) is None

def sse(*deltas, usage=None):
    events=[{'id':'c1','created':1,'model':'/m','choices':[{'index':0,'delta':d,'finish_reason':None}]} for d in deltas]
    events.append({'id':'c1','choices':[{'index':0,'delta':{},'finish_reason':'tool_calls' if any('tool_calls' in d for d in deltas) else 'stop'}]})
    if usage: events.append({'id':'c1','choices':[],'usage':usage})
    return ''.join(f'data: {json.dumps(e)}\n\n' for e in events)+'data: [DONE]\n\n'

def test_vllm_thinking_budget_is_enforced_by_the_router(running):
    client,app,manager,scanner,_=running
    app.state.config['chat_models']={'swift':{'resource':'swift','url':'http://swift.test','model':'/m','context':65536,
        'think_by_default':True,'default_reasoning_effort':'medium','thinking_budget':3}}
    app.state.config['backends']['swift']={'engine':'vllm'}
    seen,thinking=[],[]
    def handler(request):
        if request.url.path=='/metrics': return httpx.Response(200,text='')
        body=json.loads(request.content); seen.append(body)
        if body.get('continue_final_message'):
            return httpx.Response(200,json={'choices':[{'message':{'role':'assistant','content':'6:10 PM'},'finish_reason':'stop'}],
                                            'usage':{'completion_tokens':4,'total_tokens':20}})
        return httpx.Response(200,text=sse(*({'reasoning':t} for t in thinking),{'content':'done'},usage={'completion_tokens':3}))
    app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ask=lambda **extra: client.post('/v1/chat/completions',headers=AUTH,json={'model':'swift','messages':[{'role':'user','content':'when?'}],**extra})
    # Thinking within budget: one streamed request, folded, reasoning hidden.
    thinking[:]=['a','b']
    reply=ask().json()
    assert len(seen)==1 and seen[0]['stream'] and seen[0]['include_reasoning'] is True
    assert seen[0]['chat_template_kwargs']=={'enable_thinking':True,'reasoning_effort':'medium'}
    assert reply['choices'][0]['message']=={'role':'assistant','content':'done'} and reply['model']=='swift'
    # Past budget: thinking is cut and the model answers from it with thinking off.
    seen.clear(); thinking[:]=['a','b','c','d','e','f']
    reply=ask(max_tokens=1000)
    follow=seen[-1]
    assert len(seen)==2 and follow['continue_final_message'] and follow['add_generation_prompt'] is False
    assert follow['chat_template_kwargs']=={'enable_thinking':False,'reasoning_effort':'medium'}
    assert follow['messages'][-1]['role']=='assistant' and follow['messages'][-1]['reasoning'].startswith('abcd')
    assert follow['max_tokens']==996 and not follow.get('stream')
    body=reply.json()
    assert body['choices'][0]['message']['content']=='6:10 PM' and 'reasoning' not in body['choices'][0]['message']
    assert body['usage']['completion_tokens']==8 and 'bubble_thinking_cut' not in body
    assert client.get('/metrics',headers=AUTH).json()['thinking_cuts']==1
    # Streaming clients get the finished reply replayed as events.
    streamed=ask(stream=True)
    assert streamed.headers['content-type'].startswith('text/event-stream') and '6:10 PM' in streamed.text
    # Tool calls in a stream fold into one call.
    seen.clear(); thinking[:]=[]
    def tools_handler(request):
        if request.url.path=='/metrics': return httpx.Response(200,text='')
        return httpx.Response(200,text=sse({'tool_calls':[{'index':0,'id':'t1','function':{'name':'get','arguments':'{"a"'}}]},
                                             {'tool_calls':[{'index':0,'function':{'arguments':': 1}'}}]}))
    app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(tools_handler))
    message=ask().json()['choices'][0]
    assert message['finish_reason']=='tool_calls'
    assert message['message']['tool_calls']==[{'id':'t1','type':'function','function':{'name':'get','arguments':'{"a": 1}'}}]

def test_chat_jobs_record_token_usage_for_plain_and_streamed_replies(running):
    client,app,manager,scanner,_=running
    app.state.config['chat_models']={'swift':{'resource':'swift','url':'http://swift.test','model':'/m','context':65536}}
    app.state.config['backends']['swift']={'engine':'vllm'}
    seen=[]
    def handler(request):
        if request.url.path=='/metrics': return httpx.Response(200,text='')
        body=json.loads(request.content); seen.append(body)
        usage={'prompt_tokens':40,'completion_tokens':120,'total_tokens':160,'completion_tokens_details':{'reasoning_tokens':70}}
        if body.get('stream'): return httpx.Response(200,text=sse({'content':'hi'},usage=usage))
        return httpx.Response(200,json={'choices':[{'message':{'content':'hi'},'finish_reason':'stop'}],'usage':usage})
    app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    job=lambda reply: client.get(f"/v1/jobs/{reply.headers['X-Bubble-Job-ID']}",headers=AUTH).json()
    plain=client.post('/v1/chat/completions',headers=AUTH,json={'model':'swift','messages':[{'role':'user','content':'hi'}]})
    assert job(plain)['metadata']['usage']=={'prompt_tokens':40,'completion_tokens':120,'reasoning_tokens':70}
    streamed=client.post('/v1/chat/completions',headers=AUTH,json={'model':'swift','stream':True,'messages':[{'role':'user','content':'hi'}]})
    assert '"usage"' in streamed.text and seen[-1]['stream_options']=={'include_usage':True}
    assert job(streamed)['metadata']['usage']['completion_tokens']==120
    # Stored usage survives the metadata clean-up the scheduler runs on start.
    from model_router.mac.scheduler import Scheduler
    assert Scheduler._safe_metadata({'usage':{'completion_tokens':5,'bogus':1,'prompt_tokens':-1}},allow_artifact=True)=={'usage':{'completion_tokens':5}}
    assert 'usage' not in Scheduler._safe_metadata({'usage':{'completion_tokens':5}})

def test_splash_mode_falls_back_to_vllm_then_mlx():
    base={'serving_mode':'splash','backends':{
        'swift':{'command':['mlx'],'modes':{'vllm':{'engine':'vllm'},'splash':{'engine':'splash'}}},
        'vlm':{'command':['mlx'],'modes':{'vllm':{'engine':'vllm'}}},
        'audio':{'command':['mlx']}}}
    resolved=apply_serving_mode(base)['backends']
    assert [resolved[n].get('engine') for n in ('swift','vlm','audio')]==['splash','vllm',None]

def test_status_reports_serving_mode(running):
    client,app,*_=running
    assert client.get('/router/status',headers=AUTH).json()['serving_mode']=='mlx'


def test_manual_auto_request_stays_on_loaded_capable_model(routed):
    client,manager,seen=routed
    client.app.state.config.update(shelf={'prefer_loaded':['swift','arex']},route_switch_after=1,
                                   route_targets={'general':'ornith','swift':'swift','reasoner':'arex'})
    manager.is_loaded=lambda resource: resource=='swift'
    # Laya prefers AREX, but a person is waiting and Swift is loaded.
    reply=ask(client,'auto','should I quit my job')
    assert reply.json()['model']=='swift' and reply.headers['X-Bubble-Route'].endswith('manual-kept-loaded:swift')
    job=client.get(f"/v1/jobs/{reply.headers['X-Bubble-Job-ID']}",headers=AUTH).json()
    assert job['metadata']['lane']=='manual'
    # Explicit model names still switch.
    assert ask(client,'arex','should I quit my job').json()['model']=='arex'
    # Background work routes normally; let it run without the grace window.
    scheduler=client.app.state.scheduler;scheduler.shelf_status();scheduler.instance.shelf_grace_seconds=0
    reply=client.post('/v1/chat/completions',headers={**AUTH,'x-source':'background'},
                      json={'model':'auto','messages':[{'role':'user','content':'should I quit my job'}]})
    assert reply.json()['model']=='arex'
    job=client.get(f"/v1/jobs/{reply.headers['X-Bubble-Job-ID']}",headers=AUTH).json()
    assert job['metadata']['lane']=='background'
    # Ornith is not a listed tier, so a request it cannot serve still reloads.
    manager.is_loaded=lambda resource: resource=='ornith'
    assert ask(client,'auto','should I quit my job').json()['model']=='arex'


def test_manual_preference_is_off_by_default_and_other_sources_have_no_lane(routed):
    client,manager,seen=routed
    client.app.state.config.update(route_switch_after=1,route_targets={'general':'ornith','swift':'swift','reasoner':'arex'})
    manager.is_loaded=lambda resource: resource=='swift'
    assert ask(client,'auto','should I quit my job').json()['model']=='arex'
    reply=client.post('/v1/chat/completions',headers={**AUTH,'x-source':'mirofish-task-profiles'},
                      json={'model':'swift','messages':[{'role':'user','content':'hi'}]})
    job=client.get(f"/v1/jobs/{reply.headers['X-Bubble-Job-ID']}",headers=AUTH).json()
    assert 'lane' not in job['metadata']


def test_shelve_controls_and_status(running):
    client,*_=running
    admin={'Authorization':'Bearer admin-only'}
    assert client.post('/router/control/shelve',headers=AUTH,json={'seconds':60}).status_code==403
    body=client.post('/router/control/shelve',headers=admin,json={'seconds':60}).json()
    assert body['shelved'] and body['operator_hold_seconds']>59
    assert client.get('/router/status',headers=AUTH).json()['shelf']['shelved']
    assert client.post('/router/control/shelve',headers=admin,json={'seconds':'soon'}).status_code==422
    assert not client.post('/router/control/unshelve',headers=admin,json={}).json()['shelved']
    assert client.get('/router/queues',headers=AUTH).json()['shelf']['held_background_jobs']==0

def test_resource_slots_come_from_vllm_max_num_seqs():
    backends={'swift':{'engine':'vllm','command':['vllm','serve','m','--max-num-seqs','4']},
              'arex':{'command':['mlx_vlm','--max-num-seqs','1']},'ornith':{'engine':'vllm','command':['vllm','serve','m']}}
    assert resource_slots(backends)=={'swift':4}


def test_chat_kv_tokens_caps_the_output_reserve():
    prompt={'messages':[{'role':'user','content':'x'*30000}]}
    base=chat_kv_tokens(prompt)-2048
    assert 10000<=base<=10100
    assert chat_kv_tokens({**prompt,'max_tokens':100000})==base+4096
    assert chat_kv_tokens({**prompt,'max_tokens':512})==base+512
    assert vllm_kv_capacity('# HELP x\nvllm:cache_config_info{block_size="16",kv_cache_size_tokens="5000"} 1.0')==(5000,32)
    assert vllm_kv_capacity('vllm:kv_cache_usage_perc 0.1') is None


# Opening lines as the Taste Gate loop engine and OpenClaw cron write them.
LOOP_PROMPT=('TASTE GATE LOOP turn for loop "Daily AI & business trends" (loop_daily-trends), '
             'run run_1793156363510_da19fe2993.\nPHASE: write the message draft.')
LOOP_MARKERS={'loop_id':'loop_daily-trends','loop_run_id':'run_1793156363510_da19fe2993','loop_name':'Daily AI & business trends'}
CRON_PROMPT=('[cron:a99ee3c2-7964-4d58-8dff-9cae152df006 donna-daily-brief] Morning brief for Boss.\n'
             'Current time: Wednesday\n\nThis is an unattended scheduled run.')
CRON_MARKERS={'cron_id':'a99ee3c2-7964-4d58-8dff-9cae152df006','cron_name':'donna-daily-brief'}

def user(content): return {'role':'user','content':content}

def test_turn_markers_read_the_message_that_started_the_turn():
    # OpenClaw's timestamp prefix and the tool traffic after the prompt do not hide it.
    turn=[{'role':'system','content':'You are Mot.'},user('[Wed 2026-10-28 08:29 GMT+5:30] '+LOOP_PROMPT),
          {'role':'assistant','content':None,'tool_calls':[{'id':'t','function':{'name':'exec','arguments':'{}'}}]},
          {'role':'tool','content':'TASTE GATE LOOP turn for loop "Other" (loop_other), run run_2.'}]
    assert turn_markers(turn)==LOOP_MARKERS
    named=LOOP_PROMPT.replace('Daily AI & business trends','The "big" one')
    assert turn_markers([user([{'type':'text','text':named}])])['loop_name']=='The "big" one'
    # A retried loop turn opens with the engine's note; only loop-tagged requests look past it.
    retry=[user('NOTE: your previous turn in this session hit the time limit.\n\n'+LOOP_PROMPT)]
    assert turn_markers(retry)=={} and turn_markers(retry,search=True)==LOOP_MARKERS
    # Loop sessions also look past a user-role message OpenClaw added after the prompt.
    added=[user(LOOP_PROMPT),{'role':'assistant','content':'working'},user('Exec finished: ok')]
    assert turn_markers(added)=={} and turn_markers(added,search=True)==LOOP_MARKERS
    assert turn_markers([user(CRON_PROMPT)])==CRON_MARKERS
    # A person quoting a marker, a chat after an earlier cron turn, and malformed input name nothing.
    assert turn_markers([user('why did this fail? '+LOOP_PROMPT)])=={}
    assert turn_markers([user('why did this fail?\n'+LOOP_PROMPT)])=={}
    assert turn_markers([user('look: [cron:abc daily] ran twice')])=={}
    assert turn_markers([user(CRON_PROMPT),{'role':'assistant','content':'brief'},user('thanks, now the weather')])=={}
    assert turn_markers(None)=={} and turn_markers(['text',{'role':'user'}])=={}

def test_job_origin_labels_who_sent_the_work():
    policy={**SHELF_DEFAULTS,'background_sources':['background','cron','loop','mirofish-facilitated-study']}
    origin=lambda source='',agent='',markers=None: job_origin({'source':source,'agent-id':agent},markers or {},policy)
    assert origin('loop',markers=LOOP_MARKERS)=={'origin':'loop','origin_label':'Loop · Daily AI & business trends',**LOOP_MARKERS}
    assert origin('loop')=={'origin':'loop','origin_label':'Loop'}
    # An untagged loop or cron turn keeps its manual lane but says what it is.
    assert origin(markers=LOOP_MARKERS)['origin_label']=='Loop · Daily AI & business trends'
    assert origin(markers=CRON_MARKERS)=={'origin':'cron','origin_label':'Cron · donna-daily-brief',**CRON_MARKERS}
    assert origin('background',markers=CRON_MARKERS)['origin_label']=='Cron · donna-daily-brief'
    assert origin('cron')=={'origin':'cron','origin_label':'Cron'}
    assert origin()=={'origin':'manual','origin_label':'Manual'}
    assert origin('manual','coding-benchmark')['origin_label']=='Manual · coding-benchmark'
    assert origin('background','main')=={'origin':'background','origin_label':'Background · main'}
    assert origin('background')['origin_label']=='Background'
    assert origin('mirofish-facilitated-study','validator')['origin_label']=='Background · mirofish-facilitated-study'
    # Other declared sources keep their own name, even when a message looks like a loop prompt.
    assert origin('memory-poc',markers=LOOP_MARKERS)=={'origin':'experiment','origin_label':'Experiment · memory-poc'}
    assert origin('synthetic-reader-eval','context-options-eval')['origin']=='experiment'
    assert origin('dashboard','bubble-dashboard')=={'origin':'other','origin_label':'dashboard · bubble-dashboard'}
    assert origin('opencode','opencode')['origin_label']=='opencode'

def test_chat_jobs_record_their_origin_and_keep_header_lanes(routed):
    client,manager,seen=routed
    scheduler=client.app.state.scheduler;scheduler.shelf_status();scheduler.instance.shelf_grace_seconds=0
    def job(headers,text):
        reply=client.post('/v1/chat/completions',headers={**AUTH,**headers},
                          json={'model':'swift','messages':[{'role':'system','content':'You are Mot.'},user(text)]})
        assert reply.status_code==200
        return client.get(f"/v1/jobs/{reply.headers['X-Bubble-Job-ID']}",headers=AUTH).json()['metadata']
    loop=job({'x-source':'loop'},'[Wed 2026-10-28 08:29 GMT+5:30] '+LOOP_PROMPT)
    assert loop['lane']=='background' and loop['source']=='loop'
    assert {key:loop[key] for key in ('origin','origin_label',*LOOP_MARKERS)}=={'origin':'loop','origin_label':'Loop · Daily AI & business trends',**LOOP_MARKERS}
    # The fields survive the metadata clean-up the scheduler runs on start.
    assert Scheduler._safe_metadata(loop,allow_artifact=True)['loop_run_id']=='run_1793156363510_da19fe2993'
    cron=job({},CRON_PROMPT)
    assert cron['lane']=='manual' and cron['origin_label']=='Cron · donna-daily-brief' and cron['cron_name']=='donna-daily-brief'
    chat=job({},'hi')
    assert chat['lane']=='manual' and (chat['origin'],chat['origin_label'])==('manual','Manual') and 'loop_id' not in chat
    other=job({'x-source':'memory-poc'},LOOP_PROMPT)
    assert 'lane' not in other and other['origin_label']=='Experiment · memory-poc' and 'loop_id' not in other

def test_auto_routes_with_ettin_when_configured(routed):
    client,manager,seen=routed
    class Ettin:
        def route_request(self,text):
            return {'route':'reasoner','scores':{'general':0.05,'swift':0.05,'reasoner':0.9}}
    client.app.state.route_classifier=Ettin()
    r=ask(client,'auto','what is the weather like')
    assert r.json()['model']=='qwen' and r.headers['X-Bubble-Route'].startswith('qwen;ettin:reasoner:0.9')

def test_ettin_failure_falls_back_to_laya(routed):
    client,manager,seen=routed
    class Broken:
        def route_request(self,text):raise RuntimeError('boom')
    client.app.state.route_classifier=Broken()
    r=ask(client,'auto','what is the weather like')
    assert r.json()['model']=='ornith' and r.headers['X-Bubble-Route'].startswith('ornith;laya:general')
    assert client.app.state.metrics['route_ettin_errors']==1

def test_client_disconnect_cancels_its_job(routed):
    client,manager,seen=routed
    app=client.app
    async def slow(request):
        await asyncio.sleep(30); return httpx.Response(200,json={'choices':[{'message':{'content':'late'}}]})
    app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(slow))
    body=json.dumps({'model':'ornith','messages':[{'role':'user','content':'hi'}]}).encode()
    async def call():
        sent=[False]; gone_at=time.time()+0.5
        async def receive():  # like uvicorn: the disconnect is queued (immediately readable) once the socket closes
            if not sent[0]: sent[0]=True; return {'type':'http.request','body':body,'more_body':False}
            while time.time()<gone_at: await asyncio.sleep(0.05)
            return {'type':'http.disconnect'}
        async def send(message): pass
        scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST','scheme':'http','path':'/v1/chat/completions',
               'raw_path':b'/v1/chat/completions','query_string':b'','root_path':'','server':('test',80),'client':('test',1),
               'headers':[(b'authorization',b'Bearer test-only'),(b'content-type',b'application/json'),(b'content-length',str(len(body)).encode())]}
        task=asyncio.ensure_future(app(scope,receive,send))
        for _ in range(100):
            await asyncio.sleep(0.1)
            if task.done(): break
        ended=task.done(); task.cancel()
        rows=sqlite3.connect(app.state.scheduler.path).execute('select state from jobs').fetchall()
        return ended,rows
    ended,rows=client.portal.call(call)
    assert ended, 'handler should give up once the client disconnects'
    assert rows==[('cancelled',)], rows
