from contextlib import asynccontextmanager
import json
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
import httpx
import pytest
from fastapi.testclient import TestClient
from model_router.mac.app import apply_tool_reasoning_defaults, create_app, prepare_chat, safe_asset
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
        async def generate_video(self,*args): raise AssertionError('invalid payload should not queue')
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
    body={'model':'auto','tools':EXEC_TOOL,'stream':True,'messages':[{'role':'user','content':'PRINT make a fox video'}]}
    r=client.post('/v1/chat/completions',headers=AUTH,json=body)
    assert r.status_code==200 and r.headers['X-Bubble-Route'].startswith('swift;') and 'printed-command' in r.headers['X-Bubble-Route']
    assert [url for url,_ in seen][-2:]==['http://ornith.test/v1/chat/completions','http://swift.test/v1/chat/completions']
    # After tool results, paths in the answer are normal and stay on Ornith.
    after_tool=body|{'stream':False,'messages':body['messages']+[{'role':'assistant','content':'','tool_calls':[]},{'role':'tool','tool_call_id':'c1','content':'ok'}]}
    assert client.post('/v1/chat/completions',headers=AUTH,json=after_tool).json()['model']=='ornith'


def test_buffered_tool_call_replays_as_stream(routed):
    client,manager,seen=routed
    body={'model':'auto','tools':EXEC_TOOL,'stream':True,'stream_options':{'include_usage':True},'messages':[{'role':'user','content':'CALL make a fox video'}]}
    r=client.post('/v1/chat/completions',headers=AUTH,json=body)
    assert r.status_code==200 and r.headers['X-Bubble-Route'].startswith('ornith;') and r.headers['content-type'].startswith('text/event-stream')
    events=[json.loads(line[6:]) for line in r.text.splitlines() if line.startswith('data: {')]
    assert events[0]['choices'][0]['delta']['tool_calls'][0]['index']==0
    assert events[0]['choices'][0]['delta']['tool_calls'][0]['function']['name']=='exec'
    assert events[-1]['choices'][0]['finish_reason']=='tool_calls' and events[-1]['usage']=={'total_tokens':5}
    assert r.text.endswith('data: [DONE]\n\n')


def test_fabricated_media_in_first_reply_escalates(routed):
    client,manager,seen=routed
    body={'model':'auto','tools':EXEC_TOOL,'messages':[{'role':'user','content':'FAKEMEDIA generate a video'}]}
    r=client.post('/v1/chat/completions',headers=AUTH,json=body)
    assert r.status_code==200 and r.json()['model']=='swift'
    assert 'escalated-from-ornith:printed-command' in r.headers['X-Bubble-Route']
    assert manager.calls==['ornith','swift']
    assert 'MEDIA:' not in r.json()['choices'][0]['message']['content']
    # An attachment after actual tool output is allowed through unchanged.
    after=body|{'messages':body['messages']+[{'role':'assistant','content':'','tool_calls':[]},{'role':'tool','tool_call_id':'c1','content':'artifact path'}]}
    r=client.post('/v1/chat/completions',headers=AUTH,json=after)
    assert r.json()['model']=='ornith'


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
