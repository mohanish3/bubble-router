from contextlib import asynccontextmanager
import json
from pathlib import Path
import httpx
import pytest
from fastapi.testclient import TestClient
from model_router.mac.app import create_app, prepare_chat, safe_asset
from model_router.mac.scheduler import Scheduler
from model_router.mac.security import SecurityBlocked

class Manager:
    def __init__(self): self.active=0;self.calls=[]
    @asynccontextmanager
    async def lease(self,name,**kwargs):
        self.calls.append(name);self.active+=1
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
    manager=Manager();scanner=Scanner();scheduler=LazyScheduler(tmp_path/'state/jobs.sqlite3',manager)
    config={'api_key':'test-only','admin_key':'admin-only','provider_key':'provider-only','backend_key':'backend-only','qwen_model':'/local/selected-model','backends':{'qwen-vlm':{}},'audio_models':{'tts':'/local/tts','asr':'/local/asr'}}
    app=create_app(tmp_path,injected=(config,manager,scheduler,scanner,object()))
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

def test_context_is_admitted_without_silent_truncation():
    from model_router.mac.app import fits_context
    payload,_=prepare_chat({'messages':[{'role':'user','content':'A'*4000}],'context_length':1024})
    assert payload['_context_length']==1024 and not fits_context(payload,1024)
    with pytest.raises(ValueError):prepare_chat({'messages':[{'role':'user','content':'ok'}],'context_length':999999})
    with pytest.raises(ValueError):prepare_chat({'messages':[{'role':'user','content':'ok'}],'max_tokens':100000})


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
    history=[{'role':'user','content':'background '*12000},{'role':'assistant','content':'noted'}]
    r=client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','messages':history+[{'role':'user','content':'what is the weather like'}]})
    assert r.status_code==200 and r.json()['model']=='arex' and r.headers['X-Bubble-Route'].endswith(';long-context')
    too_long=[{'role':'user','content':'x'*400000}]
    assert client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','messages':too_long}).status_code==422
