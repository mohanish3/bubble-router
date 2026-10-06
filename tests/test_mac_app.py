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
    assert response.status_code==403
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
    with pytest.raises(ValueError):prepare_chat({'messages':[{'role':'user','content':'A'*4000}],'context_length':1024})
    with pytest.raises(ValueError):prepare_chat({'messages':[{'role':'user','content':'ok'}],'max_tokens':100000})
