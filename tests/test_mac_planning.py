import json
from types import SimpleNamespace
import httpx
import pytest
from fastapi.testclient import TestClient
from test_mac_app import routed,AUTH
from model_router.mac.simple_planning import PlanningClassifier,planning_issue
from model_router.mac.planning_server import create_app,validate_payload,final_text

TEXT='Draft a simple 30-minute agenda for our hobby club with introductions and questions.'

@pytest.fixture
def planning(routed):
    client,manager,seen=routed
    app=client.app
    app.state.config['planning_router']={'enabled':True,'model':'thinkingcap'}
    app.state.config['chat_models']['thinkingcap']={'resource':'thinkingcap','url':'http://planning.test',
        'model':'thinkingcap','planning_only':True,'sampling':{'temperature':0},'context':32768}
    app.state.config['escalate']['thinkingcap']='swift'
    app.state.planning_classifier=SimpleNamespace(predict=lambda text:{'eligible':True,'score':.9999})
    app.state.config['route_switch_after']=3
    return routed

def ask(client,text=TEXT,**kwargs):
    return client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto',
        'messages':[{'role':'user','content':text}],**kwargs})

def test_simple_plan_uses_specialized_model_even_when_swift_loaded(planning):
    client,manager,seen=planning
    manager.is_loaded=lambda name:name=='swift'
    r=ask(client)
    assert r.status_code==200 and r.json()['model']=='thinkingcap'
    assert 'planning:' in r.headers['X-Bubble-Route']

@pytest.mark.parametrize('text',[
 'Draft a simple plan for a production database migration.',
 'Make a simple checklist and send it to the team.',
 'Plan how to invest my retirement savings.',
 'Outline the optimal allocation with conflicting deadlines.',
 'Describe what an agenda means; do not write one.',
 'Write a plan using the attached image.',
])
def test_unsafe_or_complex_plans_use_stronger_route(planning,text):
    client,manager,_=planning
    manager.is_loaded=lambda name:name=='thinkingcap'
    client.app.state.config['shelf']={'prefer_loaded':['thinkingcap']}
    r=ask(client,text)
    assert r.status_code==200 and r.json()['model']=='swift'
    assert 'planning-fallback:' in r.headers['X-Bubble-Route']

def test_classifier_rejection_or_error_never_uses_fast_general(planning):
    client,*_=planning
    for predictor in [lambda text:{'eligible':False},lambda text:(_ for _ in ()).throw(ValueError('bad'))]:
        client.app.state.planning_classifier=SimpleNamespace(predict=predictor)
        r=ask(client);assert r.json()['model']=='swift'

def test_specialized_simple_planning_head_can_refine_old_broad_reasoning_route(planning):
    client,*_=planning
    client.app.state.scanner.route_request=lambda text:{'route':'reasoner','scores':{'general':.05,'swift':.05,'reasoner':.9}}
    client.app.state.config['route_targets']={'reasoner':'swift'}
    r=ask(client);assert r.json()['model']=='thinkingcap'

@pytest.mark.parametrize('contract',[{'response_format':{'type':'json_object'}},{'tool_choice':'required'}])
def test_caller_output_contract_uses_swift(planning,contract):
    client,*_=planning
    assert ask(client,**contract).json()['model']=='swift'

def test_strong_planning_fallback_cannot_be_downgraded_by_loaded_preference(planning):
    client,manager,_=planning
    manager.is_loaded=lambda name:name=='ornith'
    client.app.state.config['shelf']={'prefer_loaded':['ornith']}
    assert ask(client,'Draft a production database migration plan.').json()['model']=='swift'

def test_disabled_planning_and_explicit_existing_model_remain_unchanged(planning):
    client,*_=planning
    r=ask(client,model='swift');assert r.json()['model']=='swift'
    client.app.state.config['planning_router']['enabled']=False
    r=ask(client);assert r.json()['model']!='thinkingcap'
    assert ask(client,model='thinkingcap').status_code==422

def test_planning_only_model_cannot_be_sticky_for_unrelated_work(planning):
    client,manager,_=planning;manager.is_loaded=lambda name:name=='thinkingcap'
    r=ask(client,'Hi there');assert r.json()['model']=='ornith'

def test_planning_does_not_receive_tools_or_stream_and_native_call_falls_back(planning):
    client,*_=planning;seen=[]
    def handler(request):
        body=json.loads(request.content);seen.append(body)
        if request.url.host=='planning.test':
            assert not body.get('tools') and body['stream'] is False
            return httpx.Response(200,json={'choices':[{'message':{'content':'','tool_calls':[{'function':{'name':'send','arguments':'{}'}}]},'finish_reason':'tool_calls'}]})
        return httpx.Response(200,json={'choices':[{'message':{'content':'Draft only'},'finish_reason':'stop'}]})
    client.app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tools=[{'type':'function','function':{'name':'send','parameters':{'type':'object','properties':{}}}}]
    r=ask(client,tools=tools,stream=True)
    assert r.status_code==200 and 'swift;' in r.headers['X-Bubble-Route']
    assert 'tool_calls' not in r.text and '[DONE]' in r.text
    assert len(seen)==2 and seen[1]['tools']==tools

def test_planning_backend_failure_falls_back_to_swift(planning):
    client,*_=planning
    def handler(request):
        return httpx.Response(503) if request.url.host=='planning.test' else httpx.Response(200,json={'choices':[{'message':{'content':'draft'}}]})
    client.app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r=ask(client);assert r.status_code==200 and r.json()['model']=='swift'

def test_tool_continuation_and_recent_tool_context_are_excluded(planning):
    client,*_=planning
    messages=[{'role':'user','content':TEXT},{'role':'assistant','content':None,
        'tool_calls':[{'id':'x','type':'function','function':{'name':'read','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'x','content':'synthetic fixture'}]
    r=ask(client,messages=messages);assert r.json()['model']!='thinkingcap'
    r=ask(client,messages=messages+[{'role':'user','content':TEXT}]);assert r.json()['model']=='swift'

@pytest.mark.parametrize('text,issue',[
 ('Draft a household checklist; do not send or schedule anything.',None),
 ('Make a simple packing plan. Do not book anything.',None),
 ('Plan a Rust implementation.','complex-or-consequential'),
 ('Explain the quote "make a simple plan".','discussion-only'),
 ('Plan a meeting and book it.','action'),
])
def test_scope_boundaries(text,issue):
    assert planning_issue([{'role':'user','content':text}],text)==issue

def test_classifier_fails_closed_on_incomplete_or_invalid_prediction(tmp_path):
    (tmp_path/'planning_policy.json').write_text(json.dumps({'question':{'simple_planning':{}},'confidence_threshold':.99}))
    for result in [
        {'answers':{'simple_planning':{'probabilities':{'A':1,'B':0}}},'usage':{'truncated':True}},
        {'answers':{'simple_planning':{'probabilities':{'A':float('nan'),'B':0}}}},
        {'answers':{'simple_planning':{'probabilities':{'A':1}}}},
    ]:
        c=PlanningClassifier(tmp_path,loader=lambda path:SimpleNamespace(predict=lambda *args:result))
        with pytest.raises((KeyError,ValueError)):c.predict(TEXT)

@pytest.fixture
def server():
    app=create_app(loader=lambda:(None,None),generator=lambda *args:{'choices':[{'message':{'content':'Draft only'}}]},api_key='fixture-only')
    with TestClient(app) as client:yield client

def test_server_auth_health_and_text_only_contract(server):
    assert server.get('/health').status_code==401
    h={'Authorization':'Bearer fixture-only'}
    assert server.get('/health',headers=h).json()['thinking'] is False
    assert server.post('/v1/chat/completions',headers=h,json={'messages':[{'role':'user','content':TEXT}]}).status_code==200
    assert server.post('/v1/chat/completions',headers=h,json={'messages':[{'role':'user','content':[{'type':'image_url','image_url':{}}]}]}).status_code==422
    assert server.post('/v1/chat/completions',headers=h,json={'messages':[{'role':'tool','content':'fixture'}]}).status_code==422
    assert server.post('/v1/chat/completions',headers=h,json={'messages':[{'role':'user','content':TEXT}],'tools':[{}]}).status_code==422

def test_server_caps_output_and_does_not_accept_control_output():
    messages,cap=validate_payload({'messages':[{'role':'user','content':TEXT}],'max_tokens':8000})
    assert cap==2048
    with pytest.raises(ValueError):final_text('<|tool_call>call:send{}<tool_call|>',SimpleNamespace())
    with pytest.raises(ValueError):final_text('<|channel>thought\nunfinished',SimpleNamespace())
    assert final_text('Draft only',SimpleNamespace())=='Draft only'
