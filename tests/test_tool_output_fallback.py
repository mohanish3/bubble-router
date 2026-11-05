"""Regression cases use synthetic protocol data only, no real tools/models."""
import copy
import json
import httpx
import pytest
from model_router.mac.app import unexecuted_tool_output
from test_mac_app import routed, AUTH, EXEC_TOOL

CALL={'id':'fixture-call','type':'function','function':{'name':'exec','arguments':'{"command":"fixture only"}'}}
GROUP={'type':'process_group','processGroupId':'fixture','state':'Running','runningCount':1,'count':1,'results':[]}
TIMER={'type':'timer','name':'fixture','triggered':True,'next':None,'status':'ok'}
def completion(content='',calls=None):
    msg={'role':'assistant','content':content}
    if calls:msg['tool_calls']=calls
    return {'id':'fixture','choices':[{'index':0,'message':msg,'finish_reason':'tool_calls' if calls else 'stop'}]}

@pytest.mark.parametrize('content,reason',[
    (json.dumps(GROUP),'printed-tool-result'),
    ('Checking now.\n'+json.dumps(GROUP)+'\nDone.','printed-tool-result'),
    ('```json\n'+json.dumps(TIMER)+'\n```','printed-tool-result'),
    ('<tool_call>\n<function=exec>\n<parameter=command>fixture</parameter>\n</function>\n</tool_call>','printed-tool-call'),
    (json.dumps({'name':'exec','arguments':{'command':'fixture'}}),'printed-tool-call'),
    (json.dumps({'function':CALL['function']}),'printed-tool-call'),
    (json.dumps({'tool_calls':[CALL]}),'printed-tool-call'),
])
def test_known_leak_envelopes_detected(content,reason):
    assert unexecuted_tool_output(completion(content),EXEC_TOOL)==reason
    assert unexecuted_tool_output(completion(content,[CALL]),EXEC_TOOL) is None

@pytest.mark.parametrize('content',[
    'The job status is ready.',json.dumps({'status':'ok'}),json.dumps({'type':'process_group'}),
    json.dumps({'name':'other','arguments':{'x':1}}),
    'For example, '+json.dumps(GROUP),
    'Use the exec tool to check this.', '```json\n{"count":1,"results":[]}\n```',
    json.dumps({'name':[],'arguments':{}}),json.dumps({'function':{'name':{},'arguments':{}}}),
])
def test_ordinary_answers_and_invalid_names_remain_text(content):
    assert unexecuted_tool_output(completion(content),EXEC_TOOL) is None

@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('bad',[json.dumps(GROUP),json.dumps(TIMER),json.dumps({'name':'exec','arguments':{'command':'fixture'}})])
def test_leaked_output_retries_original_request_and_returns_only_fallback(routed,stream,bad):
    client,manager,_=routed;seen=[]
    def handler(request):
        body=json.loads(request.content);seen.append((request.url.host,body))
        return httpx.Response(200,json=completion(bad) if request.url.host=='ornith.test' else completion(calls=[CALL]))
    client.app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    body={'model':'auto','stream':stream,'tools':EXEC_TOOL,'messages':[{'role':'user','content':'Check fixture state with the tool.'}]}
    r=client.post('/v1/chat/completions',headers=AUTH,json=body)
    assert r.status_code==200
    assert manager.calls==['ornith','swift']
    assert seen[0][1]['messages']==seen[1][1]['messages']==body['messages']
    assert all(not request['stream'] for _,request in seen)
    assert r.headers['X-Bubble-Route'].startswith('swift;')
    assert 'printed-tool-' in r.headers['X-Bubble-Route']
    if stream:
        events=[json.loads(line[6:]) for line in r.text.splitlines() if line.startswith('data: {')]
        assert events[0]['choices'][0]['delta']['tool_calls'][0]['function']==CALL['function']
        assert events[-1]['choices'][0]['finish_reason']=='tool_calls'
        assert r.text.endswith('data: [DONE]\n\n')
    else:assert r.json()['choices'][0]['message']['tool_calls']==[CALL]
    assert 'processGroupId' not in r.text

@pytest.mark.parametrize('stream',[False,True])
def test_both_models_leaking_output_fail_without_leaking_text(routed,stream):
    client,manager,_=routed
    client.app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=completion(json.dumps(GROUP)))))
    r=client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','stream':stream,'tools':EXEC_TOOL,'messages':[{'role':'user','content':'Check the fixture.'}]})
    assert r.status_code==502 and r.json()['detail']['code']=='unexecuted_tool_output'
    assert manager.calls==['ornith','swift']
    assert 'processGroupId' not in r.text

@pytest.mark.parametrize('last_role',['tool','assistant'])
def test_existing_results_may_be_summarized_without_retry(routed,last_role):
    client,manager,_=routed
    def handler(r):
        body=json.loads(r.content)
        assert not body.get('stream')
        return httpx.Response(200,json=completion(json.dumps(GROUP)))
    client.app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    messages=[{'role':'user','content':'Check fixture.'},{'role':'assistant','content':None,'tool_calls':[CALL]},
              {'role':last_role,'tool_call_id':CALL['id'],'content':'Fixture result'}]
    r=client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','tools':EXEC_TOOL,'messages':messages})
    assert r.status_code==200 and manager.calls==['ornith']
    assert r.json()['choices'][0]['message']['content']==json.dumps(GROUP)

def test_without_tools_json_is_not_retried(routed):
    client,manager,_=routed
    client.app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=completion(json.dumps(GROUP)))))
    r=client.post('/v1/chat/completions',headers=AUTH,json={'model':'auto','messages':[{'role':'user','content':'Return a JSON object.'}]})
    assert r.status_code==200 and manager.calls==['ornith']
