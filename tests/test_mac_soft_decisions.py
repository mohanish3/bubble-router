from types import SimpleNamespace
import pytest
from model_router.mac.soft_decisions import SoftDecisions
from model_router.mac.scheduler import Scheduler

def policy(rate=1):
    return {k:{'enabled':True,'sample_rate':rate,'cooldown_seconds':600,'threshold':.97}
            for k in ('weak_answers','failed_generation')}

def test_sampling_and_cooldown(tmp_path):
    layer=SoftDecisions(tmp_path,policy(0))
    assert not layer.selected('weak_answers','a')
    layer=SoftDecisions(tmp_path,policy())
    assert layer.selected('weak_answers','a')
    assert not layer.selected('weak_answers','b')
    assert layer.selected('failed_generation','b')

@pytest.mark.asyncio
async def test_decision_failure_keeps_answer_and_refusal_never_escalates(tmp_path):
    layer=SoftDecisions(tmp_path,policy())
    def fail(*args): raise ValueError('malformed model output')
    layer._predict=fail
    result={'choices':[{'message':{'content':'This is an ordinary answer.'}}]}
    assert (await layer.weak_answer('question',result))['action']=='review'
    result['choices'][0]['message']['content']="I cannot help with that request."
    assert (await layer.weak_answer('question',result))['action']=='keep'

@pytest.mark.asyncio
async def test_failed_media_advice_is_durable_and_never_retries(tmp_path):
    layer=SoftDecisions(tmp_path,policy())
    states=[]
    def predict(kind,state):
        states.append(state)
        return {'action':'retry','available':True}
    layer._predict=predict
    scheduler=Scheduler(tmp_path/'jobs.sqlite')
    scheduler.failure_advisor=layer.failed_generation
    calls=[]
    async def fail(job):
        calls.append(job.id)
        raise TimeoutError('private prompt secret timed out')
    job=await scheduler.submit('minimax-h3',5,fail,prompt='private prompt')
    with pytest.raises(TimeoutError): await job.future
    await scheduler._idle.wait()
    assert calls==[job.id]
    assert job.metadata['failure_advice']['automatic_retry'] is False
    assert 'private' not in states[0] and 'secret' not in states[0]
    await scheduler.close()
    restored=Scheduler(tmp_path/'jobs.sqlite')
    assert (await restored.get(job.id)).metadata['failure_advice']['action']=='retry'
    await restored.close()
