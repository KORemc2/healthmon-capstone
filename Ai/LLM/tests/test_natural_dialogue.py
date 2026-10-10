import copy
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src.natural_dialogue import apply_policy
from src.model_registry import read_registry, cleanup_inventory, select
from src.persona_lab import create_app
from src.persona_runtime import PersonaRuntime, persona_messages
from src.character_lab_schema import CharacterState, Session
from src.character_lab_models import response_context


def context():
    session=Session().model_dump(mode='json')
    return response_context(session,session['state'],'오늘은 영화 봤어.',[],'casual')


def test_opt_in_preserves_baseline_and_does_not_mutate_facts():
    ctx=context(); before=copy.deepcopy(ctx)
    baseline=persona_messages(ctx,'오늘은 영화 봤어.')
    assert apply_policy(baseline,'baseline') is baseline
    natural=persona_messages(ctx,'오늘은 영화 봤어.',profile='natural-v1')
    assert natural[1:]==baseline[1:]
    assert ctx==before
    assert '일상 대화를 운동' in natural[0]['content']


def test_cleanup_protects_shared_paths_and_selection_requires_human_review(tmp_path):
    registry=read_registry();registry['models']['old']={**registry['models']['baseline-expanded'],'status':'Rejected'}
    registry['models']['natural-v1']['status']='Candidate'
    assert not next(row for row in cleanup_inventory(registry) if row['id']=='old')['deletion_candidate']
    path=tmp_path/'registry.json';path.write_text(json.dumps(registry),encoding='utf-8')
    with pytest.raises(ValueError,match='human review'):
        select('natural-v1',path)
    assert select('baseline-expanded',path)['selected_id']=='baseline-expanded'


@pytest.fixture
def client(monkeypatch,tmp_path):
    monkeypatch.setenv('APP_ENV','development');monkeypatch.setenv('PERSONA_LAB_ENABLED','1')
    monkeypatch.setattr('src.model_comparison_api.ROOT',tmp_path)
    monkeypatch.setattr('src.model_comparison_api.subprocess.check_output',lambda *a,**kw:'test-commit\n')
    app=create_app(runtime=PersonaRuntime())
    return TestClient(app,base_url='http://127.0.0.1:8001',client=('127.0.0.1',30000))


def test_comparison_fixed_controls_dummy_labels_and_no_auto_selection(client):
    response=client.post('/api/model-comparison',json={'text':'오늘 영화 봤어.','mode':'dummy'},
        headers={'Origin':'http://127.0.0.1:8001','X-Persona-Lab':'1'})
    assert response.status_code==200,response.text
    result=response.json()
    assert result['same_context_and_seed'] and not result['automatic_selection']
    assert all(not row['is_real_model'] for row in result['results'])
    feedback=client.post('/api/model-feedback',json={'evaluation_id':result['evaluation_id'],'preferred_model':'tie','scores':{}},
        headers={'Origin':'http://127.0.0.1:8001','X-Persona-Lab':'1'})
    assert feedback.status_code==422


def test_comparison_rejects_unknown_or_duplicate_models(client):
    for ids in (['baseline-expanded','unknown'],['natural-v1','natural-v1']):
        response=client.post('/api/model-comparison',json={'text':'안녕','mode':'dummy','model_ids':ids},
            headers={'Origin':'http://127.0.0.1:8001','X-Persona-Lab':'1'})
        assert response.status_code==422


def test_adapter_and_profile_restored_when_generation_fails(monkeypatch):
    import threading
    from types import SimpleNamespace
    from src.model_comparison import compare_models
    class Adapter:
        active_adapter='original'
        peft_config={}
        def load_adapter(self,path,adapter_name):self.peft_config[adapter_name]={}
        def set_adapter(self,key):self.active_adapter=key
        def eval(self):pass
    model=Adapter()
    runtime=SimpleNamespace(lock=threading.Lock(),backend=SimpleNamespace(model=model),dialogue_profile='baseline',_load=lambda:None)
    def fail(*args):raise RuntimeError('test generation failure')
    runtime._one=fail
    with pytest.raises(RuntimeError,match='test generation failure'):
        compare_models(runtime,context(),'hello',['natural-v1'],{},'qwen')
    assert model.active_adapter=='original' and runtime.dialogue_profile=='baseline'
