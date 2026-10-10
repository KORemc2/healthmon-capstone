import copy
import json
from src.persona_runtime import persona_messages
from src.character_lab_schema import Session
from src.character_lab_models import response_context
from src.sft_tasks import training_prompt


def context(text):
    session = Session().model_dump(mode='json')
    return response_context(session, session['state'], text, [], 'casual')


def test_native_turn_order_and_request_survive_projection_without_state_change():
    ctx = context('순서를 설명해줘.')
    ctx['recent'] = [{'user': '기구 없이 해보고 싶어.', 'assistant': '어깨를 천천히 돌려봐.'}]
    before = copy.deepcopy(ctx)
    messages = persona_messages(ctx, '순서를 설명해줘.', profile='natural-v2')
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'user']
    assert messages[-1]['content'] == '순서를 설명해줘.'
    assert json.loads(messages[2]['content'])['speech'] == '어깨를 천천히 돌려봐.'
    assert ctx == before
    assert training_prompt({'prompt_kind': 'character', 'context': ctx, 'user_text': '순서를 설명해줘.'}, 'natural-v2') == messages


def test_general_chat_omits_catalog_but_explicit_quest_retains_authorized_goal():
    ctx = context('오늘 뭐 하지?')
    ctx['candidates'] = [{'id': 'authorized', 'title': '걷기', 'goal': {'target': 100, 'unit': 'steps'}, 'reward_id': 'r'}]
    before = copy.deepcopy(ctx)
    casual = persona_messages(ctx, '집에서 놀고 싶어.', profile='natural-v2')[0]['content']
    quest = persona_messages(ctx, '할 만한 퀘스트 알려줘.', profile='natural-v2')[0]['content']
    assert 'authorized' not in casual
    assert 'authorized' in quest and '"target":100' in quest
    assert ctx == before


def test_memory_and_verified_facts_are_not_lost_by_compaction():
    ctx = context('내 목표가 뭐야?')
    ctx['memories'] = [{'id': 'mem1', 'slot': 'goal', 'content': '하루에 20분 걷기', 'verified': True}]
    ctx['active_quests'] = [{'id': 'q1', 'goal': {'target': 200, 'unit': 'steps'}, 'progress': 50}]
    ctx['activity_summary']['requested_period']['record_count'] = 1
    ctx['activity_summary']['requested_period']['confirmed_steps'] = 175
    system = persona_messages(ctx, '내 목표가 뭐야?', profile='natural-v2')[0]['content']
    facts = json.loads(system.split('\ncontext=', 1)[1])
    assert facts['memories'] == ctx['memories']
    assert facts['active_quests'] == ctx['active_quests']
    assert facts['activity_summary']['requested_period']['confirmed_steps'] == 175
    assert facts['state_facts']['upper_body'] == ctx['character_state']['upper_body']
    assert facts['state_facts']['level'] == ctx['character_state']['level']


def test_new_profile_does_not_change_default_or_v1_prompts():
    ctx = context('안녕.')
    old = persona_messages(ctx, '안녕.')
    persona_messages(ctx, '안녕.', profile='natural-v2')
    assert persona_messages(ctx, '안녕.') == old
    assert persona_messages(ctx, '안녕.', profile='natural-v1')[1] == old[1]
