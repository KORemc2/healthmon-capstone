import copy
import json
from pathlib import Path
import pytest
from src.prompts import messages
from src.persona_runtime import persona_messages
from src.character_lab_models import response_context, CONVERSATION_SYSTEM
from src.character_lab_schema import Session
from src.sft_tasks import training_prompt


def row():
    session=Session().model_dump(mode='json')
    return {'context':response_context(session,session['state'],'오늘은 쉬고 싶어.',[],'casual'),
            'user_text':'오늘은 쉬고 싶어.'}


def test_legacy_training_default_is_unchanged():
    path=Path(__file__).resolve().parents[1]/'data/improvement_v3/train.jsonl'
    value=json.loads(path.read_text(encoding='utf-8').splitlines()[0]);before=copy.deepcopy(value)
    assert training_prompt(value)==messages(value['context'],value['user_text'])
    assert value==before


def test_character_training_matches_serving_prompt_and_preserves_state():
    value=row();before=copy.deepcopy(value)
    assert training_prompt(value,'natural-v1','character')==persona_messages(value['context'],value['user_text'],profile='natural-v1')
    assert value==before


def test_analyzer_target_is_validated_and_uses_separate_task():
    value={'prompt_kind':'conversation_analysis','user_text':'운동하기 싫어',
           'answer':{'target':'exercise','sentiment':'negative','motivation':'low',
                     'character_affection':'neutral','acceptance':'none','analysis_confidence':.95}}
    assert training_prompt(value)[0]['content']==CONVERSATION_SYSTEM
    value['answer']['confidence']=value['answer'].pop('analysis_confidence')
    with pytest.raises(ValueError):training_prompt(value)


def test_unknown_task_fails_before_training():
    with pytest.raises(ValueError,match='Unknown SFT'):training_prompt({'prompt_kind':'invalid'})
