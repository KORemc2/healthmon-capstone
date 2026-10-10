"""Actual baseline/prompt-only/trained comparisons, including own-history turns."""
import copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.persona_runtime import PersonaRuntime,DEFAULT_GENERATION
from src.character_lab_schema import CharacterState,Session
from src.character_lab_models import response_context
from src.model_comparison import compare_models
from src.model_registry import read_registry

SCENARIOS=[
    {'id':'reluctance','turns':['아 운동하기 싫어.','그냥 오늘은 쉬고 싶어. 운동 얘기는 그만하자.','그럼 운동 말고 집에서 할 만한 거 하나만 골라줘.']},
    {'id':'recommend','turns':['운동 뭐 하지?','집에 있고 도구는 없어. 5분 정도만 할래.','방금 말한 걸 순서대로 짧게 정리해줘.']},
    {'id':'bored','turns':['하 심심하다.','운동 말고, 집에서 혼자 할 수 있는 거로.','그중 하나만 골라서 지금 바로 시작하게 해줘.']},
]

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    dest=ROOT/'runs/dialogue-repair-20261010/repair-conversations.json'
    if dest.exists():raise RuntimeError('Preserve existing evaluation; choose a new version')
    models=['baseline-expanded','baseline-prompt-v2','repair-v5']
    runtime=PersonaRuntime();session=Session().model_dump(mode='json')
    settings=dict(DEFAULT_GENERATION,seed=20261010)
    report={'models':models,'adapter_metadata':{k:read_registry()['models'][k] for k in models},
        'generation':settings,'method':'Same context and seed for first turns; each model own actual previous answers for follow-up',
        'first_turn':[],'conversations':[],'axis_probes':[],'human_review':False,'automatic_selection':False,'completed':False}
    def save():dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    def ctx(text,value=50,history=None,overrides=None):
        state=CharacterState(**dict({k:value for k in ('activity','affection','initiative')},**(overrides or {}))).model_dump(mode='json')
        context=response_context(session,state,text,[],'casual');context['recent']=copy.deepcopy(history or []);return context
    started=time.time();save()
    for scenario in SCENARIOS:
        for band,value in [('low',20),('neutral',50),('high',80)]:
            text=scenario['turns'][0]
            result=compare_models(runtime,ctx(text,value),text,models,settings)
            report['first_turn'].append({'scenario_id':scenario['id'],'profile_id':band,**result});save()
            print(json.dumps({'phase':'first','scenario':scenario['id'],'band':band,'rows':len(report['first_turn'])*len(models)}),flush=True)
    for scenario in SCENARIOS:
        first=next(x for x in report['first_turn'] if x['scenario_id']==scenario['id'] and x['profile_id']=='neutral')
        for model in models:
            row=next(x for x in first['results'] if x['model_id']==model)
            conversation={'scenario_id':scenario['id'],'model_id':model,'turns':[{'user':scenario['turns'][0],'result':row}]}
            report['conversations'].append(conversation)
            history=[{'user':scenario['turns'][0],'assistant':row['output']['speech']}]
            for index,text in enumerate(scenario['turns'][1:],2):
                result=compare_models(runtime,ctx(text,history=history),text,[model],dict(settings,seed=20261009+index))
                row=result['results'][0];conversation['turns'].append({'user':text,'result':row,'context':result['context']})
                history.append({'user':text,'assistant':row['output']['speech']});save()
            print(json.dumps({'phase':'conversation','scenario':scenario['id'],'model':model}),flush=True)
    # Separate single-axis probes avoid claiming a joint state change isolates
    # one trait. These are examples, not a population-level causal estimate.
    for axis,text in [('activity','몸을 조금 풀고 싶은데 뭐부터 하면 돼?'),('affection','오늘도 와서 네 얘기 좀 들으려 해.'),('initiative','집에서 가볍게 시간을 보낼 방법을 알려줘.')]:
        for value in (20,80):
            result=compare_models(runtime,ctx(text,overrides={axis:value}),text,models,settings)
            report['axis_probes'].append({'axis':axis,'value':value,**result});save()
    rows=[r for c in report['first_turn'] for r in c['results']]+[t['result'] for c in report['conversations'] for t in c['turns'][1:]]+[r for c in report['axis_probes'] for r in c['results']]
    assert all(r['is_real_model'] for r in rows)
    report['actual_generation_count']=len(rows);report['elapsed_seconds']=time.time()-started
    report['summary']={model:{'count':sum(r['model_id']==model for r in rows),
        'fallbacks':sum(r['metrics']['fallback'] for r in rows if r['model_id']==model),
        'raw_first_pass':sum(r['metrics']['raw_valid'] for r in rows if r['model_id']==model),
        'exact_bad_refusals':sum(r['output']['speech'].strip()=='할 수 없어.' for r in rows if r['model_id']==model)} for model in models}
    report['completed']=True;save();print(json.dumps(report['summary']),flush=True)

if __name__=='__main__':main()
