"""Actual paired offline Qwen regression with raw generations; never promotes a model."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.persona_runtime import PersonaRuntime,DEFAULT_GENERATION
from src.character_lab_schema import Session,CharacterState
from src.character_lab_models import response_context,ConversationAnalyzer
from src.model_comparison import compare_models,analyze_model


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--models',nargs='+',default=['baseline-expanded','natural-v1'])
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--reference',help='Reuse frozen baseline outputs with identical cases and generation')
    parser.add_argument('--analyze-each-model',action='store_true',help='Evaluate Analyzer with each actual candidate adapter')
    args=parser.parse_args();out=ROOT/args.output
    if out.exists():raise ValueError('Choose a new output path')
    casepath=ROOT/'data/natural_eval_v1/cases.json'
    cases=json.loads(casepath.read_text(encoding='utf-8'))['cases']
    if args.limit:cases=cases[:args.limit]
    runtime=PersonaRuntime();settings=dict(DEFAULT_GENERATION,seed=20261010,max_new_tokens=240)
    report={'experiment_ids':args.models,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'case_sha256':hashlib.sha256(casepath.read_bytes()).hexdigest(),'case_count':len(cases),'cases':[],
            'actual_qwen':True,'human_review':False,'independent_benchmark':False,'automatic_selection':False,
            'training_executed_by_this_script':False,'generation':settings}
    reference=None
    if args.reference:
        reference_path=ROOT/args.reference
        reference=json.loads(reference_path.read_text(encoding='utf-8'))
        if reference['case_sha256']!=report['case_sha256'] or reference['generation']!=settings or not reference.get('completed'):
            raise ValueError('Reference cases/settings do not match or reference is incomplete')
        report['reference']={'path':args.reference,'sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                             'baseline_git_commit':reference['git_commit'],'baseline_generated_now':False}
    out.parent.mkdir(parents=True,exist_ok=True)
    def save():out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for index,case in enumerate(cases):
        session=Session().model_dump(mode='json')
        state=CharacterState(**case['state']).model_dump(mode='json')
        context=response_context(session,state,case['text'],case['memories'],case['speech_level'])
        context['recent']=case['recent']
        result=compare_models(runtime,context,case['text'],args.models,dict(settings,seed=settings['seed']+index))
        if reference:
            source=next(row for row in reference['cases'] if row['case_id']==case['id'])
            if source['context_sha256']!=result['context_sha256'] or source['generation']!=result['generation']:
                raise ValueError('Reference context or seed changed')
            result['reference_baseline']=next(row for row in source['results'] if row['model_id']=='baseline-expanded')
        report['cases'].append({'case_id':case['id'],**result});save()
        for row in result['results']:
            print(json.dumps({'case':case['id'],'model':row['model_id'],'speech':row['output']['speech'],
                              'fallback':row['metrics']['fallback']},ensure_ascii=False),flush=True)
    # Existing Analyzer compatibility: same task prompt, separate from responder ratings.
    analyzer=ConversationAnalyzer(runtime)
    report['analysis_checks']=[]
    checks=[('운동하기 싫어',{'target':'exercise','sentiment':'negative','motivation':'low','character_affection':'neutral','acceptance':'none'}),
            ('너랑 이야기하는 거 좋아',{'target':'character','sentiment':'positive','motivation':'none','character_affection':'positive','acceptance':'none'}),
            ('스쿼트 할 생각에 신나',{'target':'exercise','sentiment':'positive','motivation':'high','character_affection':'neutral','acceptance':'none'}),
            ('그 퀘스트는 안 하겠어',{'target':'quest','sentiment':'negative','motivation':'low','character_affection':'neutral','acceptance':'reject'}),
            ('너 덕분에 즐거워',{'target':'character','sentiment':'positive','motivation':'none','character_affection':'positive','acceptance':'none'}),
            ('네가 싫어서 말하기 싫어',{'target':'character','sentiment':'negative','motivation':'none','character_affection':'negative','acceptance':'reject'})]
    if not args.analyze_each_model: checks=checks[:2]
    for model_id in args.models if args.analyze_each_model else ['selected-baseline']:
        for text,expect in checks:
            value=analyze_model(runtime,model_id,text,settings) if args.analyze_each_model else analyzer.analyze(text,'qwen',settings)
            report['analysis_checks'].append({'model_id':model_id,'text':text,'expected':expect,'result':value,
                'match':all((value.get('prediction') or {}).get(k)==v for k,v in expect.items())});save()
    report['analysis_coverage']='requested_adapters' if args.analyze_each_model else 'selected_baseline_only'
    report['summary']={model:{'count':len(report['cases']),
        'fallbacks':sum(row['metrics']['fallback'] for case in report['cases'] for row in case['results'] if row['model_id']==model),
        'questions':sum('?' in row['output']['speech'] for case in report['cases'] for row in case['results'] if row['model_id']==model),
        'quest_actions':sum(row['output']['action']['type']!='none' for case in report['cases'] for row in case['results'] if row['model_id']==model)} for model in args.models}
    report['completed']=True;save();print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
