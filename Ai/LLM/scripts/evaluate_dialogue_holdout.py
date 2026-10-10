"""Compare held-out external conversations locally; publish only aggregate metrics."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.persona_runtime import PersonaRuntime,DEFAULT_GENERATION
from src.model_comparison import compare_models,analyze_model


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    path=ROOT/args.output
    if path.exists():raise ValueError('Output already exists')
    if not path.resolve().is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Raw external evaluation must remain in ignored local runs')
    sources={}
    for name in ('tone-v1','emotion-v2','multiturn-v4'):
        source=ROOT/('data/external_sft_'+name)/'test.jsonl'
        sources[name]={'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                       'rows':[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]}
    cases=sources['tone-v1']['rows'][:2]+sources['emotion-v2']['rows'][:2]+sources['multiturn-v4']['rows'][:2]+sources['multiturn-v4']['rows'][-2:]
    analysis=sources['emotion-v2']['rows'][-2:]
    settings=dict(DEFAULT_GENERATION,seed=20261011,max_new_tokens=240)
    runtime=PersonaRuntime()
    report={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'actual_qwen':True,'human_review':False,'independent_benchmark':False,
            'scope':'local noncommercial research; raw test data and generations are not published',
            'test_sha256':{k:v['sha256'] for k,v in sources.items()},'generation':settings,'cases':[],'analysis':[]}
    path.parent.mkdir(parents=True,exist_ok=True)
    def save():path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for i,row in enumerate(cases):
        result=compare_models(runtime,row['context'],row['user_text'],args.models,dict(settings,seed=settings['seed']+i))
        expected=row['answer']['memory_ids']
        term=re.search(r'보다 (.+?) 쪽',row['answer']['speech']) if expected else None
        report['cases'].append({'sample_id':row['sample_id'],'expected_memory_ids':expected,
            'memory_term':term[1] if term else None,**result});save()
        print(json.dumps({'case_index':i,'completed_models':len(result['results'])}),flush=True)
    for model_id in args.models:
        for row in analysis:
            result=analyze_model(runtime,model_id,row['user_text'],settings)
            expected={k:v for k,v in row['answer'].items() if k!='analysis_confidence'}
            report['analysis'].append({'model_id':model_id,'sample_id':row['sample_id'],'text':row['user_text'],
                'expected':expected,'result':result,'match':all((result.get('prediction') or {}).get(k)==v for k,v in expected.items())});save()
    summary={'git_commit':report['git_commit'],'test_sha256':report['test_sha256'],'actual_qwen':True,
             'human_review':False,'independent_benchmark':False,'raw_results_local':args.output,'models':{}}
    for key in args.models:
        outputs=[r for case in report['cases'] for r in case['results'] if r['model_id']==key]
        memory=[(case,r) for case in report['cases'] if case['expected_memory_ids'] for r in case['results'] if r['model_id']==key]
        checks=[r for r in report['analysis'] if r['model_id']==key]
        summary['models'][key]={'conversation_count':len(outputs),'fallbacks':sum(r['metrics']['fallback'] for r in outputs),
            'raw_json_valid':sum(r['metrics']['raw_valid'] for r in outputs),
            'memory_case_count':len(memory),'memory_ids_exact':sum(r['output']['memory_ids']==case['expected_memory_ids'] for case,r in memory),
            'memory_terms_supported':sum(bool(case['memory_term'] and case['memory_term'] in r['output']['speech']) for case,r in memory),
            'analysis_count':len(checks),'analysis_exact_matches':sum(r['match'] for r in checks),
            'analysis_json_valid':sum(r['result']['prediction'] is not None for r in checks)}
    report['completed']=True;save()
    target=ROOT/'experiments/results/dialogue-heldout-summary.json'
    target.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
