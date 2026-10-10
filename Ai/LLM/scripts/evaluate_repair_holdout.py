"""New frozen local holdout; external text never enters public evaluation files."""
import copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.persona_runtime import PersonaRuntime,DEFAULT_GENERATION
from src.model_comparison import compare_models

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    path=ROOT/'runs/dialogue-repair-20261010/repair-heldout.json'
    summary_path=ROOT/'experiments/results/repair-v5-heldout-summary.json'
    if path.exists() or summary_path.exists():raise RuntimeError('Existing holdout results preserved')
    source=ROOT/'data/dialogue_repair_v8/test.jsonl'
    all_rows=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
    memory=[r for r in all_rows if r['answer']['memory_ids']]
    style=[r for r in all_rows if r['source_id'].startswith('agent-authored-dialogue-style')]
    dialogues=[r for r in all_rows if r['source_id'].startswith('lemon-mint') and r['context']['recent']]
    ordinary=[r for r in all_rows if r not in memory+style+dialogues]
    cases=memory+style+dialogues[:4]+ordinary[:3]
    models=['baseline-expanded','baseline-prompt-v2','repair-v5']
    runtime=PersonaRuntime();settings=dict(DEFAULT_GENERATION,seed=20261012,max_new_tokens=384)
    report={'test_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'generation':settings,'cases':[],
        'human_review':False,'scope':'Local noncommercial research; references are not gold naturalness ratings','completed':False}
    def save():path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for index,row in enumerate(cases):
        value=compare_models(runtime,copy.deepcopy(row['context']),row['user_text'],models,dict(settings,seed=settings['seed']+index))
        report['cases'].append({'sample_id':row['sample_id'],'reference':row['answer'],**value});save()
        print(json.dumps({'completed_cases':len(report['cases']),'total_cases':len(cases)}),flush=True)
    summary={'test_sha256':report['test_sha256'],'generation':settings,'actual_qwen':True,'human_review':False,
        'raw_results_local':str(path.relative_to(ROOT)),'model_automatic_selection':False,'case_count':len(cases),'models':{}}
    for model in models:
        rows=[(c,r) for c in report['cases'] for r in c['results'] if r['model_id']==model]
        memories=[(c,r) for c,r in rows if c['reference']['memory_ids']]
        summary['models'][model]={'count':len(rows),'fallbacks':sum(r['metrics']['fallback'] for _,r in rows),
            'raw_first_pass':sum(r['metrics']['raw_valid'] for _,r in rows),'memory_cases':len(memories),
            'memory_ids_exact':sum(c['reference']['memory_ids']==r['output']['memory_ids'] for c,r in memories)}
    report['completed']=True;save();summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
