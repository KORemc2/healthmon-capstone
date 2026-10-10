"""Continue this local experiment after successful training; never promote or push."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/dialogue-repair-20261010'
RUN=ROOT/'runs/dialogue-repair-v5-20261010'

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    (BASE/'evaluation-worker.pid').write_text(str(os.getpid()))
    status={'phase':'waiting_for_training','automatic_selection':False,'production_approved':False}
    def phase(name,**extra):
        status.update(phase=name,**extra);save(BASE/'evaluation-status.json',status)
        print(json.dumps(status,ensure_ascii=False),flush=True)
    phase('waiting_for_training')
    deadline=time.time()+4*3600
    while True:
        if (BASE/'cancel-evaluation.request').exists():phase('cancelled');return
        path=RUN/'metrics.jsonl'
        # The trainer appends concurrently. Only parse flushed, complete lines.
        raw=path.read_text(encoding='utf-8') if path.exists() else ''
        lines=raw.splitlines()
        if raw and not raw.endswith('\n'):lines=lines[:-1]
        rows=[json.loads(x) for x in lines]
        if rows and rows[-1]['event']=='failed':phase('training_failed',error=rows[-1]['error']);return
        if rows and rows[-1]['event']=='complete':break
        if time.time()>deadline:phase('training_wait_timeout');return
        time.sleep(5)
    from filelock import FileLock
    phase('waiting_for_gpu_release')
    with FileLock(str(ROOT/'runtime/gpu.lock'),timeout=60):
        pass
    conf=json.loads((ROOT/'experiments/repair-v5.json').read_text(encoding='utf-8'))
    baseline=ROOT/'runs/expanded-20260929/checkpoint-epoch-1/adapter_model.safetensors'
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=conf['baseline_adapter_sha256']:raise RuntimeError('Baseline changed')
    selection=json.loads((RUN/'selection.json').read_text(encoding='utf-8'))
    selected=Path(selection['selected']['checkpoint']).resolve()
    if not selected.is_relative_to(RUN.resolve()):raise RuntimeError('Unexpected checkpoint selected')
    adapter_sha=hashlib.sha256((selected/'adapter_model.safetensors').read_bytes()).hexdigest()
    registry_path=ROOT/'config/model_registry.json';registry=json.loads(registry_path.read_text(encoding='utf-8'))
    if registry['selected_id']!='baseline-expanded':raise RuntimeError('User changed selection; preserve it and inspect before continuing')
    if 'repair-v5' in registry['models']:raise RuntimeError('Existing repair entry preserved')
    registry['models']['baseline-prompt-v2']={
        'status':'Candidate','adapter_path':registry['models']['baseline-expanded']['adapter_path'],
        'prompt_profile':'natural-v2','training_status':'not_required','human_review':False,'production_approved':False,
        'compatibility_passed':False,'note':'Prompt-only control using preserved baseline weights; no new training'}
    registry['models']['repair-v5']={'status':'Candidate','adapter_path':str(selected.relative_to(ROOT)).replace('\\','/'),
        'adapter_sha256':adapter_sha,'prompt_profile':'natural-v2','training_status':'completed',
        'dataset_version':'dialogue_repair_v8','training_git_commit':json.loads((RUN/'config.json').read_text(encoding='utf-8'))['git_commit'],
        'human_review':False,'production_approved':False,'compatibility_passed':False,
        'note':'Independent clean-base NF4 LoRA; 851 rows, rank16, speech-weighted loss. Evaluation required.'}
    save(registry_path,registry)
    phase('fixed_regression',selected_checkpoint=str(selected),adapter_sha256=adapter_sha)
    tasks=[('fixed_regression',['scripts/evaluate_natural.py','--models','baseline-expanded','baseline-prompt-v2','repair-v5',
        '--output','experiments/results/repair-v5-fixed.json','--analyze-each-model']),
        ('multi_turn_regression',['scripts/evaluate_repair_conversations.py']),
        ('new_local_holdout',['scripts/evaluate_repair_holdout.py'])]
    for name,args in tasks:
        if (BASE/'cancel-evaluation.request').exists():phase('cancelled');return
        phase(name)
        with (BASE/(name+'.log')).open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    phase('completed',baseline_unchanged=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        save(BASE/'evaluation-status.json',{'phase':'failed','error':repr(error),'automatic_selection':False})
        raise
