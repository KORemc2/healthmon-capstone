"""Run four independent local experiments and optionally publish metadata to owner fork."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
FORK='https://github.com/KORemc2/healthmon-capstone.git'
BASELINE='runs/expanded-20260929/checkpoint-epoch-1'
IDS=('tone-v1','emotion-v2','mixed-v3','multiturn-v4')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    if git('remote','get-url','--push','origin')!=FORK:raise RuntimeError('Owner fork origin is required')
    integration=git('branch','--show-current')
    if integration!='feat/llm-natural-dialogue' or git('status','--porcelain'):
        raise RuntimeError('Start on clean integration branch')
    common=git('rev-parse','HEAD')
    baseline_sha=hashlib.sha256((ROOT/BASELINE/'adapter_model.safetensors').read_bytes()).hexdigest()
    for name in IDS:
        branch='exp/llm/'+name
        if subprocess.run(['git','show-ref','--verify','--quiet','refs/heads/'+branch],cwd=REPO).returncode==0:
            raise RuntimeError('Experiment branch already exists: '+branch)
    summary=[]
    try:
        for name in IDS:
            branch='exp/llm/'+name;run='dialogue-'+name+'-20261010'
            git('switch','-c',branch,common)
            manifest=json.loads((ROOT/('data/external_sft_'+name)/'manifest.json').read_text(encoding='utf-8'))
            config={'experiment_id':name,'shared_pipeline_commit':common,'branch':branch,
                'baseline_adapter':BASELINE,'baseline_adapter_sha256':baseline_sha,
                'base_model':'Qwen3.5-9B','base_revision':'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
                'dataset_version':manifest['version'],'dataset_sha256':manifest['sha256'],
                'seed':42,'epochs':1,'rank':8,'alpha':32,'lr':2e-5,'batch':1,'grad_accum':8,
                'max_length':3072,'bf16':True,'prompt_kind':'character','profile':'natural-v1',
                'assistant_logits_only':True,
                'initialized_from_previous_experiment':False,'training_status':'planned',
                'checkpoint_path':'runs/'+run+'/checkpoint-epoch-1',
                'human_review':False,'production_approved':False,'model_status':'Candidate',
                'data_and_weights_publication':'local_only_noncommercial_research'}
            config_path=ROOT/'experiments'/(name+'.json')
            save(config_path,config)
            relative=str(config_path.relative_to(REPO)).replace('\\','/')
            git('add','--',relative);git('commit','-m','Configure independent '+name+' dialogue experiment')
            training_commit=git('rev-parse','HEAD')
            command=[sys.executable,'scripts/train_lora.py','--run',run,'--data','data/external_sft_'+name,
                '--init-adapter',BASELINE,'--profile','natural-v1','--prompt-kind','character',
                '--epochs','1','--max-length','3072','--lr','2e-5','--assistant-logits-only']
            print(json.dumps({'event':'training_started','id':name,'branch':branch,'git_commit':training_commit}),flush=True)
            log_path=ROOT/'runs'/(run+'.log')
            with log_path.open('w',encoding='utf-8') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            run_path=ROOT/'runs'/run
            metrics=[json.loads(line) for line in (run_path/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            complete=result.returncode==0 and metrics[-1]['event']=='complete'
            if hashlib.sha256((ROOT/BASELINE/'adapter_model.safetensors').read_bytes()).hexdigest()!=baseline_sha:
                raise RuntimeError('Baseline changed')
            config.update(training_status='completed' if complete else 'failed',training_git_commit=training_commit,
                          evaluation_status='pending',elapsed_seconds=metrics[-1]['time']-metrics[0]['time'])
            save(config_path,config)
            public=ROOT/'experiments/results'/(name+'-training.json')
            save(public,{'experiment':config,'training_config':json.loads((run_path/'config.json').read_text(encoding='utf-8')),
                         'events':metrics,'checkpoint_published':False,'baseline_unchanged':True})
            git('add','--',relative,str(public.relative_to(REPO)).replace('\\','/'))
            subprocess.run([sys.executable,'scripts/audit_publication.py'],cwd=ROOT,check=True)
            git('diff','--cached','--check');git('commit','-m','Record actual '+name+' training results')
            if args.publish:git('-c','credential.interactive=false','push','origin',branch)
            summary.append(config)
            save(ROOT/'runs/dialogue-suite-summary.json',summary)
            print(json.dumps({'event':'training_completed' if complete else 'training_failed','id':name,
                              'elapsed_seconds':config['elapsed_seconds'],'metrics':metrics[-1]}),flush=True)
            if not complete:raise RuntimeError('Training failed; preserve failure record and inspect before continuing')
    finally:
        git('switch',integration)


if __name__=='__main__':main()
