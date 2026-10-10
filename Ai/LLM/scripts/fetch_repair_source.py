"""Pinned, limited download of public Korean SmolTalk; no paid API calls."""
import json
import sys
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fetch_dialogue_sources import get,save

REPO='lemon-mint/smol-koreantalk'
REVISION='04eb12c1f999578d74dd87720bc853cedb7fa156'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--all-shards',action='store_true');args=parser.parse_args()
    info=json.loads(get('https://huggingface.co/api/datasets/'+REPO+'/revision/'+REVISION))
    if info.get('gated') not in (None,False) or info['cardData']['license']!='apache-2.0':
        raise RuntimeError('Expected accessible Apache-2.0 source')
    folder=ROOT/'data/external_raw/smol_koreantalk'
    count=8 if args.all_shards else 1
    files=[save(folder,'README.md',f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md')]
    for index in range(count):
        name=f'train-{index:05d}-of-00008.parquet'
        files.append(save(folder,name,f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{name}'))
        print(json.dumps({'downloaded_shards':index+1,'planned_shards':count}),flush=True)
    result={'repo':REPO,'revision':REVISION,'declared_license':'apache-2.0','upstream':'HuggingFaceTB/smol-smoltalk',
            'purpose':'Korean instruction following and context-aware conversation; exclude role-play, system-override, technical and unverifiable factual examples',
            'downloaded_shards':count,'total_shards':8,'human_review':False,'files':files}
    path=ROOT/'experiments'/('dialogue-repair-source-full.json' if args.all_shards else 'dialogue-repair-source.json')
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=result:raise RuntimeError('Existing provenance differs')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
