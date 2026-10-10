"""Prepare conversation-grouped, original Korean instruction data for triage."""
import argparse,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'runtime/dataset-tools')]
from prepare_dialogue_repair import authored
from prepare_dialogue_experiments import normalized
import pyarrow.parquet as pq

INTRO=re.compile(r"^(?:Hey|Hi[!,. ]|Hello|I(?: am| have| want| need| feel| like| enjoy| love| recently| was| went| would| do|'m|'ve|'d)|What (?:can|should) I|How (?:can|do|should) I|Can you (?:help|recommend|suggest)|Could you)",re.I)
TOPIC=re.compile('헬스|운동|취미|기분|스트레스|친구|외로|심심|여가|휴식|요리|간식|음악|노래|영화|산책|청소|정리|계획|독서|그림|책|게임|놀이|하루|주말|시간 관리|시간을|할 일|잠|피곤|집에서|걷기')
BAD=re.compile(r'당신은|너는|역할|연기|AI|어시스턴트|가정하|상상해|시스템|프로그|논문|에세이|번역|자살|약물|진단|치료|코드|죽|범죄|대통령|주식|투자|우울증|질환|성폭|의학|처방|질병|데이터|통계|수학|방정식|함수|SQL|Python|마크다운|글머리|키워드|반드시|단어 미만|신경망|앨런|키스|만져|종교|정치|보험|신용|법률|재무')

def short_excerpt(text):
    """Select complete units for an explicit short-answer task, never cut tokens."""
    text=text.replace('**','').strip()
    if len(text)<=460:return text,False
    items=re.split(r'\n\s*(?=(?:[-*]|\d+[.)])\s)',text)
    bullets=[re.sub(r'^\s*(?:[-*]|\d+[.)])\s*','',x).strip() for x in items[1:]]
    if len(bullets)>=2 and all(12<=len(x)<=205 for x in bullets[:2]):
        return '1. '+bullets[0]+'\n2. '+bullets[1],True
    sentences=re.split(r'(?<=[.!?])\s+',text)
    picked=[]
    for sentence in sentences[:3]:
        if len('\n'.join(picked+[sentence]))>450:break
        picked.append(sentence)
    if len(picked)>=2:return ' '.join(picked),True
    return None,False

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser();parser.add_argument('--short-tasks',action='store_true');args=parser.parse_args()
    cache=ROOT/('runs/dialogue-repair-20261010/triage-v7-candidates.json' if args.short_tasks else 'runs/dialogue-repair-20261010/triage-v6-candidates.json')
    if cache.exists():raise RuntimeError('Existing candidate version preserved')
    data={s:{'authored':[],'candidates':[]} for s in ('train','validation','test')}
    for filename in ('dialogue_repair_catalog.json','dialogue_repair_style_catalog.json'):
        catalog=json.loads((ROOT/'data'/filename).read_text(encoding='utf-8'))
        for split in data:data[split]['authored'].extend(authored(catalog,split))
    seen=set();raw_count=0
    for path in sorted((ROOT/'data/external_raw/smol_koreantalk').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                raw_count+=1;m=row['messages']
                if not m or any(t['role']=='system' for t in m):continue
                first=m[0]['content'].strip()
                if not INTRO.search(m[0]['content_en'].strip()) or not TOPIC.search(first) or BAD.search(first) or len(first)>240:continue
                group='smol-'+hashlib.sha256(normalized(first).encode()).hexdigest()
                bucket=int(group[-8:],16)%100
                split='train' if bucket<80 else 'validation' if bucket<90 else 'test'
                history=[]
                for i in range(1,len(m),2):
                    if m[i]['role']!='assistant' or m[i-1]['role']!='user':break
                    user=m[i-1]['content'].strip();original=m[i]['content'].strip();answer=original.replace('**','');adapted=False
                    if args.short_tasks:
                        answer,adapted=short_excerpt(original)
                        if answer is None:continue
                        if adapted:
                            if re.search(r'\d+\s*(?:개|가지|단계)|세 가지|다섯|네 가지|모두|전부|자세히',user):continue
                            user+=' 핵심만 짧게 알려줘.'
                    pair=normalized(user)+'|'+normalized(answer)
                    recent=history[-2:]
                    if (4<=len(user)<=220 and 30<=len(answer)<=460 and not BAD.search(user+answer)
                            and not re.search(r'저도|나도|저는|나는|제가|내가|우리|```|[{}]|https?://|@|[\u0e00-\u0e7f\u4e00-\u9fff]',answer)
                            and len(re.findall('[가-힣]',answer))/len(answer)>.45
                            and all(len(t)<460 and not BAD.search(t) for h in recent for t in h.values()) and pair not in seen):
                        speech='formal' if re.search(r'(요|니다|세요|까요)[.!? ]*$',answer) else 'casual'
                        sample={'sample_id':'smol-'+hashlib.sha256(pair.encode()).hexdigest()[:16],
                            'scenario_group':group,'source_id':f'lemon-mint/smol-koreantalk:{row["custom_id"]}:{i}',
                            'prompt_kind':'character','user_text':user,'answer':{'speech':answer,'emotion':'neutral','animation':'idle',
                            'action':{'type':'none','target_id':None},'memory_ids':[],'personality_update':None},
                            'deferred_speech_level':speech,'deferred_recent':recent.copy(),'human_review':False,'split':split,
                            'short_answer_task_adaptation':adapted,
                            'transformation':'Explicit short-answer request; complete first two list items or up to three complete sentences; not the full original answer' if adapted else 'Original complete response, emphasis marks removed',
                            'source_text_type':'synthetic English dialogue translated to Korean'}
                        data[split]['candidates'].append(sample);seen.add(pair)
                    history.append({'user':user,'assistant':answer})
        print(json.dumps({'shard':path.name,'raw_conversations':raw_count,'candidates':{s:len(v['candidates']) for s,v in data.items()}}),flush=True)
    if args.short_tasks:
        for split in data:
            data[split]['candidates'].sort(key=lambda x:hashlib.sha256((x['sample_id']+'20261010').encode()).hexdigest())
            data[split]['candidates']=data[split]['candidates'][:2500 if split=='train' else 200]
    cache.write_text(json.dumps({'data':data,'raw_conversations':raw_count,'version':'repair-v7' if args.short_tasks else 'repair-v6'},ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
