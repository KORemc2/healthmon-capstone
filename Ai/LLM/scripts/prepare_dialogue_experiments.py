"""Prepare local, grouped Korean SFT experiments without changing baseline data."""
import copy
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.character_lab_schema import CharacterState, Session, ConversationLabel
from src.character_lab_models import response_context
from src.sft_tasks import training_prompt
from src.chatml import encode_chatml
from src.agent import validate

SEED = 42
MAX_LENGTH = 3072


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def normalized(text):
    return re.sub(r'[^가-힣a-z0-9]', '', text.lower())


def usable(text, answer=False):
    if not 4 <= len(text) <= (100 if answer else 180): return False
    if len(re.findall('[가-힣]', text))/max(1, len(text)) < .35: return False
    if re.search(r'@|https?://|\d{3}[- ]\d{3,4}[- ]\d{4}|\d{6}[- ]?[1-4]\d{6}', text): return False
    if re.search(r'자살|자해|살인|성폭|강간|시발|씨발|병신|죽어|의약|약물', text): return False
    if answer and re.search(r'저도|나도|저는|나는|제가|내가|우리|제 (남편|아내|친구)|내 (남편|아내|친구)|인공지능|AI|역할|\*|힘내|응원|대단|잘했|이해합니다|정말 힘드|운동해', text): return False
    return True


def make_context(text, recent=None, memories=None, state=None, speech='formal'):
    state = CharacterState(**(state or {})).model_dump(mode='json')
    context = response_context(Session().model_dump(mode='json'), state, text, memories or [], speech)
    context['recent'] = recent or []
    return context


def response(text):
    return {'speech': text, 'emotion': 'neutral', 'animation': 'idle',
            'action': {'type': 'none', 'target_id': None}, 'memory_ids': [], 'personality_update': None}


def external_pools():
    pools = {kind: {s: [] for s in ('train', 'validation', 'test')} for kind in ('tone', 'emotion', 'multiturn')}
    counts = Counter()
    seen_pairs, group_splits, utterance_splits = set(), {}, {}
    evaluation = json.loads((ROOT/'data/natural_eval_v1/cases.json').read_text(encoding='utf-8'))
    excluded = {normalized(x['text']) for x in evaluation['cases']}

    def add(kind, split, group, source_id, user, answer, recent=None, emotion=None):
        if len(pools[kind][split]) >= 128: counts['pool_limit_skipped'] += 1; return
        fingerprint = digest(normalized(user)+'|'+normalized(answer))
        if not usable(user) or not usable(answer, True): counts['filtered_text'] += 1; return
        if normalized(user) in excluded: counts['evaluation_overlap'] += 1; return
        if (kind, fingerprint) in seen_pairs: counts['duplicate_pairs'] += 1; return
        if group in group_splits and group_splits[group] != split: counts['cross_split_groups'] += 1; return
        # Prevent identical substantial utterances from crossing split boundaries.
        keys = [digest(normalized(t)) for t in [user, answer, *(t for r in recent or [] for t in r.values())]
                if len(normalized(t)) >= 12]
        if any(k in utterance_splits and utterance_splits[k] != split for k in keys):
            counts['cross_split_utterances'] += 1; return
        speech = 'formal' if re.search(r'(요|니다|세요|까요)[.!?]*$', answer) else 'casual'
        row = {'sample_id': kind+'-'+fingerprint[:16], 'source_id': source_id,
               'scenario_group': group, 'prompt_kind': 'character', 'user_text': user,
               'context': make_context(user, recent, speech=speech), 'answer': response(answer),
               'source_user_emotion': emotion, 'character_mood_source': 'independent_default_zero',
               'human_review': False, 'split': split}
        try: validate(json.dumps(row['answer'],ensure_ascii=False),row['context'],user)
        except ValueError: counts['contract_filtered'] += 1; return
        seen_pairs.add((kind, fingerprint)); group_splits[group] = split
        for key in keys: utterance_splits[key] = split
        pools[kind][split].append(row); counts['accepted_'+kind] += 1

    rows = list(csv.DictReader((ROOT/'data/external_raw/chatbot/ChatbotData.csv').open(encoding='utf-8')))
    for i, row in enumerate(rows):
        counts['raw_chatbot_pairs'] += 1
        if row['label'] != '0': continue
        group = 'chatbot-'+digest(normalized(row['Q']))
        bucket = int(digest(group)[:8], 16) % 100
        split = 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'
        add('tone', split, group, f'songys/Chatbot_data:{i}', row['Q'], row['A'])
    for filename, split in [('train', 'train'), ('valid', 'validation'), ('test', 'test')]:
        rows = json.loads((ROOT/f'data/external_raw/kor_empathy/{filename}.json').read_text(encoding='utf-8'))
        for row in rows:
            counts['raw_empathy_dialogues'] += 1
            turns = row['dialogue']
            group = 'empathy-'+digest('|'.join(normalized(t['utter']) for t in turns))
            for i in range(1, len(turns), 2):
                if turns[i]['user_id'] != 1 or turns[i-1]['user_id'] != 0: continue
                user, answer = turns[i-1]['utter'].strip(), turns[i]['utter'].strip()
                source = f'passing2961/KorEmpatheticDialogues:{filename}:{row["dialogue_id"]}:{i}'
                if i == 1: add('emotion', split, group, source, user, answer, emotion=row['emotion'])
                elif i >= 3:
                    history = [{'user': turns[j]['utter'], 'assistant': turns[j+1]['utter']}
                               for j in range(max(0, i-5), i-1, 2)]
                    if all(usable(t, k == 'assistant') for r in history for k,t in r.items()):
                        add('multiturn', split, group, source, user, answer, history, row['emotion'])
    for values in pools.values():
        for split, rows in values.items(): rows.sort(key=lambda x: digest(x['sample_id']+str(SEED)))
    counts['roleplay_downloaded_but_not_trained'] = 32367
    counts['roleplay_exclusion_reason'] = 'Fixed foreign identities and roleplay instructions conflict with HEALTHMON identity; no safe automatic mapping.'
    return pools, dict(counts)


def analysis_rows(split):
    cases = {
        'train': [('운동하기 싫어', 'exercise','negative','low','neutral','none'),
                  ('너랑 이야기하는 게 좋아', 'character','positive','none','positive','none'),
                  ('그 퀘스트 안 할래', 'quest','negative','low','neutral','reject'),
                  ('점심 맛있었어', 'food','positive','none','neutral','none'),
                  ('그 퀘스트 해볼게', 'quest','positive','high','neutral','accept'),
                  ('너 자꾸 말 걸지 마', 'character','negative','none','negative','reject'),
                  ('오늘 달리기 기대돼', 'exercise','positive','high','neutral','none'),
                  ('밖에 비가 오네', 'other','neutral','none','neutral','none')],
        'validation': [('스쿼트는 오늘 하기 싫다','exercise','negative','low','neutral','none'),
                       ('너 덕분에 즐거워','character','positive','none','positive','none')],
        'test': [('푸시업 할 생각에 신나','exercise','positive','high','neutral','none'),
                 ('네가 싫어서 말하기 싫어','character','negative','none','negative','reject')]}
    rows=[]
    for i,(text,target,sentiment,motivation,affection,acceptance) in enumerate(cases[split]):
        answer=ConversationLabel(target=target,sentiment=sentiment,motivation=motivation,
            character_affection=affection,acceptance=acceptance,analysis_confidence=.95).model_dump()
        rows.append({'sample_id': f'analysis-{split}-{i}', 'scenario_group': f'authored-analysis-{split}-{i}',
                     'prompt_kind': 'conversation_analysis', 'user_text': text, 'answer': answer,
                     'source_id': 'agent_authored_schema_example', 'human_review': False, 'split': split})
    return rows


def memory_rows(split):
    topics={'train': [('아침','저녁'),('산책','수영'),('딸기','복숭아'),('종이책','전자책'),
                      ('바다','산'),('강아지','고양이'),('커피','차'),('겨울','여름')],
            'validation': [('기차','버스'),('파랑','초록')], 'test': [('토요일','일요일'),('재즈','클래식')]}
    rows=[]
    for i,(old,new) in enumerate(topics[split]):
        memory_id=f'local-{split}-{i}'
        text='내가 아까 더 좋다고 한 게 뭐였지?'
        fact=f'사용자는 {old}보다 {new} 쪽을 더 좋아한다.'
        recent=[{'user':f'나는 {old}보다 {new} 쪽이 더 좋아.', 'assistant':f'{new} 쪽이 더 끌리는구나.'}]
        memories=[{'id':memory_id,'slot':'preference','content':fact,'source_id':'synthetic_user_turn'}]
        answer=response(f'{old}보다 {new} 쪽이 더 좋다고 했어.');answer['memory_ids']=[memory_id]
        rows.append({'sample_id':f'memory-{split}-{i}', 'scenario_group':f'authored-memory-{split}-{i}',
                     'prompt_kind':'character','user_text':text,'context':make_context(text,recent,memories,speech='casual'),
                     'answer':answer,'source_id':'agent_authored_memory_example','human_review':False,'split':split})
    return rows


def persona_rows():
    cases=[('그냥 잠깐 얘기하고 싶어.', '응, 편하게 말해.', {'affection':15,'initiative':15}),
           ('그냥 잠깐 얘기하고 싶어.', '좋아, 여기 있을게. 무슨 얘기든 편하게 해.', {'affection':85,'initiative':75}),
           ('운동 계획은 내일 생각할래.', '응. 오늘은 계획 안 세워도 돼.', {'activity':15,'initiative':15}),
           ('운동 계획은 내일 생각할래.', '그래. 내일 마음이 생기면 그때 같이 정해 보자.', {'activity':85,'initiative':85})]
    return [{'sample_id':f'persona-train-{i}','scenario_group':'authored-persona-'+str(i//2),
             'prompt_kind':'character','user_text':text,'context':make_context(text,state=state,speech='casual'),
             'answer':response(speech),'source_id':'agent_authored_stat_conditioned','human_review':False,'split':'train'}
            for i,(text,speech,state) in enumerate(cases)]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/qwen3.5-9b',local_files_only=True)
    pools, filters=external_pools()
    old=[json.loads(line) for line in (ROOT/'data/improvement_v3/train.jsonl').read_text(encoding='utf-8').splitlines()]
    retained=[]
    for original in old:
        if original['scenario_group'] in ('identity','style_pair','state_printer','memory_present'):
            row=copy.deepcopy(original);row['prompt_kind']='legacy';retained.append(row)
        if len(retained)==8: break
    if len(retained)<8: raise ValueError('Not enough baseline retention samples')
    report={'seed':SEED,'max_length':MAX_LENGTH,'human_review':False,'filters':filters,'experiments':{},
            'raw_or_derived_data_published':False,'limitations':['Automated filters are not a complete privacy or semantic review.',
            'Small feasibility datasets; no independent benchmark or production approval.',
            'Group split and substantial exact utterance overlap checks do not detect all semantic paraphrases.']}
    prepared={}
    for name in ('tone-v1','emotion-v2','mixed-v3','multiturn-v4'):
        dest=ROOT/('data/external_sft_'+name)
        if dest.exists(): raise RuntimeError('Dataset exists; choose a new version: '+name)
        split_rows={};payloads={}
        for split,count in [('train',32),('validation',8),('test',8)]:
            if name=='tone-v1': rows=pools['tone'][split][:count]
            elif name=='emotion-v2':
                authored=analysis_rows(split);rows=pools['emotion'][split][:count-len(authored)]+authored
            elif name=='mixed-v3':
                authored=analysis_rows(split)[:4 if split=='train' else 2]
                authored+=persona_rows() if split=='train' else []
                amount=count-len(authored)
                rows=pools['tone'][split][:amount//2]+pools['emotion'][split][:amount-amount//2]+authored
            else:
                authored=memory_rows(split);rows=pools['multiturn'][split][:count-len(authored)]+authored
            if len(rows)!=count: raise ValueError('Insufficient filtered rows: '+name+':'+split)
            rows=copy.deepcopy(rows)+(copy.deepcopy(retained) if split=='train' else [])
            lengths=[]
            for row in rows:
                if row['prompt_kind']!='conversation_analysis': validate(json.dumps(row['answer'],ensure_ascii=False),row['context'],row['user_text'])
                encoded=encode_chatml(tokenizer,training_prompt(row,'natural-v1','character'),
                    json.dumps(row['answer'],ensure_ascii=False,separators=(',',':')),MAX_LENGTH)
                lengths.append(len(encoded['input_ids']));row['token_count']=lengths[-1]
            payloads[split]=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
            split_rows[split]={'count':len(rows),'sha256':hashlib.sha256(payloads[split].encode('utf-8')).hexdigest(),
                'token_min':min(lengths),'token_max':max(lengths),
                'groups':sorted(set(r['scenario_group'] for r in rows)),
                'sources':dict(Counter(r.get('source_id','legacy').split(':')[0] for r in rows))}
        group_sets=[set(split_rows[s]['groups']) for s in ('train','validation','test')]
        if any(a & b for i,a in enumerate(group_sets) for b in group_sets[i+1:]): raise ValueError('Group leakage')
        manifest={'version':'external_sft_'+name,'sha256':{s:v['sha256'] for s,v in split_rows.items()},
                  'splits':split_rows,'prompt_kind':'character','profile':'natural-v1','human_review':False,
                  'baseline_retention_count':len(retained),'no_truncation':True,'group_leakage':False,
                  'training_scope':'noncommercial local research','sources_manifest':'experiments/external-dialogue-sources.json'}
        prepared[name]=(dest,payloads,manifest)
        report['experiments'][name]=manifest
        print(json.dumps({'experiment':name,'splits':{s:{k:v[k] for k in ('count','token_min','token_max')} for s,v in split_rows.items()}},ensure_ascii=False),flush=True)
    for dest,payloads,manifest in prepared.values():
        dest.mkdir()
        for split,payload in payloads.items():
            with (dest/(split+'.jsonl')).open('w',encoding='utf-8',newline='\n') as stream: stream.write(payload)
        (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'experiments/external-dialogue-preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__': main()
