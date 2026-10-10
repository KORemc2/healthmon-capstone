"""Freeze a local training dataset after contract, overlap and provenance checks."""
import copy,hashlib,json,random,re,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from prepare_dialogue_experiments import make_context,response,normalized
from src.agent import validate
from src.sft_tasks import training_prompt
from src.chatml import encode_chatml

def hash_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def grounded_rows(split):
    facts={
      'train': [('preference','구운 감자','간식으로 뭘 좋아한다고 했지?'),('preference','따뜻한 보리차','내가 좋아하는 차가 뭐였어?'),
                ('preference','추리 소설','내가 좋아하는 책 종류 기억해?'),('preference','조용한 피아노 음악','어떤 음악을 좋아한다고 했더라?'),
                ('preference','맵지 않은 음식','음식 취향 기억해?'),('preference','주말 오전 산책','산책 시간 취향이 뭐야?'),
                ('preference','종이접기','내가 좋아하는 취미 기억해?'),('goal','하루 3000걸음','내 하루 걸음 목표가 뭐지?'),
                ('goal','일주일에 책 한 편 읽기','독서 목표가 뭐였지?'),('goal','매일 밤 열한 시에 잠자리 준비','수면 준비 목표를 말해줘.'),
                ('experience','지난 금요일에 동네 도서관 방문','지난 금요일에 뭘 했다고 했지?'),('experience','어제 집에서 음악을 들음','어제 뭐 했다는 기억이 있어?')],
      'validation':[('preference','호박 수프','내가 좋아하는 수프 기억해?'),('goal','매일 식물에 물이 필요한지 확인','내 식물 관리 목표가 뭐야?')],
      'test':[('preference','사과 조각','내가 좋아하는 과일 간식은?'),('goal','이번 달 사진 열 장 정리','사진 정리 목표 기억해?')]}
    rows=[]
    for index,(slot,fact,question) in enumerate(facts[split]):
        mid=f'repair-memory-{split}-{index}'
        context=make_context(question,memories=[{'id':mid,'slot':slot,'content':fact,'confidence':1,
            'evidence_refs':[{'kind':'message','id':mid+'-source','revision':1}]}],speech='casual')
        speech=f'기억에 남은 취향은 {fact}야.' if slot=='preference' else f'목표는 {fact}야.' if slot=='goal' else f'기억에는 {fact}이라고 남아 있어.'
        answer=response(speech);answer['memory_ids']=[mid]
        rows.append({'sample_id':mid,'scenario_group':mid,'source_id':'agent-authored-grounding-v7','prompt_kind':'character',
            'user_text':question,'context':context,'answer':answer,'human_review':False,'split':split})
    if split=='train':
        for index,text in enumerate(('선택할 만한 퀘스트 하나 알려줘.','오늘 가능한 미션을 보여줘.','가벼운 퀘스트 한 개 추천해.')):
            for value in (20,50,80):
                context=make_context(text,state={k:value for k in ('activity','affection','initiative')},speech='casual')
                answer=response('지금 선택된 퀘스트 후보는 없어. 일반적인 활동 정보는 알려줄 수 있어.')
                if context['candidates']:
                    q=context['candidates'][0];answer=response(q['title']+' 퀘스트가 있어. 하고 싶을 때 선택해도 돼.')
                    answer['action']={'type':'offer_quest','target_id':q['id']}
                rows.append({'sample_id':f'repair-quest-{index}-{value}','scenario_group':f'repair-quest-{index}',
                    'source_id':'agent-authored-grounding-v7','prompt_kind':'character','user_text':text,
                    'context':context,'answer':answer,'human_review':False,'split':split})
    return rows

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    dest=ROOT/'data/dialogue_repair_v7'
    if dest.exists():raise RuntimeError('Existing frozen dataset preserved')
    base=ROOT/'runs/dialogue-repair-20261010'
    sources=[('triage-candidates.json','semantic-triage.jsonl'),('triage-v7-candidates.json','semantic-triage-v7.jsonl')]
    data={s:[] for s in ('train','validation','test')};counts=Counter();seen=set()
    exclude={normalized(x) for x in ('아 운동하기 싫어.','운동 뭐 하지?','하 심심하다.',
        '그냥 오늘은 쉬고 싶어. 운동 얘기는 그만하자.','그럼 운동 말고 집에서 할 만한 거 하나만 골라줘.',
        '집에 있고 도구는 없어. 5분 정도만 할래.','방금 말한 걸 순서대로 짧게 정리해줘.',
        '운동 말고, 집에서 혼자 할 수 있는 거로.','그중 하나만 골라서 지금 바로 시작하게 해줘.')}
    bad_sources={'songys/Chatbot_data:4750','songys/Chatbot_data:1680','songys/Chatbot_data:1712',
        'passing2961/KorEmpatheticDialogues:train:9745:1','passing2961/KorEmpatheticDialogues:train:16837:1',
        'passing2961/KorEmpatheticDialogues:train:8547:3','passing2961/KorEmpatheticDialogues:train:4240:1',
        'passing2961/KorEmpatheticDialogues:train:16131:1','passing2961/KorEmpatheticDialogues:train:9460:3'}
    source_hashes={}
    # Neutral base contexts are shared only while constructing data; every row
    # owns an independent copy. Current candidate suggestions are irrelevant to
    # these ordinary conversations and hidden in natural-v2 prompts.
    contexts={s:make_context('',speech=s) for s in ('formal','casual')}
    for cache_name,audit_name in sources:
        cache=json.loads((base/cache_name).read_text(encoding='utf-8'))['data']
        scores={r['sample_id']:r for r in (json.loads(x) for x in (base/audit_name).read_text(encoding='utf-8').splitlines())}
        source_hashes[cache_name]=hash_file(base/cache_name);source_hashes[audit_name]=hash_file(base/audit_name)
        for split in data:
            if 'v7' in cache_name:data[split].extend(cache[split]['authored'])
            selected=0
            for original in cache[split]['candidates']:
                score=scores.get(original['sample_id'])
                if not score or score['score_A']<=.5:counts['triage_excluded']+=1;continue
                if original['source_id'] in bad_sources:counts['manual_excluded']+=1;continue
                speech=original['answer']['speech']
                if re.search(r'할 수 없|당신|귀하|잘못.{0,5}없|질투하지|가야 (?:해|합)|반드시|살아본|정확히 \d',speech):counts['strict_text_excluded']+=1;continue
                if any(normalized(x) in exclude for x in [original['user_text'],*(h['user'] for h in original['deferred_recent'])]):counts['presentation_overlap_excluded']+=1;continue
                pair=normalized(original['user_text'])+'|'+normalized(speech)
                if pair in seen:counts['duplicate_pair']+=1;continue
                row=copy.deepcopy(original);row['context']=copy.deepcopy(contexts[row.pop('deferred_speech_level')])
                row['context']['recent']=row.pop('deferred_recent');row['semantic_triage_score']=score['score_A']
                row['semantic_triage_rule']='A logit exceeds B logit; uncalibrated model triage, not human approval'
                data[split].append(row);seen.add(pair);selected+=1
            counts[cache_name+'_'+split]=selected
    for split in data:data[split].extend(grounded_rows(split))
    # Hold-out groups take priority; drop entire training conversation groups
    # if any substantial complete utterance occurs in another split.
    utterances={};groups={};filtered={};dropped_groups=set()
    for split in ('test','validation','train'):
        by_group={}
        for row in data[split]:by_group.setdefault(row['scenario_group'],[]).append(row)
        accepted=[]
        for group,rows in by_group.items():
            keys={normalized(t) for row in rows for t in [row['user_text'],row['answer']['speech'],
                *(t for h in row['context']['recent'] for t in h.values())] if len(normalized(t))>=12}
            if group in groups and groups[group]!=split or any(k in utterances and utterances[k]!=split for k in keys):
                counts['cross_split_group_excluded']+=len(rows);continue
            groups[group]=split
            for key in keys:utterances[key]=split
            accepted.extend(rows)
        random.Random(20261010).shuffle(accepted);filtered[split]=accepted
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/qwen3.5-9b',local_files_only=True)
    result={'version':'dialogue_repair_v7','profile':'natural-v2','prompt_kind':'character','max_length':2048,
        'counts':{},'sha256':{},'statistics':{},'filters':dict(counts),'source_hashes':source_hashes,
        'group_leakage':False,'substantial_exact_utterance_overlap':False,'no_token_truncation':True,
        'human_review':False,'license_scope':'Local noncommercial research; source texts, derived data and adapter stay local',
        'triage_note':'Forced A/B scores from local base Qwen are uncalibrated selection heuristics, not ground truth or accuracy.',
        'short_task_note':'Some SmolTalk responses are complete-sentence excerpts paired with an explicit short-answer request. These are documented derived tasks, not full original responses.'}
    dest.mkdir()
    for split,rows in filtered.items():
        accepted=[];lengths=[]
        for row in rows:
            try:
                validate(json.dumps(row['answer'],ensure_ascii=False),row['context'],row['user_text'])
                encoded=encode_chatml(tok,training_prompt(row,'natural-v2','character'),
                    json.dumps(row['answer'],ensure_ascii=False,separators=(',',':')),2048)
            except ValueError:counts['final_contract_or_length_excluded']+=1;continue
            accepted.append(row);lengths.append(len(encoded['input_ids']))
        path=dest/(split+'.jsonl');path.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in accepted),encoding='utf-8')
        result['counts'][split]=len(accepted);result['sha256'][path.name]=hash_file(path)
        result['statistics'][split]={'unique_scenario_groups':len({x['scenario_group'] for x in accepted}),
            'source_families':dict(Counter(x['source_id'].split(':')[0] for x in accepted)),
            'mean_speech_characters':sum(len(x['answer']['speech']) for x in accepted)/len(accepted),
            'max_speech_characters':max(len(x['answer']['speech']) for x in accepted),
            'max_tokens':max(lengths),'mean_tokens':sum(lengths)/len(lengths),
            'multi_turn_rows':sum(bool(x['context']['recent']) for x in accepted),'memory_rows':sum(bool(x['context']['memories']) for x in accepted)}
    result['filters']=dict(counts)
    (dest/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
