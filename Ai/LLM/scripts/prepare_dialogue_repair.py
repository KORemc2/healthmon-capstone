"""Scale real unique dialogue data; keep explicit authored behavior pairs separate."""
import copy
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_dialogue_experiments import external_pools, make_context, response, normalized
from src.agent import validate
from src.sft_tasks import training_prompt
from src.chatml import encode_chatml

VERSION = 'dialogue_repair_v5'
MAX_LENGTH = 2048


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authored(catalog, split):
    rows = []
    for group in catalog[split]:
        if 'turns' in group:
            history = []
            for index, (text, speech) in enumerate(group['turns']):
                rows.append({'sample_id': group['id'] + '-' + str(index), 'scenario_group': group['id'],
                    'source_id': catalog['version'], 'prompt_kind': 'character', 'user_text': text,
                    'context': make_context(text, copy.deepcopy(history), speech='casual'),
                    'answer': response(speech), 'human_review': False, 'split': split})
                history.append({'user': text, 'assistant': speech})
        else:
            for band, value in [('low', 20), ('neutral', 50), ('high', 80)]:
                text = group['user']
                rows.append({'sample_id': group['id'] + '-' + band, 'scenario_group': group['id'],
                    'source_id': catalog['version'], 'prompt_kind': 'character', 'user_text': text,
                    'context': make_context(text, state={k: value for k in ('activity', 'affection', 'initiative')}, speech='casual'),
                    'answer': response(group['variants'][band]), 'human_review': False, 'split': split})
    return rows


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    dest = ROOT / 'data' / VERSION
    if dest.exists():
        raise RuntimeError('Dataset version exists; existing data preserved')
    catalog_path = ROOT / 'data/dialogue_repair_catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    pools, filters = external_pools(pool_limit=None, answer_max=320, defer_context=True)
    print(json.dumps({'available': {k: {s: len(v) for s, v in p.items()} for k, p in pools.items()}, 'filters': filters}), flush=True)
    # Length is not treated as quality; exclude context-free canned assurances.
    def acceptable(row):
        speech = row['answer']['speech']
        return (len(speech) >= 10 and not re.search(r'잘못.{0,5}없|당신.{0,8}이해|할 수 없|괜찮을 거|좋은 일이 생|곧 좋아|모든.{0,5}잘 될|기운 내|행운|믿어|당신은|정말|확실히', speech)
                and not re.fullmatch(r'.{0,14}(그래요|그렇군요|그러게요|좋겠어요|좋아요|괜찮아요)[.! ]*', speech))
    tokenizer = __import__('transformers').AutoTokenizer.from_pretrained(ROOT / 'models/qwen3.5-9b', local_files_only=True)
    result = {'version': VERSION, 'profile': 'natural-v2', 'prompt_kind': 'character', 'max_length': MAX_LENGTH,
              'human_review': False, 'catalog_sha256': sha(catalog_path), 'source_manifest_sha256': sha(ROOT / 'experiments/external-dialogue-sources.json'),
              'license_scope': 'local noncommercial research; no external raw/derived data or checkpoint publication',
              'filters': filters, 'sha256': {}, 'counts': {}, 'split_statistics': {}, 'no_truncation': True}
    corpus = {}
    for split in ('train', 'validation', 'test'):
        # No source row is duplicated, even across emotion/multi-turn tasks.
        seen_sources = set()
        rows = authored(catalog, split)
        for kind, quota in [('tone', 640), ('emotion', 320), ('multiturn', 320)]:
            limit = quota if split == 'train' else 24
            accepted = 0
            for row in pools[kind][split]:
                if accepted >= limit:
                    break
                if row['source_id'] in seen_sources or not acceptable(row):
                    continue
                trial = copy.deepcopy(row)
                trial['context'] = make_context(trial['user_text'], trial.pop('deferred_recent'),
                                                speech=trial.pop('deferred_speech_level'))
                # Exclude known presentation and development user strings from
                # training; evaluation contains separate actual conversations.
                if split == 'train' and normalized(trial['user_text']) in {
                    normalized(x) for x in ('아 운동하기 싫어.', '운동 뭐 하지?', '하 심심하다.',
                    '그냥 오늘은 쉬고 싶어. 운동 얘기는 그만하자.', '그럼 운동 말고 집에서 할 만한 거 하나만 골라줘.',
                    '집에 있고 도구는 없어. 5분 정도만 할래.', '방금 말한 걸 순서대로 짧게 정리해줘.',
                    '운동 말고, 집에서 혼자 할 수 있는 거로.', '그중 하나만 골라서 지금 바로 시작하게 해줘.')}: continue
                try:
                    encode_chatml(tokenizer, training_prompt(trial, 'natural-v2', 'character'),
                        json.dumps(trial['answer'], ensure_ascii=False, separators=(',', ':')), MAX_LENGTH)
                except ValueError:
                    continue
                rows.append(trial); seen_sources.add(row['source_id']); accepted += 1
        random.Random(20261010).shuffle(rows)
        corpus[split] = rows
    # Check whole conversation groups, source IDs and exact substantial user
    # messages across splits, including authored history, before writing files.
    groups, user_splits = {}, {}
    for split, rows in corpus.items():
        for row in rows:
            group = row['scenario_group']
            if group in groups and groups[group] != split: raise RuntimeError('Group overlap')
            groups[group] = split
            for text in [row['user_text'], *(x['user'] for x in row['context']['recent'])]:
                key = normalized(text)
                if len(key) < 12: continue
                if key in user_splits and user_splits[key] != split: raise RuntimeError('User utterance overlap')
                user_splits[key] = split
    dest.mkdir()
    for split, rows in corpus.items():
        lengths, targets = [], []
        for row in rows:
            validate(json.dumps(row['answer'], ensure_ascii=False), row['context'], row['user_text'])
            batch = encode_chatml(tokenizer, training_prompt(row, 'natural-v2', 'character'),
                json.dumps(row['answer'], ensure_ascii=False, separators=(',', ':')), MAX_LENGTH)
            lengths.append(len(batch['input_ids'])); targets.append(batch['target_tokens'])
        path = dest / (split + '.jsonl')
        path.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows), encoding='utf-8')
        result['sha256'][path.name] = sha(path)
        result['counts'][split] = len(rows)
        result['split_statistics'][split] = {'source_unique': len({r['source_id'] for r in rows if r['source_id'] != catalog['version']}),
            'authored_rows': sum(r['source_id'] == catalog['version'] for r in rows),
            'scenario_groups': len({r['scenario_group'] for r in rows}),
            'max_tokens': max(lengths), 'mean_tokens': sum(lengths) / len(lengths),
            'target_tokens': sum(targets), 'speech_mean_characters': sum(len(r['answer']['speech']) for r in rows) / len(rows),
            'speech_max_characters': max(len(r['answer']['speech']) for r in rows),
            'speech_levels': dict(Counter(r['context']['settings']['speech_level'] for r in rows)),
            'multi_turn_rows': sum(bool(r['context']['recent']) for r in rows)}
    result['group_leakage'] = False
    result['substantial_exact_user_overlap'] = False
    (dest / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
