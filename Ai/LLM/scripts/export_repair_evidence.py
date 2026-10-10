"""Export completed synthetic conversations for review and labeled captures.

No model calls, quality scores or publication of external held-out text.
"""
import csv
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/dialogue-repair-20261010'
LABELS = {'baseline-expanded': '기존 Baseline',
          'baseline-prompt-v2': 'Baseline + 새 프롬프트',
          'repair-v5': '새 QLoRA + 새 프롬프트'}
SCENARIOS = {'reluctance': '운동하기 싫을 때', 'recommend': '운동 추천', 'bored': '심심할 때'}
PROFILES = {'low': 20, 'neutral': 50, 'high': 80}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    source = BASE / 'repair-conversations.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    if not data.get('completed') or data.get('actual_generation_count') != 63:
        raise RuntimeError('Expected all 63 actual generations; preserve incomplete evidence')
    dest = BASE / 'output'
    gallery = BASE / 'gallery'
    if (dest / 'evidence-index.json').exists():
        raise RuntimeError('Evidence already exported; use a new version')
    dest.mkdir(exist_ok=True)
    gallery.mkdir(exist_ok=True)
    (dest / 'captures').mkdir(exist_ok=True)
    records = []
    pages = []
    lines = ['# HEALTHMON 재학습 실제 대화 기록', '',
             '기존 Baseline, 같은 가중치에 새 프롬프트, 새 QLoRA와 새 프롬프트를 비교했다.',
             '63건은 실제 모델 생성이다. 첫 반응 27건, 후속 턴 18건, 단일 축 조건 18건이다.',
             '후속 턴에는 각 모델 자신의 실제 이전 응답을 사용했다. 사람 평가 점수와 자동 채택은 없다.',
             f'원시 기록 SHA256: {sha(source)}', '']
    esc = html.escape

    def add_record(kind, case, profile, turn, user, result, digest):
        if not result.get('is_real_model'):
            raise RuntimeError('Synthetic/dummy output found')
        records.append({'kind': kind, 'case': case, 'profile': profile,
                        'model': result['model_id'], 'turn': turn, 'user': user,
                        'speech': result['output']['speech'],
                        'fallback': result['metrics']['fallback'],
                        'raw_valid': result['metrics']['raw_valid'],
                        'attempts': result['metrics']['attempts'],
                        'action': result['output']['action']['type'],
                        'memory_ids': json.dumps(result['output']['memory_ids'], ensure_ascii=False),
                        'context_sha256': digest})

    def page(key, title, subtitle, body):
        filename = f'{len(pages)+1:02d}_{key}'
        pages.append({'page': filename + '.html', 'capture': filename + '.jpg',
                      'title': title, 'subtitle': subtitle})
        text = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#F7FAFB;color:#16363E;font-family:"Malgun Gothic",sans-serif}}
main{{max-width:1320px;margin:0 auto;padding:30px 38px}}h1{{font-size:32px;margin:0 0 12px}}
.subtitle{{font-size:19px;color:#526C75;line-height:1.55;margin:0 0 20px}}
table{{border-collapse:collapse;width:100%;table-layout:fixed;background:white}}
th,td{{border:1px solid #D7E1E5;padding:16px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}
th{{background:#197462;color:white;font-size:20px}}td{{font-size:23px;line-height:1.55;white-space:pre-wrap}}
.model{{font-size:19px;font-weight:bold}}small{{display:block;font-size:15px;color:#526C75;margin-top:12px}}
footer{{font-size:15px;line-height:1.6;color:#526C75;margin-top:18px}}
</style><main><h1>{esc(title)}</h1><p class="subtitle">{esc(subtitle)}</p>{body}
<footer>실제 Qwen 생성 기록을 정리한 화면입니다. 실시간 채팅 화면과 구분합니다.<br>
첫 답변 seed 20261010 / 후속 2턴 20261011, 3턴 20261012 / temperature 0.7 / 최대 240토큰 / 반말 / Mood 0<br>
원시 기록 SHA256: {sha(source)}</footer></main></html>'''
        (gallery / (filename + '.html')).write_text(text, encoding='utf-8')

    def comparison_body(results):
        cells = []
        for result in results:
            metrics = result['metrics']
            cells.append('<td>' + esc(result['output']['speech']) +
                         f'<small>{esc(result["model_id"])} / 실제 모델 / '
                         f'fallback={metrics["fallback"]} / {metrics["attempts"]}회 시도</small></td>')
        return '<table><tr>' + ''.join('<th>' + esc(LABELS[r['model_id']]) + '</th>' for r in results) + '</tr><tr>' + ''.join(cells) + '</tr></table>'

    for case in data['first_turn']:
        scenario, profile = case['scenario_id'], case['profile_id']
        value = PROFILES[profile]
        title = f'{SCENARIOS[scenario]} / 첫 반응 / {profile}'
        subtitle = f'사용자: {case["user_text"]} / Activity·Affinity·Initiative {value} / 그 외 수치 고정'
        page(f'first_{scenario}_{profile}_A{value}_F{value}_I{value}', title, subtitle,
             comparison_body(case['results']))
        lines.extend(['## ' + title, '사용자: ' + case['user_text'], ''])
        for result in case['results']:
            add_record('first', scenario, profile, 1, case['user_text'], result, case['context_sha256'])
            lines.extend(['### ' + LABELS[result['model_id']], result['output']['speech'], ''])

    for conversation in data['conversations']:
        scenario, model = conversation['scenario_id'], conversation['model_id']
        title = f'{SCENARIOS[scenario]} / 3턴 / {LABELS[model]}'
        body = '<table><colgroup><col style="width:7%"><col style="width:32%"><col style="width:61%"></colgroup><tr><th>턴</th><th>사용자</th><th>실제 응답</th></tr>'
        lines.extend(['## ' + title, ''])
        for index, turn in enumerate(conversation['turns'], 1):
            result = turn['result']
            body += f'<tr><td>{index}</td><td>{esc(turn["user"])}</td><td>{esc(result["output"]["speech"])}</td></tr>'
            lines.extend([f'사용자{index}: ' + turn['user'], f'모델{index}: ' + result['output']['speech'], ''])
            if index > 1:
                context_digest = hashlib.sha256(json.dumps(turn['context'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                add_record('follow_up', scenario, 'neutral', index, turn['user'], result, context_digest)
        page(f'multi_{scenario}_{model}_A50_F50_I50', title,
             'Activity·Affinity·Initiative 50 / 각 모델 자신의 이전 실제 응답을 문맥으로 사용', body + '</table>')

    for case in data['axis_probes']:
        axis, value = case['axis'], case['value']
        title = f'단일 축 확인 / {axis} {value}'
        page(f'axis_{axis}_{value}', title, f'사용자: {case["user_text"]} / 해당 축만 {value}, 나머지 장기 수치 50',
             comparison_body(case['results']))
        lines.extend(['## ' + title, '사용자: ' + case['user_text'], ''])
        for result in case['results']:
            add_record('axis', axis, str(value), 1, case['user_text'], result, case['context_sha256'])
            lines.extend(['### ' + LABELS[result['model_id']], result['output']['speech'], ''])
    if len(records) != 63:
        raise RuntimeError('Evidence count differs from inference record')
    with (dest / '전체대화.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (dest / '전체대화.md').write_text('\n'.join(lines), encoding='utf-8')
    index = {'raw_path': str(source.relative_to(ROOT)), 'raw_sha256': sha(source),
             'actual_generation_count': len(records), 'labels': LABELS, 'pages': pages,
             'method': data['method'], 'generation': data['generation'], 'human_review': False}
    write_json(dest / 'evidence-index.json', index)
    links = ''.join(f'<li><a href="{esc(p["page"])}">{esc(p["title"])}</a></li>' for p in pages)
    (gallery / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>HEALTHMON 실제 재학습 비교</title><h1>실제 재학습 비교 기록</h1><ul>' + links + '</ul>', encoding='utf-8')
    print(json.dumps({'actual_generations': len(records), 'labeled_pages': len(pages)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
