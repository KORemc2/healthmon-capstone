"""Repeat the three reported failure triggers with two additional seeds."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.persona_runtime import PersonaRuntime, DEFAULT_GENERATION
from src.character_lab_schema import CharacterState, Session
from src.character_lab_models import response_context
from src.model_comparison import compare_models


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    path = ROOT / 'runs/dialogue-repair-20261010/repair-seed-checks.json'
    if path.exists():
        raise RuntimeError('Existing seed checks preserved; choose a new version')
    models = ['baseline-expanded', 'baseline-prompt-v2', 'repair-v5']
    runtime = PersonaRuntime()
    session, state = Session().model_dump(mode='json'), CharacterState().model_dump(mode='json')
    report = {'seeds': [20261011, 20261012], 'models': models, 'cases': [],
              'human_review': False, 'automatic_selection': False, 'completed': False,
              'scope': 'Development examples; two extra seeds do not establish a general refusal rate'}
    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for text in ('아 운동하기 싫어.', '운동 뭐 하지?', '하 심심하다.'):
        for seed in report['seeds']:
            context = response_context(session, state, text, [], 'casual')
            value = compare_models(runtime, context, text, models, dict(DEFAULT_GENERATION, seed=seed))
            report['cases'].append(value);save()
            print(json.dumps({'text': text, 'seed': seed,
                              'responses': {r['model_id']: r['output']['speech'] for r in value['results']}}, ensure_ascii=False), flush=True)
    report['summary'] = {}
    for model in models:
        rows = [r for c in report['cases'] for r in c['results'] if r['model_id'] == model]
        assert all(r['is_real_model'] for r in rows)
        report['summary'][model] = {'count': len(rows),
            'fallbacks': sum(r['metrics']['fallback'] for r in rows),
            'raw_first_pass': sum(r['metrics']['raw_valid'] for r in rows),
            'exact_bad_refusals': sum(r['output']['speech'].strip() == '할 수 없어.' for r in rows)}
    report['actual_generation_count'] = sum(x['count'] for x in report['summary'].values())
    report['completed'] = True;save()


if __name__ == '__main__':
    main()
