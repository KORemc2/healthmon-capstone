"""Reproducible local ablations; preserve existing datasets and checkpoints."""
import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cpu_audit():
    import torch
    from safetensors.torch import load_file
    from src.model_registry import read_registry
    registry = read_registry()
    base = load_file(str(ROOT / registry['models']['baseline-expanded']['adapter_path'] / 'adapter_model.safetensors'))
    report = {'adapters': {}, 'datasets': {}}
    for key in ('tone-v1', 'emotion-v2', 'mixed-v3', 'multiturn-v4'):
        current = load_file(str(ROOT / registry['models'][key]['adapter_path'] / 'adapter_model.safetensors'))
        assert current.keys() == base.keys()
        change2 = sum(float((current[k].float() - base[k].float()).square().sum()) for k in base)
        base2 = sum(float(base[k].float().square().sum()) for k in base)
        report['adapters'][key] = {
            'changed_tensors': sum(not torch.equal(current[k], base[k]) for k in base),
            'total_tensors': len(base), 'relative_parameter_delta_l2': (change2 / base2) ** .5}
        rows = [json.loads(line) for line in (ROOT / ('data/external_sft_' + key) / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
        speeches = [r['answer']['speech'] for r in rows if 'speech' in r['answer']]
        report['datasets'][key] = {'rows': len(rows), 'character_speech_rows': len(speeches),
            'speech_mean_characters': statistics.mean(map(len, speeches)),
            'speech_max_characters': max(map(len, speeches)),
            'exact_bad_refusal_targets': sum(s == '할 수 없어.' for s in speeches)}
    return report


def generations(v2=False):
    import torch
    from transformers import set_seed
    from src.persona_runtime import PersonaRuntime, persona_messages, DEFAULT_GENERATION
    from src.character_lab_models import response_context
    from src.character_lab_schema import CharacterState, Session
    from src.model_registry import read_registry
    runtime = PersonaRuntime()
    runtime._load()
    model, tokenizer = runtime.backend.model, runtime.backend.tokenizer
    registry = read_registry()
    model.load_adapter(str(ROOT / registry['models']['tone-v1']['adapter_path']), adapter_name='audit-tone')
    session = Session().model_dump(mode='json')
    state = CharacterState().model_dump(mode='json')
    cases = [
        ('reluctance', '아 운동하기 싫어.', []),
        ('recommend', '운동 뭐 하지?', []),
        ('bored', '하 심심하다.', []),
        ('bored-followup', '그중 하나만 골라서 지금 바로 시작하게 해줘.', [
            {'user': '하 심심하다.', 'assistant': '종이 한 장으로 그림 그리기나 끝말잇기는 어때?'},
            {'user': '운동 말고, 집에서 혼자 할 수 있는 거로.', 'assistant': '그림 그리기는 혼자 바로 할 수 있어.'}]),
        ('instructions', '방금 말한 걸 순서대로 짧게 정리해줘.', [
            {'user': '운동 뭐 하지?', 'assistant': '집에서 가볍게 스트레칭은 어때?'},
            {'user': '집에 있고 도구는 없어. 5분 정도만 할래.', 'assistant': '어깨와 발목을 천천히 풀어보자.'}]),
    ]
    variants = [('base-plain', False, 'plain'), ('base-full-baseline', False, 'baseline'),
                ('base-full-natural-v1', False, 'natural-v1'), ('baseline-full-baseline', 'default', 'baseline'),
                ('baseline-full-natural-v1', 'default', 'natural-v1'), ('tone-full-natural-v1', 'audit-tone', 'natural-v1')]
    if v2:
        variants = [('base-natural-v2',False,'natural-v2'),('baseline-natural-v2','default','natural-v2'),
                    ('tone-natural-v2','audit-tone','natural-v2')]
    results = []
    settings = dict(DEFAULT_GENERATION, seed=20261010)
    for variant, adapter, profile in variants:
        if adapter:
            model.set_adapter(adapter)
        for case_id, text, history in cases:
            context = response_context(session, state, text, [], 'casual')
            context['recent'] = copy.deepcopy(history)
            if profile == 'plain':
                prompt = [{'role': 'system', 'content': '너는 헬스몬의 펭귄 동료 몽글이야. 한국어 반말로 상대의 말에 자연스럽게 답해. 사용자가 원하는 구체적인 도움을 제공해. 실제로 하지 않은 행동을 완료했다고 말하지 마.'}]
                for turn in history:
                    prompt.extend([{'role': 'user', 'content': turn['user']}, {'role': 'assistant', 'content': turn['assistant']}])
                prompt.append({'role': 'user', 'content': text})
            else:
                prompt = persona_messages(context, text, profile=profile)
            batch = tokenizer.apply_chat_template(prompt, tokenize=True, enable_thinking=False,
                add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
            set_seed(settings['seed'])
            with torch.inference_mode():
                if adapter:
                    out = model.generate(**batch, **{k: settings[k] for k in ('max_new_tokens', 'temperature', 'top_p', 'repetition_penalty')}, do_sample=True)
                else:
                    with model.disable_adapter():
                        out = model.generate(**batch, **{k: settings[k] for k in ('max_new_tokens', 'temperature', 'top_p', 'repetition_penalty')}, do_sample=True)
            raw = tokenizer.decode(out[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
            row = {'variant': variant, 'case': case_id, 'user_text': text, 'raw': raw,
                   'input_tokens': batch['input_ids'].shape[1], 'prompt': prompt}
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'prompt'}, ensure_ascii=False), flush=True)
            del out, batch
            torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--v2', action='store_true')
    args = parser.parse_args()
    path = ROOT / args.output
    if path.exists():
        raise RuntimeError('Existing audit preserved; choose a new output')
    report = {'cpu_audit': cpu_audit(), 'generations': [], 'completed': False}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['cpu_audit'], ensure_ascii=False), flush=True)
    if args.generate:
        report['generations'] = generations(args.v2)
    report['completed'] = True
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
