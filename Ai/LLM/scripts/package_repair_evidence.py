"""Package the editable deck and synthetic run evidence; omit external holdouts."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/dialogue-repair-20261010'


def main():
    index = json.loads((BASE / 'output/evidence-index.json').read_text(encoding='utf-8'))
    conversations = BASE / 'repair-conversations.json'
    if hashlib.sha256(conversations.read_bytes()).hexdigest() != index['raw_sha256']:
        raise RuntimeError('Conversation evidence changed after export')
    seeds = json.loads((BASE / 'repair-seed-checks.json').read_text(encoding='utf-8'))
    fixed = json.loads((ROOT / 'experiments/results/repair-v5-fixed.json').read_text(encoding='utf-8'))
    if not seeds.get('completed') or not fixed.get('completed'):
        raise RuntimeError('Complete the actual evaluations first')
    entries = [
        (BASE / 'output/HEALTHMON_재학습_실제대화비교.pptx', 'HEALTHMON_재학습_실제대화비교.pptx'),
        (BASE / 'output/전체대화.csv', '전체대화.csv'),
        (BASE / 'output/전체대화.md', '전체대화.md'),
        (BASE / 'output/evidence-index.json', 'evidence-index.json'),
        (conversations, '실행기록/repair-conversations.json'),
        (BASE / 'repair-seed-checks.json', '실행기록/repair-seed-checks.json'),
        (ROOT / 'experiments/results/repair-v5-fixed.json', '실행기록/repair-v5-fixed.json'),
        (ROOT / 'experiments/repair-v5.json', '실행기록/training-metadata.json'),
    ]
    for page in index['pages']:
        entries.append((BASE / 'output/captures' / page['capture'], '라벨별캡처/' + page['capture']))
    for path, _ in entries:
        if not path.is_file():
            raise RuntimeError(f'Missing completed evidence: {path.name}')
    archive = BASE / 'HEALTHMON_재학습_발표자료와_캡처.zip'
    readme = (
        'HEALTHMON 재학습 실제 대화 비교\n\n'
        '기존 Baseline / 같은 가중치에 새 프롬프트 / 새 QLoRA와 새 프롬프트를 비교합니다.\n'
        '전체대화.csv와 전체대화.md에는 첫 반응27건, 후속18건, 단일 축18건의 총63건이 있습니다.\n'
        '후속 대화는 각 모델 자신의 실제 이전 응답을 포함합니다. 첫 답변은 후속 기록에서 재사용합니다.\n'
        '추가 seed 검사는18응답, 고정 회귀는48대화 응답과18분석기 출력입니다.\n'
        '사람 평가 점수나 자동 서비스 채택은 없습니다. 정성 검토와 형식 검사를 구분합니다.\n\n'
        '라벨별캡처의 파일명으로 모델·상황·수치를 구분할 수 있습니다.\n'
        '캡처는 실제 생성 결과를 정리한 브라우저 기록 화면이며 실시간 채팅 화면과 구분합니다.\n'
        '외부 데이터셋의 로컬 홀드아웃 원문과 모델 가중치는 이 파일에 포함하지 않습니다.\n'
    )
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path, label in entries:
            zipped.write(path, label)
        zipped.writestr('읽어주세요.txt', readme)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise RuntimeError('Archive verification failed')
    print(json.dumps({'archive': str(archive), 'files': len(entries) + 1,
                      'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
