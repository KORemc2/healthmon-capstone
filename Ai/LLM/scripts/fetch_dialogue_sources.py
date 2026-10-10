"""Download pinned public dialogue sources; raw text remains local and ignored."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/external_raw'


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'HEALTHMON-research/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def save(folder, name, url):
    target = folder / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + '.download')
        temporary.write_bytes(get(url))
        temporary.replace(target)
    return {'file': str(target.relative_to(ROOT)).replace('\\', '/'), 'url': url,
            'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path=ROOT/'experiments/external-dialogue-sources.json'
    if manifest_path.exists():
        cached=json.loads(manifest_path.read_text(encoding='utf-8'))
        files=[f for source in cached['sources'] for f in source['files']]
        for item in files:
            if hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest()!=item['sha256']:
                raise RuntimeError('Pinned local source changed; preserve it and use a new version')
        print(json.dumps({'pinned_manifest_reused':True,'sha256_verified_files':len(files)}))
        return
    if any(RAW.iterdir()):
        raise RuntimeError('Unrecorded partial download preserved; choose a fresh version directory')
    records = []
    github = json.loads(get('https://api.github.com/repos/songys/Chatbot_data/commits/master'))
    revision = github['sha']
    base = f'https://raw.githubusercontent.com/songys/Chatbot_data/{revision}/'
    records.append({'id': 'songys/Chatbot_data', 'revision': revision, 'license': 'MIT',
        'use': 'Authored Korean daily dialogue; retain author and license.',
        'files': [save(RAW/'chatbot', name, base+name) for name in ('ChatbotData.csv', 'README.md', 'LICENSE')]})
    for repo, folder, license_name in (
            ('passing2961/KorEmpatheticDialogues', 'kor_empathy', 'cc-by-nc-4.0'),
            ('huggingface-KREW/korean-role-playing', 'roleplay', 'apache-2.0')):
        info = json.loads(get('https://huggingface.co/api/datasets/'+repo))
        if info.get('gated') not in (False, None):
            raise RuntimeError('Gated source skipped: '+repo)
        declared = info.get('cardData', {}).get('license')
        if declared != license_name:
            raise RuntimeError('License metadata changed: '+repo)
        revision = info['sha']
        files = ['README.md']
        if folder == 'kor_empathy':
            files += ['train.json', 'valid.json', 'test.json']
        else:
            files += [item['rfilename'] for item in info['siblings']
                      if item['rfilename'].startswith('general-roleplay-data/')
                      and item['rfilename'].endswith('.parquet')]
            if len(files) == 1:
                raise RuntimeError('General-roleplay files not found')
        records.append({'id': repo, 'revision': revision, 'license': declared,
            'use': 'Local noncommercial research only; no raw data, derived rows or adapters published.'
                   if folder == 'kor_empathy' else 'Only general-roleplay subset; exclude couple/YouTube/EXA identities.',
            'files': [save(RAW/folder, name,
                f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}') for name in files]})
    report = {'version': 'external-dialogue-20261010', 'sources': records,
              'paid_api_used': False, 'download_authentication_used': False,
              'raw_data_published': False, 'derivatives_published': False,
              'scope': 'Local noncommercial research experiments; no production/commercial approval.'}
    (ROOT/'experiments/external-dialogue-sources.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
