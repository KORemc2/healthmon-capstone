"""Check tracked publication candidates against local secrets without printing values."""
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parents[1]


def main():
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True).strip())
    names=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
    secrets=[]
    for path in (ROOT/'.env',Path('E:/qwen-llm/.env')):
        if not path.is_file():continue
        for key,value in dotenv_values(path).items():
            if not value:continue
            if any(word in key for word in ('TOKEN','PASSWORD','SECRET','DATABASE_URL')):
                secrets.append(value)
                if key=='DEV_TOKENS':secrets.extend(json.loads(value))
                if key=='DATABASE_URL':secrets.append(urlparse(value).password)
    hits=[]
    for name in names:
        path=repo/name
        if path.name=='.env' or path.suffix in ('.safetensors','.pt','.sqlite3','.gguf'):
            hits.append({'file':name,'kind':'private_runtime_file'})
        try:value=path.read_text(encoding='utf-8')
        except (UnicodeDecodeError,IsADirectoryError):continue
        if any(secret and len(secret)>10 and secret in value for secret in secrets):
            hits.append({'file':name,'kind':'actual_local_secret'})
        if re.search(r'(?m)^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----$|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{30,}',value):
            hits.append({'file':name,'kind':'key_pattern'})
    report={'files_checked':len(names),'secret_hits':hits,'passed':not hits,
            'scope':'Secret/ignored-file audit only; synthetic dataset provenance is documented separately.'}
    print(json.dumps(report,indent=2))
    if hits:raise SystemExit(1)


if __name__=='__main__':main()
