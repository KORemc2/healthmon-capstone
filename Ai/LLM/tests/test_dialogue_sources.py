import hashlib
import importlib.util
import json
from pathlib import Path
import pytest


def loader(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/fetch_dialogue_sources.py'
    spec=importlib.util.spec_from_file_location('dialogue_sources_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'RAW',tmp_path/'data/external_raw')
    payload='한국어 공개 대화 테스트'.encode('utf-8')
    item={'file':'data/external_raw/test/data.json','url':'https://example.invalid/pinned-revision/data.json',
          'sha256':hashlib.sha256(payload).hexdigest()}
    (tmp_path/'experiments').mkdir()
    (tmp_path/'experiments/external-dialogue-sources.json').write_text(json.dumps({'sources':[{'files':[item]}]}))
    return module,payload,item


def test_fresh_clone_downloads_recorded_revision(monkeypatch,tmp_path):
    module,payload,item=loader(tmp_path,monkeypatch);urls=[]
    def get(url):urls.append(url);return payload
    monkeypatch.setattr(module,'get',get)
    module.main()
    assert urls==[item['url']]
    assert (tmp_path/item['file']).read_bytes()==payload


def test_modified_cached_source_is_preserved(monkeypatch,tmp_path):
    module,payload,item=loader(tmp_path,monkeypatch)
    target=tmp_path/item['file'];target.parent.mkdir(parents=True);target.write_bytes(b'changed')
    def get(url):raise AssertionError('Modified cache must not be overwritten')
    monkeypatch.setattr(module,'get',get)
    with pytest.raises(RuntimeError,match='Pinned local source changed'):module.main()
    assert target.read_bytes()==b'changed'
