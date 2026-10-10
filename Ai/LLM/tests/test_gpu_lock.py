import pytest
from filelock import FileLock
from src.model import QwenBackend

def test_second_model_does_not_load_when_gpu_reserved(tmp_path,monkeypatch):
    # No CUDA allocation in this test; the lock must fail before model construction.
    # Isolate the unit-test lease from a real running model or trainer.
    monkeypatch.setattr('src.model.ROOT',tmp_path)
    (tmp_path/'runtime').mkdir()
    with FileLock(str(tmp_path/'runtime/gpu.lock'),timeout=0):
        with pytest.raises(RuntimeError,match='Another Healthmon process'):
            QwenBackend().load()
