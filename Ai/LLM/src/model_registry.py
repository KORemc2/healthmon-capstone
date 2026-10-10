"""Local model inventory and explicit, reviewed selection. Never deletes weights."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'config/model_registry.json'
STATUSES = {'Baseline', 'Candidate', 'Selected', 'Rejected', 'Archived'}


def read_registry(path=REGISTRY):
    registry = json.loads(Path(path).read_text(encoding='utf-8'))
    if registry['selected_id'] not in registry['models']:
        raise ValueError('Selected model is missing')
    for key, model in registry['models'].items():
        if model['status'] not in STATUSES or model['prompt_profile'] not in ('baseline', 'natural-v1', 'natural-v2'):
            raise ValueError('Invalid registry entry: ' + key)
        relative = Path(model['adapter_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Adapter must be within the local LLM workspace')
    return registry


def entry(model_id, registry=None, require_files=True, allow_rejected=False):
    registry = read_registry() if registry is None else registry
    if model_id not in registry['models']:
        raise ValueError('Unknown model ID')
    model = dict(registry['models'][model_id], id=model_id)
    if model['status'] == 'Archived' or model['status'] == 'Rejected' and not allow_rejected:
        raise ValueError('Model is not available for inference')
    if require_files:
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            if not (ROOT / model['adapter_path'] / name).is_file():
                raise ValueError('Local adapter is missing: ' + model_id)
    return model


def cleanup_inventory(registry=None):
    registry = read_registry() if registry is None else registry
    protected = {model['adapter_path'] for key, model in registry['models'].items()
                 if model['status'] in ('Baseline', 'Selected', 'Candidate') or key == registry['selected_id']}
    return [{'id': key, 'adapter_path': model['adapter_path'], 'status': model['status'],
             'deletion_candidate': model['status'] in ('Rejected', 'Archived') and model['adapter_path'] not in protected,
             'automatic_deletion': False} for key, model in registry['models'].items()]


def select(model_id, path=REGISTRY):
    registry = read_registry(path)
    model = entry(model_id, registry)
    if model['status'] != 'Baseline' and not (model.get('human_review') and model.get('compatibility_passed')):
        raise ValueError('Selection requires recorded human review and compatibility checks')
    for value in registry['models'].values():
        if value['status'] == 'Selected':
            value['status'] = 'Candidate'
    registry['selected_id'] = model_id
    if model['status'] != 'Baseline':
        registry['models'][model_id]['status'] = 'Selected'
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return registry
