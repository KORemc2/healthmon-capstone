"""Compare registered adapters using one immutable context and existing validation."""
import copy
import hashlib
import json
from pathlib import Path
from .model_registry import read_registry, entry
from .persona_runtime import persona_messages


def analyze_model(runtime, model_id, text, settings):
    """Run the existing Analyzer using the requested adapter, then restore selection."""
    from types import SimpleNamespace
    from .character_lab_models import ConversationAnalyzer
    from .model_registry import ROOT
    model = entry(model_id, allow_rejected=True)
    previous = []
    def load():
        runtime._load()
        loaded = runtime.backend.model
        previous.append(loaded.active_adapter)
        key = 'compare_' + hashlib.sha256(model['adapter_path'].encode()).hexdigest()[:12]
        if key not in loaded.peft_config:
            loaded.load_adapter(str(ROOT / model['adapter_path']), adapter_name=key)
        loaded.set_adapter(key)
        loaded.eval()
        return model['adapter_path']
    proxy = SimpleNamespace(lock=runtime.lock, backend=runtime.backend, _load=load)
    try:
        return ConversationAnalyzer(proxy).analyze(text, 'qwen', settings)
    finally:
        if previous:
            with runtime.lock:
                runtime.backend.model.set_adapter(previous[-1])


def compare_models(runtime, context, text, model_ids, settings, mode='qwen'):
    registry = read_registry()
    models = [entry(key, registry, require_files=mode == 'qwen', allow_rejected=True) for key in model_ids]
    context = copy.deepcopy(context)
    digest = hashlib.sha256(json.dumps(context, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    results = []
    with runtime.lock:
        previous_profile = getattr(runtime, 'dialogue_profile', 'baseline')
        previous_adapter = None
        try:
            if mode == 'qwen':
                runtime._load()
                previous_profile = getattr(runtime, 'dialogue_profile', 'baseline')
                previous_adapter = runtime.backend.model.active_adapter
            for model in models:
                if mode == 'qwen':
                    loaded = runtime.backend.model
                    # PEFT keeps the shared base in memory; only small LoRA weights are switched.
                    key = 'compare_' + hashlib.sha256(model['adapter_path'].encode()).hexdigest()[:12]
                    if key not in loaded.peft_config:
                        from .model_registry import ROOT
                        loaded.load_adapter(str(ROOT / model['adapter_path']), adapter_name=key)
                    loaded.set_adapter(key)
                    loaded.eval()
                runtime.dialogue_profile = model['prompt_profile']
                result = runtime._one(copy.deepcopy(context), text, dict(settings), mode)
                results.append({'model_id': model['id'], 'status': model['status'],
                    'adapter_path': model['adapter_path'], 'prompt_profile': model['prompt_profile'],
                    'prompt': persona_messages(context, text, profile=model['prompt_profile']), **result})
        finally:
            runtime.dialogue_profile = previous_profile
            if mode == 'qwen' and previous_adapter is not None:
                runtime.backend.model.set_adapter(previous_adapter)
    return {'user_text': text, 'context': context, 'context_sha256': digest,
            'generation': dict(settings), 'results': results, 'mode': mode,
            'same_context_and_seed': True, 'human_review': False,
            'storage': 'synthetic_lab_only', 'automatic_selection': False}
