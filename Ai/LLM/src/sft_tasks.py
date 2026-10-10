"""Use existing runtime prompts for opt-in character and analysis SFT."""
import json
from .prompts import messages


def training_prompt(row, profile='baseline', default_kind='legacy'):
    kind = row.get('prompt_kind', default_kind)
    if kind == 'conversation_analysis':
        from .character_lab_models import CONVERSATION_SYSTEM
        from .character_lab_schema import ConversationLabel
        ConversationLabel.model_validate(row['answer'])
        return [{'role': 'system', 'content': CONVERSATION_SYSTEM},
                {'role': 'user', 'content': json.dumps({'text': row['user_text']}, ensure_ascii=False)}]
    if kind == 'character':
        from .persona_runtime import persona_messages
        return persona_messages(row['context'], row['user_text'], profile=profile)
    if kind == 'legacy':
        return messages(row['context'], row['user_text'], profile=profile)
    raise ValueError('Unknown SFT prompt kind')
