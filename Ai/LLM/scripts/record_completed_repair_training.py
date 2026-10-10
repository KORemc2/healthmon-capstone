"""Publish metadata from a genuinely completed run, without model/data files."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/dialogue-repair-v5-20261010'


def main():
    metrics = [json.loads(line) for line in (RUN / 'metrics.jsonl').read_text().splitlines()]
    if metrics[-1]['event'] != 'complete':
        raise RuntimeError('Training has not completed successfully')
    config = json.loads((RUN / 'config.json').read_text(encoding='utf-8'))
    selection = json.loads((RUN / 'selection.json').read_text(encoding='utf-8'))
    path = ROOT / 'experiments/repair-v5.json'
    experiment = json.loads(path.read_text(encoding='utf-8'))
    if experiment['dataset']['sha256'] != config['data_manifest']['sha256']:
        raise RuntimeError('Dataset metadata changed')
    if experiment['base_revision'] != config['base_revision']:
        raise RuntimeError('Base model revision changed')
    epochs = []
    for row in metrics:
        if row['event'] != 'validation':
            continue
        checkpoint = Path(row['checkpoint']).resolve()
        if not checkpoint.is_relative_to(RUN.resolve()):
            raise RuntimeError('Unexpected checkpoint path')
        batches = [x for x in metrics if x['event'] == 'backward' and x['epoch'] == row['epoch']]
        epochs.append({'epoch': row['epoch'], 'train_microbatch_count': len(batches),
                       'train_microbatch_mean_loss': statistics.mean(x['loss'] for x in batches),
                       'validation_mean_loss': row['loss'],
                       'checkpoint': str(checkpoint.relative_to(ROOT)).replace('\\', '/'),
                       'adapter_sha256': hashlib.sha256((checkpoint / 'adapter_model.safetensors').read_bytes()).hexdigest()})
    selected = Path(selection['selected']['checkpoint']).resolve()
    selected_epoch = next(row for row in epochs if (ROOT / row['checkpoint']).resolve() == selected)
    baseline = ROOT / 'runs/expanded-20260929/checkpoint-epoch-1/adapter_model.safetensors'
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != experiment['baseline_adapter_sha256']:
        raise RuntimeError('Preserved baseline changed')
    experiment.update(training_status='completed', training_git_commit=config['git_commit'],
                      optimizer_steps=metrics[-1]['optimizer_steps'], completed_epochs=len(epochs),
                      trainable_parameters=config['trainable'],
                      peak_allocated_gib=metrics[-1]['peak_allocated_gib'],
                      peak_reserved_gib=metrics[-1]['peak_reserved_gib'],
                      reload_validation_loss=metrics[-1]['reload_validation_loss'],
                      epoch_metrics=epochs, selected_checkpoint=selected_epoch['checkpoint'],
                      selected_adapter_sha256=selected_epoch['adapter_sha256'],
                      checkpoint_selection_basis='Minimum validation loss; no model promotion or dialogue-quality conclusion')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    print(json.dumps({'training_status': 'completed', 'optimizer_steps': experiment['optimizer_steps'],
                      'selected_checkpoint': experiment['selected_checkpoint'], 'epochs': epochs}, ensure_ascii=False))


if __name__ == '__main__':
    main()
