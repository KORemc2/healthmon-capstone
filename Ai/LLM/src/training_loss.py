"""Assistant-only causal loss without allocating unused prompt vocabulary logits."""
import torch
from torch.nn import functional as F


def assistant_only_loss(model, batch):
    labels=batch['labels']
    if labels.ndim!=2:
        raise ValueError('Selected-logit training requires a batch of causal labels')
    positions=torch.nonzero((labels[:,1:]!=-100).any(dim=0),as_tuple=False).flatten()
    if positions.numel()==0:raise ValueError('No supervised assistant tokens')
    output=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                 logits_to_keep=positions,use_cache=False)
    targets=labels[:,positions+1]
    flat_logits=output.logits.float().reshape(-1,output.logits.shape[-1])
    if 'loss_weights' not in batch:
        return F.cross_entropy(flat_logits,targets.reshape(-1),ignore_index=-100)
    losses=F.cross_entropy(flat_logits,targets.reshape(-1),ignore_index=-100,reduction='none').reshape_as(targets)
    weights=batch['loss_weights'][:,positions+1]*(targets!=-100)
    return (losses*weights).sum()/weights.sum()
