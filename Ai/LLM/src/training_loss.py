"""Assistant-only causal loss without allocating unused prompt vocabulary logits."""
import torch
from torch.nn import functional as F


def assistant_only_loss(model, batch):
    labels=batch['labels']
    if labels.ndim!=2 or labels.shape[0]!=1:
        raise ValueError('Selected-logit training requires batch size one')
    positions=torch.nonzero(labels[0,1:]!=-100,as_tuple=False).flatten()
    if positions.numel()==0:raise ValueError('No supervised assistant tokens')
    output=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
                 logits_to_keep=positions,use_cache=False)
    targets=labels[0,positions+1]
    return F.cross_entropy(output.logits[0].float(),targets)
