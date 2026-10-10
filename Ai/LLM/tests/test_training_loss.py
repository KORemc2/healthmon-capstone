from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from src.training_loss import assistant_only_loss


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(19,8);self.head=torch.nn.Linear(8,19)
    def forward(self,input_ids,attention_mask,logits_to_keep=0,use_cache=False):
        hidden=self.embedding(input_ids).cumsum(dim=1)
        if isinstance(logits_to_keep,torch.Tensor):hidden=hidden[:,logits_to_keep,:]
        return SimpleNamespace(logits=self.head(hidden))


def test_selected_logits_preserve_masked_causal_loss_and_gradients():
    torch.manual_seed(42);model=SmallModel()
    batch={'input_ids':torch.tensor([[1,2,3,4,5,6,7]]),'attention_mask':torch.ones(1,7,dtype=torch.long),
           'labels':torch.tensor([[-100,-100,-100,4,5,-100,7]])}
    logits=model(batch['input_ids'],batch['attention_mask']).logits
    reference=F.cross_entropy(logits[:,:-1,:].reshape(-1,19),batch['labels'][:,1:].reshape(-1),ignore_index=-100)
    reference.backward();expected=[p.grad.clone() for p in model.parameters()];model.zero_grad(set_to_none=True)
    efficient=assistant_only_loss(model,batch);efficient.backward()
    torch.testing.assert_close(efficient,reference)
    for parameter,gradient in zip(model.parameters(),expected):torch.testing.assert_close(parameter.grad,gradient)


def test_empty_supervision_is_rejected():
    batch={'input_ids':torch.tensor([[1,2,3]]),'attention_mask':torch.ones(1,3),
           'labels':torch.full((1,3),-100)}
    with pytest.raises(ValueError,match='No supervised'):assistant_only_loss(SmallModel(),batch)
