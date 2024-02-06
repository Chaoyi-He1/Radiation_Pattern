import torch
import torch.nn as nn
from torch import Tensor


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.lsoftmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x: Tensor, y: Tensor, temperature: float = 1.0) -> Tensor:
        x = x / temperature
        y = y / temperature
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        x = x.expand(batch_size, batch_size, x.size(2))
        y = y.expand(batch_size, batch_size, y.size(2))
        logits = torch.sum(x * y, dim=-1)
        logits = self.lsoftmax(logits)
        loss = -torch.diag(logits).mean()
        return loss