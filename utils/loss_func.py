import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss_CE(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction=False):
        super(FocalLoss_CE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets.long(), reduction="none")
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        if self.reduction:
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)
