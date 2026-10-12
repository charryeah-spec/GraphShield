# utils/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        if self.label_smoothing > 0:
            confidence = 1.0 - self.label_smoothing
            smooth_value = self.label_smoothing / (num_classes - 1)
            one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)
            # Put ``confidence`` on the target class and distribute smoothing
            # only across non-target classes. The previous expression added
            # ``smooth_value`` to every class, so each target row summed to
            # 1 + smoothing and unintentionally rescaled the loss.
            smooth_labels = one_hot * confidence + (1.0 - one_hot) * smooth_value
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(smooth_labels * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
