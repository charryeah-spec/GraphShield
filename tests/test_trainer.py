import unittest

import torch
import torch.nn.functional as F

from utils.trainer import FocalLoss


class TrainerTests(unittest.TestCase):
    def test_zero_gamma_matches_alpha_weighted_smoothed_cross_entropy(self):
        logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
        targets = torch.tensor([0, 1])
        loss = FocalLoss(alpha=0.8, gamma=0.0, label_smoothing=0.1)(logits, targets)

        labels = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        cross_entropy = -(labels * F.log_softmax(logits, dim=1)).sum(dim=1)
        alpha = torch.tensor([0.2, 0.8])
        expected = (alpha * cross_entropy).mean()
        self.assertTrue(torch.allclose(loss, expected))


if __name__ == "__main__":
    unittest.main()
