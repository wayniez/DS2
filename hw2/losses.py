"""
Loss functions for the Siamese autoencoder.

Total loss = recon_loss + alpha * contrastive_loss
"""

import torch.nn as nn
import torch.nn.functional as F


class ReconLoss(nn.Module):
    """MSE"""
    def forward(self, x, x_hat):
        return F.mse_loss(x_hat, x)


class ContrastiveLoss(nn.Module):
    """
    It pulls z1 and z2 towards each other (both are normal).
    On the trainer, we always feed only normal pairs,
    so no margin is needed — we simply minimise the distance.
    """
    def forward(self, z1, z2):
        return F.mse_loss(z1, z2)


class TotalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.recon = ReconLoss()
        self.contrast = ContrastiveLoss()
        self.alpha = alpha

    def forward(self, x1, x2, x1_hat, x2_hat, z1, z2):
        l_recon = self.recon(x1, x1_hat) + self.recon(x2, x2_hat)
        l_contrast = self.contrast(z1, z2)
        total = l_recon + self.alpha * l_contrast
        return total, l_recon.item(), l_contrast.item()
