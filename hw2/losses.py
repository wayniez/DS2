"""
Loss functions for the Siamese autoencoder.

Total loss = recon_loss + alpha * contrastive_loss
recon_loss = (1 - ssim_weight) * MSE + ssim_weight * (1 - SSIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# SSIM
# ------------------------------------------------------------------

def _gaussian_kernel(channels: int, kernel_size: int = 11, sigma: float = 1.5):
    """1D Gaussian → 2D kernel, repeated for each channel."""
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel = g.outer(g)                          # [k, k]
    kernel = kernel.unsqueeze(0).unsqueeze(0)    # [1, 1, k, k]
    kernel = kernel.expand(channels, 1, -1, -1)  # [C, 1, k, k]
    return kernel


def _denorm(t):
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1,3,1,1)
    return (t * std + mean).clamp(0, 1)

def ssim(x, x_hat, kernel_size: int = 11, sigma: float = 1.5,
         C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
    """
    Structural Similarity Index between x and x_hat.
    Returns a scalar mean across the batch, channels, and space.
    """
    B, C, H, W = x.shape
    kernel = _gaussian_kernel(C, kernel_size, sigma).to(x.device)
    pad = kernel_size // 2

    x = _denorm(x)      
    x_hat = _denorm(x_hat)

    mu_x    = F.conv2d(x,     kernel, padding=pad, groups=C)
    mu_xhat = F.conv2d(x_hat, kernel, padding=pad, groups=C)

    mu_x_sq    = mu_x    ** 2
    mu_xhat_sq = mu_xhat ** 2
    mu_x_xhat  = mu_x * mu_xhat

    sigma_x_sq    = F.conv2d(x     * x,     kernel, padding=pad, groups=C) - mu_x_sq
    sigma_xhat_sq = F.conv2d(x_hat * x_hat, kernel, padding=pad, groups=C) - mu_xhat_sq
    sigma_x_xhat  = F.conv2d(x     * x_hat, kernel, padding=pad, groups=C) - mu_x_xhat

    numerator   = (2 * mu_x_xhat  + C1) * (2 * sigma_x_xhat  + C2)
    denominator = (mu_x_sq + mu_xhat_sq + C1) * (sigma_x_sq + sigma_xhat_sq + C2)

    ssim_map = numerator / (denominator + 1e-8)  # [B, C, H, W]
    return ssim_map.mean()


# ------------------------------------------------------------------
# Loss modules
# ------------------------------------------------------------------

class ReconLoss(nn.Module):
    """
    Combined MSE + SSIM.
    ssim_weight=0 → pure MSE (old behavior).
    ssim_weight=0.5 → 50% MSE + 50% (1-SSIM).
    """
    def __init__(self, ssim_weight: float = 0.5):
        super().__init__()
        self.ssim_weight = ssim_weight

    def forward(self, x, x_hat):
        mse  = F.mse_loss(x_hat, x)
        if self.ssim_weight == 0:
            return mse
        ssim_val  = ssim(x, x_hat)
        ssim_loss = 1.0 - ssim_val
        return (1 - self.ssim_weight) * mse + self.ssim_weight * ssim_loss


class VICRegLoss(nn.Module):
    """
    VICReg (Variance-Invariance-Covariance Regularization).
    Addresses three issues related to MSE between latent variables:

    1. Invariance  — pulls z1 and z2 toward each other (like MSE)
    2. Variance    — prevents all z values from collapsing into a single point
                     (imposes a penalty if the batch standard deviation falls below 1)
    3. Covariance  — decorrelates the dimensions of z
                     (each dimension encodes a different feature)
    """
    def __init__(self, lambda_inv=25.0, mu_var=25.0, nu_cov=1.0, eps=1e-4):
        super().__init__()
        self.lambda_inv = lambda_inv  # invariance weight
        self.mu_var     = mu_var      # variance weight
        self.nu_cov     = nu_cov      # covariance weight
        self.eps        = eps

    def forward(self, z1, z2):
        # For spatial maps [B, D, H, W]:
        # Reshape to [B*H*W, D] — compute VICReg along the D channels
        # The covariance matrix will be [D, D] = [256, 256] — memory-safe
        if z1.dim() == 4:
            B, D, H, W = z1.shape
            z1 = z1.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
            z2 = z2.permute(0, 2, 3, 1).reshape(-1, D)
        B, D = z1.shape

        # --- Invariance: MSE between pairs ---
        l_inv = F.mse_loss(z1, z2)

        # --- Variance: std for each channel, at least 1 per batch ---
        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.eps)
        l_var = (
            torch.mean(F.relu(1.0 - std_z1)) +
            torch.mean(F.relu(1.0 - std_z2))
        ) / 2.0

        # --- Covariance: [D, D] — memory-safe ---
        z1_c = z1 - z1.mean(dim=0)
        z2_c = z2 - z2.mean(dim=0)
        cov_z1 = (z1_c.T @ z1_c) / (B - 1)
        cov_z2 = (z2_c.T @ z2_c) / (B - 1)
        mask = ~torch.eye(D, dtype=torch.bool, device=z1.device)
        l_cov = (
            cov_z1[mask].pow(2).sum() +
            cov_z2[mask].pow(2).sum()
        ) / D

        total = (
            self.lambda_inv * l_inv +
            self.mu_var     * l_var +
            self.nu_cov     * l_cov
        )
        return total


class TotalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, ssim_weight: float = 0.5):
        super().__init__()
        self.recon   = ReconLoss(ssim_weight=ssim_weight)
        self.vicreg  = VICRegLoss()
        self.alpha   = alpha

    def forward(self, x1, x2, x1_hat, x2_hat, z1, z2):
        l_recon    = self.recon(x1, x1_hat) + self.recon(x2, x2_hat)
        l_contrast = self.vicreg(z1, z2)
        total      = l_recon + self.alpha * l_contrast
        return total, l_recon.item(), l_contrast.item()