"""
Evidential Loss for Uncertainty-aware Learning

This module implements the evidential loss function for training models with
uncertainty estimation using Dirichlet distributions.

Reference:
    Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018
    Kim et al., "E2-BKI: Evidential Ellipsoidal Bayesian Kernel Inference", 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import OPENOCC_LOSS


@OPENOCC_LOSS.register_module()
class EvidentialLoss(nn.Module):
    """
    Evidential Loss for uncertainty-aware semantic learning.
    
    The loss consists of two components:
    1. Evidential loss: encourages correct predictions with high evidence
    2. KL divergence: regularizes the Dirichlet distribution to prevent overconfidence
    
    Args:
        lambda_kl (float): Weight for KL divergence regularization (default: 0.1)
        use_kl_annealing (bool): Whether to use KL annealing (default: False)
        kl_annealing_max (float): Maximum KL weight for annealing (default: 0.1)
    """
    
    def __init__(self, weight=1.0, lambda_kl=0.1, use_kl_annealing=False, kl_annealing_max=0.1):
        super().__init__()
        self.weight = weight
        self.lambda_kl = lambda_kl
        self.use_kl_annealing = use_kl_annealing
        self.kl_annealing_max = kl_annealing_max
        self.register_buffer('current_epoch', torch.tensor(0.0))
        # Initialize epoch tracking
        self._current_epoch = 0
        self._current_iter = 0
    
    def set_epoch(self, epoch):
        """Update current epoch for KL annealing."""
        self.current_epoch = torch.tensor(float(epoch))
        self._current_epoch = epoch
    
    def forward(self, inputs, reduction='mean'):
        """
        Compute evidential loss.
        
        Args:
            inputs: Dictionary containing:
                - 'alphas': [B, N, C] or [B, G, C] - Dirichlet distribution parameters
                - 'sampled_label': [B, N] or [B, G] - Ground truth class labels
            reduction: 'mean' or 'sum' - Reduction method for loss
        
        Returns:
            loss: Scalar loss value
        """
        # Extract inputs from dictionary
        if isinstance(inputs, dict):
            alphas = inputs.get('alphas', None)
            labels = inputs.get('sampled_label', None)
            if alphas is None or labels is None:
                # If alphas or labels are not in inputs, return zero loss
                # (This loss might not be used in all configurations)
                return torch.tensor(0.0, device=next(self.parameters()).device if list(self.parameters()) else 'cpu')
        else:
            raise TypeError("EvidentialLoss.forward() expects an input dictionary")
        
        return self._compute_loss(alphas, labels, reduction)
    
    def _compute_loss(self, alphas, labels, reduction='mean'):
        """
        Compute evidential loss.
        
        Args:
            alphas: [B, N, C] or [B, G, C] - Dirichlet distribution parameters
                   B: batch size
                   N/G: number of points/Gaussians
                   C: number of classes
            labels: [B, N] or [B, G] - Ground truth class labels (long tensor)
            reduction: 'mean' or 'sum' - Reduction method for loss
        
        Returns:
            loss: Scalar loss value
        """
        """
        Internal method to compute evidential loss.
        
        Args:
            alphas: [B, N, C] or [B, G, C] - Dirichlet distribution parameters
            labels: [B, N] or [B, G] - Ground truth class labels (long tensor)
            reduction: 'mean' or 'sum' - Reduction method for loss
        
        Returns:
            loss: Scalar loss value
        """
        if alphas.ndim != 3 or alphas.shape[-1] != 2:
            raise ValueError(f"RIGS evidential alpha must be [B,N,2], got {tuple(alphas.shape)}")
        if labels.shape != alphas.shape[:-1]:
            raise ValueError("evidential labels/alpha shape mismatch")
        if labels.numel() and (labels.min() < 0 or labels.max() > 2):
            raise ValueError("RIGS labels must be 0, 1, 2")

        # Empty has no semantic Dirichlet target. Train only occupied voxels,
        # remapping background/foreground labels 1/2 to indices 0/1.
        occupied = labels > 0
        if not occupied.any():
            return alphas.sum() * 0.0
        alphas = alphas[occupied]
        labels = labels[occupied] - 1

        # Ensure labels are long tensor
        if labels.dtype != torch.long:
            labels = labels.long()
        
        # Compute alpha_sum: sum of all alpha values for each sample
        alpha_sum = torch.sum(alphas, dim=-1)  # [B, N] or [B, G]
        
        # Get alpha_y: alpha value for the true class
        labels_expanded = labels.unsqueeze(-1)  # [B, N, 1] or [B, G, 1]
        alpha_y = torch.gather(alphas, dim=-1, index=labels_expanded).squeeze(-1)  # [B, N] or [B, G]
        
        # Evidential loss: L_ev = sum(log(sum(alpha)) - log(alpha_y))
        # This encourages high evidence (large alpha_sum) and correct predictions (large alpha_y)
        log_alpha_sum = torch.log(alpha_sum + 1e-8)
        log_alpha_y = torch.log(alpha_y + 1e-8)
        L_ev = log_alpha_sum - log_alpha_y  # [B, N] or [B, G]
        
        # KL divergence: L_KL = KL(Dir(alpha) || Dir(1))
        # This regularizes the Dirichlet distribution to prevent overconfidence
        L_KL = self.compute_kl_divergence(alphas)  # [B, N] or [B, G]
        
        # Compute KL weight (with annealing if enabled)
        if self.use_kl_annealing:
            # Linear annealing: lambda_KL = epoch / max_epoch * kl_annealing_max
            # Following E2-BKI paper: lambda_KL = epoch / 120
            kl_weight = (self.current_epoch.item() / 120.0) * self.kl_annealing_max
            kl_weight = min(kl_weight, self.kl_annealing_max)
        else:
            kl_weight = self.lambda_kl
        
        # Total loss: L = L_ev + lambda_KL * L_KL
        total_loss = L_ev + kl_weight * L_KL
        
        # Apply reduction
        if reduction == 'mean':
            loss = torch.mean(total_loss)
        elif reduction == 'sum':
            loss = torch.sum(total_loss)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        # Apply weight
        return self.weight * loss
    
    def compute_kl_divergence(self, alphas):
        """
        Compute KL divergence: KL(Dir(alpha) || Dir(1))
        
        Formula:
            KL(Dir(alpha) || Dir(1)) = log(Gamma(sum(alpha))) - sum(log(Gamma(alpha))) +
                                      (C - sum(alpha)) * psi(sum(alpha)) +
                                      sum((alpha - 1) * (psi(alpha) - psi(sum(alpha))))
        
        Args:
            alphas: [B, N, C] or [B, G, C] - Dirichlet parameters
        
        Returns:
            kl_div: [B, N] or [B, G] - KL divergence values
        """
        # Compute sum of alphas
        alpha_sum = torch.sum(alphas, dim=-1, keepdim=True)  # [B, N, 1] or [B, G, 1]
        
        # Compute log(Gamma(sum(alpha))) - sum(log(Gamma(alpha)))
        log_gamma_alpha_sum = torch.lgamma(alpha_sum).squeeze(-1)  # [B, N] or [B, G]
        log_gamma_alphas = torch.lgamma(alphas)  # [B, N, C] or [B, G, C]
        sum_log_gamma_alphas = torch.sum(log_gamma_alphas, dim=-1)  # [B, N] or [B, G]
        term1 = log_gamma_alpha_sum - sum_log_gamma_alphas
        
        # Compute (C - sum(alpha)) * psi(sum(alpha))
        num_classes = alphas.shape[-1]
        psi_alpha_sum = torch.digamma(alpha_sum).squeeze(-1)  # [B, N] or [B, G]
        term2 = (num_classes - alpha_sum.squeeze(-1)) * psi_alpha_sum
        
        # Compute sum((alpha - 1) * (psi(alpha) - psi(sum(alpha))))
        psi_alphas = torch.digamma(alphas)  # [B, N, C] or [B, G, C]
        psi_diff = psi_alphas - psi_alpha_sum.unsqueeze(-1)  # [B, N, C] or [B, G, C]
        term3 = torch.sum((alphas - 1.0) * psi_diff, dim=-1)  # [B, N] or [B, G]
        
        # Total KL divergence
        kl_div = term1 + term2 + term3
        
        return kl_div
