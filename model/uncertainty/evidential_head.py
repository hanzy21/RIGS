"""
Evidential Head for Uncertainty Estimation

This module implements evidential deep learning for semantic uncertainty estimation.
It converts semantic logits to Dirichlet distribution parameters (alpha) and computes
uncertainty based on the evidential framework.

Reference:
    Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """
    Evidential Head for converting semantic logits to Dirichlet parameters and uncertainty.
    
    The evidential framework models uncertainty by treating the network outputs as
    evidence for a Dirichlet distribution over class probabilities. Higher evidence
    (larger alpha values) indicates lower uncertainty.
    
    Args:
        num_classes (int): Number of semantic classes
        evidence_scale (float): Scale factor for evidence (default: 1.0)
        min_alpha (float): Minimum value for alpha parameters (default: 1.0)
    """
    
    def __init__(self, num_classes, evidence_scale=1.0, min_alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_scale = evidence_scale
        self.min_alpha = min_alpha
    
    def forward(self, logits):
        """
        Convert semantic logits to Dirichlet parameters and compute uncertainty.
        
        Args:
            logits: [B, N, C] or [B, G, C] - Semantic logits from network
                   B: batch size
                   N/G: number of points/Gaussians
                   C: number of classes
        
        Returns:
            alphas: [B, N, C] or [B, G, C] - Dirichlet distribution parameters
            uncertainties: [B, N] or [B, G] - Semantic uncertainty values
        """
        # Convert logits to evidence (non-negative values)
        # Using softplus ensures alpha > 0, which is required for Dirichlet distribution
        evidence = F.softplus(logits) * self.evidence_scale
        
        # Compute alpha parameters (Dirichlet parameters)
        # Adding min_alpha ensures numerical stability and proper Dirichlet distribution
        alphas = evidence + self.min_alpha  # [B, N, C] or [B, G, C]
        
        # Compute uncertainty: u = C / sum(alpha)
        # Higher sum(alpha) means more evidence and lower uncertainty
        alpha_sum = torch.sum(alphas, dim=-1, keepdim=False)  # [B, N] or [B, G]
        uncertainties = self.num_classes / (alpha_sum + 1e-8)  # [B, N] or [B, G]
        probs = alphas / alpha_sum.unsqueeze(-1)  # [B, N, C] or [B, G, C]
        
        # Clamp uncertainty to [0, 1] range for interpretability  TODO：可能带来>1时的梯度问题，先保留后续确认
        uncertainties = torch.clamp(uncertainties, min=0.0, max=1.0)
        
        return alphas, uncertainties
    
    def compute_entropy_uncertainty(self, logits):
        """
        Alternative uncertainty computation using entropy (for comparison).
        
        Args:
            logits: [B, N, C] or [B, G, C] - Semantic logits
        
        Returns:
            uncertainties: [B, N] or [B, G] - Entropy-based uncertainty
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        # Normalize entropy to [0, 1] range
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=probs.dtype, device=probs.device))
        uncertainties = entropy / (max_entropy + 1e-8)
        return uncertainties

