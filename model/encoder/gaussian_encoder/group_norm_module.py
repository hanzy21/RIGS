"""
GroupNorm module for (B, N, C) tensors
Replaces LayerNorm with GroupNorm for better batch-size independence
"""
import torch.nn as nn
from mmengine.registry import MODELS


@MODELS.register_module()
class GroupNorm1D(nn.Module):
    """
    GroupNorm wrapper for 1D sequences (B, N, C)
    Replaces LayerNorm with GroupNorm for better batch-size independence
    """
    def __init__(self, num_channels=None, num_groups=None, eps=1e-5, affine=True, normalized_shape=None, **kwargs):
        """
        Args:
            num_channels: number of channels (embed_dims). If None, will use normalized_shape if provided
            num_groups: number of groups for GroupNorm. If None, use 32 groups
            eps: epsilon for numerical stability
            affine: whether to use learnable affine parameters
            normalized_shape: compatibility with LayerNorm config (will be converted to num_channels)
        """
        super().__init__()
        # Handle compatibility with LayerNorm config format
        if num_channels is None:
            if normalized_shape is not None:
                # If normalized_shape is provided (from LayerNorm config), use it as num_channels
                if isinstance(normalized_shape, (list, tuple)):
                    num_channels = normalized_shape[0] if len(normalized_shape) > 0 else 128
                else:
                    num_channels = normalized_shape
            else:
                raise ValueError("Either num_channels or normalized_shape must be provided")
        
        if num_groups is None:
            # Default: use 32 groups, but ensure at least 4 channels per group
            num_groups = min(32, num_channels // 4)
        if num_groups < 1:
            num_groups = 1
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, C] tensor
        Returns:
            x: [B, N, C] tensor after GroupNorm
        """
        B, N, C = x.shape
        # Reshape to [B*N, C, 1] for GroupNorm
        x = x.view(B * N, C, 1)
        x = self.gn(x)
        # Reshape back to [B, N, C]
        x = x.view(B, N, C)
        return x

