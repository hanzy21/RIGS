"""
LayerNorm2d for 2D convolutional layers.
This module provides LayerNorm normalization for 2D feature maps [B, C, H, W].
"""
import torch
import torch.nn as nn
from mmcv.cnn.bricks.norm import MODELS as NORM_LAYERS


@NORM_LAYERS.register_module()
class LN2d(nn.Module):
    """LayerNorm2d: Layer Normalization for 2D feature maps.
    
    This normalizes over the channel and spatial dimensions [C, H, W] for each sample.
    Unlike GroupNorm which normalizes within groups, LayerNorm normalizes across
    all channels and spatial locations, which may preserve more input differences.
    
    Args:
        num_channels (int): Number of channels.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5.
        affine (bool): If True, learnable affine parameters (weight and bias) are added. Default: True.
    """
    
    def __init__(self, num_channels, eps=1e-5, affine=True, init_weight_scale=1.0, init_bias=0.0):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            # 方案4：使用可配置的初始值，默认weight=1.0, bias=0.0（恒等变换）
            # 可以通过init_weight_scale和init_bias调整初始值
            # 例如：init_weight_scale=2.0 会让weight从更大的值开始，可能保留更多输入差异
            self.weight = nn.Parameter(torch.ones(num_channels) * init_weight_scale)
            self.bias = nn.Parameter(torch.zeros(num_channels) + init_bias)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
        
        Returns:
            torch.Tensor: Normalized tensor of shape [B, C, H, W].
        """
        B, C, H, W = x.shape
        
        # Reshape to [B, C, H*W] for normalization
        x = x.view(B, C, H * W)
        
        # Compute mean and variance over [C, H*W] dimensions
        mean = x.mean(dim=[1, 2], keepdim=True)
        var = x.var(dim=[1, 2], keepdim=True, unbiased=False)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back to [B, C, H, W]
        x = x.view(B, C, H, W)
        
        # Apply affine transformation if enabled
        if self.affine:
            x = x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)
        
        return x
    
    def __repr__(self):
        return f'{self.__class__.__name__}(num_channels={self.num_channels}, eps={self.eps}, affine={self.affine})'

