from ...utils.safe_ops import safe_sigmoid, safe_inverse_sigmoid
import torch, torch.nn as nn
from torch import Tensor
from typing import NamedTuple


def spherical2cartesian(anchor, pc_range, phi_activation='loop'):
    if phi_activation == 'sigmoid':
        xyz = safe_sigmoid(anchor[..., :3])
    elif phi_activation == 'loop':
        xy = safe_sigmoid(anchor[..., :2])
        z = torch.remainder(anchor[..., 2:3], 1.0)
        xyz = torch.cat([xy, z], dim=-1)
    else:
        raise NotImplementedError
    rrr = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    theta = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    phi = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    xxx = rrr * torch.sin(theta) * torch.cos(phi)
    yyy = rrr * torch.sin(theta) * torch.sin(phi)
    zzz = rrr * torch.cos(theta)
    xyz = torch.stack([xxx, yyy, zzz], dim=-1)
    
    return xyz

def cartesian(anchor, pc_range, use_sigmoid=True):
    if use_sigmoid:
        xyz = safe_sigmoid(anchor[..., :3])
    else:
        xyz = anchor[..., :3].clamp(min=1e-6, max=1-1e-6)
    xxx = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    yyy = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    zzz = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack([xxx, yyy, zzz], dim=-1)
    
    return xyz

def reverse_cartesian(xyz, pc_range, use_sigmoid=True):
    xxx = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    yyy = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    zzz = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    unitxyz = torch.stack([xxx, yyy, zzz], dim=-1)
    if use_sigmoid:
        anchor = safe_inverse_sigmoid(unitxyz)
    else:
        anchor = unitxyz.clamp(min=1e-6, max=1-1e-6)
    return anchor

def get_group_norm_groups(num_channels, default_groups=32):
    """
    Calculate appropriate number of groups for GroupNorm
    
    Args:
        num_channels: number of channels
        default_groups: default number of groups (default: 32)
    
    Returns:
        num_groups: number of groups for GroupNorm
    """
    num_groups = min(default_groups, num_channels // 4)  # Ensure at least 4 channels per group
    if num_groups < 1:
        num_groups = 1
    return num_groups


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    """
    Create MLP layers with GroupNorm instead of LayerNorm
    Returns layers suitable for Sequential that work with [B, N, C] tensors
    """
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        # Replace LayerNorm with GroupNorm
        num_groups = min(32, embed_dims // 4)
        if num_groups < 1:
            num_groups = 1
        layers.append(GroupNormWrapper(num_groups, embed_dims))
    return layers


class GroupNormWrapper(nn.Module):
    """Wrapper to use GroupNorm with (B, N, C) tensors"""
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)
    
    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        # Reshape to [B*N, C, 1] for GroupNorm
        x = x.view(B * N, C, 1)
        x = self.gn(x)
        # Reshape back to [B, N, C]
        x = x.view(B, N, C)
        return x


class GaussianPrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    semantics: Tensor
    uncertainties: Tensor = None  # 新增：[B, G] 语义不确定性
    velocities: Tensor = None     # 新增：[B, G, 3] 3D速度向量
    evidential_alphas: Tensor = None  # 新增：[B, G, C] Dirichlet参数（用于evidential learning）
    original_means: Tensor = None
    delta_means: Tensor = None
