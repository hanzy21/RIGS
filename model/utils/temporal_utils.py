"""
Temporal utilities for transforming Gaussian parameters between frames using pose information
"""
import torch
import numpy as np
from ..utils.safe_ops import safe_sigmoid, safe_inverse_sigmoid


def transform_gaussian_xyz(
    prev_xyz_normalized,  # [B, N, 3] - 上一帧的归一化xyz（参数空间）
    prev_ego2global,      # [B, 4, 4] or [4, 4] - 上一帧的ego2global变换矩阵
    curr_ego2global,      # [B, 4, 4] or [4, 4] - 当前帧的ego2global变换矩阵
    pc_range,             # [6] - 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
    xyz_activation="sigmoid"
):
    """
    将上一帧的高斯xyz坐标变换到当前帧坐标系
    
    Args:
        prev_xyz_normalized: [B, N, 3] - 上一帧的归一化xyz（经过sigmoid逆变换的值）
        prev_ego2global: [B, 4, 4] or [4, 4] - 上一帧的ego2global变换矩阵
        curr_ego2global: [B, 4, 4] or [4, 4] - 当前帧的ego2global变换矩阵
        pc_range: [6] - 点云范围
        xyz_activation: "sigmoid" 或 "loop"
    
    Returns:
        curr_xyz_normalized: [B, N, 3] - 变换后的归一化xyz（参数空间）
    """
    B, N, _ = prev_xyz_normalized.shape
    device = prev_xyz_normalized.device
    
    # 确保pc_range是tensor
    if isinstance(pc_range, (list, np.ndarray)):
        pc_range = torch.tensor(pc_range, dtype=torch.float32, device=device)
    
    # 确保位姿矩阵是tensor且维度正确
    if isinstance(prev_ego2global, np.ndarray):
        prev_ego2global = torch.from_numpy(prev_ego2global).float().to(device)
    if isinstance(curr_ego2global, np.ndarray):
        curr_ego2global = torch.from_numpy(curr_ego2global).float().to(device)
    
    # 如果位姿矩阵是[4, 4]，扩展为[B, 4, 4]
    if prev_ego2global.dim() == 2:
        prev_ego2global = prev_ego2global.unsqueeze(0).expand(B, -1, -1)
    if curr_ego2global.dim() == 2:
        curr_ego2global = curr_ego2global.unsqueeze(0).expand(B, -1, -1)
    
    # Step 1: 将归一化参数转换为真实3D坐标
    if xyz_activation == "sigmoid":
        prev_xyz_normalized_sigmoid = safe_sigmoid(prev_xyz_normalized)
    else:
        # loop模式
        xy = safe_sigmoid(prev_xyz_normalized[..., :2])
        z = torch.remainder(prev_xyz_normalized[..., 2:3], 1.0)
        prev_xyz_normalized_sigmoid = torch.cat([xy, z], dim=-1)
    
    # 转换为真实坐标
    prev_xyz_real = torch.zeros_like(prev_xyz_normalized)
    prev_xyz_real[..., 0] = prev_xyz_normalized_sigmoid[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    prev_xyz_real[..., 1] = prev_xyz_normalized_sigmoid[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    prev_xyz_real[..., 2] = prev_xyz_normalized_sigmoid[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    
    # Step 2: 转换为齐次坐标 [B, N, 4]
    prev_xyz_homo = torch.cat([
        prev_xyz_real,
        torch.ones(B, N, 1, device=device)
    ], dim=-1)  # [B, N, 4]
    
    # Step 3: 变换到global坐标系，再变换到当前帧ego坐标系
    # prev_xyz_global = prev_ego2global @ prev_xyz_homo^T
    prev_xyz_homo_T = prev_xyz_homo.transpose(1, 2)  # [B, 4, N]
    prev_xyz_global_homo = torch.bmm(prev_ego2global, prev_xyz_homo_T)  # [B, 4, N]
    prev_xyz_global = prev_xyz_global_homo[:, :3, :].transpose(1, 2)  # [B, N, 3]
    
    # 变换到当前帧ego坐标系
    curr_ego2global_inv = torch.inverse(curr_ego2global)  # [B, 4, 4]
    curr_xyz_homo = torch.cat([
        prev_xyz_global,
        torch.ones(B, N, 1, device=device)
    ], dim=-1)  # [B, N, 4]
    
    curr_xyz_homo_T = curr_xyz_homo.transpose(1, 2)  # [B, 4, N]
    curr_xyz_real_homo = torch.bmm(curr_ego2global_inv, curr_xyz_homo_T)  # [B, 4, N]
    curr_xyz_real = curr_xyz_real_homo[:, :3, :].transpose(1, 2)  # [B, N, 3]
    
    # Step 4: 转换回归一化坐标
    curr_xyz_normalized_sigmoid = torch.zeros_like(curr_xyz_real)
    curr_xyz_normalized_sigmoid[..., 0] = (curr_xyz_real[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0] + 1e-8)
    curr_xyz_normalized_sigmoid[..., 1] = (curr_xyz_real[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1] + 1e-8)
    curr_xyz_normalized_sigmoid[..., 2] = (curr_xyz_real[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2] + 1e-8)
    
    # 限制在[0, 1]范围内
    curr_xyz_normalized_sigmoid = torch.clamp(curr_xyz_normalized_sigmoid, 0.0, 1.0)
    
    # Step 5: 转换回参数空间（sigmoid逆变换）
    if xyz_activation == "sigmoid":
        curr_xyz_normalized = safe_inverse_sigmoid(curr_xyz_normalized_sigmoid)
    else:
        # loop模式
        xy = safe_inverse_sigmoid(curr_xyz_normalized_sigmoid[..., :2])
        z = curr_xyz_normalized_sigmoid[..., 2:3]
        curr_xyz_normalized = torch.cat([xy, z], dim=-1)
    
    return curr_xyz_normalized




