from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmengine import build_from_cfg
from mmengine.model import xavier_init, constant_init
import torch, torch.nn as nn
import torch.nn.functional as F_torch
import numpy as np
import math
from typing import List, Optional
from ...utils.safe_ops import safe_sigmoid
from ...utils.utils import get_rotation_matrix
from .utils import linear_relu_ln
try:
    from .ops import DeformableAggregationFunction as DAF
except:
    DAF = None


# ==================== Radar 点云下采样 & 空间近邻注意力工具 ====================

def voxel_downsample_radar(radar_points, radar_features, radar_mask,
                           voxel_size: float = 1.6,
                           max_points: int = 4096):
    """
    体素下采样：将 radar 点按空间体素分组，每个 voxel 内保留功率最强的一个点。
    保证空间均匀性，优于随机采样。

    Args:
        radar_points: [B, M, 3]
        radar_features: [B, M, F]
        radar_mask: [B, M] bool
        voxel_size: 体素大小（米）
        max_points: 下采样后上限
    Returns:
        radar_points, radar_features, radar_mask （下采样后）
    """
    B, M, _ = radar_points.shape
    device = radar_points.device

    if M <= max_points:
        return radar_points, radar_features, radar_mask

    out_pts = []
    out_feats = []
    out_masks = []

    for b in range(B):
        valid = radar_mask[b]  # [M]
        pts_b = radar_points[b][valid]     # [V, 3]
        feats_b = radar_features[b][valid] # [V, F]
        V = pts_b.shape[0]

        if V <= max_points:
            # 不需要下采样，直接 pad
            pad_n = max_points - V
            out_pts.append(F_torch.pad(pts_b, (0, 0, 0, pad_n)))
            out_feats.append(F_torch.pad(feats_b, (0, 0, 0, pad_n)))
            m = torch.zeros(max_points, dtype=torch.bool, device=device)
            m[:V] = True
            out_masks.append(m)
            continue

        # 体素化：计算每个点所属的体素索引
        voxel_coords = torch.floor(pts_b / voxel_size).long()  # [V, 3]
        # 偏移到非负
        vmin = voxel_coords.min(dim=0)[0]
        voxel_coords = voxel_coords - vmin  # [V, 3], all >= 0
        vmax = voxel_coords.max(dim=0)[0] + 1
        hash_val = (voxel_coords[:, 0] * vmax[1] * vmax[2]
                     + voxel_coords[:, 1] * vmax[2]
                     + voxel_coords[:, 2])  # [V]

        # 每个 voxel 内保留功率最强的点（向量化 scatter 实现）
        unique_voxels, inverse_idx = hash_val.unique(return_inverse=True)
        n_voxels = unique_voxels.shape[0]
        power = feats_b[:, 0] if feats_b.shape[1] > 0 else torch.zeros(V, device=device)

        # 向量化找每个 voxel 内最大 power 的点索引
        # 按 voxel 分组排序，同 voxel 内按 power 降序
        sort_key = inverse_idx.float() * 1e10 - power  # 同 voxel 内 power 大的排前面
        sorted_order = sort_key.argsort()
        sorted_voxel = inverse_idx[sorted_order]
        # 取每个 voxel 的第一个（即 power 最大的）
        first_mask = torch.ones(V, dtype=torch.bool, device=device)
        first_mask[1:] = sorted_voxel[1:] != sorted_voxel[:-1]
        best_idx = sorted_order[first_mask]  # [n_voxels]

        if n_voxels <= max_points:
            sel_pts = pts_b[best_idx]
            sel_feats = feats_b[best_idx]
            n_sel = n_voxels
        else:
            # voxel 数量超过 max_points，取 power 最强的 top-K voxel 代表点
            best_power = power[best_idx]
            _, topk_vi = best_power.topk(min(max_points, n_voxels))
            best_idx = best_idx[topk_vi]
            sel_pts = pts_b[best_idx]
            sel_feats = feats_b[best_idx]
            n_sel = sel_pts.shape[0]

        pad_n = max_points - n_sel
        out_pts.append(F_torch.pad(sel_pts, (0, 0, 0, pad_n)))
        out_feats.append(F_torch.pad(sel_feats, (0, 0, 0, pad_n)))
        m = torch.zeros(max_points, dtype=torch.bool, device=device)
        m[:n_sel] = True
        out_masks.append(m)

    return (torch.stack(out_pts), torch.stack(out_feats), torch.stack(out_masks))


def chunked_cross_attention(query, key, value, key_padding_mask,
                            num_heads, query_xyz=None, key_xyz=None,
                            spatial_radius=None,
                            topk_neighbors=64,
                            chunk_size=4096,
                            qkv_proj=None,
                            out_proj=None):
    """
    分块 Top-K 近邻交叉注意力。
    参考 e2bki 的 blockwise spatial pruning 思路：
    1. 对 query 分 chunk 处理
    2. 每个 chunk 中，每个 query 只和最近的 K 个 radar 点做注意力
    3. 通过 gather 操作只取用局部 K/V，注意力矩阵从 [B,H,chunk,M] 缩小到 [B,H,chunk,K]

    显存优化分析（4 decoder layers × 2 modules × 7 chunks）：
    - 全局注意力: [1,8,4096,4096]*4B × 56 = 28 GB（OOM）
    - Top-K=64:   [1,8,4096,64]*4B × 56 = 448 MB（✓）

    Args:
        query: [B, N, C]
        key: [B, M, C]
        value: [B, M, C]
        key_padding_mask: [B, M] bool, True=忽略
        num_heads: int
        query_xyz: [B, N, 3] query 的 3D 坐标
        key_xyz: [B, M, 3] key 的 3D 坐标
        spatial_radius: 不再使用（保留接口兼容），已被 topk_neighbors 替代
        topk_neighbors: 每个 query 最多注意的近邻数（默认 64）
        chunk_size: query 分块大小
        qkv_proj: dict with 'in_proj_weight', 'in_proj_bias'
        out_proj: nn.Linear
    Returns:
        [B, N, C]
    """
    B, N, C = query.shape
    M = key.shape[1]
    head_dim = C // num_heads
    scale = 1.0 / math.sqrt(head_dim)
    K = min(topk_neighbors, M)  # 近邻数不超过 key 数量

    # 全局 QKV 投影
    w = qkv_proj['in_proj_weight']  # [3*C, C]
    b = qkv_proj['in_proj_bias']    # [3*C]
    w_q, w_k, w_v = w.chunk(3, dim=0)
    b_q, b_k, b_v = b.chunk(3, dim=0)

    K_proj = F_torch.linear(key, w_k, b_k)     # [B, M, C]
    V_proj = F_torch.linear(value, w_v, b_v)    # [B, M, C]

    # 将 padding 点的 key 设为极大值，防止被 topk 选中
    if key_padding_mask is not None and key_xyz is not None:
        # 无效点 xyz 设为极远处
        key_xyz_for_dist = key_xyz.clone()
        key_xyz_for_dist[key_padding_mask] = 1e6

    BH = B * num_heads

    output_chunks = []

    for c_start in range(0, N, chunk_size):
        c_end = min(c_start + chunk_size, N)
        q_chunk = query[:, c_start:c_end]  # [B, chunk, C]
        chunk_n = c_end - c_start

        # Q 投影
        Q_chunk = F_torch.linear(q_chunk, w_q, b_q)  # [B, chunk, C]
        Q_chunk = Q_chunk.view(B, chunk_n, num_heads, head_dim)  # [B, chunk, H, D]
        Q_chunk = Q_chunk.permute(0, 2, 1, 3)  # [B, H, chunk, D]

        # ====== Top-K 近邻选择（no grad，不进入 autograd 图）======
        if query_xyz is not None and key_xyz is not None:
            q_xyz_chunk = query_xyz[:, c_start:c_end]  # [B, chunk, 3]
            with torch.no_grad():
                dist = torch.cdist(q_xyz_chunk.float(),
                                   key_xyz_for_dist.float())  # [B, chunk, M]
                _, topk_idx = dist.topk(K, dim=-1, largest=False)  # [B, chunk, K]
        else:
            # 无坐标信息时取前 K 个
            topk_idx = torch.arange(K, device=query.device).unsqueeze(0).unsqueeze(0).expand(B, chunk_n, -1)

        # ====== 通过 flattened gather 取局部 K/V（避免 expand 的大张量）======
        # K_proj: [B, M, C], V_proj: [B, M, C]
        # topk_idx: [B, chunk, K]
        idx_flat = topk_idx.reshape(B, -1)                          # [B, chunk*K]
        idx_for_kv = idx_flat.unsqueeze(-1).expand(-1, -1, C)       # [B, chunk*K, C]
        K_local = K_proj.gather(1, idx_for_kv).reshape(B, chunk_n, K, C)  # [B, chunk, K, C]
        V_local = V_proj.gather(1, idx_for_kv).reshape(B, chunk_n, K, C)  # [B, chunk, K, C]

        # reshape for multi-head: [B, chunk, K, C] -> [B, H, chunk, K, D]
        K_local = K_local.view(B, chunk_n, K, num_heads, head_dim).permute(0, 3, 1, 2, 4)  # [B, H, chunk, K, D]
        V_local = V_local.view(B, chunk_n, K, num_heads, head_dim).permute(0, 3, 1, 2, 4)  # [B, H, chunk, K, D]

        # ====== 局部注意力 ======
        # Q_chunk: [B, H, chunk, D], K_local: [B, H, chunk, K, D]
        attn_scores = torch.einsum('bhqd,bhqkd->bhqk', Q_chunk, K_local) * scale  # [B, H, chunk, K]

        # padding mask for topk neighbors
        if key_padding_mask is not None:
            # 检查选出的 topk 中哪些是 padding
            topk_pad = key_padding_mask.unsqueeze(1).expand(-1, chunk_n, -1).gather(2, topk_idx)  # [B, chunk, K]
            topk_pad = topk_pad.unsqueeze(1)  # [B, 1, chunk, K]
            attn_scores = attn_scores.masked_fill(topk_pad, float('-inf'))

        attn_weights = F_torch.softmax(attn_scores, dim=-1)  # [B, H, chunk, K]
        attn_weights = attn_weights.nan_to_num(0.0)

        # V_local: [B, H, chunk, K, D]
        chunk_out = torch.einsum('bhqk,bhqkd->bhqd', attn_weights, V_local)  # [B, H, chunk, D]
        chunk_out = chunk_out.permute(0, 2, 1, 3).reshape(B, chunk_n, C)  # [B, chunk, C]

        if out_proj is not None:
            chunk_out = out_proj(chunk_out)

        output_chunks.append(chunk_out)

    return torch.cat(output_chunks, dim=1)  # [B, N, C]


@MODELS.register_module()
class SparseGaussian3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        learnable_fixed_scale=1,
        fix_scale=None,
        pc_range=None,
        scale_range=None,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        **kwargs,
    ):
        super(SparseGaussian3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        self.learnable_fixed_scale = learnable_fixed_scale
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 3)

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,              # [B, N, anchor_dim] - anchor参数
        instance_feature=None,  # [B, N, embed_dims] - 可选，用于生成可学习采样点
    ):
        """
        SparseGaussian3DKeyPointsGenerator的前向传播流程：
        
        功能：为每个Gaussian anchor生成多个3D采样点（key points），用于后续的可变形注意力
        
        输入：
            anchor: [B, N, anchor_dim] - anchor参数（包含xyz, scale, rotation等）
            instance_feature: [B, N, embed_dims] - 可选，用于生成可学习的采样点
        
        过程：
            1. 生成固定的相对偏移（fix_scale）和可学习的相对偏移（learnable_scale）
            2. 结合Gaussian的scale和rotation，将相对偏移转换为3D坐标
            3. 加上anchor的中心位置（xyz），得到最终的3D采样点
        
        输出：
            key_points: [B, N, num_pts, 3] - 每个anchor的3D采样点坐标
        
        示例：如果num_pts=13（7个固定点 + 6个可学习点）
          - 固定点：中心点 + 6个方向的固定偏移
          - 可学习点：根据instance_feature学习的位置
        """
        bs, num_anchor = anchor.shape[:2]
        
        # ========== Step 1: 生成固定的相对偏移 ==========
        # fix_scale: 固定的相对偏移模式
        # 例如：[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0], ...]
        # 这些是相对于Gaussian中心的标准化偏移
        fix_scale = anchor.new_tensor(self.fix_scale)  # [num_fix_pts, 3]
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1])  # [B, N, num_fix_pts, 3]
        
        # ========== Step 2: 生成可学习的相对偏移（可选）==========
        # 如果配置了num_learnable_pts > 0，使用MLP从instance_feature学习采样点位置
        if self.num_learnable_pts > 0 and instance_feature is not None:
            # 2.1 MLP预测相对偏移
            learnable_scale = self.learnable_fc(instance_feature)  # [B, N, num_learnable_pts * 3]
            learnable_scale = learnable_scale.reshape(bs, num_anchor, self.num_learnable_pts, 3)
            
            # 2.2 将输出映射到[-0.5, 0.5]范围，然后乘以learnable_fixed_scale
            # 这样可以将可学习偏移限制在合理范围内
            learnable_scale = (
                safe_sigmoid(learnable_scale) - 0.5  # [B, N, num_learnable_pts, 3] - 范围[-0.5, 0.5]
            ) * self.learnable_fixed_scale  # 例如：乘以1.0，范围变为[-0.5, 0.5]
            
            # 2.3 拼接固定点和可学习点
            scale = torch.cat([scale, learnable_scale], dim=-2)  # [B, N, num_pts, 3]
        
        # ========== Step 3: 获取Gaussian的scale并转换为真实单位 ==========
        # 3.1 提取anchor的scale参数（归一化的参数）
        gs_scales = anchor[..., None, 3:6]  # [B, N, 1, 3] - scale参数
        
        # 3.2 如果scale_act是sigmoid，先进行sigmoid归一化
        if self.scale_act == "sigmoid":
            gs_scales = safe_sigmoid(gs_scales)  # [B, N, 1, 3] - 映射到[0, 1]
        
        # 3.3 映射到scale_range（真实单位：米）
        # 例如：scale_range=[0.01, 1.8]
        #   [0, 1] → [0.01, 1.8]米
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        # gs_scales: [B, N, 1, 3] - Gaussian的scale（真实单位）

        # ========== Step 4: 将相对偏移乘以Gaussian的scale ==========
        # 相对偏移（标准化） × Gaussian scale（真实单位） = 相对于中心的偏移（真实单位）
        # 例如：相对偏移[0.45, 0, 0] × scale[0.5m] = 偏移[0.225m, 0, 0]
        key_points = scale * gs_scales  # [B, N, num_pts, 3] - 相对于中心的偏移
        
        # ========== Step 5: 应用Gaussian的旋转 ==========
        # 5.1 提取旋转四元数并转换为旋转矩阵
        rots = anchor[..., 6:10]  # [B, N, 4] - 四元数 [w, x, y, z]
        rotation_mat = get_rotation_matrix(rots)  # [B, N, 3, 3] - 旋转矩阵
        rotation_mat = rotation_mat.transpose(-1, -2)  # [B, N, 3, 3] - 转置（逆旋转）
        
        # 5.2 应用旋转：将偏移从局部坐标系转换到全局坐标系
        # key_points是相对于Gaussian中心、在Gaussian局部坐标系中的偏移
        # 需要旋转到全局坐标系（世界坐标系）
        key_points = torch.matmul(
            rotation_mat[:, :, None],  # [B, N, 1, 3, 3]
            key_points[..., None]      # [B, N, num_pts, 3, 1]
        ).squeeze(-1)  # [B, N, num_pts, 3] - 旋转后的偏移（全局坐标系）

        # ========== Step 6: 获取anchor的中心位置 ==========
        # 6.1 提取anchor的xyz参数（归一化的参数）
        xyz = anchor[..., :3]  # [B, N, 3] - 归一化参数
        
        # 6.2 如果xyz_act是sigmoid，先进行sigmoid归一化
        if self.xyz_act == 'sigmoid':
            xyz = safe_sigmoid(xyz)  # [B, N, 3] - 映射到[0, 1]
        
        # 6.3 转换为3D真实坐标（在pc_range范围内）
        xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]  # X坐标
        yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]  # Y坐标
        zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]  # Z坐标
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)  # [B, N, 3] - 3D真实坐标
        
        # ========== Step 7: 将偏移加到中心位置 ==========
        # 最终的key points = anchor中心位置 + 旋转后的偏移
        # [B, N, 3] + [B, N, num_pts, 3] → [B, N, num_pts, 3]
        key_points = key_points + xyz.unsqueeze(2)  # [B, N, num_pts, 3] - 最终的3D采样点
        
        return key_points


@MODELS.register_module()
class RadarFeatureAggregation(BaseModule):
    """Radar–Gaussian 交叉注意力（体素下采样 + 分块空间近邻注意力版本）。

    改进点（相比随机采样 + 全局注意力）：
    1. 体素下采样：保证空间均匀性，每个 voxel 取功率最强点
    2. 分块处理：query 分 chunk 计算注意力，避免显存爆炸
    3. 空间近邻掩码：每个 Gaussian 只注意半径内的 radar 点（参考 e2bki blockwise pruning）
    """

    def __init__(
        self,
        embed_dims: int = 256,
        radar_feat_dim: int = 10,
        num_heads: int = 8,
        proj_drop: float = 0.0,
        pc_range: Optional[List[float]] = None,
        max_radar_points: int = 4096,
        voxel_size: float = 1.6,
        topk_neighbors: int = 64,
        attn_chunk_size: int = 4096,
        **kwargs,
    ):
        super(RadarFeatureAggregation, self).__init__()
        self.embed_dims = embed_dims
        self.radar_feat_dim = radar_feat_dim
        self.num_heads = num_heads
        self.pc_range = pc_range or [0.0, -25.6, -5.0, 51.2, 25.6, 3.0]
        self.max_radar_points = max_radar_points
        self.voxel_size = voxel_size
        self.topk_neighbors = topk_neighbors
        self.attn_chunk_size = attn_chunk_size
        # radar 编码：xyz(3) + radar_feats(F) -> embed_dims
        self.radar_encoder = nn.Sequential(
            nn.Linear(3 + radar_feat_dim, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=proj_drop, batch_first=True
        )
        self.output_norm = nn.LayerNorm(embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weight(self):
        for m in self.radar_encoder.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution="uniform", bias=0.0)
        xavier_init(self.cross_attn.in_proj_weight, distribution="uniform", bias=0.0)
        xavier_init(self.cross_attn.out_proj.weight, distribution="uniform", bias=0.0)

    def _unify_radar_inputs(self, radar_points, radar_features, radar_mask, bs, device, dtype):
        """统一 radar 输入格式为 [B, M, *]"""
        if isinstance(radar_points, (list, tuple)):
            max_n = max(p.shape[0] for p in radar_points)
            rp = torch.zeros(bs, max_n, 3, device=device, dtype=dtype)
            mask = torch.zeros(bs, max_n, dtype=torch.bool, device=device)
            for b_idx in range(bs):
                pt = radar_points[b_idx]
                if not torch.is_tensor(pt):
                    pt = torch.from_numpy(pt)
                n = pt.shape[0]
                rp[b_idx, :n] = pt.to(device=device, dtype=dtype)
                mask[b_idx, :n] = True
            radar_points = rp
            if radar_features is None:
                radar_features = torch.zeros(bs, max_n, self.radar_feat_dim, device=device, dtype=dtype)
            elif isinstance(radar_features, (list, tuple)):
                rf = torch.zeros(bs, max_n, radar_features[0].shape[-1], device=device, dtype=dtype)
                for b_idx in range(bs):
                    ft = radar_features[b_idx]
                    if not torch.is_tensor(ft):
                        ft = torch.from_numpy(ft)
                    rf[b_idx, :ft.shape[0]] = ft.to(device=device, dtype=dtype)
                radar_features = rf
            radar_mask = mask
        else:
            radar_points = radar_points.to(device=device, dtype=dtype)
            if radar_features is None:
                radar_features = torch.zeros(bs, radar_points.shape[1], self.radar_feat_dim, device=device, dtype=dtype)
            else:
                radar_features = radar_features.to(device=device, dtype=dtype)
            if radar_mask is None:
                radar_mask = torch.ones(bs, radar_points.shape[1], dtype=torch.bool, device=device)
            else:
                radar_mask = radar_mask.to(device=device)
        return radar_points, radar_features, radar_mask

    def forward(
        self,
        instance_feature: torch.Tensor,   # [B, N, C]
        anchor: torch.Tensor,            # [B, N, anchor_dim]
        anchor_embed: torch.Tensor,      # [B, N, C]
        radar_points: torch.Tensor,      # [B, N_radar, 3] 或 list
        radar_features: Optional[torch.Tensor] = None,
        radar_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        bs, num_anchor, _ = instance_feature.shape
        device = instance_feature.device
        dtype = instance_feature.dtype

        if kwargs.get("_radar_already_downsampled", False):
            # 已由 GaussianOccEncoder 预计算下采样，直接使用
            radar_mask = kwargs.get("radar_mask", radar_mask)
            radar_points = radar_points.to(device=device, dtype=dtype)
            if radar_features is not None:
                radar_features = radar_features.to(device=device, dtype=dtype)
            if radar_mask is not None:
                radar_mask = radar_mask.to(device=device)
        else:
            # 1. 统一输入格式
            radar_points, radar_features, radar_mask = self._unify_radar_inputs(
                radar_points, radar_features, radar_mask, bs, device, dtype)
            # 2. 体素下采样（替代随机采样）
            radar_points, radar_features, radar_mask = voxel_downsample_radar(
                radar_points, radar_features, radar_mask,
                voxel_size=self.voxel_size, max_points=self.max_radar_points)

        # 3. 编码 radar 点
        pr = torch.tensor(self.pc_range, device=device, dtype=dtype)
        xyz_norm = (radar_points - pr[:3]) / (pr[3:6] - pr[:3] + 1e-6)
        xyz_norm = xyz_norm.clamp(0.0, 1.0)
        F_dim = radar_features.shape[-1]
        if F_dim < self.radar_feat_dim:
            radar_features = F_torch.pad(radar_features, (0, self.radar_feat_dim - F_dim))
        elif F_dim > self.radar_feat_dim:
            radar_features = radar_features[..., :self.radar_feat_dim]
        radar_input = torch.cat([xyz_norm, radar_features], dim=-1)
        radar_encoded = self.radar_encoder(radar_input)  # [B, M', C]

        # 4. 分块空间近邻注意力
        query = instance_feature + anchor_embed  # [B, N, C]

        # 获取 Gaussian 中心坐标（从 anchor 的前 3 维解码）
        anchor_xyz = anchor[:, :, :3].clone()
        # anchor xyz 是归一化的 [0,1]，转回真实坐标用于距离计算
        query_xyz = anchor_xyz * (pr[3:6] - pr[:3]) + pr[:3]

        attn_out = chunked_cross_attention(
            query=query,
            key=radar_encoded,
            value=radar_encoded,
            key_padding_mask=~radar_mask,
            num_heads=self.num_heads,
            query_xyz=query_xyz,
            key_xyz=radar_points,
            topk_neighbors=self.topk_neighbors,
            chunk_size=self.attn_chunk_size,
            qkv_proj={
                'in_proj_weight': self.cross_attn.in_proj_weight,
                'in_proj_bias': self.cross_attn.in_proj_bias,
            },
            out_proj=self.cross_attn.out_proj,
        )
        out = self.output_norm(attn_out)
        return self.proj_drop(out)


@MODELS.register_module()
class RadarVelocityNet(BaseModule):
    """
    专用的雷达多普勒速度网络：从雷达点云的多普勒特征中学习每个高斯椭球的3D速度。

    设计思路：
    1. 用独立的 radar_doppler_encoder 编码雷达点（xyz_norm + Doppler/功率特征）
    2. 用 cross-attention 让每个 Gaussian 聚合附近雷达点的多普勒信息
    3. 用专用 velocity head 从聚合特征 + instance_feature + 初始速度 → 预测 3D 速度增量
    4. 残差设计：refined_vel = initial_vel + delta_vel（vel_head 末层零初始化，初始输出≈0）

    与现有 RadarFeatureAggregation 的区别：
    - RadarFeatureAggregation 是通用的雷达-高斯交叉注意力，输出混入 instance_feature
    - RadarVelocityNet 专门为速度预测设计，有独立参数，直接输出 3D 速度
    """

    def __init__(
        self,
        embed_dims: int = 256,
        radar_feat_dim: int = 10,
        num_heads: int = 4,
        pc_range: Optional[List[float]] = None,
        vel_hidden_dim: int = 64,
        proj_drop: float = 0.0,
        residual: bool = True,
        max_radar_points: int = 4096,
        voxel_size: float = 1.6,
        topk_neighbors: int = 64,
        attn_chunk_size: int = 4096,
        **kwargs,
    ):
        super(RadarVelocityNet, self).__init__()
        self.embed_dims = embed_dims
        self.radar_feat_dim = radar_feat_dim
        self.num_heads = num_heads
        self.pc_range = pc_range or [0.0, -25.6, -5.0, 51.2, 25.6, 3.0]
        self.residual = residual
        self.max_radar_points = max_radar_points
        self.voxel_size = voxel_size
        self.topk_neighbors = topk_neighbors
        self.attn_chunk_size = attn_chunk_size

        # ========== 雷达多普勒编码器 ==========
        # 输入: xyz_norm(3) + radar_feats(F) → embed_dims
        self.radar_doppler_encoder = nn.Sequential(
            nn.Linear(3 + radar_feat_dim, embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
        )

        # ========== 交叉注意力: Gaussian queries → Radar Doppler keys/values ==========
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=proj_drop, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dims)

        # ========== 速度预测头 ==========
        # 输入: instance_feature(C) + attn_output(C) + initial_vel(3) → 3
        self.vel_head = nn.Sequential(
            nn.Linear(embed_dims * 2 + 3, vel_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(vel_hidden_dim, vel_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(vel_hidden_dim, 3),
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weight(self):
        for m in self.radar_doppler_encoder.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution="uniform", bias=0.0)
        # 末层小随机初始化: 避免 vel_delta=0 死区，加速学习
        nn.init.normal_(self.vel_head[-1].weight, std=0.01)
        nn.init.zeros_(self.vel_head[-1].bias)

    def forward(
        self,
        instance_feature: torch.Tensor,    # [B, N, C]   高斯特征
        gaussian_means: torch.Tensor,       # [B, N, 3]   高斯中心位置（真实坐标）
        initial_velocity: torch.Tensor,     # [B, N, 3]   RefineModule 预测的初始速度
        radar_points: torch.Tensor,         # [B, M, 3] 或 list
        radar_features: Optional[torch.Tensor] = None,  # [B, M, F] 或 list
        radar_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """返回 refined velocity [B, N, 3]"""
        bs, num_anchor, _ = instance_feature.shape
        device = instance_feature.device
        dtype = instance_feature.dtype

        # ========== 1. 统一 radar 输入格式 + 下采样 ==========
        if kwargs.get("_radar_already_downsampled", False):
            # 已由 GaussianOccEncoder 预计算下采样，直接使用
            radar_mask = kwargs.get("radar_mask", radar_mask)
            radar_points = radar_points.to(device=device, dtype=dtype)
            if radar_features is not None:
                radar_features = radar_features.to(device=device, dtype=dtype)
            if radar_mask is not None:
                radar_mask = radar_mask.to(device=device)
            N_radar = radar_points.shape[1]
            if radar_features is None:
                radar_features = torch.zeros(bs, N_radar, self.radar_feat_dim, device=device, dtype=dtype)
        else:
            if isinstance(radar_points, (list, tuple)):
                max_n = max(p.shape[0] for p in radar_points)
                rp = torch.zeros(bs, max_n, 3, device=device, dtype=dtype)
                mask = torch.zeros(bs, max_n, dtype=torch.bool, device=device)
                for b in range(bs):
                    pt = radar_points[b]
                    if not torch.is_tensor(pt):
                        pt = torch.from_numpy(pt)
                    n = pt.shape[0]
                    rp[b, :n] = pt.to(device=device, dtype=dtype)
                    mask[b, :n] = True
                radar_points = rp
                if radar_features is not None and isinstance(radar_features, (list, tuple)):
                    rf = torch.zeros(bs, max_n, radar_features[0].shape[-1], device=device, dtype=dtype)
                    for b in range(bs):
                        ft = radar_features[b]
                        if not torch.is_tensor(ft):
                            ft = torch.from_numpy(ft)
                        rf[b, :ft.shape[0]] = ft.to(device=device, dtype=dtype)
                    radar_features = rf
                radar_mask = mask
            else:
                radar_points = radar_points.to(device=device, dtype=dtype)
                if radar_features is not None:
                    radar_features = radar_features.to(device=device, dtype=dtype)
                if radar_mask is None:
                    radar_mask = torch.ones(bs, radar_points.shape[1], dtype=torch.bool, device=device)

            N_radar = radar_points.shape[1]
            if radar_features is None:
                radar_features = torch.zeros(bs, N_radar, self.radar_feat_dim, device=device, dtype=dtype)

            # 体素下采样
            radar_points, radar_features, radar_mask = voxel_downsample_radar(
                radar_points, radar_features, radar_mask,
                voxel_size=self.voxel_size, max_points=self.max_radar_points)

        # ========== 3. 编码雷达点 ==========
        pr = torch.tensor(self.pc_range, device=device, dtype=dtype)
        xyz_norm = (radar_points - pr[:3]) / (pr[3:6] - pr[:3] + 1e-6)
        xyz_norm = xyz_norm.clamp(0.0, 1.0)
        F_dim = radar_features.shape[-1]
        if F_dim < self.radar_feat_dim:
            radar_features = F_torch.pad(radar_features, (0, self.radar_feat_dim - F_dim))
        elif F_dim > self.radar_feat_dim:
            radar_features = radar_features[..., :self.radar_feat_dim]
        radar_input = torch.cat([xyz_norm, radar_features], dim=-1)
        radar_encoded = self.radar_doppler_encoder(radar_input)

        # ========== 4. 分块空间近邻注意力 ==========
        attn_out = chunked_cross_attention(
            query=instance_feature,
            key=radar_encoded,
            value=radar_encoded,
            key_padding_mask=~radar_mask,
            num_heads=self.num_heads,
            query_xyz=gaussian_means,
            key_xyz=radar_points,
            topk_neighbors=self.topk_neighbors,
            chunk_size=self.attn_chunk_size,
            qkv_proj={
                'in_proj_weight': self.cross_attn.in_proj_weight,
                'in_proj_bias': self.cross_attn.in_proj_bias,
            },
            out_proj=self.cross_attn.out_proj,
        )
        attn_out = self.attn_norm(attn_out)
        attn_out = self.proj_drop(attn_out)

        # ========== 5. 速度预测 ==========
        vel_input = torch.cat([instance_feature, attn_out, initial_velocity], dim=-1)
        vel_delta = self.vel_head(vel_input)

        if self.residual:
            return initial_velocity + vel_delta
        else:
            return vel_delta


@MODELS.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
        use_radar_branch: bool = False,
        radar_aggregation: dict = None,
        pc_range=None,
        **kwargs,
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_deformable_func = use_deformable_func and DAF is not None
        assert self.use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, MODELS)
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.use_radar_branch = use_radar_branch and radar_aggregation is not None
        if self.use_radar_branch:
            self.radar_aggregation = build_from_cfg(radar_aggregation, MODELS)
            self.radar_fusion_proj = nn.Linear(embed_dims, embed_dims)
            # 可学习门控：初始 sigmoid(0)=0.5，让 radar 从一开始就有 50% 的贡献权重
            self.radar_gate_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.radar_aggregation = None
            self.radar_fusion_proj = None
            self.radar_gate_logit = None

        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2, 12)
            )
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_levels * self.num_pts
            )
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        if self.radar_fusion_proj is not None:
            # 小非零初始化：保证 radar_fusion_proj(radar_out)≠0，gate 能收到梯度从而有机会增大
            constant_init(self.radar_fusion_proj, val=0.01, bias=0.0)
        if self.radar_aggregation is not None and hasattr(self.radar_aggregation, "init_weight"):
            self.radar_aggregation.init_weight()

    def forward(
        self,
        instance_feature: torch.Tensor,  # [B, N, embed_dims] - 当前anchor的特征
        anchor: torch.Tensor,             # [B, N, anchor_dim] - anchor参数
        anchor_embed: torch.Tensor,       # [B, N, embed_dims] - anchor的embedding
        feature_maps: List[torch.Tensor], # List[[B, N_cam, C, H, W]] - 多层级图像特征
        metas: dict,                      # dict - 元数据（投影矩阵、相机参数等）
        radar_points=None,
        radar_features=None,
        **kwargs: dict,
    ):
        """
        DeformableFeatureAggregation的前向传播流程：
        
        功能：从多视图、多层级图像特征中采样和融合信息到3D anchor特征中
        
        这是整个系统的核心操作，连接3D Gaussian和2D图像特征
        
        输入：
            instance_feature: [B, N, embed_dims] - 当前anchor的特征
            anchor: [B, N, anchor_dim] - anchor参数（包含xyz, scale, rotation等）
            anchor_embed: [B, N, embed_dims] - anchor参数的embedding表示
            feature_maps: List[[B, N_cam, C, H, W]] - 多层级图像特征（FPN输出）
            metas: dict - 元数据（包含projection_mat等）
        
        过程：
            1. 生成key points（每个anchor周围的3D采样点）
            2. 将3D key points投影到每个相机的图像平面
            3. 从图像特征中采样（可变形注意力）
            4. 计算注意力权重（多视图、多层级、多key points）
            5. 加权融合特征
            6. 残差连接
        
        输出：
            [B, N, embed_dims] 或 [B, N, 2*embed_dims] - 更新后的特征（取决于residual_mode）
        """
        bs, num_anchor = instance_feature.shape[:2]
        
        # ========== Step 1: 生成3D采样点（key points）==========
        # 为每个anchor生成多个3D采样点（通常13个：7个固定点 + 6个可学习点）
        # key_points: [B, N, num_pts, 3] - 每个anchor的3D采样点坐标
        key_points = self.kps_generator(anchor, instance_feature)
        
        # ========== Step 2: 初始化队列（用于多帧融合，当前实现为空）==========
        # feature_queue, meta_queue等用于存储历史帧信息（【当前未使用】）
        temp_key_points_list = feature_queue = meta_queue = temp_anchor_embeds = []
        
        # ========== Step 3: 格式化特征图（如果使用deformable function）==========
        # 将特征图格式化为CUDA优化的格式
        if self.use_deformable_func:
            feature_maps = DAF.feature_maps_format(feature_maps)

        # ========== Step 4: 处理特征（当前循环只执行一次，因为队列为空）==========
        # 遍历feature_queue和当前帧，进行特征采样和融合
        for (
            temp_feature_maps,    # 当前帧或历史帧的特征图
            temp_metas,          # 当前帧或历史帧的元数据
            temp_key_points,     # 当前帧或历史帧的key points
            temp_anchor_embed,   # 当前帧或历史帧的anchor embedding
        ) in zip(
            feature_queue[::-1] + [feature_maps],      # 历史帧（倒序） + 当前帧
            meta_queue[::-1] + [metas],                # 历史帧元数据（倒序） + 当前帧元数据
            temp_key_points_list[::-1] + [key_points], # 历史帧key points（倒序） + 当前帧key points
            temp_anchor_embeds[::-1] + [anchor_embed], # 历史帧embedding（倒序） + 当前帧embedding
        ):
            # ========== Step 4.1: 计算注意力权重 ==========
            # 【重要设计特点】权重计算是content-agnostic的：
            # 
            # 1. 权重计算只基于3D anchor信息，不依赖图像特征内容：
            #    - instance_feature: anchor的3D特征
            #    - anchor_embed: anchor参数（xyz, scale, rotation等）的embedding
            #    - camera_embed: 相机参数的embedding（可选）
            #    - 不使用feature_maps（图像特征）来计算权重
            # 
            # 2. 这种设计的优点：
            #    - 权重是基于几何位置和anchor属性预测的
            #    - 不依赖图像内容，更稳定和可解释
            #    - 类似Deformable DETR的设计，是一种"位置先验"注意力
            # 
            # 3. 流程：
            #    Step 1: 基于anchor预测权重（位置和属性信息）
            #    Step 2: 使用权重从图像特征中采样（content-based sampling）
            # 
            # 这是一种"先预测去哪里采样，再去采样"的两阶段设计
            weights, weight_mask = self._get_weights(
                instance_feature, temp_anchor_embed, temp_metas
            )
            if self.use_deformable_func:
                # ========== Step 4.2: 重新排列权重维度 ==========
                # 从 [B, N, num_cams, num_levels, num_pts, num_groups]
                # 到 [B, N, num_pts, num_cams, num_levels, num_groups]
                # 这样便于后续处理
                weights = weights.permute(0, 1, 4, 2, 3, 5).contiguous()
                weights = weights.reshape(
                    bs, num_anchor, self.num_pts, self.num_cams, 
                    self.num_levels, self.num_groups
                )
                weight_mask = weight_mask.permute(0, 1, 4, 2, 3, 5).contiguous()
                weight_mask = weight_mask.reshape(
                    bs, num_anchor, self.num_pts, self.num_cams, 
                    self.num_levels, self.num_groups
                )
                
                # ========== Step 4.3: 将3D key points投影到2D图像平面 ==========
                # 使用投影矩阵将3D点投影到每个相机的图像平面
                # points_2d: [B, N, num_pts, num_cams, 2] - 2D图像坐标（归一化到[0, 1]）
                # mask: [B, num_cams, N, num_pts] - 掩码（True表示点在图像范围内）
                points_2d, mask = self.project_points(
                    temp_key_points,                      # [B, N, num_pts, 3] - 3D点
                    temp_metas["projection_mat"],         # [B, num_cams, 3, 4] - 投影矩阵
                    temp_metas.get("image_wh"),           # [B, num_cams, 2] - 图像宽高（可选）
                )
                
                # ========== Step 4.4: 重新排列points_2d和mask的维度 ==========
                # points_2d: [B, N, num_pts, num_cams, 2] → [B, N*num_pts, num_cams, 2]
                # mask: [B, N, num_pts, num_cams] → [B, N, num_pts, num_cams]
                points_2d = points_2d.permute(0, 2, 3, 1, 4).reshape(
                    bs, num_anchor * self.num_pts, self.num_cams, 2
                )
                mask = mask.permute(0, 2, 3, 1)  # [B, N, num_pts, num_cams]
                
                # ========== Step 4.5: 合并mask ==========
                # 将投影mask和权重mask合并
                # mask表示点是否在图像范围内
                # weight_mask表示权重是否有效
                mask = mask[..., None, None] & weight_mask  # [B, N, num_pts, num_cams, num_levels, num_groups]
                
                # ========== Step 4.6: 处理完全miss的情况 ==========
                # all_miss: 如果某个key point在所有相机、所有层级都没有有效采样点
                # 则将其权重设为0，避免softmax后的无效权重
                # 
                # all_miss的计算：
                #   - 对每个key_point，在所有camera和level上求和mask
                #   - 如果sum=0，说明这个key_point在所有camera/level都无效
                #   - all_miss维度：[B, N, 1, 1, 1, 1]（每个key_point一个标志）
                all_miss = mask.sum(dim=[2, 3, 4], keepdim=True) == 0  # [B, N, 1, 1, 1, 1]
                all_miss = all_miss.expand(-1, -1, self.num_pts, self.num_cams, 
                                          self.num_levels, -1)  # [B, N, num_pts, num_cams, num_levels, num_groups]
                
                # ========== Step 4.7: Softmax归一化权重 ==========
                # 
                # 【关键区别】为什么~mask和all_miss的处理方式不同？
                # 
                # 1. ~mask（部分无效情况）：
                #    - 表示：某个特定的(key_point, camera, level)组合无效
                #    - 例如：key_point_1在camera_2的level_3上无效，但在其他camera/level可能有效
                #    - 处理：设为-inf
                #    - 原因：
                #      * softmax会在同一key_point的所有有效camera/level组合之间进行归一化
                #      * 设为-inf后，softmax(inf, val1, val2, ...) → (0, prob1, prob2, ...)
                #      * 这样可以排除无效位置，同时保留有效位置的相对重要性
                # 
                # 2. all_miss（完全无效情况）：
                #    - 表示：某个key_point在所有camera和所有level都没有有效采样点
                #    - 例如：key_point_2在所有camera的所有level都投影到图像外
                #    - 处理：设为0（而不是-inf）
                #    - 原因：
                #      * 如果所有位置都是-inf，softmax会导致数值问题：
                #        softmax([-inf, -inf, -inf, ...]) → [NaN, NaN, NaN, ...]
                #        （因为exp(-inf) = 0，sum = 0，除以0 = NaN）
                #      * 直接设为0更安全，后续在第340行会再次确保这些位置的权重为0
                # 
                # 执行顺序很重要：
                #   1. 先设置~mask为-inf（处理部分无效）
                #   2. 再设置all_miss为0（覆盖完全无效的情况，避免NaN）
                #   3. 然后进行softmax（在有效位置之间归一化）
                #   4. 最后再次确保all_miss位置为0（第340行）
                weights[~mask] = -torch.inf  # 部分无效：设为-inf，softmax后为0
                weights[all_miss] = 0.       # 完全无效：设为0，避免softmax时的NaN问题
                
                # 在num_cams、num_levels、num_pts维度上进行softmax归一化
                weights = weights.flatten(2, 4)  # [B, N, num_pts*num_cams*num_levels, num_groups]
                weights = weights.softmax(dim=-2)  # Softmax归一化
                weights = weights.reshape(
                    bs, num_anchor * self.num_pts, self.num_cams, 
                    self.num_levels, self.num_groups
                )
                
                # 将all_miss位置的权重设为0
                weights = weights * (1 - all_miss.flatten(1, 2).float())

                # ========== Step 4.8: 使用CUDA优化的可变形采样 ==========
                # 【两阶段设计】先预测权重，再采样特征
                # 
                # DAF.apply: 使用可变形注意力从图像特征中采样
                #   输入：
                #     - feature_maps: 多层级图像特征（content，之前未用于计算权重）
                #     - points_2d: 2D采样位置（基于3D key points投影得到）
                #     - weights: 注意力权重（基于3D anchor信息预测，不依赖图像内容）
                #   输出：采样后的特征 [B*N*num_pts, embed_dims]
                # 
                # 过程：
                #   1. 对每个key point，在每个相机的每个层级特征图上采样
                #   2. 使用bilinear interpolation进行可变形采样
                #   3. 根据之前预测的注意力权重加权融合（权重已经softmax归一化）
                # 
                # 关键点：
                #   - 权重是基于3D anchor几何信息预测的（位置、属性、相机参数）
                #   - 图像特征只用于采样，不用于权重计算
                #   - 这是一种"geometry-driven"而非"content-driven"的注意力机制
                temp_features_next = DAF.apply(
                    *temp_feature_maps,  # 多层级特征图（图像内容）
                    points_2d,           # [B*N*num_pts, num_cams, 2] - 2D采样位置（几何信息）
                    weights              # [B*N*num_pts, num_cams, num_levels, num_groups] - 注意力权重（几何驱动的预测）
                ).reshape(bs, num_anchor, self.num_pts, self.embed_dims)  # [B, N, num_pts, embed_dims]
            else:
                # ========== 备用路径：不使用CUDA优化的采样（当前未使用）==========
                # 使用PyTorch的grid_sample进行特征采样
                temp_features_next = self.feature_sampling(
                    temp_feature_maps,
                    temp_key_points,
                    temp_metas["projection_mat"],
                    temp_metas.get("image_wh"),
                )
                # 多视图、多层级特征融合
                temp_features_next = self.multi_view_level_fusion(
                    temp_features_next, weights
                )

            features = temp_features_next  # [B, N, num_pts, embed_dims]

        # ========== Step 5: 融合多个key points的特征 ==========
        # 对每个anchor的所有key points的特征求和，得到最终的anchor特征
        # [B, N, num_pts, embed_dims] → [B, N, embed_dims]
        features = features.sum(dim=2)  # [B, N, embed_dims]
        
        # ========== Step 5.5: Radar 分支（radar/image/gaussian 三模态交叉注意力）==========
        if self.use_radar_branch and radar_points is not None:
            # 过滤 kwargs 中已显式传递的参数，避免 multiple values 错误
            radar_extra_kwargs = {k: v for k, v in kwargs.items()
                                  if k not in ("radar_points", "radar_features", "radar_mask")}
            radar_out = self.radar_aggregation(
                instance_feature, anchor, anchor_embed,
                radar_points=radar_points,
                radar_features=radar_features,
                radar_mask=kwargs.get("radar_mask"),
                **radar_extra_kwargs,
            )  # [B, N, embed_dims]
            # 可学习门控：初始 sigmoid(0)=0.5，radar 有足够贡献权重
            gate = torch.sigmoid(self.radar_gate_logit.to(features.device))
            output = self.output_proj(features) + gate * self.radar_fusion_proj(radar_out)
        else:
            output = self.output_proj(features)
        
        # ========== Step 6: Dropout ==========
        output = self.proj_drop(output)      # [B, N, embed_dims]
        
        # ========== Step 7: 残差连接 ==========
        # 将采样融合后的特征与原始instance_feature相加或拼接
        if self.residual_mode == "add":
            # 相加模式：output = output + instance_feature
            # 输出维度：[B, N, embed_dims]
            output = output + instance_feature
        elif self.residual_mode == "cat":
            # 拼接模式：output = concat([output, instance_feature])
            # 输出维度：[B, N, 2*embed_dims]
            output = torch.cat([output, instance_feature], dim=-1)
        
        return output  # [B, N, embed_dims] 或 [B, N, 2*embed_dims]

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        """
        计算注意力权重
        
        【关键特点】权重计算是content-agnostic的，只基于3D anchor信息，不依赖图像特征内容
        
        功能：根据instance_feature、anchor_embed和相机信息，预测每个采样点的注意力权重
        
        输入：
            instance_feature: [B, N, embed_dims] - anchor的3D特征（来自前面的decoder层）
            anchor_embed: [B, N, embed_dims] - anchor参数（xyz, scale, rotation等）的embedding
            metas: dict - 元数据（包含projection_mat）
            【注意】不输入feature_maps（图像特征），权重是位置和属性驱动的
        
        过程：
            1. 融合instance_feature和anchor_embed（3D anchor信息）
            2. 如果使用camera_embed，融合相机参数信息
            3. MLP预测权重logits（基于3D信息预测采样权重）
            4. 添加attention dropout（训练时）
        
        输出：
            weights: [B, N, num_cams, num_levels, num_pts, num_groups] - 注意力权重logits
            weight_mask: [B, N, num_cams, num_levels, num_pts, num_groups] - 权重掩码
        
        设计思想：
            - 这是一种"位置先验"注意力机制（类似Deformable DETR）
            - 权重预测不依赖图像内容，而是基于：
              * 3D anchor的位置和属性（xyz, scale, rotation）
              * 相机参数（不同相机的视角不同）
              * anchor的3D特征（instance_feature，来自decoder的积累）
            - 然后使用这些预测的权重去采样图像特征
            - 相比content-based attention（如self-attention），这种方法：
              * 更稳定（不依赖图像内容的变化）
              * 更高效（不需要计算query-key相似度）
              * 更适合3D场景（几何先验更重要）
        """
        bs, num_anchor = instance_feature.shape[:2]
        
        # ========== Step 1: 融合anchor特征和embedding ==========
        # 将instance_feature和anchor_embed相加，作为query
        feature = instance_feature + anchor_embed  # [B, N, embed_dims]
        
        # ========== Step 2: 融合相机信息（可选）==========
        # 如果使用camera_embed，将相机参数编码并融合到特征中
        # 这样可以让模型区分不同的相机视角
        if self.camera_encoder is not None:
            # 2.1 提取投影矩阵的前3列（3x3部分，包含旋转和缩放信息）
            proj_mat = metas["projection_mat"][:, :, :3]  # [B, num_cams, 3, 3]
            proj_mat = proj_mat.reshape(bs, self.num_cams, -1)  # [B, num_cams, 9]
            
            # 2.2 使用MLP编码相机参数
            camera_embed = self.camera_encoder(proj_mat)  # [B, num_cams, embed_dims]
            
            # 2.3 将相机embedding加到特征上
            # [B, N, embed_dims] + [B, 1, num_cams, embed_dims] → [B, N, num_cams, embed_dims]
            feature = feature[:, :, None] + camera_embed[:, None]  # [B, N, num_cams, embed_dims]
        
        # ========== Step 3: MLP预测注意力权重logits ==========
        # 使用MLP预测每个采样点、每个相机、每个层级、每个group的注意力权重
        weights = self.weights_fc(feature)  # [B, N, num_cams, num_groups*num_levels*num_pts] 或 [B, N, num_groups*num_levels*num_pts]
        
        # ========== Step 4: 重新排列权重维度 ==========
        weights = weights.reshape(bs, num_anchor, -1, self.num_groups)
        weights = weights.reshape(
            bs, num_anchor, self.num_cams, self.num_levels, 
            self.num_pts, self.num_groups
        )  # [B, N, num_cams, num_levels, num_pts, num_groups]
        
        # ========== Step 5: Attention Dropout（训练时）==========
        # 在训练时，随机dropout一些注意力权重，提高模型泛化能力
        if self.training and self.attn_drop > 0:
            # 生成随机mask，以attn_drop的概率dropout
            mask = torch.rand_like(weights) > self.attn_drop  # [B, N, num_cams, num_levels, num_pts, num_groups]
        else:
            # 测试时或attn_drop=0时，所有位置都有效
            mask = torch.ones_like(weights) > 0  # [B, N, num_cams, num_levels, num_pts, num_groups]
        
        return weights, mask

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        depth = points_2d[..., 2]
        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        mask = (depth > 1e-5) & (points_2d[..., 0] > 0) & (points_2d[..., 0] < 1) & \
                                (points_2d[..., 1] > 0) & (points_2d[..., 1] < 1)
        return points_2d, mask

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],  # List[[B, N_cam, C, H, W]] - 多层级特征图
        key_points: torch.Tensor,          # [B, N, num_pts, 3] - 3D采样点
        projection_mat: torch.Tensor,      # [B, num_cams, 3, 4] - 投影矩阵
        image_wh: Optional[torch.Tensor] = None,  # [B, num_cams, 2] - 图像宽高（可选）
    ) -> torch.Tensor:
        """
        特征采样（备用实现，当前未使用）
        
        功能：使用PyTorch的grid_sample从图像特征中采样
        
        输入：
            feature_maps: List[[B, N_cam, C, H, W]] - 多层级特征图
            key_points: [B, N, num_pts, 3] - 3D采样点
            projection_mat: [B, num_cams, 3, 4] - 投影矩阵
            image_wh: [B, num_cams, 2] - 可选，图像宽高
        
        过程：
            1. 将3D点投影到2D图像平面
            2. 将坐标转换到grid_sample所需的[-1, 1]范围
            3. 使用grid_sample进行双线性插值采样
            4. 重新排列维度
        
        输出：
            [B, N, num_cams, num_levels, num_pts, embed_dims] - 采样后的特征
        """
        num_levels = len(feature_maps)  # 特征层级数（例如：4）
        num_cams = feature_maps[0].shape[1]  # 相机数量（例如：6）
        bs, num_anchor, num_pts = key_points.shape[:3]

        # ========== Step 1: 将3D点投影到2D图像平面 ==========
        # points_2d: [B, N, num_pts, num_cams, 2] - 归一化到[0, 1]的2D坐标
        points_2d, _ = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        
        # ========== Step 2: 转换坐标范围 ==========
        # grid_sample需要的坐标范围是[-1, 1]
        # 所以需要将[0, 1]转换到[-1, 1]: x_new = x * 2 - 1
        points_2d = points_2d * 2 - 1  # [B, N, num_pts, num_cams, 2] - 范围[-1, 1]
        
        # ========== Step 3: 展平batch和anchor维度 ==========
        # grid_sample需要输入格式：[B*N_cam, C, H, W] 和 [B*N*num_pts*num_cams, 1, 2]
        points_2d = points_2d.flatten(end_dim=1)  # [B*N*num_pts*num_cams, 2]

        # ========== Step 4: 对每个层级进行采样 ==========
        features = []
        for fm in feature_maps:
            # fm: [B, N_cam, C, H, W]
            # 展平batch和相机维度：[B*N_cam, C, H, W]
            fm_flat = fm.flatten(end_dim=1)
            
            # 使用grid_sample进行双线性插值采样
            # 输入：特征图 [B*N_cam, C, H, W]，采样坐标 [B*N*num_pts*num_cams, 1, 2]
            # 输出：[B*N*num_pts*num_cams, C, 1, 1]
            sampled = torch.nn.functional.grid_sample(
                fm_flat,                    # [B*N_cam, C, H, W]
                points_2d.unsqueeze(1),     # [B*N*num_pts*num_cams, 1, 2]
                mode='bilinear',            # 双线性插值
                padding_mode='zeros',       # 超出边界时填充0
                align_corners=False
            ).squeeze(-1).squeeze(-1)  # [B*N*num_pts*num_cams, C]
            
            features.append(sampled)
        
        # ========== Step 5: 重新排列维度 ==========
        # Stack: [num_levels, B*N*num_pts*num_cams, C]
        features = torch.stack(features, dim=1)  # [B*N*num_pts*num_cams, num_levels, C]
        
        # Reshape: 恢复batch、anchor、pts、cam维度
        features = features.reshape(
            bs, num_anchor, num_pts, num_cams, num_levels, -1
        )  # [B, N, num_pts, num_cams, num_levels, C]
        
        # Permute: 重新排列为最终格式
        features = features.permute(0, 1, 3, 4, 2, 5)  # [B, N, num_cams, num_levels, num_pts, C]

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,  # [B, N, num_cams, num_levels, num_pts, embed_dims] - 采样后的特征
        weights: torch.Tensor,   # [B, N, num_cams, num_levels, num_pts, num_groups] - 注意力权重
    ):
        """
        多视图、多层级特征融合
        
        功能：根据注意力权重，融合多个相机、多个层级的特征
        
        输入：
            features: [B, N, num_cams, num_levels, num_pts, embed_dims] - 采样后的特征
            weights: [B, N, num_cams, num_levels, num_pts, num_groups] - 注意力权重
        
        过程：
            1. 将特征按group分组
            2. 使用权重加权融合（在num_cams和num_levels维度）
            3. 重新reshape为最终格式
        
        输出：
            [B, N, num_pts, embed_dims] - 融合后的特征
        """
        bs, num_anchor = weights.shape[:2]
        
        # ========== Step 1: 将特征按group分组 ==========
        # 将embed_dims维度分解为num_groups * group_dims
        # 例如：embed_dims=256, num_groups=8, group_dims=32
        # features: [B, N, num_cams, num_levels, num_pts, 256]
        # → [B, N, num_cams, num_levels, num_pts, 8, 32]
        features = features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )  # [B, N, num_cams, num_levels, num_pts, num_groups, group_dims]
        
        # ========== Step 2: 应用注意力权重 ==========
        # 使用权重对每个group的特征进行加权
        # weights: [B, N, num_cams, num_levels, num_pts, num_groups, 1]
        # features: [B, N, num_cams, num_levels, num_pts, num_groups, group_dims]
        features = weights[..., None] * features  # [B, N, num_cams, num_levels, num_pts, num_groups, group_dims]
        
        # ========== Step 3: 融合多视图和多层级特征 ==========
        # 在num_cams和num_levels维度上求和，实现加权融合
        # sum(dim=2): 融合多个相机的特征
        # sum(dim=2): 融合多个层级的特征（注意：第一个dim=2是num_cams，sum后dim=2变成num_levels）
        features = features.sum(dim=2)  # [B, N, num_levels, num_pts, num_groups, group_dims] - 融合相机
        features = features.sum(dim=2)  # [B, N, num_pts, num_groups, group_dims] - 融合层级
        
        # ========== Step 4: 重新reshape为最终格式 ==========
        # 将group_dims合并回embed_dims
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )  # [B, N, num_pts, embed_dims]
        
        return features
