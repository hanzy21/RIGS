from mmengine.registry import MODELS
from mmengine.model import BaseModule
import spconv.pytorch as spconv
import torch.nn as nn, torch
from functools import partial
from .utils import spherical2cartesian, cartesian


class GroupNormForSparseConv(nn.Module):
    """
    GroupNorm wrapper for spconv features [B*N, C]
    Replaces LayerNorm with GroupNorm for better batch-size independence
    """
    def __init__(self, num_channels, num_groups=None, eps=1e-5, affine=True):
        """
        Args:
            num_channels: number of channels (embed_channels)
            num_groups: number of groups for GroupNorm. If None, use 32 groups
            eps: epsilon for numerical stability
            affine: whether to use learnable affine parameters
        """
        super().__init__()
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
            x: [B*N, C] tensor (spconv features format)
        Returns:
            x: [B*N, C] tensor after GroupNorm
        """
        # Reshape to [B*N, C, 1] for GroupNorm (needs 3D input)
        x = x.unsqueeze(-1)  # [B*N, C, 1]
        x = self.gn(x)
        # Reshape back to [B*N, C]
        x = x.squeeze(-1)  # [B*N, C]
        return x


@MODELS.register_module()
class SparseConv3D(BaseModule):
    def __init__(
        self, 
        in_channels,
        embed_channels,
        pc_range,
        grid_size,
        xyz_activation="sigmoid",
        use_out_proj=False,
        kernel_size=5,
        use_multi_layer=False,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        
        if use_multi_layer:
            self.layer = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
                GroupNormForSparseConv(embed_channels),
                nn.ReLU(True),
                spconv.SubMConv3d(embed_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
                GroupNormForSparseConv(embed_channels),
                nn.ReLU(True),
                spconv.SubMConv3d(embed_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
                GroupNormForSparseConv(embed_channels),
                nn.ReLU(True),
            )        
        else:
            self.layer = spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False)
        if use_out_proj:
            self.output_proj = nn.Linear(embed_channels, embed_channels)
        else:
            self.output_proj = nn.Identity()
        self.get_xyz = partial(cartesian, pc_range=pc_range, use_sigmoid=(xyz_activation=="sigmoid"))
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float))

    def forward(self, instance_feature, anchor):
        """
        SparseConv3D的前向传播流程：
        
        功能：在3D空间中聚合邻域anchor的特征信息，利用空间局部性增强特征表示
        
        输入：
            instance_feature: [B, N, C] - 每个anchor的特征
            anchor: [B, N, anchor_dim] - anchor参数（包含xyz坐标）
        
        过程：
            1. 提取anchor的xyz坐标并转换为3D真实空间坐标
            2. 将3D坐标转换为3D网格索引
            3. 构建稀疏卷积张量（SparseConvTensor）
            4. 执行稀疏3D卷积（SubMConv3d）
            5. 输出投影（可选）
        
        输出：
            [B, N, embed_channels] - 聚合邻域信息后的特征
        """
        # ========== Step 1: 获取输入维度 ==========
        # bs: batch size
        # g: num_anchor (每个样本的anchor数量)
        # c: in_channels (输入特征维度)
        bs, g, _ = instance_feature.shape  # instance_feature: [B, N, C]

        # ========== Step 2: 提取并转换xyz坐标 ==========
        # 2.1 从anchor中提取xyz参数（归一化后的参数空间）
        anchor_xyz = anchor[..., :3]  # [B, N, 3] - 归一化参数
        
        # 2.2 将归一化参数转换为3D真实空间坐标（单位：米）
        # get_xyz = partial(cartesian, ...)
        #   输入：[B, N, 3] 归一化参数
        #   输出：[B, N, 3] 3D真实坐标（在pc_range范围内）
        anchor_xyz = self.get_xyz(anchor_xyz)  # [B, N, 3] - 3D真实坐标

        # 2.3 展平batch和anchor维度，便于后续处理
        anchor_xyz = anchor_xyz.flatten(0, 1)  # [B*N, 3] - 展平后的3D坐标

        # ========== Step 3: 计算3D网格索引 ==========
        # 3.1 将3D坐标转换为相对于pc_range的偏移
        # pc_range格式: [x_min, y_min, z_min, x_max, y_max, z_max]
        # 例如：[0.0, -25.6, -5.0, 51.2, 25.6, 3.0]
        indices = anchor_xyz - self.pc_range[None, :3]  # [B*N, 3] - 相对于最小值的偏移
        
        # 3.2 将偏移转换为网格索引（除以grid_size）
        # grid_size: 每个网格的大小（例如：[0.2, 0.2, 0.2]米）
        # indices: [B*N, 3] - 网格索引（浮点数）
        indices = indices / self.grid_size[None, :]  # [B*N, 3] - 网格索引（浮点数）
        indices = indices.to(torch.int32)  # [B*N, 3] - 转换为整数索引
        
        # 3.3 添加batch索引，构建完整的稀疏张量索引
        # batched_indices格式: [batch_idx, x_idx, y_idx, z_idx]
        # 例如：[[0, 10, 20, 5], [0, 11, 20, 5], [1, 10, 20, 5], ...]
        batch_indices = torch.arange(
            bs, device=indices.device, dtype=torch.int32
        ).reshape(bs, 1, 1).expand(-1, g, -1).flatten(0, 1)  # [B*N, 1]
        batched_indices = torch.cat([batch_indices, indices], dim=-1)  # [B*N, 4]
        
        # ========== Step 4: 计算3D网格的空间形状 ==========
        # spatial_shape: 3D网格在x, y, z三个方向的网格数量
        # 例如：如果pc_range=[0, -25.6, -5, 51.2, 25.6, 3]，grid_size=[0.2, 0.2, 0.2]
        #   x方向: (51.2 - 0) / 0.2 = 256
        #   y方向: (25.6 - (-25.6)) / 0.2 = 256
        #   z方向: (3 - (-5)) / 0.2 = 40
        # spatial_shape = [256, 256, 40]
        spatial_shape = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        spatial_shape = spatial_shape.to(torch.int32)  # [3] - 网格形状

        # ========== Step 5: 构建稀疏卷积张量 ==========
        # SparseConvTensor是spconv库的核心数据结构，用于表示稀疏的3D张量
        # 
        # 输入参数：
        #   features: [B*N, C] - 每个anchor的特征（展平后）
        #   indices: [B*N, 4] - 每个anchor在3D网格中的位置 [batch, x, y, z]
        #   spatial_shape: [3] - 3D网格的形状 [X, Y, Z]
        #   batch_size: int - batch大小
        #
        # 注意：只有indices指定的位置才有特征，其他位置为0（稀疏表示）
        input = spconv.SparseConvTensor(
            instance_feature.flatten(0, 1),  # [B*N, C] - 特征
            indices=batched_indices,          # [B*N, 4] - 索引
            spatial_shape=spatial_shape,      # [3] - 空间形状
            batch_size=bs                     # batch大小
        )

        # ========== Step 6: 执行稀疏3D卷积 ==========
        # 使用SubMConv3d（Submanifold Convolution）进行3D卷积
        # 
        # SubMConv3d特点：
        #   - 只在有特征的位置进行卷积（保持稀疏性）
        #   - 使用3x3x3或5x5x5的卷积核
        #   - 在3D空间中聚合邻域anchor的特征信息
        #
        # 过程：
        #   1. 对每个anchor，在3D网格中查找其邻居（根据kernel_size）
        #   2. 使用卷积核权重对邻居特征进行加权求和
        #   3. 输出新的特征表示
        #
        # 例如：kernel_size=5，每个anchor会聚合其周围5x5x5网格内的特征
        output = self.layer(input)  # SparseConvTensor with features [B*N, embed_channels]
        
        # ========== Step 7: 恢复batch和anchor维度 ==========
        # 将展平的特征重新组织为 [B, N, embed_channels]
        output = output.features.unflatten(0, (bs, g))  # [B, N, embed_channels]

        # ========== Step 8: 输出投影（可选）==========
        # 如果use_out_proj=True，使用Linear层进行投影
        # 如果use_out_proj=False，使用Identity()，不做任何变换
        return self.output_proj(output)  # [B, N, embed_channels]

        # 使用SubMConv3d（Submanifold Convolution）进行3D卷积
        # 
        # SubMConv3d特点：
        #   - 只在有特征的位置进行卷积（保持稀疏性）
        #   - 使用3x3x3或5x5x5的卷积核
        #   - 在3D空间中聚合邻域anchor的特征信息
        #
        # 过程：
        #   1. 对每个anchor，在3D网格中查找其邻居（根据kernel_size）
        #   2. 使用卷积核权重对邻居特征进行加权求和
        #   3. 输出新的特征表示
        #
        # 例如：kernel_size=5，每个anchor会聚合其周围5x5x5网格内的特征
        output = self.layer(input)  # SparseConvTensor with features [B*N, embed_channels]
        
        # ========== Step 7: 恢复batch和anchor维度 ==========
        # 将展平的特征重新组织为 [B, N, embed_channels]
        output = output.features.unflatten(0, (bs, g))  # [B, N, embed_channels]

        # ========== Step 8: 输出投影（可选）==========
        # 如果use_out_proj=True，使用Linear层进行投影
        # 如果use_out_proj=False，使用Identity()，不做任何变换
        return self.output_proj(output)  # [B, N, embed_channels]
