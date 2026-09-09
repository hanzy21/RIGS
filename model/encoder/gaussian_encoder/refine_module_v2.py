from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
from functools import partial
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction, cartesian, reverse_cartesian
from ...utils.safe_ops import safe_sigmoid


@MODELS.register_module()
class SparseGaussian3DRefinementModuleV2(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        pc_range=None,
        scale_range=None,
        unit_xyz=None,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        semantics_activation='softmax',
        xyz_activation="sigmoid",
        scale_activation="sigmoid",
        enable_velocity=True,
        vel_scale=30.0,
        **kwargs,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.enable_velocity = enable_velocity

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        
        vel_dim = 3 if enable_velocity else 0
        self.output_dim = 10 + int(include_opa) + semantic_dim + vel_dim
        self.semantic_start = 10 + int(include_opa)
        self.velocity_start = self.semantic_start + semantic_dim
        self.semantic_dim = semantic_dim
        self.include_opa = include_opa
        self.semantics_activation = semantics_activation
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation

        self.pc_range = pc_range
        # scale_range: [float, float] - Gaussian scale参数的最小值和最大值（真实单位：米）
        # 例如：[0.01, 1.8] 表示 Gaussian 的 scale 取值范围为 [0.01, 1.8] 米
        # 
        # 物理含义：
        # - scale 表示 Gaussian 在三个轴上的标准差（或半高宽）
        # - 决定了 Gaussian 在3D空间中的"大小"和"范围"
        # - 较小的 scale：Gaussian 更紧凑，适合表示小物体或细节
        # - 较大的 scale：Gaussian 更扩散，适合表示大物体或背景区域
        # 
        # 设置依据：
        # 1. 物体大小：根据场景中物体的典型大小
        #    - 小物体（行人、自行车）：0.1-0.5米
        #    - 中等物体（汽车）：0.5-2.0米
        #    - 大物体（卡车、建筑物）：1.0-3.0米
        # 2. Anchor数量：anchor越多，每个anchor的scale可以更小
        # 3. 场景分辨率：更高的分辨率可以使用更小的scale
        # 4. 经验值：通过实验调优得到的最优范围
        # 
        # 在代码中的使用：
        # - MLP输出 -> sigmoid -> [0, 1] -> 线性映射到 [scale_range[0], scale_range[1]]
        # - scale = scale_range[0] + (scale_range[1] - scale_range[0]) * sigmoid(output)
        self.scale_range = scale_range
        # unit_xyz: [float, float, float] - 每个decoder层中xyz坐标的最大更新步长（真实单位：米）
        # 例如：[4.0, 4.0, 1.0] 表示：
        #   - X方向：每次refine最多偏移±4.0米
        #   - Y方向：每次refine最多偏移±4.0米
        #   - Z方向：每次refine最多偏移±1.0米
        # 
        # 设置依据：
        # 1. 根据pc_range的空间范围：通常设置为空间范围的5-10%
        # 2. 根据场景特点：
        #    - 水平方向（X, Y）：通常较大（如2-4米），因为物体在水平方向分布较广
        #    - 垂直方向（Z）：通常较小（如0.5-1米），因为物体高度变化相对较小
        # 作用：
        # - 限制每次refine的更新幅度，使训练更稳定
        # - 防止anchor位置在训练初期跳变过大
        # - 通过多层decoder逐步细化，最终达到精确位置
        assert unit_xyz is not None, "unit_xyz must be specified (e.g., [4.0, 4.0, 1.0])"
        self.register_buffer("unit_xyz", torch.tensor(unit_xyz, dtype=torch.float), False)
        self.get_xyz = partial(
            cartesian, pc_range=pc_range, use_sigmoid=(xyz_activation=="sigmoid"))
        self.reverse_xyz = partial(
            reverse_cartesian, pc_range=pc_range, use_sigmoid=(xyz_activation=="sigmoid"))
        
        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim))

        # I1: 独立 vel_mlp，梯度隔离避免速度 loss 干扰 OCC。共享 MLP 结构不变，checkpoint 兼容。
        if enable_velocity:
            self.vel_scale = vel_scale
            self.vel_mlp = nn.Sequential(
                nn.Linear(embed_dims, embed_dims // 4),
                nn.GELU(),
                nn.Linear(embed_dims // 4, 3),
            )

    def forward(
        self,
        instance_feature: torch.Tensor,  # [B, N, embed_dims] - 当前decoder层的特征
        anchor: torch.Tensor,             # [B, N, anchor_dim] - 当前anchor参数（归一化后的参数空间）
        anchor_embed: torch.Tensor,       # [B, N, embed_dims] - anchor参数的embedding表示
    ):
        """
        Refine Module V2 的完整流程：
        
        1. 特征融合：将instance_feature和anchor_embed相加，输入MLP预测参数delta
        2. XYZ处理：在3D真实空间中计算delta，然后转换回参数空间
        3. 其他参数处理：直接使用预测值（scale, rotation, opacity, semantic）
        4. 参数变换：将归一化参数转换为真实空间的值（用于Gaussian渲染）
        5. 构建GaussianPrediction对象
        """
        
        # ========== Step 1: MLP预测参数delta ==========
        # 融合特征：instance_feature + anchor_embed
        # 输入MLP：Linear → ReLU → GroupNorm → Linear → ReLU → GroupNorm → Linear → Scale
        # 输出：output [B, N, output_dim]
        #   - output_dim = 10 + include_opa + semantic_dim
        #   - 包括：xyz_delta(3) + scale(3) + rotation(4) + opacity(1) + semantic(semantic_dim)
        output = self.layers(instance_feature + anchor_embed)

        # ========== Step 2: XYZ坐标的处理（在3D真实空间中计算）==========
        # 2.1 计算delta_xyz（在3D真实空间中，单位：米）
        # output[..., :3]经过sigmoid后映射到[0, 1]，然后映射到[-1, 1]
        # 再乘以unit_xyz（每个方向的更新步长，例如[0.5, 0.5, 0.3]米）
        # delta_xyz: [B, N, 3] - 3D空间中的偏移量（真实单位）
        delta_xyz = (2 * safe_sigmoid(output[..., :3]) - 1.) * self.unit_xyz[None, None]
        
        # 2.2 将当前anchor的xyz从参数空间转换到3D真实空间
        # anchor[..., :3]是归一化的参数（经过sigmoid逆变换的值）
        # original_xyz: [B, N, 3] - 3D空间中的原始位置（真实单位：米）
        original_xyz = self.get_xyz(anchor[..., :3])
        
        # 2.3 在3D空间中更新位置：original_xyz + delta_xyz
        # anchor_xyz: [B, N, 3] - 更新后的3D位置（真实单位：米）
        anchor_xyz = original_xyz + delta_xyz
        
        # 2.4 将更新后的3D位置转换回参数空间（归一化后的值）
        # anchor_xyz: [B, N, 3] - 更新后的归一化参数（用于存储和下一层使用）
        anchor_xyz = self.reverse_xyz(anchor_xyz)

        # ========== Step 3: Scale参数处理 ==========
        # 直接使用MLP的输出，后续会经过sigmoid和range映射
        # anchor_scale: [B, N, 3] - 归一化后的scale参数
        anchor_scale = output[..., 3:6]

        # ========== Step 4: Rotation参数处理 ==========
        # 直接使用MLP的输出，进行L2归一化以保持四元数的单位约束
        # anchor_rotation: [B, N, 4] - 归一化后的四元数 [w, x, y, z]
        anchor_rotation = output[..., 6:10]
        anchor_rotation = torch.nn.functional.normalize(anchor_rotation, 2, -1)

        # ========== Step 5: Opacity参数处理 ==========
        # 直接使用MLP的输出，后续会经过sigmoid
        # anchor_opa: [B, N, 1] - 未归一化的opacity参数
        anchor_opa = output[..., 10:(10 + int(self.include_opa))]

        # ========== Step 6: Semantic参数处理 ==========
        # 直接使用MLP的输出，后续会经过softmax或softplus
        # anchor_sem: [B, N, semantic_dim] - 未归一化的semantic logits
        anchor_sem = output[..., self.semantic_start:self.velocity_start]

        # ========== Step 7: Velocity参数处理（受 enable_velocity 控制）==========
        if self.enable_velocity:
            # I1: 使用独立 vel_mlp，梯度隔离避免速度 loss 干扰 OCC
            # DETACH: 切断速度梯度向 backbone 的回传
            vel_input = (instance_feature + anchor_embed).detach()
            anchor_vel = self.vel_scale * self.vel_mlp(vel_input)
        else:
            anchor_vel = None

        # ========== Step 8: 拼接所有参数 ==========
        # 将所有处理后的参数拼接，形成新的anchor参数
        # output: [B, N, anchor_dim] - 更新后的完整anchor参数（参数空间）
        parts = [anchor_xyz, anchor_scale, anchor_rotation, anchor_opa, anchor_sem]
        if anchor_vel is not None:
            parts.append(anchor_vel)
        output = torch.cat(parts, dim=-1)
        
        # ========== Step 9: 转换为真实空间的值（用于Gaussian渲染）==========
        # 9.1 XYZ转换：将归一化参数转换为3D真实坐标
        # xyz: [B, N, 3] - 3D空间中的位置（真实单位：米）
        xyz = self.get_xyz(anchor_xyz)

        # 9.2 Scale转换：sigmoid后映射到scale_range
        # 如果scale_act == 'sigmoid'，先进行sigmoid
        if self.scale_act == 'sigmoid':
            scale = safe_sigmoid(anchor_scale)
        # 映射到scale_range：例如 [0.05, 0.5] 米
        # scale: [B, N, 3] - Gaussian的scale（真实单位：米）
        scale = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scale
        
        # 9.3 Semantic转换：softmax（分类）或softplus（回归）
        if self.semantics_activation == 'softmax':
            semantics = anchor_sem.softmax(dim=-1)  # [B, N, semantic_dim] - 概率分布
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(anchor_sem)      # [B, N, semantic_dim] - 正值
        else:
            semantics = anchor_sem                  # [B, N, semantic_dim] - 原始logits
        
        # 9.4 Velocity转换：仅在 enable_velocity 时生成
        velocities = anchor_vel  # None if velocity disabled

        # ========== Step 10: 构建GaussianPrediction对象 ==========
        # 包含所有Gaussian渲染所需的参数，以及用于调试的原始值和delta
        gaussian = GaussianPrediction(
            means=xyz,                    # [B, N, 3] - Gaussian中心位置（3D真实坐标）
            scales=scale,                 # [B, N, 3] - Gaussian的scale（3D真实单位）
            rotations=anchor_rotation,    # [B, N, 4] - Gaussian的旋转（四元数）
            opacities=safe_sigmoid(anchor_opa),  # [B, N, 1] - Gaussian的不透明度 [0, 1]
            semantics=semantics,          # [B, N, semantic_dim] - Gaussian的语义信息
            velocities=velocities,        # [B, N, 3] - Gaussian的速度信息
            original_means=original_xyz,  # [B, N, 3] - 更新前的原始位置（用于调试/可视化）
            delta_means=delta_xyz         # [B, N, 3] - 位置偏移量（用于调试/可视化）
        )
        return output, gaussian  # output: 更新后的anchor参数（参数空间），gaussian: Gaussian渲染参数（真实空间）

    # def get_gaussian(self, output):
    #     if self.phi_activation == 'sigmoid':
    #         xyz = safe_sigmoid(output[..., :3])
    #     elif self.phi_activation == 'loop':
    #         xy = safe_sigmoid(output[..., :2])
    #         z = torch.remainder(output[..., 2:3], 1.0)
    #         xyz = torch.cat([xy, z], dim=-1)
    #     else:
    #         raise NotImplementedError
        
    #     if self.xyz_coordinate == 'polar':
    #         rrr = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
    #         theta = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
    #         phi = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
    #         xxx = rrr * torch.sin(theta) * torch.cos(phi)
    #         yyy = rrr * torch.sin(theta) * torch.sin(phi)
    #         zzz = rrr * torch.cos(theta)
    #     else:
    #         xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
    #         yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
    #         zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
    #     xyz = torch.stack([xxx, yyy, zzz], dim=-1)

    #     gs_scales = safe_sigmoid(output[..., 3:6])
    #     gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        
    #     semantics = output[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
    #     if self.semantics_activation == 'softmax':
    #         semantics = semantics.softmax(dim=-1)
    #     elif self.semantics_activation == 'softplus':
    #         semantics = F.softplus(semantics)
        
    #     gaussian = GaussianPrediction(
    #         means=xyz,
    #         scales=gs_scales,
    #         rotations=output[..., 6:10],
    #         opacities=safe_sigmoid(output[..., 10: (10 + int(self.include_opa))]),
    #         semantics=semantics
    #     )
    #     return gaussian
