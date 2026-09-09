from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmengine import build_from_cfg
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
import torch.nn as nn, torch

@MODELS.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            # Try to build using build_from_cfg first (for custom modules like GroupNorm1D)
            # If that fails, fall back to build_norm_layer
            try:
                if isinstance(pre_norm, dict) and 'type' in pre_norm:
                    # Check if it's a custom module registered in MODELS
                    if pre_norm.get('type') == 'GroupNorm1D':
                        # Ensure num_channels is set
                        pre_norm_config = pre_norm.copy()
                        if 'num_channels' not in pre_norm_config:
                            pre_norm_config['num_channels'] = in_channels
                        self.pre_norm = build_from_cfg(pre_norm_config, MODELS)
                    else:
                        # Use build_norm_layer for standard norm types
                        self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]
                else:
                    self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]
            except Exception as e:
                # Fall back to build_norm_layer if build_from_cfg fails
                import warnings
                warnings.warn(f"Failed to build pre_norm with build_from_cfg, falling back to build_norm_layer: {e}")
                self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else nn.Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        """
        AsymmetricFFN的前向传播流程：
        
        功能：对输入特征进行非线性变换，通常用于Transformer/Encoder中的前馈网络
        
        输入：
            x: [B, N, in_channels] 或 [B, N, embed_dims] - 输入特征
            identity: [B, N, embed_dims] - 可选，用于残差连接的特征
        
        过程：
            1. Pre-normalization（可选）：对输入进行GroupNorm
            2. 多层MLP变换：Linear → ReLU → Dropout → ... → Linear → Dropout
            3. 残差连接（可选）：将变换后的特征与identity相加
        
        输出：
            [B, N, embed_dims] - 变换后的特征
        """
        # ========== Step 1: Pre-normalization（可选）==========
        # 如果配置了pre_norm，先对输入特征进行GroupNorm归一化
        # 作用：稳定训练，加速收敛
        if self.pre_norm is not None:
            x = self.pre_norm(x)  # [B, N, in_channels] 或 [B, N, embed_dims]
        
        # ========== Step 2: MLP变换 ==========
        # layers结构（num_fcs=2的典型情况）：
        #   Linear(in_channels → feedforward_channels) 
        #   → ReLU 
        #   → Dropout
        #   → Linear(feedforward_channels → embed_dims)
        #   → Dropout
        # 
        # 维度变化：
        #   x: [B, N, in_channels]
        #   → [B, N, feedforward_channels]  (第一次Linear + ReLU + Dropout)
        #   → [B, N, embed_dims]           (第二次Linear + Dropout)
        # 
        # 例如：embed_dims=256, feedforward_channels=1024
        #   [B, N, 256] → [B, N, 1024] → [B, N, 256]
        out = self.layers(x)  # [B, N, embed_dims]
        
        # ========== Step 3: 残差连接处理 ==========
        # 如果不使用残差连接，直接返回Dropout后的结果
        if not self.add_identity:
            return self.dropout_layer(out)  # [B, N, embed_dims]
        
        # ========== Step 4: 准备identity用于残差连接 ==========
        # 如果没有提供identity，使用输入x作为identity
        # 否则使用提供的identity（通常是operation_order中identity操作保存的特征）
        if identity is None:
            identity = x  # [B, N, in_channels] 或 [B, N, embed_dims]
        
        # ========== Step 5: Identity维度适配 ==========
        # 如果identity的维度与embed_dims不同，需要线性投影
        # 例如：如果输入是[256]，输出是[128]，则需要投影
        # 如果维度相同，则identity_fc是Identity()，不做任何变换
        identity = self.identity_fc(identity)  # [B, N, embed_dims]
        
        # ========== Step 6: 残差连接 + Dropout ==========
        # 将变换后的特征与identity相加，然后应用Dropout
        # 公式：output = identity + Dropout(MLP(x))
        # 
        # 作用：
        #   - 允许梯度直接传播，缓解梯度消失
        #   - 使深层网络更容易训练
        #   - 保留原始信息，同时增加非线性变换能力
        return identity + self.dropout_layer(out)  # [B, N, embed_dims]
