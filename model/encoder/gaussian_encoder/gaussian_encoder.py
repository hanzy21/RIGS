from typing import List, Optional
import torch, torch.nn as nn

from mmseg.registry import MODELS
from mmengine import build_from_cfg
from ..base_encoder import BaseEncoder


@MODELS.register_module()
class GaussianOccEncoder(BaseEncoder):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        mid_refine_layer: dict = None,
        spconv_layer: dict = None,
        radar_vel_net: dict = None,
        num_decoder: int = 6,
        operation_order: Optional[List[str]] = None,
        embed_dims: int = 128,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder
        self.embed_dims = embed_dims

        # operation_order: 定义每个decoder层的操作序列
        # 默认操作顺序（如果没有自定义）：
        #   ["spconv", "norm", "deformable", "norm", "ffn", "norm", "refine"] * num_decoder
        # 
        # 操作类型说明：
        # - "spconv": 稀疏3D卷积，在3D空间中聚合邻域信息（可选，配置中通常为None）
        # - "norm": LayerNorm，特征归一化
        # - "deformable": 可变形注意力，从多视图图像特征中采样和融合
        # - "ffn": 前馈网络，非线性变换（通常包含残差连接）
        # - "refine": 精炼模块，更新Gaussian anchor参数（每个decoder层最后执行）
        # - "identity": 保存当前特征（用于残差连接）
        # - "add": 残差连接，将保存的特征与当前特征相加
        # 
        # K-Radar 使用的自定义 operation_order（带残差连接）：
        #   ["identity", "deformable", "add", "norm",
        #    "identity", "ffn", "add", "norm",
        #    "identity", "spconv", "add", "norm",
        #    "identity", "ffn", "add", "norm",
        #    "refine"] * num_decoder
        # 
        # 执行流程（以默认顺序为例，num_decoder=1）：
        # 1. spconv: 3D空间卷积（如果启用）
        # 2. norm: 归一化
        # 3. deformable: 从图像特征中采样（核心操作）
        # 4. norm: 归一化
        # 5. ffn: 非线性变换
        # 6. norm: 归一化
        # 7. refine: 更新anchor参数，输出Gaussian
        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.anchor_encoder = build(anchor_encoder, MODELS)
        self.op_config_map = {
            "norm": [norm_layer, MODELS],
            "ffn": [ffn, MODELS],
            "deformable": [deformable_model, MODELS],
            "refine": [refine_layer, MODELS],
            "mid_refine":[mid_refine_layer, MODELS],
            "spconv": [spconv_layer, MODELS],
            "radar_vel": [radar_vel_net, MODELS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        # 注入逐帧全局图像上下文，解决 instance_feature 全零导致各帧起点相同的问题
        self.global_context_proj = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
        )
        
    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(
        self,
        representation,      # [B, N, anchor_dim] - 初始anchor参数（来自Lifter）
        rep_features,        # [B, N, embed_dims] - 初始特征（通常全零或可学习参数）
        ms_img_feats=None,   # List[[B, N_cam, C, H, W]] - 多层级图像特征
        metas=None,          # dict - 元数据（投影矩阵、相机参数等）
        radar_points=None,   # [B, N_radar, 3] 或 list，供 radar/image/gaussian 交叉注意力
        radar_features=None, # [B, N_radar, F] 或 list（power_val/Doppler 等）
        **kwargs
    ):
        """
        GaussianOccEncoder 的完整前向传播流程：
        
        ========== 输入准备 ==========
        1. representation: 初始Gaussian anchor参数（xyz, scale, rotation, opacity, semantic）
        2. rep_features: 初始特征（可学习参数，初始化为全零）
        3. ms_img_feats: 多层级图像特征（从图像backbone和neck提取）
        4. metas: 元数据（投影矩阵、相机内参等）
        
        ========== 执行流程（以默认operation_order，num_decoder=2为例）==========
        
        【初始化】
        anchor_embed = anchor_encoder(anchor)  # 将anchor参数编码为embedding
        
        【Decoder Layer 1】
        1. spconv: 3D空间卷积聚合邻域信息（可选）
        2. norm: 归一化
        3. deformable: 从图像特征中采样（核心：多视图、多层级融合）
        4. norm: 归一化
        5. ffn: 非线性变换
        6. norm: 归一化
        7. refine: 更新anchor参数，输出Gaussian_1
           - anchor = refine(instance_feature, anchor, anchor_embed)
           - anchor_embed = anchor_encoder(anchor)  # 更新embedding
        
        【Decoder Layer 2】
        1. spconv: 3D空间卷积（从Layer 1的anchor位置开始）
        2. norm: 归一化
        3. deformable: 从图像特征中采样（基于更新后的anchor位置）
        4. norm: 归一化
        5. ffn: 非线性变换
        6. norm: 归一化
        7. refine: 再次更新anchor参数，输出Gaussian_2
        
        【输出】
        prediction = [{'gaussian': Gaussian_1}, {'gaussian': Gaussian_2}, ...]
        
        ========== 数据流维度变化 ==========
        instance_feature: [B, N, embed_dims] (保持不变)
        anchor: [B, N, anchor_dim] (每次refine后更新)
        anchor_embed: [B, N, embed_dims] (每次refine后重新编码)
        """
        feature_maps = ms_img_feats
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        
        # ========== 阶段1.1: 检查图像backbone输出 ==========
        # 通过环境变量控制是否启用调试（避免过于频繁的打印）
        import os
        debug_image_usage = os.environ.get('DEBUG_MODE', '0') == '1'
        debug_image_usage_iter = 1  # 每N个iter打印一次
        
        if debug_image_usage:
            # 获取当前iter（从metas中获取，如果没有则使用默认值）
            current_iter = -1
            if isinstance(metas, dict):
                current_iter = metas.get('global_iter', -1)
            elif hasattr(metas, 'get'):
                current_iter = metas.get('global_iter', -1)
            
            # 如果current_iter为-1，则每次都打印（用于测试）
            should_print = (current_iter == -1) or (current_iter % debug_image_usage_iter == 0)
            
            if should_print:
                print(f"\n[DEBUG_IMAGE_USAGE] ========== Iter {current_iter} ==========")
                print(f"[DEBUG_IMAGE_USAGE] Feature maps info:")
                print(f"  - Number of feature map levels: {len(feature_maps)}")
                for i, fm in enumerate(feature_maps):
                    print(f"  - Level {i}: shape={fm.shape}, mean={fm.mean().item():.6f}, std={fm.std().item():.6f}, "
                          f"min={fm.min().item():.6f}, max={fm.max().item():.6f}")
                    # 检查是否接近全零（可能是图像被置零）
                    if fm.abs().max().item() < 1e-5:
                        print(f"    ⚠️  WARNING: Feature map level {i} is nearly zero!")
                    # 检查是否所有值相同（可能是backbone没有提取特征）
                    if fm.std().item() < 1e-5:
                        print(f"    ⚠️  WARNING: Feature map level {i} has very low std (nearly constant)!")
        
        instance_feature = rep_features  # [B, N, embed_dims] - 初始特征
        # 注入逐帧全局图像上下文，解决各帧输出趋同问题
        if feature_maps is not None and len(feature_maps) > 0:
            global_feat = feature_maps[0].mean(dim=[-2, -1])  # [B, n_cam, 128]
            global_feat = global_feat.mean(dim=1)              # [B, 128]
            global_ctx = self.global_context_proj(global_feat)  # [B, 128]
            instance_feature = instance_feature + global_ctx.unsqueeze(1)
        anchor = representation          # [B, N, anchor_dim] - 初始anchor参数
        kwargs = kwargs.copy() if kwargs else {}
        kwargs.setdefault("radar_points", radar_points)
        kwargs.setdefault("radar_features", radar_features)

        # ========== 预计算 radar 体素下采样（避免每层 decoder 重复计算）==========
        # 原本每个 decoder 层的 RadarFeatureAggregation 和 RadarVelocityNet 都各自调用
        # voxel_downsample_radar，共 num_decoder*2 次。radar 原始点云在各层间不变，
        # 下采样结果完全相同，因此只需计算一次。
        _rp = kwargs.get("radar_points")
        if _rp is not None:
            from .deformable_module import voxel_downsample_radar
            bs_r = anchor.shape[0]
            device_r = anchor.device
            dtype_r = anchor.dtype
            # Unify inputs: handle list or tensor format
            if isinstance(_rp, (list, tuple)):
                max_n = max(p.shape[0] for p in _rp)
                rp_u = torch.zeros(bs_r, max_n, 3, device=device_r, dtype=dtype_r)
                mask_u = torch.zeros(bs_r, max_n, dtype=torch.bool, device=device_r)
                for b_idx in range(bs_r):
                    pt = _rp[b_idx]
                    if not torch.is_tensor(pt):
                        pt = torch.from_numpy(pt)
                    n = pt.shape[0]
                    rp_u[b_idx, :n] = pt.to(device=device_r, dtype=dtype_r)
                    mask_u[b_idx, :n] = True
                _rf = kwargs.get("radar_features")
                if _rf is not None and isinstance(_rf, (list, tuple)):
                    f_dim = _rf[0].shape[-1]
                    rf_u = torch.zeros(bs_r, max_n, f_dim, device=device_r, dtype=dtype_r)
                    for b_idx in range(bs_r):
                        ft = _rf[b_idx]
                        if not torch.is_tensor(ft):
                            ft = torch.from_numpy(ft)
                        rf_u[b_idx, :ft.shape[0]] = ft.to(device=device_r, dtype=dtype_r)
                    _rf = rf_u
                elif _rf is None:
                    _rf = torch.zeros(bs_r, max_n, 10, device=device_r, dtype=dtype_r)
                _rm = mask_u
            else:
                _rp_t = _rp.to(device=device_r, dtype=dtype_r)
                _rf = kwargs.get("radar_features")
                if _rf is not None:
                    _rf = _rf.to(device=device_r, dtype=dtype_r)
                else:
                    _rf = torch.zeros(bs_r, _rp_t.shape[1], 10, device=device_r, dtype=dtype_r)
                _rm = torch.ones(bs_r, _rp_t.shape[1], dtype=torch.bool, device=device_r)
                _rp = _rp_t
            # 从任一 radar 子模块获取下采样参数（voxel_size, max_points）
            _vs, _mp = 1.6, 4096  # 默认值
            for lyr in self.layers:
                if lyr is not None and hasattr(lyr, 'voxel_size'):
                    _vs = lyr.voxel_size
                    _mp = lyr.max_radar_points
                    break
                # deformable 层的 radar_aggregation 子模块
                if lyr is not None and hasattr(lyr, 'radar_aggregation') and lyr.radar_aggregation is not None:
                    _vs = lyr.radar_aggregation.voxel_size
                    _mp = lyr.radar_aggregation.max_radar_points
                    break
            ds_pts, ds_feats, ds_mask = voxel_downsample_radar(
                _rp, _rf, _rm, voxel_size=_vs, max_points=_mp)
            kwargs["radar_points"] = ds_pts
            kwargs["radar_features"] = ds_feats
            kwargs["radar_mask"] = ds_mask
            kwargs["_radar_already_downsampled"] = True

        # 初始编码：将anchor参数编码为embedding
        # anchor_embed: [B, N, embed_dims] - anchor参数的embedding表示
        anchor_embed = self.anchor_encoder(anchor)

        # prediction: 存储每个decoder层refine后的Gaussian结果
        # 每个元素：{'gaussian': GaussianPrediction对象}
        prediction = []
        
        # ========== 按operation_order顺序执行每个操作 ==========
        for i, op in enumerate(self.operation_order):
            
            # ========== 操作1: spconv - 稀疏3D卷积 ==========
            # 功能：在3D空间中聚合邻域anchor的特征信息
            # 输入：instance_feature [B, N, C], anchor [B, N, anchor_dim]
            # 过程：
            #   1. 将anchor的xyz转换为3D网格索引
            #   2. 构建稀疏卷积张量（SparseConvTensor）
            #   3. 使用SubMConv3d进行3D卷积
            #   4. 输出：更新的instance_feature [B, N, C]
            # 注意：如果spconv_layer为None，此操作会被跳过
            if op == 'spconv':
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor)
            
            # ========== 操作2: norm - GroupNorm归一化 ==========
            # 功能：对instance_feature进行GroupNorm归一化（替代LayerNorm）
            # 输入/输出：instance_feature [B, N, C]
            # 作用：稳定训练，加速收敛
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            
            # ========== 操作3: identity - 保存特征（用于残差连接）==========
            # 功能：保存当前instance_feature的副本，用于后续的残差连接
            # 输入：instance_feature [B, N, C]
            # 输出：保存到identity变量（不修改instance_feature）
            # 使用场景：在deformable/ffn/spconv之前保存，然后在add中恢复
            elif op == "identity":
                identity = instance_feature
            
            # ========== 操作4: add - 残差连接 ==========
            # 功能：将之前保存的identity与当前instance_feature相加
            # 输入：instance_feature [B, N, C], identity [B, N, C]
            # 输出：instance_feature + identity [B, N, C]
            # 作用：允许梯度直接传播，缓解梯度消失，使深层网络更容易训练
            elif op == "add":
                instance_feature = instance_feature + identity
            
            # ========== 操作5: deformable - 可变形注意力（核心操作）==========
            # 功能：从多视图、多层级图像特征中采样和融合信息
            # 输入：
            #   - instance_feature [B, N, C] - 当前特征
            #   - anchor [B, N, anchor_dim] - anchor参数
            #   - anchor_embed [B, N, C] - anchor的embedding
            #   - feature_maps List[[B, N_cam, C, H, W]] - 多层级图像特征
            #   - metas dict - 元数据（投影矩阵等）
            # 过程：
            #   1. 生成key points（每个anchor周围13个采样点）
            #   2. 将3D key points投影到每个相机的图像平面
            #   3. 使用可变形注意力从图像特征中采样
            #   4. 多视图、多层级特征加权融合
            #   5. 残差连接（residual_mode="cat"或"add"）
            # 输出：更新的instance_feature [B, N, C] 或 [B, N, 2C]
            elif op == "deformable":
                # 只通过显式参数传 radar_points/radar_features，避免 **kwargs 中重复传入导致 multiple values
                deformable_kwargs = {k: v for k, v in kwargs.items() if k not in ("radar_points", "radar_features")}
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                    radar_points=kwargs.get("radar_points"),
                    radar_features=kwargs.get("radar_features"),
                    **deformable_kwargs,
                )
            
            # ========== 操作6: refine - 精炼anchor参数（每个decoder层的最后操作）==========
            # 功能：根据当前instance_feature更新Gaussian anchor的参数
            # 输入：
            #   - instance_feature [B, N, C] - 当前特征
            #   - anchor [B, N, anchor_dim] - 当前anchor参数
            #   - anchor_embed [B, N, C] - anchor的embedding
            # 过程：
            #   1. MLP预测参数delta：delta = MLP(instance_feature + anchor_embed)
            #   2. 更新anchor参数：
            #      - xyz: 在3D空间中计算delta，更新位置
            #      - scale, rotation, opacity, semantic: 直接使用预测值
            #   3. 转换为真实空间的值（用于Gaussian渲染）
            #   4. 构建GaussianPrediction对象
            # 输出：
            #   - anchor [B, N, anchor_dim] - 更新后的anchor参数
            #   - gaussian: GaussianPrediction对象 - 用于渲染的Gaussian参数
            elif "refine" in op:
                anchor, gaussian = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                )
            
                # 保存当前层的refine结果
                prediction.append({'gaussian': gaussian})
                
                # 如果不是最后一层，更新anchor_embed（用于下一层）
                # 因为anchor已经被refine，需要重新编码
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)

            elif op == "radar_vel":
                # ========== 操作: radar_vel - 用雷达多普勒数据细化高斯速度 ==========
                # 设计：RefineModule 预测初始速度（从 instance_feature），
                #        RadarVelocityNet 用雷达多普勒的交叉注意力施加残差修正
                # I1 DETACH: 速度梯度不回传到 backbone，避免 VelocityLoss 干扰 OCC 学习
                if self.layers[i] is not None and kwargs.get("radar_points") is not None and len(prediction) > 0:
                    last_gaussian = prediction[-1]['gaussian']
                    refined_vel = self.layers[i](
                        instance_feature.detach(),
                        last_gaussian.means.detach(),  # detach: 速度梯度不回传到位置
                        last_gaussian.velocities,
                        kwargs["radar_points"],
                        kwargs.get("radar_features"),
                        radar_mask=kwargs.get("radar_mask"),
                        _radar_already_downsampled=kwargs.get("_radar_already_downsampled", False),
                    )
                    # 更新 GaussianPrediction 中的 velocities
                    prediction[-1]['gaussian'] = last_gaussian._replace(velocities=refined_vel)
                    # I2 DETACH: refined_vel 拼接到 anchor 时 detach，防止梯度泄漏到 anchor_embed
                    # 同步更新 anchor 中的速度维度（最后 3 维）
                    anchor = torch.cat([anchor[..., :-3], refined_vel.detach()], dim=-1)
                    # 重新编码 anchor_embed，让下一个 decoder 层看到雷达修正后的速度
                    if i != len(self.operation_order) - 1:
                        anchor_embed = self.anchor_encoder(anchor)

            else:
                raise NotImplementedError(f"{op} is not supported.")

        # 返回refined_anchor用于时序初始化（最后一个refine后的anchor）
        return {
            "representation": prediction,
            "refined_anchor": anchor,  # [B, N, anchor_dim] - 最后一个decoder层refine后的anchor参数
            "refined_rep_features": instance_feature  # [B, N, embed_dims] - 最后一个decoder层的特征
        }