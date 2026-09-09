import os

_base_ = [
    './_base_/misc.py',
    './_base_/kradar_surroundocc.py'
]

# =========== 速度场功能一键开关 ==============
# True:  启用速度场预测、RadarVelocityNet、VelocityLoss（不影响 radar 交叉注意力和深度初始化）
# False: 关闭所有速度场相关功能，减少显存占用和训练时间
enable_velocity = True  # 保持模型架构不变（含 velocity 分支），仅通过 VelocityLoss weight=0 关闭速度梯度

# =========== Radar Depth 辅助 Loss 一键开关 ==============
# True:  启用 RadarDepthLoss，利用 radar 点云深度监督 Gaussian anchor 位置
# False: 关闭 RadarDepthLoss
enable_radar_depth_loss = True
# RadarDepthLoss 超参数（仅在 enable_radar_depth_loss=True 时生效）
radar_depth_loss_config = dict(
    weight=1.0,             # loss 权重，建议从 0.5~2.0 开始调
    topk=8,                 # 每个 radar 点匹配的最近 K 个 Gaussian
    max_distance=5.0,       # 匹配距离阈值（米），超出的不计入 loss
    loss_type='SmoothL1',   # 'SmoothL1' / 'L1' / 'L2'
    sigma=2.0,              # 距离加权的 sigma（米），越近权重越大
    use_opacity_weight=True, # 是否用 Gaussian 不透明度加权 loss
    warmup_epochs=0,        # 前 N 个 epoch 不启用此 loss（让模型先收敛）
)

# 训练总 epoch 数（同时控制 cosine lr schedule 的衰减周期）
max_epochs = 20
# cosine schedule 最终 lr = initial_lr * min_lr_ratio
min_lr_ratio = 0.01
# 每多少个 epoch 做一次可视化（保存 combined/occ/gaussian 图到 vis_dir）
vis_every_epochs = 1   # 1=每个 epoch 都可视化；5=每 5 个 epoch 可视化一次
# 全量数据集训练时，可视化的样本索引步长（0, vis_step, 2*vis_step, ...）；固定样本过拟合时忽略，会画全部固定样本
vis_step = 100

# =========== data config ==============
# KRadar数据集相关参数（保留KRadar特有的配置）
dataset_type = 'kradar'
input_shape = (720, 1280)
data_aug_conf = {
    "resize_lim": (0.95, 1.05),  # 轻微的缩放增强
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.0, 5.0),  # 添加轻微旋转增强（±5度）
    "H": 720,  # KRadar图像高度
    "W": 2560,  # KRadar图像宽度
    "rand_flip": True,  # 保持随机水平翻转
    # 添加颜色抖动增强
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    },
}
val_dataset_config = dict(
    data_aug_conf=data_aug_conf
)
train_dataset_config = dict(
    data_aug_conf=data_aug_conf
)
# =========== misc config ==============
# 优化器配置
# 针对全数据集训练优化：降低学习率，提高稳定性
optimizer = dict(
    optimizer = dict(
        type="AdamW", 
        lr=1e-4,  # 降低学习率从2e-4到1e-4，提高训练稳定性
        weight_decay=0.05, # 0.01→0.05 增加权重衰减减轻过拟合
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),  # backbone用更小的学习率
            # 'lifter.anchor': dict(lr_mult=2.0),  # anchor点需要更快学习
            # 'lifter.instance_feature': dict(lr_mult=2.0),  # feature也需要更快学习
            'vel_mlp': dict(lr_mult=10.0),  # 独立速度 MLP，需要更快学习
            'radar_vel': dict(lr_mult=5.0),  # RadarVelocityNet: 50x→5x 降低以减轻速度过拟合
            'radar_gate_logit': dict(lr_mult=2.0),  # gate 参数: 10x→2x 降低以减轻过拟合
            'radar_aggregation': dict(lr_mult=2.0),  # radar 交叉注意力: 5x→2x 降低以减轻过拟合
            'radar_fusion_proj': dict(lr_mult=2.0),  # radar 融合投影: 5x→2x 降低以减轻过拟合
        }
    )
)
grad_max_norm = 35
# 梯度累积步数：每个iter训练完后累积梯度，累积指定步数后再一起反向传播和更新参数
# 例如：gradient_accumulation=4 表示每4个iter才更新一次参数，等效于batch_size×4
gradient_accumulation = 1


# Native release contract: empty plus two non-empty semantic classes.
semantic_dim = 2

# ---------- 显存优化 (OOM 时优先尝试) ----------
# 混合精度训练：FP16 前向/部分计算，显著降低显存
# 若 amp=True 出现梯度全 NaN，请改回 amp=False（部分算子已做 float32 保护，仍可能不稳定）
amp = False  # AMP 在当前模型下导致 NaN 梯度，暂不启用；其他优化（radar 缓存、向量化）已生效

# ========= model config ===============
# =========== E2-BKI配置 ===========
# Evidential Learning和E2-BKI相关配置
e2bki_config = dict(
    use_e2bki=True,
    use_evidential=True,
    evidential_loss_weight=1.0,
    lambda_kl=0.1,
    use_kl_annealing=True,
    kl_annealing_max=0.1,
    evidential_head_config=dict(evidence_scale=10.0, min_alpha=1.0),
    kernel_scale=0.8,
    max_euclidean_distance=3.2,
    kernel_type='anisotropic',
    refinement_enabled=True,
    dL=1.0,
    dS=0.2,
    epsilon=2.5,
    use_uncertainty_decomposition=True,
    use_uncertainty_adaptive_kernel=True,
    beta=0.75,
    u_percentile=0.1,
    global_voxel_size=0.4,
    use_sparse_kernel=True,
    kernel_threshold=1e-6,
    max_kernels_per_point=10,
    block_size=8192,
    blockwise_preselect_gaussians=128,
    evidence_weights=[0.15, 1.1],
    alpha_m_cap=[5.0, 8.0],
    alpha_0=[0.0005, 0.0005],
    semantic_input_mode='evidential',
    aggregation_mode='standard',
    use_temporal_recursion=True,
    temporal_alpha_decay=0.9,
    temporal_alpha_write_aggregation='last_writer',
    temporal_require_frame_id=True,
    use_non_empty_only=False,
    empty_label=0,
    use_base_model_prior=False,
    e2bki_hybrid_mode='class_replace',
    e2bki_use_original_occupancy_gate=True,
    use_morphological_recovery=False,
    verbose=False,
)

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='OccupancyLoss',
            weight=1.0,
            empty_label=0,
            num_classes=semantic_dim + 1,
            use_focal_loss=True,
            focal_loss_args=dict(
                use_sigmoid=False,
                gamma=3.0,
                alpha=0.5,
            ),
            use_dice_loss=True,
            balance_cls_weight=True,
            multi_loss_weights=dict(
                loss_voxel_ce_weight=1.0,
                loss_voxel_sem_scal_weight=3.0,
                loss_voxel_geo_scal_weight=1.0,
                loss_voxel_lovasz_weight=6.0,
            ),
            use_sem_geo_scal_loss=True,
            use_lovasz_loss=True,
            lovasz_ignore=0,
            manual_class_weight=[0.05, 0.15, 3.0],
            ignore_empty=False,
            lovasz_use_softmax=True,
            voxel_sample_rates=[0.03, 0.1, 1.0],
        ),
        dict(
            type="PixelDistributionLoss",
            weight=10.0,  # 降低权重从100.0到10.0，让OccupancyLoss更重要
            use_sigmoid=False),
        # 新增：Evidential Loss（如果启用evidential learning）
        dict(
            type='EvidentialLoss',
            weight=e2bki_config['evidential_loss_weight'],
            lambda_kl=e2bki_config['lambda_kl'],
            use_kl_annealing=e2bki_config['use_kl_annealing'],
            kl_annealing_max=e2bki_config['kl_annealing_max'],
        ) if e2bki_config['use_evidential'] else None,
        # Four-term velocity objective. The shared pseudo-target
        # builder performs ego compensation, component matching and Doppler
        # de-aliasing; objects without a reliable solution are masked out.
        dict(
            type='VelocityLoss',
            weight=0.5,  # 启用速度监督（bki 迁移）
            pc_range=[0.0, -25.6, -5.0, 51.2, 25.6, 3.0],  # K-Radar范围
            voxel_size=0.4,
            occ_resolution=[128, 128, 20],
            num_doppler_bins=64,          # K-Radar: 64 个 Doppler bin
            doppler_resolution=0.06,      # 0.06 m/s per bin → v_max = ±1.92 m/s
            max_fold=8,
            min_dynamic_speed=0.1,
            min_component_voxels=20,
            max_match_distance=5.0,
            lambda_point=1.0,
            lambda_voxel=1.0,
            lambda_consistency=0.1,
            lambda_smooth=0.1,
        ) if enable_velocity else None,
        # 新增：Radar Depth 辅助 Loss（受 enable_radar_depth_loss 控制）
        # 利用 radar 点云的精确深度监督 Gaussian anchor 的 3D 位置
        dict(
            type='RadarDepthLoss',
            pc_range=[0.0, -25.6, -5.0, 51.2, 25.6, 3.0],
            **radar_depth_loss_config,
        ) if enable_radar_depth_loss else None,
    ])
# 移除None值（如果evidential learning未启用）
loss['loss_cfgs'] = [cfg for cfg in loss['loss_cfgs'] if cfg is not None]

loss_input_convertion = dict(
    pred_occ="pred_occ",
    sampled_xyz="sampled_xyz",
    sampled_label="sampled_label",
    occ_mask="occ_mask",
    bin_logits="bin_logits",
    density="density",
    pixel_logits="pixel_logits",
    pixel_gt="pixel_gt",
    # Evidential loss: voxel-level Dirichlet alphas [B, N, C] from GaussianHead
    alphas="voxel_evidential_alphas",
    # Gaussian 对象和 radar 点云（RadarDepthLoss / VelocityLoss 共用）
    gaussian="gaussian",
    radar_points="radar_points",
)
# 速度场相关的 loss 输入映射（受 enable_velocity 控制）
if enable_velocity:
    loss_input_convertion.update(dict(
        pred_vel="pred_vel",
        radar_velocity="radar_velocity",
        curr_ego2global="curr_ego2global",
        prev_ego2global="prev_ego2global",
        timestamp="timestamp",
        prev_timestamp="prev_timestamp",
        prev_occ_label="prev_occ_label",
    ))
# ========= model config ===============
# K-Radar网格参数
embed_dims = 128
num_decoder = 4
pc_range = [0.0, -25.6, -5.0, 51.2, 25.6, 3.0]  # K-Radar范围: X=[0, 51.2], Y=[-25.6, 25.6], Z=[-5, 3]
scale_range = [0.01, 1.8]
xyz_coordinate = 'cartesian'
phi_activation = 'sigmoid'
include_opa = True
# Optional image-backbone initialization.
load_from = os.environ.get('RIGS_BACKBONE_CHECKPOINT') or None
require_backbone_checkpoint = True
require_complete_backbone_checkpoint = True
backbone_checkpoint_prefixes = (
    'img_backbone.',
    'lifter.initialize_backbone.img_backbone.',
)
semantics = True

model = dict(
    type='BEVSegmentor',
    # 从头训练时不要冻结lifter！只有使用预训练权重时才冻结
    freeze_lifter=False,  # 改为False，让lifter可以学习
    img_backbone_out_indices=[0, 1, 2, 3],
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        # 使用LayerNorm2d替代GroupNorm：LayerNorm对所有通道和空间维度归一化，可能保留更多输入差异
        # 注意：LN和BN/GN参数不兼容，预训练权重中的BN/GN参数无法直接加载到LN
        # 方案1：从头训练（不使用预训练权重，设置init_cfg=None）
        # 方案2：只加载卷积层权重，LN层随机初始化（当前配置会自动跳过不兼容的层）
        # 方案4：调整LayerNorm2d初始化，使用更大的weight scale以保留更多输入差异
        # init_weight_scale: weight的初始scale，默认1.0（恒等变换），增大此值可能保留更多差异
        # init_bias: bias的初始值，默认0.0
        # norm_cfg=dict(type='LN2d', requires_grad=True, init_weight_scale=2.0, init_bias=0.0),
        # norm_eval=False,  # LayerNorm不需要eval模式（没有running statistics）
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=False,
        style='caffe',
        with_cp = True,  # gradient checkpointing: 省显存但 backward 较慢；若 AMP 开启后显存有余量，可尝试 False
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True),
        # 预训练权重：使用ImageNet预训练的ResNet101
        # BatchNorm在batch_size=1时，如果使用预训练权重，eval模式下可以保留更多差异
        # （因为running statistics是从ImageNet预训练中来的，更稳定）
        # 如果希望完全从头训练，可以设置 init_cfg=None（但BN在batch_size=1时会有问题）
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
        # init_cfg=None,
        # 方案5：替换ReLU为LeakyReLU或SiLU以保留更多负值信息
        # 选项：'leakyrelu' (LeakyReLU), 'silu' (SiLU), '' 或 None (不替换，使用ReLU)
        # LeakyReLU参数：negative_slope，默认0.1
        # replace_relu_with='leakyrelu',  # 默认不替换，设置为 'leakyrelu' 或 'silu' 来启用
        # leaky_relu_slope=0.1,  # LeakyReLU的negative_slope参数，仅在replace_relu_with='leakyrelu'时生效
    ),
    img_neck=dict(
        type='FPN',
        num_outs=4,
        start_level=1,
        out_channels=embed_dims,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 1024, 2048]),
    lifter=dict(
        type='GaussianLifterV2',
        num_anchor=19200,
        embed_dims=embed_dims,
        anchor_grad=True,
        feat_grad=False,
        semantics=semantics,
        semantic_dim=semantic_dim,
        include_opa=include_opa,
        num_samples=256,
        anchors_per_pixel=1,
        random_sampling=False,  # 禁用随机采样，避免使用farthest_point_sampling
        projection_in=None,
        pc_range=pc_range,  # K-Radar范围: [0.0, -25.6, -5.0, 51.2, 25.6, 3.0]
        voxel_size=0.4,  # K-Radar体素大小: 0.4m
        occ_resolution=[128, 128, 20],  # K-Radar网格: 128×128×20
        empty_label=0,  # K-Radar使用0作为empty标签
        initializer=dict(
            type="ResNetSecondFPN",
            img_backbone_out_indices=[0, 1, 2, 3],
            img_backbone_config=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                # 使用LayerNorm2d替代GroupNorm：LayerNorm对所有通道和空间维度归一化，可能保留更多输入差异
                # 方案4：调整LayerNorm2d初始化，使用更大的weight scale以保留更多输入差异
                # norm_cfg=dict(type='LN2d', requires_grad=True, init_weight_scale=2.0, init_bias=0.0),
                # norm_eval=False,  # LayerNorm不需要eval模式（没有running statistics）
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                with_cp=True,  # gradient checkpointing: 省显存但 backward 较慢；若 AMP 开启后显存有余量，可尝试 False
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
                stage_with_dcn=(False, False, True, True),
                # 预训练权重：卷积层权重可以加载，但BN→GN的参数会被跳过（使用随机初始化）
                # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
                # init_cfg=None
                ),
            neck_config=dict(
                type='SECONDFPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=[embed_dims] * 4,
                upsample_strides=[0.5, 1, 2, 4])),
        initializer_img_downsample=None,
        pretrained_path=None,  # 不使用初始化权重，让模型自己初始化
        deterministic=False,
        random_samples=6400,
        use_radar_depth=True,
        radar_depth_fallback=False,
        radar_depth_ratio=0.3,
        use_radar_as_anchor=False,     # 是否直接用radar点云作为anchor（跳过深度采样，每个样本使用自己的radar点）
        use_temporal_init=True,
        temporal_init_weight=0.2,      # 时序初始化权重（0-1之间，0=只用图像，1=只用时序）- 用于xyz坐标 [降低: 0.5→0.2]
        temporal_feature_weight=0.2,   # 时序特征权重（用于rep_features的混合）[降低: 0.5→0.2]
        temporal_scale_weight=0.1,     # Scale参数的时序权重 [降低: 0.3→0.1]
        temporal_rotation_weight=0.1,  # Rotation参数的时序权重 [降低: 0.2→0.1]
        temporal_opacity_weight=0.1,   # Opacity参数的时序权重 [降低: 0.3→0.1]
        reuse_semantic=False,          # 不直接复用语义，使用混合 [改为False: 避免复制错误预测]
        temporal_match_threshold=5.0,  # 时序匹配距离阈值（米），只有距离小于此值的才进行融合
        temporal_match_method='nearest',  # 匹配方法：'nearest'（最近邻，推荐）或 'index'（索引对应，不推荐）
        temporal_warmup_epochs=1,     # 前10个epoch不用时序初始化，之后线性升温
    ),
    encoder=dict(
        type='GaussianOccEncoder',
        embed_dims=embed_dims,
        anchor_encoder=dict(
            type='SparseGaussian3DEncoder',
            embed_dims=embed_dims, 
            include_opa=include_opa,
            semantics=semantics,
            semantic_dim=semantic_dim
        ),
        norm_layer=dict(type="GroupNorm1D", num_channels=embed_dims, num_groups=32),
        ffn=dict(
            type="AsymmetricFFN",
            in_channels=embed_dims,
            embed_dims=embed_dims,
            feedforward_channels=embed_dims * 4,
            ffn_drop=0.2, # single-all (0.1→0.2 增加正则化减轻过拟合)
            add_identity=False,
        ),
        deformable_model=dict(
            type='DeformableFeatureAggregation',
            embed_dims=embed_dims,
            num_groups=4,
            num_levels=4,
            residual_mode="none",
            num_cams=1,  # KRadar有1个相机 （只用camfront）
            proj_drop=0.0,
            attn_drop=0.15,
            use_deformable_func=True,
            use_camera_embed=True,
            use_radar_branch=True,  # 启用radar交叉注意力分支
            radar_aggregation=dict(
                type='RadarFeatureAggregation',
                embed_dims=embed_dims,
                radar_feat_dim=10,  # power_val 8 + intensity + velocity
                num_heads=8,
                proj_drop=0.1,  # 0.0→0.1 增加attention dropout减轻过拟合
                pc_range=pc_range,
                max_radar_points=4096,   # 体素下采样后上限
                voxel_size=1.6,          # 体素大小（米）
                topk_neighbors=64,       # 每个 Gaussian 注意最近 64 个 radar 点
                attn_chunk_size=4096,    # query 分块大小
            ),
            pc_range=pc_range,
            kps_generator=dict(
                type='SparseGaussian3DKeyPointsGenerator',
                embed_dims=embed_dims,
                phi_activation=phi_activation,
                xyz_coordinate=xyz_coordinate,
                num_learnable_pts=6,
                pc_range=pc_range,
                scale_range=scale_range,
                learnable_fixed_scale=6.0,
            ),
        ),
        refine_layer=dict(
            type='SparseGaussian3DRefinementModuleV2',
            embed_dims=embed_dims,
            pc_range=pc_range,
            scale_range=scale_range,
            unit_xyz=[4.0, 4.0, 1.0],
            semantics=semantics,
            semantic_dim=semantic_dim,
            include_opa=include_opa,
            xyz_coordinate=xyz_coordinate,
            semantics_activation='identity',
            enable_velocity=enable_velocity,
        ),
        spconv_layer=dict(
            type="SparseConv3D",
            in_channels=embed_dims,
            embed_channels=embed_dims,
            pc_range=pc_range,
            grid_size=[0.4, 0.4, 0.4],  # K-Radar体素大小: 0.4m
            phi_activation=phi_activation,
            xyz_coordinate=xyz_coordinate,
            use_out_proj=True,
            use_multi_layer=True,
        ),
        # 雷达多普勒速度网络：专用于从雷达Doppler特征学习高斯椭球速度
        # 受 enable_velocity 控制
        radar_vel_net=dict(
            type='RadarVelocityNet',
            embed_dims=embed_dims,
            radar_feat_dim=10,   # power_val 8 + intensity + velocity
            num_heads=4,
            pc_range=pc_range,
            vel_hidden_dim=64,
            proj_drop=0.1,  # 0.0→0.1 增加attention dropout减轻过拟合
            residual=True,       # 残差模式：output = init_vel + vel_delta，init_vel 提供合理初始量级（~0.5 m/s），加速收敛
            max_radar_points=4096,
            voxel_size=1.6,
            topk_neighbors=64,
            attn_chunk_size=4096,
        ) if enable_velocity else None,
        num_decoder=num_decoder,
        operation_order=([
            "identity",
            "deformable",
            "add",
            "norm",

            "identity",
            "ffn",
            "add",
            "norm",

            "identity",
            "spconv",
            "add",
            "norm",

            "identity",
            "ffn",
            "add",
            "norm",
            
            "refine",
        ] + (["radar_vel"] if enable_velocity else [])) * num_decoder,
    ),
        head=dict(
        type='GaussianHead',
        apply_loss_type='fixed_3',
        num_classes=semantic_dim + 1,  # empty/background/foreground
        empty_label=0,  # K-Radar使用0作为empty标签
        dataset_type='kradar',  # 设置为kradar，这样empty会添加到前面而不是后面
        empty_args=dict(
            mean=[0, 0, -1.0],
            scale=[100, 100, 8.0],
        ),
        with_empty=False,
        use_localaggprob=True,
        use_localaggprob_fast=False,
        combine_geosem=True,
        # E2-BKI相关配置
        use_evidential=e2bki_config['use_evidential'],  # 是否使用evidential learning
        evidential_config=e2bki_config['evidential_head_config'],  # Evidential head配置
        cuda_kwargs=dict(
            scale_multiplier=3,  # 从4降到3: 3σ截断(标准做法)，减少Gaussian溢出导致的预测范围偏大
            H=128, W=128, D=20,  # K-Radar网格: 128 x 128 x 20
            pc_min=[0.0, -25.6, -5.0],  # K-Radar范围最小值
            grid_size=0.4),  # K-Radar体素大小: 0.4m
    )
)
