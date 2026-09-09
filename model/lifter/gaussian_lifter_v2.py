import torch, torch.nn as nn, math
import numpy as np
from einops import rearrange
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid
from ..utils.sampler import DistributionSampler
from ..utils.temporal_utils import transform_gaussian_xyz

try:
    from pointops import farthest_point_sampling
except ImportError:
    def farthest_point_sampling(points, start_len, end_len):
        """Deterministic uniform fallback for environments without pointops."""
        if start_len.ndim == 0:
            num_points = int(start_len.item())
            num_samples = int(end_len.item())
        else:
            num_points = int(start_len.reshape(-1)[0].item())
            num_samples = int(end_len.reshape(-1)[0].item())
        if num_points <= num_samples:
            return torch.arange(num_points, device=points.device, dtype=torch.long)
        return torch.linspace(
            0, num_points - 1, num_samples, device=points.device, dtype=torch.long
        )


def deterministic_anchor_sampling(configured: bool, training: bool) -> bool:
    """Validation/inference are deterministic regardless of training policy."""
    return bool(configured or not training)


def repeat_anchor_scan(scan, count):
    """Deterministically fill a short scan to exactly ``count`` entries."""
    if scan.shape[0] <= 0:
        raise ValueError("cannot fill an empty anchor scan")
    repeats = int(math.ceil(count / scan.shape[0]))
    return scan.repeat(repeats, 1)[:count]


@MODELS.register_module()
class GaussianLifterV2(BaseLifter):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        xyz_activation="sigmoid",
        scale_activation="sigmoid",

        num_samples=64,
        pc_range=[-50, -50, -5, 50, 50, 3],
        voxel_size=0.5,
        occ_resolution=[200, 200, 16],
        empty_label=0,
        anchors_per_pixel=1,
        random_sampling=True,
        projection_in=None,
        initializer=None,
        initializer_img_downsample=None,
        pretrained_path=None,
        deterministic=True,
        random_samples=0,
        use_radar_depth=False,  # 是否使用radar点云初始化深度采样
        radar_depth_fallback=False,
        radar_depth_ratio=0.3,  # 混合模式：radar bins 占总 bins 的比例 (0.0=全固定, 1.0=全radar, 0.3=30%radar+70%固定)
        use_radar_as_anchor=False,  # 是否直接用radar点云作为anchor（跳过深度采样，每个样本使用自己的radar点）
        use_temporal_init=False,      # 是否启用时序初始化
        temporal_init_weight=0.5,     # 时序初始化权重（0-1之间，0=只用图像，1=只用时序）- 用于xyz坐标
        temporal_feature_weight=0.5,  # 时序特征权重（用于rep_features的混合）
        temporal_scale_weight=0.3,    # Scale参数的时序权重（通常比xyz权重低，因为尺度变化较小）
        temporal_rotation_weight=0.2, # Rotation参数的时序权重（通常最低，因为旋转需要根据位姿调整）
        temporal_opacity_weight=0.3,  # Opacity参数的时序权重（不透明度可能变化）
        reuse_semantic=True,          # 是否直接复用上一帧的semantic（True=直接复用，False=混合）
        temporal_match_threshold=5.0, # 时序匹配距离阈值（米），只有距离小于此值的才进行融合
        temporal_match_method='nearest',  # 匹配方法：'nearest'（最近邻）或 'index'（索引对应，不推荐）
        temporal_warmup_epochs=0,     # 时序初始化热身epoch数：前N个epoch不使用时序，之后线性升温到目标权重
        **kwargs,
    ):
        super().__init__()
        if semantic_dim != 2 or empty_label != 0:
            raise ValueError(
                "RIGS lifter requires semantic_dim=2 and empty_label=0; "
                f"got semantic_dim={semantic_dim}, empty_label={empty_label}"
            )
        if use_radar_depth and radar_depth_fallback:
            raise ValueError(
                "RIGS does not allow fixed-depth fallback when radar depth is enabled"
            )
        self.embed_dims = embed_dims
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation
        self.include_opa = include_opa
        self.semantics = semantics
        self.semantic_dim = semantic_dim

        self.random_samples = random_samples
        if random_samples > 0:
            self.random_anchors = self.init_random_anchors()
        
        # 时序初始化参数
        self.use_temporal_init = use_temporal_init
        self.temporal_init_weight = temporal_init_weight
        self.temporal_feature_weight = temporal_feature_weight
        self.temporal_scale_weight = temporal_scale_weight
        self.temporal_rotation_weight = temporal_rotation_weight
        self.temporal_opacity_weight = temporal_opacity_weight
        self.reuse_semantic = reuse_semantic
        self.temporal_match_threshold = temporal_match_threshold
        self.temporal_match_method = temporal_match_method
        self.temporal_warmup_epochs = temporal_warmup_epochs
                    
        scale = torch.ones(num_anchor, 3, dtype=torch.float) * 0.5
        if scale_activation == "sigmoid":
            scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
        anchor = torch.cat([scale, rots, opacity, semantic], dim=-1)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            anchor,
            requires_grad=anchor_grad,
        )
        self.instance_feature = nn.Parameter(
            torch.zeros([num_anchor + random_samples, self.embed_dims]),
            requires_grad=feat_grad,
        )
        projection_in = embed_dims * 4 if projection_in is None else projection_in
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(projection_in, num_samples + 1),
        )
        self.sampler = DistributionSampler()
        self.num_samples = num_samples
        self.register_buffer("depth_bins", torch.linspace(
            1.0, 72.0, self.num_samples, dtype=torch.float), persistent=False)
        self.register_buffer("pc_start", torch.tensor(
            pc_range[:3], dtype=torch.float), persistent=False)
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.occ_resolution = occ_resolution
        self.empty_label = empty_label
        self.anchors_per_pixel = anchors_per_pixel
        self.random_sampling = random_sampling
        self.use_radar_depth = use_radar_depth
        self.radar_depth_fallback = radar_depth_fallback
        self.radar_depth_ratio = radar_depth_ratio
        self.use_radar_as_anchor = use_radar_as_anchor
        if initializer is not None:
            self.initialize_backbone = MODELS.build(initializer)
        else:
            self.initialize_backbone = None
        self.initializer_img_downsample = initializer_img_downsample
        
        self.pretrained_path = pretrained_path
        self.deterministic = deterministic
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location='cpu')
            ckpt = ckpt.get("state_dict", ckpt)
            if 'instance_feature' in ckpt:
                del ckpt['instance_feature']
            if 'anchor' in ckpt:
                del ckpt['anchor']
            print(self.load_state_dict(ckpt, strict=False))
            print("Gaussian Initializer Weight Loaded Successfully.")

    def init_random_anchors(self):
        num_anchor = self.random_samples

        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        if self.xyz_act == "sigmoid":
            xyz = safe_inverse_sigmoid(xyz)
        
        scale = torch.ones(num_anchor, 3, dtype=torch.float) * 0.5
        if self.scale_act == "sigmoid":
            scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if self.include_opa:
            opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if self.semantics:
            semantic_dim = self.semantic_dim
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)
        anchor = torch.cat([xyz, scale, rots, opacity, semantic], dim=-1)
        anchor = nn.Parameter(anchor, True)
        return anchor

    def init_weights(self):
        if self.pretrained_path is not None:
            return
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def forward(self, metas, **kwargs):
        # ========== 时序初始化检查 ==========
        prev_representation = None
        prev_rep_features = None
        use_temporal = False
        
        if self.use_temporal_init:
            # 检查是否有上一帧的高斯参数和位姿信息
            # 注意：collate 将 None 转换为 [None]，而 [None] is not None → True
            # 所以需要额外检查是否真的是 Tensor（而不是 [None] 之类的列表）
            prev_repr_val = metas.get('prev_gaussian_representation')
            prev_ego_val = metas.get('prev_ego2global')
            curr_ego_val = metas.get('curr_ego2global')
            if (prev_repr_val is not None and torch.is_tensor(prev_repr_val) and
                prev_ego_val is not None and torch.is_tensor(prev_ego_val) and
                curr_ego_val is not None and torch.is_tensor(curr_ego_val)):
                        # 检查是否是连续帧
                        # NOTE: After collation, is_continuous_frame may be a list [bool, ...].
                        # A non-empty list like [False] is truthy in Python, so we must
                        # explicitly resolve it to a scalar bool.
                        is_continuous = metas.get('is_continuous_frame', False)
                        if isinstance(is_continuous, (list, tuple)):
                            is_continuous = all(is_continuous)  # all items must be True
                        if is_continuous:
                            prev_representation = metas['prev_gaussian_representation']  # [B, N, anchor_dim]
                            prev_rep_features = metas.get('prev_gaussian_rep_features', None)  # [B, N, embed_dims]
                            use_temporal = True
        
        # ========== 时序权重热身：前 temporal_warmup_epochs 个 epoch 不使用时序 ==========
        # 之后线性升温到目标权重（经过 warmup 个 epoch 达到满权重）
        temporal_scale = 1.0
        if use_temporal and self.temporal_warmup_epochs > 0:
            current_epoch = metas.get('_current_epoch', None)
            if current_epoch is not None:
                if current_epoch < self.temporal_warmup_epochs:
                    # 在热身阶段，完全禁用时序初始化
                    use_temporal = False
                elif current_epoch < self.temporal_warmup_epochs * 2:
                    # 线性升温阶段
                    temporal_scale = (current_epoch - self.temporal_warmup_epochs) / self.temporal_warmup_epochs
        
        if self.initialize_backbone is not None:
            b, n = kwargs["imgs"].shape[:2]
            initialize_input = kwargs["imgs"].flatten(0, 1)
            if self.initializer_img_downsample is not None:
                initialize_input = nn.functional.interpolate(
                    initialize_input, scale_factor=self.initializer_img_downsample, 
                    mode='bilinear', align_corners=True)
            secondfpn_out = self.initialize_backbone(initialize_input)
            secondfpn_out = secondfpn_out.unflatten(0, (b, n))
        else:
            secondfpn_out = kwargs["secondfpn_out"]
        
        b, n, _, h, w = secondfpn_out.shape
        # h, w 是特征图的高度和宽度
        # 当anchors_per_pixel=1时，最终anchor数量应该是 h * w
        # num_anchor应该等于 h * w（如果不使用random_samples）
        feature = rearrange(secondfpn_out, 'b n c h w -> b n h w c')
        logits = self.projection(feature) # b, n, h, w, d + 1
        # TODO：这里的采样或许可以参考radar
        projection_mat = metas["projection_mat"].inverse() # img2lidar
        u = (torch.arange(w, dtype=feature.dtype, device=feature.device) + 0.5) / w
        v = (torch.arange(h, dtype=feature.dtype, device=feature.device) + 0.5) / h
        uv = torch.stack([
            u[None, :].expand(h, w), v[:, None].expand(h, w)], dim=-1) # h, w, 2
        uv = uv[None, None].expand(b, n, h, w, 2) * metas['image_wh'][:, :, None, None] # b, n, h, w, 2
        
        # 使用radar点云初始化深度采样
        if self.use_radar_depth and 'radar_points' not in metas and not (
            isinstance(metas.get('metas'), dict) and 'radar_points' in metas['metas']
        ):
            raise KeyError(
                "GaussianLifterV2 requires radar_points when use_radar_depth=True; "
                f"available keys: {sorted(metas.keys())}"
            )
        
        if self.use_radar_depth:
            # 检查radar_points是否在metas中（可能在metas['radar_points']或直接在metas中）
            radar_points_raw = None
            if 'radar_points' in metas:
                radar_points_raw = metas['radar_points']
            elif isinstance(metas, dict) and 'metas' in metas and 'radar_points' in metas['metas']:
                radar_points_raw = metas['metas']['radar_points']
            
            if radar_points_raw is not None:
                # radar_points: [B, N_radar, 3] 或 numpy array [N_radar, 3] - radar点云在LiDAR坐标系
                
                # 转换为tensor（如果是numpy array）
                if isinstance(radar_points_raw, np.ndarray):
                    radar_points = torch.from_numpy(radar_points_raw).float().to(feature.device)
                    # 如果是 [N_radar, 3]，添加batch维度
                    if len(radar_points.shape) == 2:
                        radar_points = radar_points.unsqueeze(0)  # [1, N_radar, 3]
                        # 扩展到batch size
                        if radar_points.shape[0] != b:
                            radar_points = radar_points.expand(b, -1, -1)  # [B, N_radar, 3]
                elif isinstance(radar_points_raw, torch.Tensor):
                    radar_points = radar_points_raw.to(feature.device)
                    # 确保数据类型是float32
                    if radar_points.dtype != torch.float32:
                        radar_points = radar_points.float()
                    # 确保有batch维度
                    if len(radar_points.shape) == 2:
                        radar_points = radar_points.unsqueeze(0)  # [1, N_radar, 3]
                    if radar_points.shape[0] != b:
                        radar_points = radar_points.expand(b, -1, -1)  # [B, N_radar, 3]
                else:
                    raise TypeError(f"radar_points must be numpy array or tensor, got {type(radar_points_raw)}")
            else:
                if not self.radar_depth_fallback:
                    raise ValueError("radar-assisted depth requires at least one valid radar point")
                radar_points = None
            
            if radar_points is not None and radar_points.shape[1] > 0:
                # print("use radar depth")  # removed: spams log every forward pass
                lidar2img = metas["projection_mat"]  # [B, N, 4, 4] - lidar到图像的投影矩阵
                # 确保lidar2img是float32类型，与radar_points一致
                if lidar2img.dtype != radar_points.dtype:
                    lidar2img = lidar2img.to(radar_points.dtype)
                
                # 将radar点云投影到图像坐标系
                radar_pts_homo = torch.cat([
                    radar_points, 
                    torch.ones(b, radar_points.shape[1], 1, device=radar_points.device, dtype=radar_points.dtype)
                ], dim=-1)  # [B, N_radar, 4]
                
                # 投影到图像坐标 (只使用第一个相机的投影矩阵)
                img_pts_homo = lidar2img[:, 0:1, :, :] @ radar_pts_homo.transpose(1, 2)  # [B, 1, 4, N_radar]
                img_pts_homo = img_pts_homo.squeeze(1).transpose(1, 2)  # [B, N_radar, 4]
                
                # 转换为像素坐标和深度
                img_pts = img_pts_homo[..., :2] / (img_pts_homo[..., 2:3] + 1e-6)  # [B, N_radar, 2]
                depths = img_pts_homo[..., 2]  # [B, N_radar] - 深度值
                
                # 过滤：只保留在图像范围内的radar点
                image_wh = metas['image_wh'][:, 0, :]  # [B, 2]
                valid_mask = (
                    (img_pts[..., 0] >= 0) & (img_pts[..., 0] < image_wh[:, 0:1]) &
                    (img_pts[..., 1] >= 0) & (img_pts[..., 1] < image_wh[:, 1:2]) &
                    (depths > 0) & (depths < 100.0)  # 合理深度范围
                )  # [B, N_radar]
                
                # 为每个像素位置构建深度bins
                pixel_coords_flat = uv[:, 0, :, :, :].reshape(b, h * w, 2)  # [B, H*W, 2]
                depth_bins_per_pixel_list = []
                
                for bi in range(b):
                    valid_depths = depths[bi][valid_mask[bi]]  # [N_valid]
                    valid_img_pts = img_pts[bi][valid_mask[bi]]  # [N_valid, 2]
                    
                    if len(valid_depths) == 0:
                        # 没有有效radar点，使用固定depth_bins或回退
                        pass  # print("no valid radar points")  # removed: spams log
                        if self.radar_depth_fallback:
                            depth_bins_per_pixel_list.append(
                                self.depth_bins.unsqueeze(0).expand(h * w, -1).to(feature.device)  # [H*W, num_samples]
                            )
                        else:
                            # 如果不需要回退，使用空深度bins（可能导致错误）
                            depth_bins_per_pixel_list.append(
                                torch.zeros(h * w, self.num_samples, device=feature.device)
                            )
                    else:
                        # 混合深度采样策略：radar bins + 固定 bins
                        # radar_depth_ratio 控制 radar bins 占总 bins 的比例
                        n_radar_bins = max(1, int(self.num_samples * self.radar_depth_ratio))
                        n_fixed_bins = self.num_samples - n_radar_bins
                        
                        # 计算每个像素到radar投影点的距离
                        dists = torch.cdist(
                            pixel_coords_flat[bi],  # [H*W, 2]
                            valid_img_pts,  # [N_valid, 2]
                            p=2
                        )  # [H*W, N_valid]
                        
                        # === Radar bins: 取最近的 n_radar_bins 个 radar 深度 ===
                        k_radar = min(n_radar_bins, len(valid_depths))
                        _, nearest_indices = torch.topk(dists, k_radar, dim=-1, largest=False)  # [H*W, k_radar]
                        radar_bins = valid_depths[nearest_indices]  # [H*W, k_radar]
                        
                        # 如果有效radar点不够 n_radar_bins，用固定bins补齐radar部分
                        if k_radar < n_radar_bins:
                            extra_fixed = self.depth_bins[k_radar:n_radar_bins].unsqueeze(0).expand(h * w, -1).to(feature.device)
                            radar_bins = torch.cat([radar_bins, extra_fixed], dim=-1)  # [H*W, n_radar_bins]
                        
                        # === Fixed bins: 从固定 depth_bins 中均匀采样 n_fixed_bins 个 ===
                        if n_fixed_bins > 0:
                            fixed_indices = torch.linspace(0, self.num_samples - 1, n_fixed_bins, device=feature.device).long()
                            fixed_bins = self.depth_bins[fixed_indices].unsqueeze(0).expand(h * w, -1).to(feature.device)  # [H*W, n_fixed_bins]
                            
                            # 合并 radar bins + fixed bins
                            pixel_depth_bins = torch.cat([radar_bins, fixed_bins], dim=-1)  # [H*W, num_samples]
                        else:
                            pixel_depth_bins = radar_bins  # [H*W, num_samples]
                        
                        # 对深度值进行排序，确保单调递增
                        pixel_depth_bins, _ = torch.sort(pixel_depth_bins, dim=-1)  # [H*W, num_samples]
                        depth_bins_per_pixel_list.append(pixel_depth_bins)
                
                # Stack并reshape: [B, H*W, num_samples] -> [B, N, H, W, num_samples]
                depth_bins_per_pixel = torch.stack(depth_bins_per_pixel_list, dim=0)  # [B, H*W, num_samples]
                depth_bins_per_pixel = depth_bins_per_pixel.reshape(b, 1, h, w, self.num_samples).expand(b, n, h, w, self.num_samples)
            else:
                # radar_points为空或格式不正确，使用固定depth_bins
                if self.radar_depth_fallback:
                    depth_bins_per_pixel = self.depth_bins.view(1, 1, 1, 1, -1).to(feature.device)  # [1, 1, 1, 1, num_samples]
                else:
                    raise ValueError("radar_points is required but not available or invalid")
        else:
            # 使用固定depth_bins
            depth_bins_per_pixel = self.depth_bins.view(1, 1, 1, 1, -1).to(feature.device)  # [1, 1, 1, 1, num_samples]
        
        # 构建UV坐标并添加深度
        uvd = uv.unsqueeze(4).expand(b, n, h, w, self.num_samples, 2)
        uvd1 = torch.cat([uvd, torch.ones_like(uvd)], dim=-1) # b, n, h, w, d, 4
        
        # 使用深度值（与原代码逻辑保持一致）
        # 原代码: uvd1[..., :3] = uvd1[..., :3] * depth_bins
        # 这意味着：uvd1的3D坐标 = [U*D, V*D, D]，然后通过投影矩阵转换
        if len(depth_bins_per_pixel.shape) == 5:  # [B, N, H, W, num_samples]
            # 每个像素有不同的深度值
            depth_bins_expanded = depth_bins_per_pixel.unsqueeze(-1)  # [B, N, H, W, num_samples, 1]
            uvd1[..., :3] = uvd1[..., :3] * depth_bins_expanded
        else:
            # 全局depth_bins（原始逻辑）
            uvd1[..., :3] = uvd1[..., :3] * depth_bins_per_pixel.view(1, 1, 1, 1, -1, 1)
        
        anchor_pts = projection_mat[:, :, None, None, None] @ uvd1[..., None]
        anchor_pts = anchor_pts.squeeze(-1)[..., :3]
        if not self.training or kwargs.get("benchmarking", False):
            anchor_gt = None
        else:
            if "occ_label" not in metas or "occ_cam_mask" not in metas:
                raise KeyError("training requires occ_label and occ_cam_mask for pixel supervision")
            oob_mask = (anchor_pts[..., 0] < self.pc_range[0]) | (anchor_pts[..., 0] >= self.pc_range[3]) | \
                       (anchor_pts[..., 1] < self.pc_range[1]) | (anchor_pts[..., 1] >= self.pc_range[4]) | \
                       (anchor_pts[..., 2] < self.pc_range[2]) | (anchor_pts[..., 2] >= self.pc_range[5])
            anchor_idx = (anchor_pts - self.pc_start.view(1, 1, 1, 1, 1, 3)) / self.voxel_size
            anchor_idx = anchor_idx.to(torch.int)
            anchor_idx[..., 0].clamp_(0, self.occ_resolution[0] - 1)
            anchor_idx[..., 1].clamp_(0, self.occ_resolution[1] - 1)
            anchor_idx[..., 2].clamp_(0, self.occ_resolution[2] - 1)

            occupancy = metas["occ_label"]
            valid_mask = metas["occ_cam_mask"]
            anchor_occ = torch.stack([occ[idx[..., 0], idx[..., 1], idx[..., 2]] for occ, idx in zip(occupancy, anchor_idx)])
            anchor_occ[oob_mask] = self.empty_label
            anchor_valid = torch.stack([occ[idx[..., 0], idx[..., 1], idx[..., 2]] for occ, idx in zip(valid_mask, anchor_idx)])
            anchor_valid[oob_mask] = False
            anchor_gt = (anchor_occ != self.empty_label) & anchor_valid
            anchor_gt = torch.cat([anchor_gt, ~torch.any(anchor_gt, dim=-1, keepdim=True)], dim=-1)
        
        pdfs = torch.softmax(logits, dim=-1)
        # Training may sample from the learned distribution, but validation and
        # inference must be repeatable for checkpoint comparison and reporting.
        deterministic = deterministic_anchor_sampling(
            getattr(self, 'deterministic', True), self.training)
        index, pdf_i = self.sampler.sample(pdfs, deterministic, self.anchors_per_pixel) # b, n, h, w, a
        disable_mask = (pdfs.argmax(dim=-1, keepdim=True) == self.num_samples).expand(
            -1, -1, -1, -1, self.anchors_per_pixel)
        # disable_mask = index == self.num_samples
        sampled_anchor = self.sampler.gather(index.clamp(max=(self.num_samples-1)), anchor_pts) # b, n, h, w, a, 3
        
        # 如果直接用radar点云作为anchor，跳过深度采样流程
        if self.use_radar_as_anchor and 'radar_points' in metas:
            radar_points_raw = metas['radar_points']
            
            # 转换为tensor（如果是numpy array）
            if isinstance(radar_points_raw, np.ndarray):
                radar_points = torch.from_numpy(radar_points_raw).float().to(sampled_anchor.device)
                if len(radar_points.shape) == 2:
                    radar_points = radar_points.unsqueeze(0)  # [1, N_radar, 3]
            elif isinstance(radar_points_raw, torch.Tensor):
                radar_points = radar_points_raw.to(sampled_anchor.device)
                # 确保数据类型是float32
                if radar_points.dtype != torch.float32:
                    radar_points = radar_points.float()
                if len(radar_points.shape) == 2:
                    radar_points = radar_points.unsqueeze(0)  # [1, N_radar, 3]
            else:
                radar_points = None
            
            if radar_points is not None:
                # 确保batch size匹配
                if radar_points.shape[0] != b:
                    if radar_points.shape[0] == 1:
                        radar_points = radar_points.expand(b, -1, -1)
                    else:
                        radar_points = radar_points[:b]
                
                anchor_xyz = []
                for i in range(b):
                    # 每个样本使用自己的radar点云
                    cur_radar = radar_points[i]  # [N_radar, 3]
                    # 过滤pc_range范围内的点
                    oob_mask = (
                        (cur_radar[..., 0] < self.pc_range[0]) | (cur_radar[..., 0] >= self.pc_range[3]) |
                        (cur_radar[..., 1] < self.pc_range[1]) | (cur_radar[..., 1] >= self.pc_range[4]) |
                        (cur_radar[..., 2] < self.pc_range[2]) | (cur_radar[..., 2] >= self.pc_range[5])
                    )
                    valid_radar = cur_radar[~oob_mask]  # [N_valid, 3]
                    
                    if len(valid_radar) == 0:
                        # 如果没有有效radar点，回退到深度采样
                        cur_sampled_anchor = sampled_anchor[i][~disable_mask[i]]
                        cur_oob_mask = (
                            (cur_sampled_anchor[..., 0] < self.pc_range[0]) | (cur_sampled_anchor[..., 0] >= self.pc_range[3]) |
                            (cur_sampled_anchor[..., 1] < self.pc_range[1]) | (cur_sampled_anchor[..., 1] >= self.pc_range[4]) |
                            (cur_sampled_anchor[..., 2] < self.pc_range[2]) | (cur_sampled_anchor[..., 2] >= self.pc_range[5])
                        )
                        scan = cur_sampled_anchor[~cur_oob_mask]
                        # 如果scan为空，使用随机初始化
                        if scan.shape[0] == 0:
                            scan = torch.rand(self.num_anchor, 3, device=cur_radar.device)
                            scan[:, 0] = scan[:, 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                            scan[:, 1] = scan[:, 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
                            scan[:, 2] = scan[:, 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
                    elif len(valid_radar) >= self.num_anchor:
                        # radar点足够，使用FPS采样选择num_anchor个点
                        scanidx = farthest_point_sampling(
                            valid_radar,
                            torch.tensor([len(valid_radar)], device=valid_radar.device, dtype=torch.int),
                            torch.tensor([self.num_anchor], device=valid_radar.device, dtype=torch.int)
                        )
                        scan = valid_radar[scanidx]
                    else:
                        # radar点不足，补充后使用所有点
                        multi = int(math.ceil(self.num_anchor * 1.0 / len(valid_radar))) - 1
                        scan_ = valid_radar.repeat(multi, 1)
                        scan_ = scan_ + torch.randn_like(scan_) * 0.1
                        scan_[:, 0].clamp_(self.pc_range[0], self.pc_range[3])
                        scan_[:, 1].clamp_(self.pc_range[1], self.pc_range[4])
                        scan_[:, 2].clamp_(self.pc_range[2], self.pc_range[5])
                        scan = torch.cat([valid_radar, scan_], dim=0)
                        # 如果还是不够，截取前num_anchor个
                        if len(scan) > self.num_anchor:
                            scanidx = farthest_point_sampling(
                                scan,
                                torch.tensor([len(scan)], device=scan.device, dtype=torch.int),
                                torch.tensor([self.num_anchor], device=scan.device, dtype=torch.int)
                            )
                            scan = scan[scanidx]
                    
                    anchor_xyz.append(scan)
            else:
                # radar_points不可用，回退到深度采样
                self.use_radar_as_anchor = False
        else:
            # 使用深度采样流程（原有逻辑）
            anchor_xyz = []
            for i in range(b):
                cur_sampled_anchor = sampled_anchor[i][~disable_mask[i]]
                cur_oob_mask = (cur_sampled_anchor[..., 0] < self.pc_range[0]) | (cur_sampled_anchor[..., 0] >= self.pc_range[3]) | \
                       (cur_sampled_anchor[..., 1] < self.pc_range[1]) | (cur_sampled_anchor[..., 1] >= self.pc_range[4]) | \
                       (cur_sampled_anchor[..., 2] < self.pc_range[2]) | (cur_sampled_anchor[..., 2] >= self.pc_range[5])
                scan = cur_sampled_anchor[~cur_oob_mask]
                
                # An empty inference scan must not silently become random output.
                if scan.shape[0] == 0:
                    if deterministic:
                        raise RuntimeError("deterministic anchor sampling produced no valid points")
                    print(f"Warning: No valid anchor points found for batch {i}. Using random initialization.")
                    scan = torch.rand(self.num_anchor, 3, device=cur_sampled_anchor.device)
                    scan[:, 0] = scan[:, 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
                    scan[:, 1] = scan[:, 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
                    scan[:, 2] = scan[:, 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
                    anchor_xyz.append(scan)
                    continue
                
                if deterministic:
                    # One candidate is generated per image feature location, so
                    # this normally only fills locations disabled as no-occupancy.
                    # Exact repetition preserves determinism without inventing
                    # jittered geometry at validation/inference time.
                    scan = repeat_anchor_scan(scan, self.num_anchor)
                elif self.random_sampling:
                    if scan.shape[0] < self.num_anchor:
                        multi = int(math.ceil(self.num_anchor * 1.0 / scan.shape[0])) - 1
                        scan_ = scan.repeat(multi, 1)
                        scan_ = scan_ + torch.randn_like(scan_) * 0.1
                        scan_ = scan_[np.random.choice(scan_.shape[0], self.num_anchor - scan.shape[0], False)]
                        scan_[:, 0].clamp_(self.pc_range[0], self.pc_range[3])
                        scan_[:, 1].clamp_(self.pc_range[1], self.pc_range[4])
                        scan_[:, 2].clamp_(self.pc_range[2], self.pc_range[5])
                        scan = torch.cat([scan, scan_], 0)
                    else:
                        scan = scan[np.random.choice(scan.shape[0], self.num_anchor, False)]
                else:
                    if scan.shape[0] < self.num_anchor:
                        multi = int(math.ceil(self.num_anchor * 1.0 / scan.shape[0])) - 1
                        scan_ = scan.repeat(multi, 1)
                        scan_ = scan_ + torch.randn_like(scan_) * 0.1
                        scan_[:, 0].clamp_(self.pc_range[0], self.pc_range[3])
                        scan_[:, 1].clamp_(self.pc_range[1], self.pc_range[4])
                        scan_[:, 2].clamp_(self.pc_range[2], self.pc_range[5])
                        scan = torch.cat([scan, scan_], 0)
                    # breakpoint()
                    if kwargs.get("benchmarking", False):
                        scan = scan[np.random.permutation(scan.shape[0])]
                        num_subsets = 3
                        sublens = torch.linspace(0, scan.shape[0], num_subsets + 1, dtype=torch.int, device=scan.device)[1:]
                        new_sublens = torch.linspace(0, self.num_anchor, num_subsets + 1, dtype=torch.int, device=scan.device)[1:]
                        scanidx = farthest_point_sampling(scan, sublens, new_sublens)
                    else:
                        # breakpoint()
                        scanidx = farthest_point_sampling(
                            scan, 
                            torch.tensor([scan.shape[0]], device=scan.device, dtype=torch.int),
                            torch.tensor([self.num_anchor], device=scan.device, dtype=torch.int))
                        scan = scan[scanidx, :]
                
                anchor_xyz.append(scan)

        anchor_xyz = torch.stack(anchor_xyz)
        anchor_xyz[..., 0] = (anchor_xyz[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        anchor_xyz[..., 1] = (anchor_xyz[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        anchor_xyz[..., 2] = (anchor_xyz[..., 2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
        
        # ========== 时序初始化：混合上一帧和当前帧的xyz ==========
        if use_temporal:
            # 从上一帧的representation中提取xyz（归一化后的参数空间）
            prev_xyz_normalized = prev_representation[..., :3]  # [B, N, 3]
            
            # 获取位姿变换矩阵
            prev_ego2global = metas['prev_ego2global']  # numpy array or tensor
            curr_ego2global = metas['curr_ego2global']  # numpy array or tensor
            
            # 转换为tensor
            if isinstance(prev_ego2global, np.ndarray):
                prev_ego2global_tensor = torch.from_numpy(prev_ego2global).float().to(anchor_xyz.device)
            else:
                prev_ego2global_tensor = prev_ego2global.to(anchor_xyz.device)
            
            if isinstance(curr_ego2global, np.ndarray):
                curr_ego2global_tensor = torch.from_numpy(curr_ego2global).float().to(anchor_xyz.device)
            else:
                curr_ego2global_tensor = curr_ego2global.to(anchor_xyz.device)
            
            # 如果位姿矩阵是[4, 4]，扩展为[B, 4, 4]
            if prev_ego2global_tensor.dim() == 2:
                prev_ego2global_tensor = prev_ego2global_tensor.unsqueeze(0).expand(b, -1, -1)
            if curr_ego2global_tensor.dim() == 2:
                curr_ego2global_tensor = curr_ego2global_tensor.unsqueeze(0).expand(b, -1, -1)
            
            # 确保prev_xyz_normalized的batch size匹配
            if prev_xyz_normalized.shape[0] != b:
                # 如果batch size不匹配，取第一个batch或重复
                if prev_xyz_normalized.shape[0] == 1:
                    prev_xyz_normalized = prev_xyz_normalized.expand(b, -1, -1)
                else:
                    prev_xyz_normalized = prev_xyz_normalized[:b]
            
            # 变换上一帧的xyz到当前帧坐标系（先变换，再匹配）
            prev_xyz_transformed = transform_gaussian_xyz(
                prev_xyz_normalized,
                prev_ego2global_tensor,
                curr_ego2global_tensor,
                torch.tensor(self.pc_range, device=anchor_xyz.device),
                self.xyz_act
            )  # [B, N_prev, 3]
            
            # ========== 关键改进：基于最近邻匹配，而不是直接索引对应 ==========
            # 将归一化坐标转换为真实坐标用于距离计算
            if self.xyz_act == "sigmoid":
                anchor_xyz_sigmoid = torch.sigmoid(anchor_xyz) if anchor_xyz.min() < 0 else anchor_xyz
                prev_xyz_transformed_sigmoid = torch.sigmoid(prev_xyz_transformed) if prev_xyz_transformed.min() < 0 else prev_xyz_transformed
            else:
                anchor_xyz_sigmoid = anchor_xyz
                prev_xyz_transformed_sigmoid = prev_xyz_transformed
            
            # 转换为真实坐标（用于距离计算）
            pc_range_tensor = torch.tensor(self.pc_range, device=anchor_xyz.device)
            anchor_xyz_real = torch.zeros_like(anchor_xyz)
            anchor_xyz_real[..., 0] = anchor_xyz_sigmoid[..., 0] * (pc_range_tensor[3] - pc_range_tensor[0]) + pc_range_tensor[0]
            anchor_xyz_real[..., 1] = anchor_xyz_sigmoid[..., 1] * (pc_range_tensor[4] - pc_range_tensor[1]) + pc_range_tensor[1]
            anchor_xyz_real[..., 2] = anchor_xyz_sigmoid[..., 2] * (pc_range_tensor[5] - pc_range_tensor[2]) + pc_range_tensor[2]
            
            prev_xyz_transformed_real = torch.zeros_like(prev_xyz_transformed)
            prev_xyz_transformed_real[..., 0] = prev_xyz_transformed_sigmoid[..., 0] * (pc_range_tensor[3] - pc_range_tensor[0]) + pc_range_tensor[0]
            prev_xyz_transformed_real[..., 1] = prev_xyz_transformed_sigmoid[..., 1] * (pc_range_tensor[4] - pc_range_tensor[1]) + pc_range_tensor[1]
            prev_xyz_transformed_real[..., 2] = prev_xyz_transformed_sigmoid[..., 2] * (pc_range_tensor[5] - pc_range_tensor[2]) + pc_range_tensor[2]
            
            # 对每个batch和每个当前anchor，找到最近的上一帧高斯
            _, N_curr, _ = anchor_xyz_real.shape
            _, N_prev, _ = prev_xyz_transformed_real.shape
            
            # 计算距离矩阵 [B, N_curr, N_prev]
            # Use torch.cdist which is much more memory-efficient than the naive
            # broadcast approach (avoids materializing [B, N_curr, N_prev, 3] intermediate).
            distances = torch.cdist(anchor_xyz_real, prev_xyz_transformed_real)  # [B, N_curr, N_prev]
            
            # 找到最近邻的索引 [B, N_curr]
            nearest_indices = torch.argmin(distances, dim=-1)  # [B, N_curr]
            
            # 获取最近邻的距离
            nearest_distances = torch.gather(distances, dim=-1, index=nearest_indices.unsqueeze(-1)).squeeze(-1)  # [B, N_curr]
            
            # 创建匹配mask：只有距离小于阈值的才进行融合
            match_mask = nearest_distances < self.temporal_match_threshold  # [B, N_curr]
            
            # 根据匹配方法选择对应的上一帧高斯
            if self.temporal_match_method == 'nearest':
                # 使用最近邻匹配
                # 为每个当前anchor选择对应的上一帧高斯
                batch_indices = torch.arange(b, device=anchor_xyz.device).unsqueeze(1).expand(-1, N_curr)  # [B, N_curr]
                matched_prev_xyz = prev_xyz_transformed[batch_indices, nearest_indices]  # [B, N_curr, 3]
            else:
                # 使用索引对应（不推荐，但保留兼容性）
                if prev_xyz_transformed.shape[1] >= anchor_xyz.shape[1]:
                    matched_prev_xyz = prev_xyz_transformed[:, :anchor_xyz.shape[1], :]
                else:
                    padding = prev_xyz_transformed[:, -1:, :].expand(-1, anchor_xyz.shape[1] - prev_xyz_transformed.shape[1], -1)
                    matched_prev_xyz = torch.cat([prev_xyz_transformed, padding], dim=1)
                match_mask = torch.ones(b, N_curr, dtype=torch.bool, device=anchor_xyz.device)
            
            # 只在匹配mask为True的位置进行融合
            # 对于不匹配的位置，只使用当前帧的anchor
            # Apply warmup scaling to all temporal weights
            eff_xyz_w = self.temporal_init_weight * temporal_scale
            anchor_xyz = torch.where(
                match_mask.unsqueeze(-1),
                (1 - eff_xyz_w) * anchor_xyz + eff_xyz_w * matched_prev_xyz,
                anchor_xyz  # 不匹配的位置，保持当前帧的anchor
            )
        
        # 只有xyz是基于radar随机生成的
        if self.xyz_act == "sigmoid":
            xyz = safe_inverse_sigmoid(anchor_xyz)
        else:
            # loop模式处理
            xy = safe_inverse_sigmoid(anchor_xyz[..., :2])
            z = torch.remainder(anchor_xyz[..., 2:3], 1.0)
            xyz = torch.cat([xy, z], dim=-1)
        
        # ========== 时序初始化：混合其他参数（scale, rotation, opacity, semantic）==========
        if use_temporal:
            # 提取其他参数（scale, rot, opa, sem）
            #
            # 关键维度对齐：
            #   self.anchor = [scale(3), rot(4), opa(0或1), sem(semantic_dim)]  ← 不含xyz，不含velocity
            #   prev_representation (refined_anchor) = [xyz(3), scale(3), rot(4), opa(0或1), sem(semantic_dim), vel(0或3)]
            #
            # 所以：
            #   curr_other_params = self.anchor 整体（已经是 scale+rot+opa+sem）
            #   prev_other_params = prev_representation[..., 3:] 去掉xyz后，还需截断velocity部分
            curr_other_params = torch.tile(self.anchor[None], (b, 1, 1))  # [B, N_curr, scale+rot+opa+sem]
            curr_other_dim = curr_other_params.shape[-1]
            
            # prev_representation[..., 3:] 去掉xyz后是 [scale, rot, opa, sem, vel(可能有)]
            # 只取前 curr_other_dim 维，这样和 curr_other_params 对齐（自动去掉末尾的velocity）
            prev_other_params = prev_representation[..., 3:3+curr_other_dim]  # [B, N_prev, scale+rot+opa+sem]
            
            # 确保batch维度匹配
            if prev_other_params.shape[0] != b:
                if prev_other_params.shape[0] == 1:
                    prev_other_params = prev_other_params.expand(b, -1, -1)
                else:
                    prev_other_params = prev_other_params[:b]
            
            # 使用与xyz相同的匹配索引（如果已经计算过）
            if self.temporal_match_method == 'nearest' and 'nearest_indices' in locals():
                # 使用最近邻匹配：为每个当前anchor选择对应的上一帧参数
                batch_indices = torch.arange(b, device=prev_other_params.device).unsqueeze(1).expand(-1, N_curr)  # [B, N_curr]
                matched_prev_other_params = prev_other_params[batch_indices, nearest_indices]  # [B, N_curr, anchor_dim-3]
            else:
                # 使用索引对应（不推荐）
                if prev_other_params.shape[1] >= curr_other_params.shape[1]:
                    matched_prev_other_params = prev_other_params[:, :curr_other_params.shape[1], :]
                else:
                    padding = prev_other_params[:, -1:, :].expand(-1, curr_other_params.shape[1] - prev_other_params.shape[1], -1)
                    matched_prev_other_params = torch.cat([prev_other_params, padding], dim=1)
                if 'match_mask' not in locals():
                    match_mask = torch.ones(b, N_curr, dtype=torch.bool, device=curr_other_params.device)
            
            # 分别处理不同参数：scale, rotation, opacity, semantic
            # 参数顺序：scale(3), rotation(4), opacity(1或0), semantic(semantic_dim)
            # Apply warmup scaling to all temporal weights
            eff_scale_w = self.temporal_scale_weight * temporal_scale
            eff_rot_w = self.temporal_rotation_weight * temporal_scale
            eff_opa_w = self.temporal_opacity_weight * temporal_scale
            eff_sem_w = self.temporal_init_weight * temporal_scale
            param_idx = 0
            
            # 1. Scale (3维) - 使用匹配的索引
            scale_dim = 3
            prev_scale_matched = matched_prev_other_params[..., param_idx:param_idx+scale_dim]
            curr_scale = curr_other_params[..., param_idx:param_idx+scale_dim]
            mixed_scale = torch.where(
                match_mask.unsqueeze(-1),
                (1 - eff_scale_w) * curr_scale + eff_scale_w * prev_scale_matched,
                curr_scale
            )
            param_idx += scale_dim
            
            # 2. Rotation (4维，四元数) - 使用匹配的索引
            rotation_dim = 4
            prev_rotation_matched = matched_prev_other_params[..., param_idx:param_idx+rotation_dim]
            curr_rotation = curr_other_params[..., param_idx:param_idx+rotation_dim]
            # 注意：旋转四元数需要归一化，混合后需要重新归一化
            mixed_rotation = torch.where(
                match_mask.unsqueeze(-1),
                (1 - eff_rot_w) * curr_rotation + eff_rot_w * prev_rotation_matched,
                curr_rotation
            )
            # 归一化四元数（保持单位长度）
            mixed_rotation = mixed_rotation / (mixed_rotation.norm(dim=-1, keepdim=True) + 1e-8)
            param_idx += rotation_dim
            
            # 3. Opacity (1维或0维) - 使用匹配的索引
            opacity_dim = 1 if self.include_opa else 0
            if opacity_dim > 0:
                prev_opacity_matched = matched_prev_other_params[..., param_idx:param_idx+opacity_dim]
                curr_opacity = curr_other_params[..., param_idx:param_idx+opacity_dim]
                mixed_opacity = torch.where(
                    match_mask.unsqueeze(-1),
                    (1 - eff_opa_w) * curr_opacity + eff_opa_w * prev_opacity_matched,
                    curr_opacity
                )
                param_idx += opacity_dim
            else:
                # opacity_dim == 0，创建空的tensor
                mixed_opacity = torch.empty(*curr_other_params.shape[:-1], 0, device=curr_other_params.device)
            
            # 4. Semantic (semantic_dim维) - 使用匹配的索引
            semantic_dim = self.semantic_dim if self.semantics else 0
            if semantic_dim > 0:
                prev_semantic_matched = matched_prev_other_params[..., param_idx:param_idx+semantic_dim]
                curr_semantic = curr_other_params[..., param_idx:param_idx+semantic_dim]
                if self.reuse_semantic:
                    # 直接复用上一帧的semantic（语义标签应该保持不变），但只在匹配的位置
                    mixed_semantic = torch.where(
                        match_mask.unsqueeze(-1),
                        (1 - eff_sem_w) * curr_semantic + eff_sem_w * prev_semantic_matched,
                        curr_semantic
                    )
                else:
                    # 混合semantic（如果希望允许语义变化）
                    mixed_semantic = torch.where(
                        match_mask.unsqueeze(-1),
                        (1 - eff_sem_w) * curr_semantic + eff_sem_w * prev_semantic_matched,
                        curr_semantic
                    )
            else:
                mixed_semantic = torch.empty(*curr_other_params.shape[:-1], 0, device=curr_other_params.device)
            
            # 拼接所有参数
            mixed_parts = [mixed_scale, mixed_rotation]
            if opacity_dim > 0:
                mixed_parts.append(mixed_opacity)
            if semantic_dim > 0:
                mixed_parts.append(mixed_semantic)
            mixed_other_params = torch.cat(mixed_parts, dim=-1)
            
            anchor = torch.cat([xyz, mixed_other_params], dim=-1)
        else:
            anchor = torch.cat([
                xyz, torch.tile(self.anchor[None], (b, 1, 1))], dim=-1)
        
        if self.random_samples > 0:
            random_anchors = torch.tile(self.random_anchors[None], (b, 1, 1)) #将相同的 random_anchors 复制到 batch 中的每个样本，使每个样本都有相同的随机 anchor 参数。
            anchor = torch.cat([anchor, random_anchors], dim=1)

        # ========== 时序初始化：混合rep_features ==========
        if use_temporal and prev_rep_features is not None:
            # self.instance_feature 包含 num_anchor + random_samples 个特征
            # 时序匹配只对前 num_anchor 个做了（nearest_indices 的形状是 [B, num_anchor]）
            # 所以特征混合也只对前 num_anchor 个做，random_samples 部分保持不变
            curr_rep_features_all = torch.tile(
                self.instance_feature[None], (b, 1, 1)
            )  # [B, num_anchor + random_samples, embed_dims]
            
            # 分离: 前 num_anchor 个参与时序混合，后 random_samples 个不参与
            curr_rep_features = curr_rep_features_all[:, :self.num_anchor, :]  # [B, num_anchor, embed_dims]
            curr_rep_features_random = curr_rep_features_all[:, self.num_anchor:, :]  # [B, random_samples, embed_dims]
            
            # 确保batch维度匹配
            if prev_rep_features.shape[0] != b:
                if prev_rep_features.shape[0] == 1:
                    prev_rep_features = prev_rep_features.expand(b, -1, -1)
                else:
                    prev_rep_features = prev_rep_features[:b]
            
            # 使用与xyz相同的匹配索引（nearest_indices shape: [B, num_anchor]）
            if self.temporal_match_method == 'nearest' and 'nearest_indices' in locals():
                batch_indices = torch.arange(b, device=prev_rep_features.device).unsqueeze(1).expand(-1, N_curr)  # [B, N_curr=num_anchor]
                # 注意：prev_rep_features 可能包含 random_samples 部分，nearest_indices 可以正确索引
                matched_prev_rep_features = prev_rep_features[batch_indices, nearest_indices]  # [B, num_anchor, embed_dims]
            else:
                if prev_rep_features.shape[1] >= curr_rep_features.shape[1]:
                    matched_prev_rep_features = prev_rep_features[:, :curr_rep_features.shape[1], :]
                else:
                    padding = prev_rep_features[:, -1:, :].expand(-1, curr_rep_features.shape[1] - prev_rep_features.shape[1], -1)
                    matched_prev_rep_features = torch.cat([prev_rep_features, padding], dim=1)
                if 'match_mask' not in locals():
                    match_mask = torch.ones(b, N_curr, dtype=torch.bool, device=curr_rep_features.device)
            
            # 只对前 num_anchor 个特征做时序融合
            eff_feat_w = self.temporal_feature_weight * temporal_scale
            mixed_rep_features = torch.where(
                match_mask.unsqueeze(-1),
                (1 - eff_feat_w) * curr_rep_features + eff_feat_w * matched_prev_rep_features,
                curr_rep_features
            )  # [B, num_anchor, embed_dims]
            
            # 拼回 random_samples 部分
            instance_feature = torch.cat([mixed_rep_features, curr_rep_features_random], dim=1)  # [B, num_anchor + random_samples, embed_dims]
        else:
            instance_feature = torch.tile(
                self.instance_feature[None], (b, 1, 1)
            )
        return {
            # rep_features: [B, num_anchor, embed_dims]
            # Anchor点的特征表示，初始化为全零的可学习参数
            # 在后续的Encoder中会被更新，用于编码每个Gaussian anchor的特征
            # 维度: embed_dims (通常为128)
            'rep_features': instance_feature,
            
            # representation: [B, num_anchor, anchor_dim]
            # Gaussian anchor的参数表示，包含：
            #   - xyz坐标 (3维): 归一化后的3D坐标，经过sigmoid逆变换
            #   - scale (3维): Gaussian的尺度参数
            #   - rotation (4维): Gaussian的旋转四元数
            #   - opacity (1维): Gaussian的不透明度
            #   - semantic (semantic_dim维): Gaussian的语义特征
            # 总维度: anchor_dim = 3 + 3 + 4 + 1 + semantic_dim
            # 这个representation会传递给Encoder进行进一步编码和refine
            'representation': anchor,
            
            # anchor_init: [num_anchor, anchor_dim]
            # 第一个batch的anchor参数的副本，用于：
            #   - 初始化参考（可能用于可视化或调试）
            #   - 保存初始anchor状态
            # 注意：只保存第一个batch的anchor，不包含batch维度
            'anchor_init': anchor[0].clone(),
            
            # pixel_logits: [B, N, H, W, num_depth_bins+1]
            # 每个像素位置沿光轴方向的深度分布预测logits
            #   - 前num_depth_bins个值：对应不同深度的占用概率logits
            #   - 最后一个值：表示"无占用"的logits
            # 用于PixelDistributionLoss，学习像素到3D点的深度分布
            # 通过softmax后得到深度分布概率，用于采样3D anchor点
            'pixel_logits': logits,
            
            # pixel_gt: [B, N, H, W, num_depth_bins+1]
            # 每个像素位置沿光轴方向的真实占用分布（二值标签）
            #   - 前num_depth_bins个值：对应深度位置是否有占用（True/False）
            #   - 最后一个值：表示"无占用"类别
            # 通过查询ground truth占用标签生成，用于监督pixel_logits的预测
            # 用于PixelDistributionLoss计算深度分布预测的损失
            'pixel_gt': anchor_gt,
        }
        
        
# 图像特征 → Lifter（初始化）
#     ↓
# 生成初始anchor + 初始feature
#     ↓
# Encoder（学习+refine）
#     ↓
# 更新anchor参数 + 更新feature
#     ↓
# 多层decoder representation
