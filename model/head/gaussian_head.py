import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from ..utils.utils import get_rotation_matrix
try:
    from ...uncertainty.evidential_head import EvidentialHead
except ImportError:
    # Fallback for relative import issues
    from model.uncertainty.evidential_head import EvidentialHead


@MODELS.register_module()
class GaussianHead(BaseTaskHead):
    def __init__(
        self, 
        init_cfg=None,
        apply_loss_type=None,
        num_classes=3,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        dataset_type='kradar',
        empty_label=0,
        use_localaggprob=False,
        use_localaggprob_fast=False,
        combine_geosem=False,
        use_evidential=False,  # 新增：是否使用evidential learning
        evidential_config=None,  # 新增：evidential配置
        class_logit_bias=None,  # 新增：推理阶段 per-class logit 偏置，用于校正类别预测偏差
        **kwargs,
    ):
        super().__init__(init_cfg)
        
        if num_classes != 3 or empty_label != 0 or dataset_type != 'kradar':
            raise ValueError(
                "RIGS release requires num_classes=3, empty_label=0 and "
                "dataset_type='kradar'; "
                f"got {num_classes=}, {empty_label=}, {dataset_type=}"
            )
        self.num_classes = num_classes
        self.use_evidential = use_evidential
        # class_logit_bias: list of floats, length=num_classes, e.g. [0, 0, -1.5, 0, ...]
        # 在 argmax 之前将 prediction 转为 log-odds 后加上此偏置，用于校正训练阶段
        # class_weight 导致的类别预测偏差（如 Sedan 过预测）。仅影响推理，不影响 loss。
        self._class_logit_bias = None
        if class_logit_bias is not None:
            self._class_logit_bias = torch.tensor(class_logit_bias, dtype=torch.float32)
        
        # 初始化Evidential Head（如果启用）
        if self.use_evidential:
            evidential_config = evidential_config or {}
            self.evidential_head = EvidentialHead(
                num_classes=num_classes - 1,
                evidence_scale=evidential_config.get('evidence_scale', 1.0),
                min_alpha=evidential_config.get('min_alpha', 1.0)
            )
        else:
            self.evidential_head = None
        self.use_localaggprob = use_localaggprob
        if use_localaggprob:
            if use_localaggprob_fast:
                import local_aggregate_prob_fast
                self.aggregator = local_aggregate_prob_fast.LocalAggregator(**cuda_kwargs)
            else:
                import local_aggregate_prob
                self.aggregator = local_aggregate_prob.LocalAggregator(**cuda_kwargs)
        else:
            import local_aggregate
            self.aggregator = local_aggregate.LocalAggregator(**cuda_kwargs)
        
        self.combine_geosem = combine_geosem
        # 初始化调试信息
        self._current_epoch = 0
        self._current_iter = 0
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label

        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        elif 'fixed' in apply_loss_type:
            self.apply_loss_type = 'fixed'
            self.fixed_apply_loss_layers = [int(item) for item in apply_loss_type.split('_')[1:]]
            print(f"Supervised fixed layers: {self.fixed_apply_loss_layers}")
        else:
            raise NotImplementedError
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))
        if self.num_classes != 3:
            raise ValueError(f"RIGS GaussianHead requires 3 output classes, got {self.num_classes}")

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        """
        对ground truth数据进行采样/展平处理
        
        功能：将occupancy grid格式的GT数据转换为点云格式，便于后续计算loss
        
        输入：
            gt_xyz: [B, H, W, D, 3] - occupancy grid中的3D坐标（每个voxel的中心坐标）
            gt_label: [B, H, W, D] - occupancy grid中的类别标签
            gt_mask: [B, H, W, D] - 可选，掩码（True表示有效voxel）
        
        输出：
            gt_xyz: [B, N] 或 [1, N] - 展平后的3D坐标（如果是masked模式，只在有效位置）
            gt_label: [B, N] 或 [1, N] - 展平后的类别标签
        """
        if gt_mask is None:
            # ========== 模式1: 展平所有voxel ==========
            # 将occupancy grid展平为点列表
            # gt_label: [B, H, W, D] → [B, H*W*D]
            gt_label = gt_label.flatten(1)  # [B, N] where N = H*W*D
            
            # gt_xyz: [B, H, W, D, 3] → [B, H*W*D, 3]
            gt_xyz = gt_xyz.flatten(1, 3)  # [B, N, 3] where N = H*W*D
        else:
            # ========== 模式2: 只保留有效voxel（masked）==========
            # 只保留mask为True的voxel，减少计算量
            # 注意：masked模式只支持batch_size=1
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            
            # gt_label: [1, H, W, D] → [1, N_valid] (N_valid = 有效voxel数量)
            gt_label = gt_label[gt_mask].reshape(1, -1)  # [1, N_valid]
            
            # gt_xyz: [1, H, W, D, 3] → [1, N_valid, 3]
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)  # [1, N_valid, 3]
        
        return gt_xyz, gt_label

    def _inference_grid(self, batch_size, device):
        """Build the canonical full grid without consulting ground truth."""
        pc_min = self.aggregator.pc_min.reshape(-1).to(device=device)
        grid_size = float(self.aggregator.grid_size)
        x = pc_min[0] + (torch.arange(self.aggregator.H, device=device) + 0.5) * grid_size
        y = pc_min[1] + (
            torch.arange(self.aggregator.W - 1, -1, -1, device=device) + 0.5
        ) * grid_size
        z = pc_min[2] + (torch.arange(self.aggregator.D, device=device) + 0.5) * grid_size
        grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

    def prepare_gaussian_args(self, gaussians):
        """
        准备Gaussian参数用于渲染
        
        功能：从GaussianPrediction对象中提取参数，进行维度调整和变换，计算协方差矩阵的逆
        
        输入：
            gaussians: GaussianPrediction对象，包含：
                - means: [B, G, 3] - Gaussian中心位置
                - scales: [B, G, 3] - Gaussian的scale（3个轴的标准差）
                - rotations: [B, G, 4] - 旋转四元数 [w, x, y, z]
                - opacities: [B, G, 1] - 不透明度
                - semantics: [B, G, C] - 语义logits（注意：代码中opacities实际是semantics）
        
        过程：
            1. 提取Gaussian参数
            2. 处理semantics维度（自动调整以匹配num_classes）
            3. 添加empty类别（如果需要）
            4. 计算协方差矩阵的逆（用于Gaussian渲染）
        
        输出：
            means: [B, G, 3] - Gaussian中心位置
            origi_opa: [B, G] - 原始不透明度（展平）
            opacities: [B, G, C] - 语义概率（包含empty类别）
            scales: [B, G, 3] - Gaussian的scale
            CovInv: [B, G, 3, 3] - 协方差矩阵的逆
        """
        # ========== Step 1: 提取Gaussian参数 ==========
        means = gaussians.means        # [B, G, 3] - Gaussian中心位置（3D坐标）
        scales = gaussians.scales      # [B, G, 3] - Gaussian的scale（3个轴的标准差）
        rotations = gaussians.rotations # [B, G, 4] - 旋转四元数 [w, x, y, z]
        opacities = gaussians.semantics # [B, G, C] - 语义logits（注意：命名有误，实际是semantics）
        origi_opa = gaussians.opacities # [B, G, 1] - 原始不透明度
        velocities = getattr(gaussians, 'velocities', None) # [B, G, 3] - 速度向量
        
        # Native RIGS keeps only the two non-empty semantic channels here.
        # Empty is produced by the geometric occupancy probability after CUDA
        # aggregation and must never be passed through the semantic kernel.
        if self.num_classes != 3 or opacities.shape[-1] != 2:
            raise ValueError(
                f"RIGS requires three voxel classes and two Gaussian semantic channels; "
                f"got num_classes={self.num_classes}, semantics={opacities.shape}"
            )
        
        # ========== Step 3: 处理原始不透明度 ==========
        # 如果opacities为空（某些配置下），初始化为全1
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)  # [B, G, 1]
        
        # ========== Step 4: 处理empty类别 ==========
        if self.with_emtpy:
            raise ValueError("RIGS represents empty geometrically; with_empty must be False")
        elif self.use_localaggprob:
            opacities = opacities.softmax(dim=-1)

        # ========== Step 6: 计算协方差矩阵的逆 ==========
        # Gaussian的协方差矩阵：Cov = M^T * M
        # 其中 M = S * R (scale矩阵 × 旋转矩阵)
        # CovInv用于Gaussian渲染中的距离计算
        bs, g, _ = means.shape  # g可能是G或G+1（取决于是否添加了empty）
        
        # 6.1 构建scale矩阵 S（对角矩阵）
        # S = diag(scale_x, scale_y, scale_z)
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)  # [B, G, 3, 3]
        S[..., 0, 0] = scales[..., 0]  # scale_x
        S[..., 1, 1] = scales[..., 1]  # scale_y
        S[..., 2, 2] = scales[..., 2]  # scale_z
        
        # 6.2 将四元数转换为旋转矩阵 R
        R = get_rotation_matrix(rotations)  # [B, G, 3, 3] - 旋转矩阵
        
        # 6.3 计算变换矩阵 M = S * R
        # 先应用scale，再应用旋转
        M = torch.matmul(S, R)  # [B, G, 3, 3]
        
        # 6.4 计算协方差矩阵 Cov = M^T * M
        # 这是Gaussian在3D空间中的协方差矩阵
        Cov = torch.matmul(M.transpose(-1, -2), M)  # [B, G, 3, 3]
        
        # 6.5 计算协方差矩阵的逆 CovInv
        # 注意：linalg.inv 不支持 float16，需先转 float32 再求逆（AMP 下 Cov 可能为 Half）
        cov_f32 = Cov.float()
        CovInv = cov_f32.inverse().to(dtype=Cov.dtype)  # [B, G, 3, 3]，求逆后恢复原 dtype
        
        # ========== Step 7: 计算语义不确定性（如果启用evidential learning）==========
        uncertainties = None
        evidential_alphas = None
        
        if self.use_evidential and self.evidential_head is not None:
            # print(f"Using evidential learning for uncertainty calculation")
            # 使用evidential learning计算不确定性
            # 注意：opacities此时已经是概率分布（如果use_localaggprob）或logits
            # 如果是概率，需要先转换为logits（使用logit函数）
            if opacities.min() >= 0 and opacities.max() <= 1:
                # 是概率分布，转换为logits（float32 下算，AMP 时 float16 易产生 NaN）
                p = opacities.float().clamp(1e-7, 1 - 1e-7)
                opacities_logits = (torch.log(p) - torch.log(1 - p)).to(opacities.dtype)
            else:
                # 已经是logits
                opacities_logits = opacities
            
            if opacities.shape[-1] != 2:
                raise ValueError("Gaussian evidential posterior must have two non-empty channels")
            evidential_alphas, uncertainties = self.evidential_head(opacities_logits)
        else:
            print(f"Using entropy for uncertainty calculation")
            # 使用entropy作为不确定性代理（fallback）
            if opacities.min() >= 0 and opacities.max() <= 1:
                # 已经是概率分布
                probs = opacities
            else:
                # 转换为概率
                probs = F.softmax(opacities, dim=-1)
            
            # 计算entropy作为不确定性（float32 下算，AMP 时 float16 的 log 易产生 NaN/Inf）
            probs_f = probs.float().clamp(1e-7, 1.0)
            entropy = -torch.sum(probs_f * torch.log(probs_f), dim=-1)  # [B, G]
            max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32, device=probs.device))
            uncertainties = (entropy / (max_entropy + 1e-8)).to(probs.dtype).clamp(min=0.0, max=1.0)
        
        # ========== 返回处理后的Gaussian参数 ==========
        return means, origi_opa, opacities, scales, CovInv, uncertainties, evidential_alphas, velocities

    def forward(
        self,
        representation,  # List[{'gaussian': GaussianPrediction}] - 来自encoder的Gaussian预测
        metas=None,      # dict - 元数据（包含occ_xyz, occ_label等）
        **kwargs
    ):
        """
        GaussianHead的前向传播流程
        
        功能：将Gaussian参数渲染为occupancy grid预测
        
        输入：
            representation: List[{'gaussian': GaussianPrediction}]
                - 每个元素对应一个decoder层的Gaussian输出
                - 长度 = num_decoder（decoder层的数量）
            metas: dict
                - 'occ_xyz': [B, H, W, D, 3] - occupancy grid中的3D坐标
                - 'occ_label': [B, H, W, D] - occupancy grid中的类别标签
                - 'occ_cam_mask': [B, H, W, D] - 相机掩码（可选）
        
        过程：
            1. 确定要计算loss的decoder层
            2. 从metas中获取ground truth数据
            3. 对每个选中的decoder层：
               a. 准备Gaussian参数
               b. 使用LocalAggregator渲染为occupancy预测
               c. 处理输出格式
            4. 生成最终预测
        
        输出：
            dict包含：
                - 'pred_occ': List[[B, N, C]] - 每个decoder层的occupancy预测
                - 'bin_logits': List[[B, 1, N]] - 二值occupancy logits（如果use_localaggprob）
                - 'density': List[[B, 1, N]] - density预测（如果use_localaggprob）
                - 'sampled_label': [B, N] - 采样后的GT标签
                - 'sampled_xyz': [B, N, 3] - 采样后的GT坐标
                - 'final_occ': [B, N] - 最终预测的类别
                - 'gaussian': GaussianPrediction - 最后一层的Gaussian
        """
        # ========== Step 1: 确定要计算loss的decoder层 ==========
        num_decoder = len(representation)  # decoder层的数量
        
        if not self.training:
            # 测试时：只使用最后一层的输出
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            # 训练时：对所有decoder层计算loss
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            # 训练时：随机选择一些decoder层计算loss（包括最后一层）
            if self.random_apply_loss_layers > 1:
                # 随机选择num_layers-1层（不包括最后一层）
                apply_loss_layers = np.random.choice(num_decoder - 1, self.random_apply_loss_layers - 1, False)
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]  # 确保包含最后一层
            else:
                apply_loss_layers = [num_decoder - 1]  # 只使用最后一层
        elif self.apply_loss_type == 'fixed':
            # 训练时：使用固定指定的decoder层
            apply_loss_layers = self.fixed_apply_loss_layers
        else:
            raise NotImplementedError

        # ========== Step 2: 初始化输出列表 ==========
        prediction = []  # 存储每个decoder层的occupancy预测
        velocity_prediction = [] # 存储每个decoder层的velocity预测
        bin_logits = []  # 存储每个decoder层的二值occupancy logits（use_localaggprob模式）
        density = []     # 存储每个decoder层的density预测（use_localaggprob模式）
        
        # Inference builds the canonical full grid without reading ground truth.
        batch_size = representation[-1]['gaussian'].means.shape[0]
        if 'occ_xyz' in metas:
            occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        else:
            if self.training:
                raise KeyError('training requires occ_xyz')
            occ_xyz = self._inference_grid(batch_size, self.zero_tensor.device)
        expected = (batch_size, self.aggregator.H, self.aggregator.W, self.aggregator.D, 3)
        if tuple(occ_xyz.shape) != expected:
            raise ValueError(f'occ_xyz must have shape {expected}, got {tuple(occ_xyz.shape)}')
        sampled_xyz = occ_xyz.flatten(1, 3)

        occ_label = metas.get('occ_label')
        if occ_label is None:
            if self.training:
                raise KeyError('training requires occ_label')
            sampled_label = None
        else:
            occ_label = occ_label.to(self.zero_tensor.device)
            if tuple(occ_label.shape) != expected[:-1]:
                raise ValueError(f'occ_label must have shape {expected[:-1]}, got {tuple(occ_label.shape)}')
            sampled_label = occ_label.flatten(1)
            if sampled_label.numel() and (sampled_label.min() < 0 or sampled_label.max() > 2):
                raise ValueError('occ_label must use RIGS classes 0, 1, 2')

        occ_cam_mask = metas.get('occ_cam_mask')
        if occ_cam_mask is None:
            occ_cam_mask = torch.ones(expected[:-1], dtype=torch.bool, device=self.zero_tensor.device)
        else:
            occ_cam_mask = occ_cam_mask.to(self.zero_tensor.device)

        # ========== Step 5: 对每个选中的decoder层进行渲染 ==========
        for idx in apply_loss_layers:
            # 5.1 获取当前decoder层的Gaussian预测
            gaussians = representation[idx]['gaussian']  # GaussianPrediction对象

            # 5.2 准备Gaussian参数（维度调整、计算协方差矩阵逆等）
            means, origi_opa, opacities, scales, CovInv, uncertainties, evidential_alphas, velocities = self.prepare_gaussian_args(gaussians)
            # means: [B, G, 3]
            # origi_opa: [B, G, 1]
            # opacities: [B, G, 2] (background/foreground only)
            # scales: [B, G, 3]
            # CovInv: [B, G, 3, 3]
            # uncertainties: [B, G] - 语义不确定性（新增）
            # evidential_alphas: [B, G, C] - Dirichlet参数（新增，如果使用evidential learning）
            # velocities: [B, G, 3] - 速度向量 (新增)
            
            bs, g = means.shape[:2]  # bs = batch_size, g = num_gaussians

            # 5.3 使用LocalAggregator进行Gaussian渲染
            # 将3D Gaussian渲染为occupancy grid预测
            # 
            # 输入：
            #   - sampled_xyz: [B, N, 3] - 要预测的3D点坐标（occupancy grid的voxel中心）
            #   - means: [B, G, 3] - Gaussian中心位置
            #   - origi_opa: [B, G] - Gaussian不透明度（展平）
            #   - opacities: [B, G, C] - Gaussian语义概率
            #   - scales: [B, G, 3] - Gaussian scale
            #   - CovInv: [B, G, 3, 3] - 协方差矩阵的逆
            #
            # 输出（取决于use_localaggprob）：
            #   - 如果use_localaggprob=False:
            #     semantics: [1, C, N] - 每个voxel的语义预测
            #   - 如果use_localaggprob=True:
            #     semantics[0]: [N, C] 或 [C, N] - 语义预测
            #     semantics[1]: [N] 或 [1, N] - 二值occupancy logits
            #     semantics[2]: [N] 或 [1, N] - density预测
            # local_aggregate_prob CUDA 扩展只支持 float32，AMP 下需转成 float32 再调，输出再转回原 dtype
            out_dtype = means.dtype
            semantics = self.aggregator(
                sampled_xyz.clone().float(),              # [B, N, 3]
                means.float(),                            # [B, G, 3]
                origi_opa.reshape(bs, g).float(),         # [B, G]
                opacities.float(),                        # [B, G, C]
                scales.float(),                           # [B, G, 3]
                CovInv.float()                            # [B, G, 3, 3]
            )
            if out_dtype not in (torch.float32, torch.float):
                if isinstance(semantics, (list, tuple)):
                    semantics = tuple(s.to(dtype=out_dtype) if isinstance(s, torch.Tensor) and s.is_floating_point() else s for s in semantics)
                elif isinstance(semantics, torch.Tensor) and semantics.is_floating_point():
                    semantics = semantics.to(dtype=out_dtype)
            # ========== Step 6: 处理aggregator的输出 ==========
            
            # ========== 新增：Velocity渲染（纯 PyTorch scatter 实现）==========
            # localagg_prob_fast 的 CUDA kernel 仅适用于 semantic+bin+density 输出，
            # 不能直接用于任意维度的特征聚合。
            # 这里用 scatter_add + opacity 加权实现最近邻 Gaussian splatting。
            # 注意：不能使用 in-place 操作（scatter_add_）修改需要梯度的张量视图，
            #       否则会导致 autograd 报 "inplace operation" 错误。
            if velocities is not None:
                N = sampled_xyz.shape[1]  # H*W*D
                vel_results = []
                
                for bi in range(bs):
                    means_b = means[bi]        # [G, 3]
                    vel_b = velocities[bi]     # [G, 3]
                    with torch.no_grad():
                        opa_b = origi_opa[bi].reshape(-1).clamp(0.0, 1.0)
                    
                    # 将 Gaussian 中心映射到体素索引
                    grid_size = self.grid_size if hasattr(self, 'grid_size') else 0.4
                    pc_min = self.aggregator.pc_min if hasattr(self.aggregator, 'pc_min') else sampled_xyz[bi].min(dim=0)[0]
                    voxel_idx = ((means_b.detach() - pc_min) / grid_size).long()  # [G, 3]
                    
                    H_dim = self.aggregator.H if hasattr(self.aggregator, 'H') else 128
                    W_dim = self.aggregator.W if hasattr(self.aggregator, 'W') else 128
                    D_dim = self.aggregator.D if hasattr(self.aggregator, 'D') else 20
                    
                    # 过滤越界的 Gaussian
                    valid = (voxel_idx[:, 0] >= 0) & (voxel_idx[:, 0] < H_dim) & \
                            (voxel_idx[:, 1] >= 0) & (voxel_idx[:, 1] < W_dim) & \
                            (voxel_idx[:, 2] >= 0) & (voxel_idx[:, 2] < D_dim)
                    
                    vel_dense = torch.zeros(N, 3, device=means.device, dtype=means.dtype)
                    if valid.any():
                        vi = voxel_idx[valid]  # [V, 3]
                        # ========== 关键修复：Y 轴翻转 ==========
                        # OCC GT 在 LoadOccupancyKRadar 中对 Y 轴做了 flip（axis=1），
                        # 导致 sampled_label/sampled_xyz 中 j=0 对应 y_max, j=W-1 对应 y_min。
                        # velocity scatter 渲染必须使用翻转后的 Y 索引才能与 OCC 对齐。
                        vi[:, 1] = W_dim - 1 - vi[:, 1]
                        flat_idx = vi[:, 0] * (W_dim * D_dim) + vi[:, 1] * D_dim + vi[:, 2]  # [V]
                        # E1: 速度幅值加权 scatter —— 让有速度的 Gaussian 贡献大，零速度贡献小
                        with torch.no_grad():
                            vel_mag = vel_b[valid].norm(dim=1, keepdim=True).clamp(min=0.01)  # [V, 1]
                        vel_weight = opa_b[valid].unsqueeze(-1) * vel_mag  # [V, 1]
                        weighted_vel = vel_b[valid] * vel_weight  # [V, 3]
                        
                        # 使用非 in-place scatter_add
                        vel_dense = vel_dense.scatter_add(0, flat_idx.unsqueeze(-1).expand(-1, 3), weighted_vel)
                        weight_sum = torch.zeros(N, 1, device=means.device, dtype=means.dtype)
                        weight_sum = weight_sum.scatter_add(0, flat_idx.unsqueeze(-1), vel_weight)
                        # 归一化（避免除零）
                        vel_dense = vel_dense / weight_sum.clamp(min=1e-6)
                    
                    vel_results.append(vel_dense)
                
                # [B, N, 3] -> [B, 3, N]
                pred_vel_dense = torch.stack(vel_results, dim=0)  # [B, N, 3]
                velocity_prediction.append(pred_vel_dense.transpose(1, 2))  # [B, 3, N]
                
                # 诊断日志
                if not hasattr(self, '_vel_render_count'):
                    self._vel_render_count = 0
                self._vel_render_count += 1
                if self._vel_render_count % 20 == 1:
                    import logging
                    _logger = logging.getLogger('mmengine')
                    with torch.no_grad():
                        _nz = (pred_vel_dense.abs().sum(dim=-1) > 1e-8).float().mean()
                        _logger.info(
                            f'[VelRender Diag] call={self._vel_render_count}, '
                            f'dense_vel: mean_abs={pred_vel_dense.abs().mean():.6f} '
                            f'max={pred_vel_dense.abs().max():.6f} '
                            f'non_zero_frac={_nz:.4f}, '
                            f'#valid_gaussians={valid.sum().item()}/{len(valid)}, '
                            f'opa_mean={opa_b.mean():.4f} opa_max={opa_b.max():.4f}'
                        )
            else:
                velocity_prediction.append(None)

            if self.use_localaggprob:
                # ========== 模式：使用概率聚合（use_localaggprob=True）==========
                # aggregator输出是一个tuple：
                #   semantics[0]: 语义预测 [N, C] 或 [C, N]
                #   semantics[1]: 二值occupancy logits [N] 或 [1, N]
                #   semantics[2]: density预测 [N] 或 [1, N]
                
                # 6.1 处理语义预测的维度
                # aggregator输出的semantics[0]形状可能是 (n, c) 或 (c, n)，统一为 (c, n)
                # if len(semantics[0].shape) == 2:
                #     if semantics[0].shape[0] > semantics[0].shape[1]:
                #         # (n, c) -> (c, n)
                #         sem_0 = semantics[0].transpose(0, 1)  # [C, N]
                #     else:
                #         # 已经是 (c, n)
                #         sem_0 = semantics[0]  # [C, N]
                # else:
                #     sem_0 = semantics[0]  # 其他维度情况（通常是 [C, N]）
                
                # 6.2 处理geometric和semantic的组合
                if self.combine_geosem:
                    # ========== 模式：组合几何和语义预测 ==========
                    # sem_0: [C, N] - 语义预测（包含empty类别）
                    # semantics[1]: [N] 或 [1, N] - 几何occupancy logits（是否有物体）
                    
                    # 6.2.1 统一semantics[1]的维度
                    # if len(semantics[1].shape) == 1:
                    #     # (N,) -> (1, N)
                    #     sem_1 = semantics[1].unsqueeze(0)  # [1, N]
                    # else:
                    #     sem_1 = semantics[1]  # [1, N] 或其他
                    
                    # CUDA returns two non-empty probabilities. Its second
                    # output is already geometric occupancy probability.
                    bin_occupancy_prob = semantics[1]
                    if semantics[0].shape[-1] != 2:
                        raise ValueError(
                            f"CUDA semantic output must be [N,2], got {semantics[0].shape}"
                        )
                    sem = semantics[0] * bin_occupancy_prob.unsqueeze(-1)
                    geo = 1 - bin_occupancy_prob.unsqueeze(-1)
                    geosem = torch.cat([geo, sem], dim=-1)
                    
                    # 6.2.3 确保geosem的类别数等于num_classes
                    # assert geosem.shape[0] == self.num_classes, f"geosem类别数({geosem.shape[0]}) != num_classes({self.num_classes})"
                    # if geosem.shape[0] != self.num_classes:
                    #     print(f"Warning: geosem类别数({geosem.shape[0]}) != num_classes({self.num_classes}), 调整中...")
                    #     if geosem.shape[0] > self.num_classes:
                    #         geosem = geosem[:self.num_classes, :]  # 截断
                    #     elif geosem.shape[0] < self.num_classes:
                    #         empty_channel = torch.zeros_like(geosem[:1, :])
                    #         geosem = torch.cat([geosem, empty_channel], dim=0)  # 填充
                else:
                    # ========== 模式：直接使用语义预测（不组合geo）==========
                    geosem = sem_0  # [C, N] - 直接使用语义预测
                    
                    # 确保geosem的类别数等于num_classes
                    if geosem.shape[-1] != self.num_classes:
                        raise ValueError(
                            f"aggregated semantic dimension must be {self.num_classes}, "
                            f"got {geosem.shape}"
                        )
                
                if geosem.ndim != 2 or geosem.shape[-1] != 3:
                    raise ValueError(f"combined occupancy output must be [N,3], got {geosem.shape}")

                # 6.3 转换geosem维度为最终格式
                # geosem当前: [N, C]（从409行的cat操作得到）
                # 目标格式: [B, C, N] (B=1) - 用于loss计算（loss期望[B,C,N]格式）
                # 转换: [N, C] -> [1, N, C] -> [1, C, N]
                prediction.append(geosem[None].transpose(1, 2))  # [1, C, N] - 最终格式为[B,C,N]
                bin_logits.append(semantics[1][None])            # [1, N] 
                density.append(semantics[2][None])               # [1, N]
                # if len(geosem.shape) == 2:
                #     prediction.append(geosem.unsqueeze(0).transpose(1, 2))  # [1, N, C]
                # elif len(geosem.shape) == 3:
                #     # 已经是 (1, C, N) 或 (B, C, N)，转置为 (B, N, C)
                #     prediction.append(geosem.transpose(-2, -1))  # [B, N, C]
                # else:
                #     raise ValueError(f"Unexpected geosem shape: {geosem.shape}")
                
                # # 6.4 处理bin_logits和density（用于loss计算）
                # # bin_logits: 二值occupancy logits（是否有物体）
                # if len(semantics[1].shape) == 1:
                #     bin_logits.append(semantics[1].unsqueeze(0).unsqueeze(0))  # [N] -> [1, 1, N]
                # elif len(semantics[1].shape) == 2:
                #     bin_logits.append(semantics[1].unsqueeze(0))  # [1, N] -> [1, 1, N]
                # else:
                #     bin_logits.append(semantics[1])  # 保持原样
                
                # # density: density预测
                # if len(semantics[2].shape) == 1:
                #     density.append(semantics[2].unsqueeze(0).unsqueeze(0))  # [N] -> [1, 1, N]
                # elif len(semantics[2].shape) == 2:
                #     density.append(semantics[2].unsqueeze(0))  # [1, N] -> [1, 1, N]
                # else:
                #     density.append(semantics[2])  # 保持原样
            else:
                # ========== 模式：不使用概率聚合（use_localaggprob=False）==========
                # semantics: [1, C, N] - 直接使用aggregator输出
                # 转换: [1, C, N] -> [1, N, C]
                prediction.append(semantics[None].transpose(1, 2))  # [1, N, C]
        
        # ========== Step 6.5: Compute voxel-level evidential alphas for loss ==========
        # 将最后一层 decoder 的 per-voxel prediction 转换为 Dirichlet alpha 参数，
        # 供 EvidentialLoss 训练。这样 evidential loss 的梯度可以回传到整个 pipeline。
        voxel_evidential_alphas = None
        if self.use_evidential and self.evidential_head is not None and len(prediction) > 0:
            try:
                # The Dirichlet posterior is defined only over non-empty
                # background/foreground semantics. Empty remains geometric.
                last_pred_bnc = semantics[0].unsqueeze(0)
                if last_pred_bnc.shape[-1] != 2:
                    raise ValueError(f'voxel non-empty semantics must be [B,N,2], got {last_pred_bnc.shape}')
                
                ev_scale = self.evidential_head.evidence_scale
                min_alpha_v = self.evidential_head.min_alpha
                
                # 检测是否是概率（combine_geosem 模式输出 [0,1] 的准概率）
                with torch.no_grad():
                    pred_min = last_pred_bnc.min().item()
                    pred_max = last_pred_bnc.max().item()
                
                if pred_min >= -0.01 and pred_max <= 1.01:
                    # 概率 → logits: log(p / (1-p))
                    p = last_pred_bnc.clamp(min=1e-6, max=1.0 - 1e-6)
                    voxel_logits = torch.log(p / (1.0 - p))
                else:
                    # 已经是 logits
                    voxel_logits = last_pred_bnc
                
                # softplus → evidence → Dirichlet alpha
                voxel_evidence = F.softplus(voxel_logits) * ev_scale
                voxel_evidential_alphas = voxel_evidence + min_alpha_v  # [B, N, C]
            except Exception:
                raise
        
        # ========== Step 7: 生成最终预测 ==========
        if self.use_localaggprob and not self.combine_geosem:
            # ========== 模式：使用概率聚合但不组合geo ==========
            # 在这种情况下，需要结合语义预测和几何occupancy预测
            threshold = kwargs.get("sigmoid_thresh", 0.5)  # 默认阈值0.5
            
            # 7.1 从语义预测中获取类别（argmax）
            # prediction[-1]: [B, C, N] - 最后一层的预测（根据445行，geosem[None].transpose(1,2)得到[B,C,N]）
            final_semantics = prediction[-1].argmax(dim=1)  # [B, N] - 预测的类别索引（在C维度上argmax）
            
            # 7.2 从bin_logits中获取occupancy（是否有物体）
            # bin_logits[-1]: [1, N] 或 [B, 1, N] - 二值occupancy logits（logit值，需要sigmoid转换为概率）
            bin_logits_raw = bin_logits[-1]  # [1, N] 或 [B, 1, N]
            # 统一维度为[1, N]或[B, N]（去掉多余的维度）
            if len(bin_logits_raw.shape) == 3:
                # [B, 1, N] -> [B, N]
                bin_logits_flat = bin_logits_raw.squeeze(1)
            elif len(bin_logits_raw.shape) == 2:
                # [1, N] -> 保持[1, N]
                bin_logits_flat = bin_logits_raw
            else:
                # [N] -> [1, N]
                bin_logits_flat = bin_logits_raw.unsqueeze(0) if len(bin_logits_raw.shape) == 1 else bin_logits_raw
            
            # ⚠️ 关键修复：将logits转换为概率（bin_logits是logit值，必须先sigmoid转换为概率[0,1]）
            # 之前错误地直接比较 logit > threshold，这是错误的！
            bin_probs = torch.sigmoid(bin_logits_flat)  # [1, N] 或 [B, N] - 概率值[0, 1]
            final_occupancy = bin_probs > threshold  # [1, N] 或 [B, N] - 是否有物体（布尔）
            
            # 7.3 组合语义和occupancy
            # 初始化为empty_label（没有物体）
            final_prediction = torch.ones_like(final_semantics) * self.empty_label  # [B, N]
            # 在occupancy为True的位置，使用语义预测的类别
            final_prediction[final_occupancy] = final_semantics[final_occupancy]  # [B, N]
        else:
            # ========== 模式：直接使用语义预测 ==========
            # prediction[-1]: [B, C, N] - 最后一层的预测（根据445/476行，transpose后得到[B,C,N]）
            pred_logits = prediction[-1]  # [B, C, N]
            
            # 应用 class_logit_bias（仅推理阶段，不影响 loss 计算）
            if self._class_logit_bias is not None and not self.training:
                bias = self._class_logit_bias.to(pred_logits.device)  # [C]
                # prediction[-1] 在 combine_geosem 模式下是概率值 [0,1]
                # 转换为 log-odds → 加 bias → argmax
                p = pred_logits.clamp(min=1e-6, max=1.0 - 1e-6)
                pred_logits = torch.log(p / (1.0 - p)) + bias.view(1, -1, 1)
            
            final_prediction = pred_logits.argmax(dim=1)  # [B, N] - 预测的类别索引（在C维度上argmax）
        
        # ========== 注释：现在mask已经改为全True（包含所有体素），所以不需要强制设置empty ==========
        # 如果mask是全True，那么所有区域都会参与训练，模型会学习预测empty和非empty
        # 如果未来需要使用相机可见性mask，可以在这里添加相应的逻辑
        # if occ_cam_mask is not None:
        #     occ_mask_flat = occ_cam_mask.flatten(1)  # [B, H, W, D] -> [B, N]
        #     # 对不可见区域（occ_mask_flat=False）强制设为empty
        #     final_prediction = torch.where(occ_mask_flat, final_prediction, 
        #                                   torch.full_like(final_prediction, self.empty_label))
        
        # ========== Step 8: 调试输出：final_prediction的统计信息 ==========
        # 只在特定条件下输出（避免过于频繁）
        import os
        debug_final = os.environ.get('DEBUG_FINAL_PRED', 'false').lower() == 'true'
        if debug_final:
            try:
                epoch = getattr(self, '_current_epoch', 0)
                iter_idx = getattr(self, '_current_iter', 0)
                # 只在最开始几次和特定epoch输出
                if (epoch == 0 and iter_idx < 5) or (iter_idx % 50 == 0):
                    print(f"\n[DEBUG final_prediction] Epoch {epoch}, Iter {iter_idx}")
                    if len(final_prediction.shape) >= 2 and final_prediction.shape[0] > 0:
                        final_pred_batch = final_prediction[0]  # [N]
                        print(f"  final_prediction shape: {final_prediction.shape}")
                        print(f"  final_prediction value range: [{final_pred_batch.min()}, {final_pred_batch.max()}]")
                        print(f"  final_prediction unique values: {torch.unique(final_pred_batch).tolist()}")
                        
                        # 统计预测类别分布
                        for cls in range(self.num_classes):
                            count = (final_pred_batch == cls).sum().item()
                            pct = count / final_pred_batch.numel() * 100 if final_pred_batch.numel() > 0 else 0
                            print(f"    Pred Class {cls}: {count} voxels ({pct:.2f}%)")
                        
                        # 检查sampled_label
                        if sampled_label is not None and len(sampled_label.shape) >= 2 and sampled_label.shape[0] > 0:
                            gt_label_batch = sampled_label[0]  # [N]
                            print(f"\n  sampled_label shape: {sampled_label.shape}")
                            print(f"  sampled_label value range: [{gt_label_batch.min()}, {gt_label_batch.max()}]")
                            print(f"  sampled_label unique values: {torch.unique(gt_label_batch).tolist()}")
                            
                            # 统计GT类别分布
                            for cls in range(self.num_classes):
                                count = (gt_label_batch == cls).sum().item()
                                pct = count / gt_label_batch.numel() * 100 if gt_label_batch.numel() > 0 else 0
                                print(f"    GT Class {cls}: {count} voxels ({pct:.2f}%)")
                            
                            # 检查对齐情况（随机采样1000个点）
                            sample_size = min(1000, final_pred_batch.numel())
                            if sample_size > 0:
                                sample_indices = torch.randperm(final_pred_batch.numel())[:sample_size]
                                matches = (final_pred_batch[sample_indices] == gt_label_batch[sample_indices]).sum().item()
                                match_rate = matches / sample_size * 100
                                print(f"\n  随机采样{sample_size}个点的匹配率: {matches}/{sample_size} = {match_rate:.2f}%")
                                
                                # 打印一些不匹配的例子
                                mismatches = []
                                for idx in sample_indices[:50]:
                                    pred_cls = final_pred_batch[idx].item()
                                    gt_cls = gt_label_batch[idx].item()
                                    if pred_cls != gt_cls:
                                        mismatches.append((idx.item(), int(pred_cls), int(gt_cls)))
                                if mismatches:
                                    print(f"  不匹配的例子（前50个中的）: {mismatches[:10]}")
                        
                        # 如果使用combine_geosem，检查prediction[-1]的数值范围
                        if self.combine_geosem and len(prediction) > 0:
                            pred_last = prediction[-1][0]  # [C, N]
                            print(f"\n  [combine_geosem模式] prediction[-1]统计:")
                            print(f"    Shape: {pred_last.shape}")
                            print(f"    Value range: [{pred_last.min():.4f}, {pred_last.max():.4f}]")
                            print(f"    Mean per channel:")
                            for c in range(min(5, pred_last.shape[0])):  # 只打印前5个通道
                                print(f"      Channel {c}: mean={pred_last[c].mean():.4f}, std={pred_last[c].std():.4f}")
                            # 检查第一个通道（geo）是否总是最大
                            if pred_last.shape[0] > 1:
                                argmax_result = pred_last.argmax(dim=0)  # [N]
                                geo_is_max = (argmax_result == 0).sum().item()
                                geo_pct = geo_is_max / argmax_result.numel() * 100
                                print(f"    第一个通道（geo）是最大值的比例: {geo_is_max}/{argmax_result.numel()} = {geo_pct:.2f}%")
                                if geo_pct > 90:
                                    print(f"    ⚠️ 警告：geo通道几乎总是最大，这会导致预测几乎全是empty！")
            except Exception as e:
                print(f"[WARNING] Failed to debug final_prediction: {e}")
                import traceback
                traceback.print_exc()
        
        # ========== Step 9: 准备返回的Gaussian对象（包含不确定性）==========
        # 获取最后一层的Gaussian
        last_gaussian = representation[-1]['gaussian']
        
        # 如果计算了不确定性，创建包含不确定性的新Gaussian对象
        # 注意：uncertainties和evidential_alphas是在最后一次调用prepare_gaussian_args时计算的
        # 我们需要重新计算一次以获取最后一层的不确定性
        if self.use_evidential:
            _, _, _, _, _, last_uncertainties, last_alphas, last_velocities = self.prepare_gaussian_args(last_gaussian)

            # 创建包含不确定性的新Gaussian对象
            from ..encoder.gaussian_encoder.utils import GaussianPrediction
            gaussian_with_uncertainty = GaussianPrediction(
                means=last_gaussian.means,
                scales=last_gaussian.scales,
                rotations=last_gaussian.rotations,
                opacities=last_gaussian.opacities,
                semantics=last_gaussian.semantics,
                velocities=last_velocities, # 新增
                uncertainties=last_uncertainties,
                evidential_alphas=last_alphas,
                original_means=last_gaussian.original_means,
                delta_means=last_gaussian.delta_means
            )
        else:
            gaussian_with_uncertainty = last_gaussian
        
        # ========== Step 10: 返回结果 ==========
        return {
            'pred_occ': prediction,           # List[[B, C, N]] - 每个decoder层的occupancy预测
            'pred_vel': velocity_prediction,  # List[[B, 3, N]] - 每个decoder层的velocity预测
            'bin_logits': bin_logits,         # List[[B, 1, N]] - 每个decoder层的二值occupancy logits（use_localaggprob模式）
            'density': density,               # List[[B, 1, N]] - 每个decoder层的density预测（use_localaggprob模式）
            'sampled_label': sampled_label,   # [B, N] - 采样后的GT标签
            'sampled_xyz': sampled_xyz,       # [B, N, 3] - 采样后的GT坐标
            'occ_mask': occ_cam_mask,         # [B, H, W, D] - 相机掩码
            'final_occ': final_prediction,    # [B, N] - 最终预测的类别索引
            'gaussian': gaussian_with_uncertainty,  # 最后一层的Gaussian（包含不确定性）
            'gaussians': [r['gaussian'] for r in representation],  # 所有层的Gaussian列表
            'voxel_evidential_alphas': voxel_evidential_alphas,  # [B, N, C] - voxel-level Dirichlet alphas (for EvidentialLoss)
        }
