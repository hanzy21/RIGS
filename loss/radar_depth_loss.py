"""
RadarDepthLoss: 利用雷达点云的深度信息作为辅助监督信号。

核心思想：
  对于每个 radar 点（有精确的 3D 位置），找到其最近的 K 个 Gaussian anchor，
  计算它们之间的 3D 距离作为 loss。这鼓励 Gaussian 分布与 radar 观测到的
  真实物体位置对齐。

可配置参数：
  - weight:         loss 权重（默认 1.0）
  - topk:           每个 radar 点匹配的最近 K 个 Gaussian（默认 8）
  - max_distance:   匹配距离阈值（米），超出的不计入 loss（默认 5.0）
  - loss_type:      'SmoothL1' / 'L1' / 'L2'（默认 'SmoothL1'）
  - sigma:          距离加权的 sigma（米），越近权重越大（默认 2.0）
  - pc_range:       点云范围，用于过滤 radar 点（默认 None=不过滤）
  - use_opacity_weight: 是否用 Gaussian 不透明度加权 loss（默认 True）
  - warmup_epochs:  前 N 个 epoch 不启用此 loss（默认 0）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class RadarDepthLoss(BaseLoss):

    @staticmethod
    def opacity_weights(opacities):
        """Use the already activated Gaussian opacity without re-sigmoiding it."""
        if opacities.numel() and (opacities.min() < 0 or opacities.max() > 1):
            raise ValueError("Gaussian opacities must already be in [0, 1]")
        return opacities

    def __init__(
            self,
            weight=1.0,
            topk=8,
            max_distance=5.0,
            loss_type='SmoothL1',
            sigma=2.0,
            pc_range=None,
            use_opacity_weight=True,
            warmup_epochs=0,
            input_dict=None,
            **kwargs):
        super().__init__(weight=weight, input_dict=input_dict)

        if input_dict is None:
            self.input_dict = {
                'gaussian': 'gaussian',
                'radar_points': 'radar_points',
            }

        self.topk = topk
        self.max_distance = max_distance
        self.loss_type = loss_type
        self.sigma = sigma
        self.pc_range = pc_range
        self.use_opacity_weight = use_opacity_weight
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0

        self.loss_func = self.compute_radar_depth_loss

    def set_epoch(self, epoch):
        """外部调用设置当前 epoch，用于 warmup 控制。"""
        self._current_epoch = epoch

    def compute_radar_depth_loss(self, gaussian, radar_points, **kwargs):
        """
        计算 Radar Depth Auxiliary Loss。

        Args:
            gaussian: GaussianPrediction (NamedTuple)，包含 means [B, G, 3] 等
            radar_points: [B, M, 3] 或 List[Tensor]，radar 点云的 3D 坐标（LiDAR 系）

        Returns:
            loss: scalar tensor
        """
        # Warmup: 前几个 epoch 返回 0
        if self._current_epoch < self.warmup_epochs:
            if isinstance(gaussian.means, torch.Tensor):
                return gaussian.means.new_zeros(1).squeeze()
            return torch.tensor(0.0)

        gaussian_means = gaussian.means  # [B, G, 3]
        gaussian_opacities = gaussian.opacities  # [B, G, 1]

        if gaussian_means is None or radar_points is None:
            return gaussian_means.new_zeros(1).squeeze() if gaussian_means is not None else torch.tensor(0.0)

        B = gaussian_means.shape[0]
        total_loss = gaussian_means.new_zeros(1).squeeze()
        valid_count = 0

        for b in range(B):
            g_means = gaussian_means[b]  # [G, 3]

            # 处理 radar_points 的不同格式
            if isinstance(radar_points, (list, tuple)):
                r_pts = radar_points[b]  # [M, 3] or [M, >=3]
            elif isinstance(radar_points, torch.Tensor):
                if radar_points.dim() == 3:
                    r_pts = radar_points[b]  # [M, 3]
                else:
                    r_pts = radar_points  # [M, 3]
            else:
                continue

            if r_pts is None or (isinstance(r_pts, torch.Tensor) and r_pts.numel() == 0):
                continue

            # 确保是 tensor
            if not isinstance(r_pts, torch.Tensor):
                r_pts = torch.tensor(r_pts, device=g_means.device, dtype=g_means.dtype)

            # 只取前 3 维（x, y, z），忽略其他属性
            r_pts = r_pts[:, :3].to(g_means.device).to(g_means.dtype)

            M = r_pts.shape[0]
            if M == 0:
                continue

            # 可选：根据 pc_range 过滤 radar 点
            if self.pc_range is not None:
                pc = self.pc_range
                mask = (
                    (r_pts[:, 0] >= pc[0]) & (r_pts[:, 0] <= pc[3]) &
                    (r_pts[:, 1] >= pc[1]) & (r_pts[:, 1] <= pc[4]) &
                    (r_pts[:, 2] >= pc[2]) & (r_pts[:, 2] <= pc[5])
                )
                r_pts = r_pts[mask]
                M = r_pts.shape[0]
                if M == 0:
                    continue

            # 计算 radar 点到所有 Gaussian 的 3D 距离
            # r_pts: [M, 3], g_means: [G, 3]
            # dists: [M, G]
            dists = torch.cdist(r_pts.unsqueeze(0), g_means.unsqueeze(0)).squeeze(0)  # [M, G]

            # 找每个 radar 点最近的 topk 个 Gaussian
            k = min(self.topk, g_means.shape[0])
            topk_dists, topk_indices = torch.topk(dists, k, dim=-1, largest=False)  # [M, k]

            # 距离阈值过滤
            valid_mask = topk_dists < self.max_distance  # [M, k]

            if valid_mask.sum() == 0:
                continue

            # 距离加权：近距离的 Gaussian 权重更大
            dist_weights = torch.exp(-topk_dists.pow(2) / (2 * self.sigma ** 2))  # [M, k]

            # 可选：用 Gaussian 不透明度加权
            if self.use_opacity_weight and gaussian_opacities is not None:
                opa = gaussian_opacities[b].squeeze(-1)  # [G]
                topk_opa = opa[topk_indices]  # [M, k]
                # GaussianPrediction already stores safe_sigmoid(anchor_opa).
                # Applying another sigmoid maps zero opacity to weight 0.5 and
                # defeats the intended confidence gate.
                opa_weights = self.opacity_weights(topk_opa)
                combined_weights = dist_weights * opa_weights * valid_mask.float()
            else:
                combined_weights = dist_weights * valid_mask.float()

            # 计算 loss
            if self.loss_type == 'SmoothL1':
                element_loss = F.smooth_l1_loss(
                    topk_dists, torch.zeros_like(topk_dists), reduction='none')
            elif self.loss_type == 'L1':
                element_loss = topk_dists.abs()
            elif self.loss_type == 'L2':
                element_loss = topk_dists.pow(2)
            else:
                element_loss = F.smooth_l1_loss(
                    topk_dists, torch.zeros_like(topk_dists), reduction='none')

            # 加权平均
            weighted_loss = (element_loss * combined_weights).sum()
            weight_sum = combined_weights.sum().clamp(min=1e-6)
            batch_loss = weighted_loss / weight_sum

            total_loss = total_loss + batch_loss
            valid_count += 1

        if valid_count > 0:
            total_loss = total_loss / valid_count

        return total_loss
