"""Paper-aligned Doppler-supervised voxel velocity loss (Eq. 10--16)."""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from scipy.optimize import linear_sum_assignment

from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class VelocityLoss(BaseLoss):
    def __init__(self, weight=1.0, pc_range=(0., -25.6, -5., 51.2, 25.6, 3.),
                 voxel_size=.4, occ_resolution=(128, 128, 20),
                 num_doppler_bins=64, doppler_resolution=.06, max_fold=8,
                 min_component_voxels=20, max_match_distance=5.,
                 min_dynamic_speed=.1,
                 lambda_point=1., lambda_voxel=1., lambda_consistency=.1,
                 lambda_smooth=.1, input_dict=None):
        input_dict = input_dict or {
            'pred_vel': 'pred_vel', 'radar_points': 'radar_points',
            'radar_velocity': 'radar_velocity', 'curr_ego2global': 'curr_ego2global',
            'prev_ego2global': 'prev_ego2global', 'timestamp': 'timestamp',
            'prev_timestamp': 'prev_timestamp', 'sampled_label': 'sampled_label',
            'prev_occ_label': 'prev_occ_label'}
        super().__init__(weight=weight, input_dict=input_dict)
        self.loss_func = self.loss_velocity
        self.pc_range = tuple(map(float, pc_range)); self.voxel_size = float(voxel_size)
        self.occ_resolution = tuple(map(int, occ_resolution))
        if self.occ_resolution != (128, 128, 20):
            raise ValueError('RIGS requires a 128x128x20 grid')
        self.v_unambiguous = num_doppler_bins * doppler_resolution
        self.max_fold = max_fold; self.min_component_voxels = min_component_voxels
        self.max_match_distance = max_match_distance
        self.min_dynamic_speed = float(min_dynamic_speed)
        self.lambdas = dict(point=lambda_point, voxel=lambda_voxel,
                            consistency=lambda_consistency, smooth=lambda_smooth)
        self.last_components = {}

    @staticmethod
    def _item(value, index):
        if isinstance(value, (list, tuple)): return value[index]
        if torch.is_tensor(value) and value.ndim and value.shape[0] > index: return value[index]
        return value

    @staticmethod
    def _pose(value):
        if torch.is_tensor(value): value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError(f'pose must be finite [4,4], got {value.shape}')
        return value

    @staticmethod
    def _time(value):
        if torch.is_tensor(value): value = value.detach().cpu().reshape(-1)[0].item()
        return float(value)

    def _centres(self, indices):
        return np.column_stack((
            self.pc_range[0] + (indices[:, 0] + .5) * self.voxel_size,
            self.pc_range[4] - (indices[:, 1] + .5) * self.voxel_size,
            self.pc_range[2] + (indices[:, 2] + .5) * self.voxel_size,
        )).astype(np.float32)

    def _components(self, labels):
        grid = labels.detach().cpu().numpy().reshape(self.occ_resolution)
        if grid.size and (grid.min() < 0 or grid.max() > 2):
            raise ValueError('VelocityLoss expects labels 0, 1, 2')
        cc, count = connected_components(grid == 2, structure=np.ones((3, 3, 3)))
        ids, centres = [], []
        for cid in range(1, count + 1):
            indices = np.argwhere(cc == cid)
            if len(indices) < self.min_component_voxels:
                cc[cc == cid] = 0; continue
            ids.append(cid); centres.append(self._centres(indices).mean(0))
        return cc, np.asarray(ids, np.int32), np.asarray(centres, np.float32)

    def _matches(self, ids, centres, previous_centres, current_pose, previous_pose, dt):
        if not len(ids) or not len(previous_centres): return {}
        previous_h = np.c_[previous_centres, np.ones(len(previous_centres))]
        previous_current = ((np.linalg.inv(current_pose) @ previous_pose) @ previous_h.T).T[:, :3]
        distance = np.linalg.norm(centres[:, None] - previous_current[None], axis=-1)
        rows, cols = linear_sum_assignment(distance)
        return {int(ids[r]): (centres[r] - previous_current[c]) / dt
                for r, c in zip(rows, cols) if distance[r, c] <= self.max_match_distance}

    def _radar_voxels(self, points):
        x = torch.floor((points[:, 0] - self.pc_range[0]) / self.voxel_size).long()
        yn = torch.floor((points[:, 1] - self.pc_range[1]) / self.voxel_size).long()
        y = self.occ_resolution[1] - 1 - yn
        z = torch.floor((points[:, 2] - self.pc_range[2]) / self.voxel_size).long()
        indices = torch.stack((x, y, z), 1)
        bounds = torch.tensor(self.occ_resolution, device=points.device)
        return indices, ((indices >= 0) & (indices < bounds)).all(1)

    def build_pseudo_targets(self, labels, previous_labels, points, doppler,
                             current_pose, previous_pose, dt):
        """Return component grid, radar voxel indices and valid object velocities."""
        cc, ids, centres = self._components(labels)
        _, _, previous_centres = self._components(previous_labels)
        displacement = self._matches(ids, centres, previous_centres,
                                     current_pose, previous_pose, dt)
        point_indices, in_bounds = self._radar_voxels(points)
        pi = point_indices.detach().cpu().numpy(); valid_np = in_bounds.detach().cpu().numpy()
        point_cc = np.zeros(len(points), np.int32)
        point_cc[valid_np] = cc[tuple(pi[valid_np].T)]
        current_h = torch.cat((points, torch.ones_like(points[:, :1])), 1)
        transform = torch.as_tensor(np.linalg.inv(previous_pose) @ current_pose,
                                    device=points.device, dtype=points.dtype)
        previous_points = (transform @ current_h.T).T[:, :3]
        v_static = (previous_points.norm(dim=1) - points.norm(dim=1)) / dt
        v_base = v_static - doppler
        targets = {}
        for cid, vdisp_np in displacement.items():
            if np.linalg.norm(vdisp_np) < self.min_dynamic_speed:
                continue
            mask_np = point_cc == cid
            if mask_np.sum() < 3:
                continue
            mask = torch.as_tensor(mask_np, device=points.device)
            rays = points[mask] / points[mask].norm(dim=1, keepdim=True).clamp_min(1e-6)
            if int(torch.linalg.matrix_rank(rays.float()).item()) < 3:
                continue
            vdisp = torch.as_tensor(vdisp_np, device=points.device, dtype=points.dtype)
            base = v_base[mask]
            fold = torch.round(((rays @ vdisp) - base) / self.v_unambiguous)
            if (fold.abs() > self.max_fold).any():
                continue
            dealiased = base + fold * self.v_unambiguous
            vls = torch.linalg.lstsq(rays.float(), dealiased[:, None].float()).solution[:, 0]
            vls = vls.to(points.dtype)
            mean_ray = rays.mean(0); mean_ray = mean_ray / mean_ray.norm().clamp_min(1e-6)
            tangent = vdisp - (vdisp @ mean_ray) * mean_ray
            vstar = (vls @ mean_ray) * mean_ray + tangent
            if torch.isfinite(vstar).all(): targets[cid] = (vstar, mask)
        return cc, point_indices, targets

    def _batch_losses(self, pred, labels, previous_labels, points, doppler,
                      current_pose, previous_pose, dt):
        cc, point_indices, targets = self.build_pseudo_targets(
            labels, previous_labels, points, doppler, current_pose, previous_pose, dt)
        zero = pred.sum() * 0.
        if not targets: return {name: zero for name in self.lambdas}
        field = pred.reshape(self.occ_resolution + (3,))
        point_sum = voxel_sum = consistency_sum = zero
        point_count = voxel_count = consistency_count = 0
        for cid, (vstar, point_mask) in targets.items():
            voxels = point_indices[point_mask]
            point_error = torch.abs(field[tuple(voxels.T)] - vstar)
            point_sum = point_sum + point_error.sum()
            point_count += point_error.numel()
            object_mask = torch.as_tensor(cc == cid, device=pred.device)
            object_pred = field[object_mask]
            voxel_error = torch.abs(object_pred - vstar)
            voxel_sum = voxel_sum + voxel_error.sum()
            voxel_count += voxel_error.numel()
            consistency_error = torch.abs(object_pred - object_pred.mean(0))
            consistency_sum = consistency_sum + consistency_error.sum()
            consistency_count += consistency_error.numel()
        smooth_sum = zero
        smooth_count = 0
        ids = torch.as_tensor(cc, device=pred.device)
        for axis in range(3):
            left = [slice(None)] * 3; right = [slice(None)] * 3
            left[axis] = slice(None, -1); right[axis] = slice(1, None)
            li, ri = ids[tuple(left)], ids[tuple(right)]
            for cid in targets:
                mask = (li == cid) & (ri == cid)
                if mask.any():
                    error = torch.abs(field[tuple(left)][mask] - field[tuple(right)][mask])
                    smooth_sum = smooth_sum + error.sum()
                    smooth_count += error.numel()
        normalize = lambda total, count: total / count if count else zero
        return dict(
            point=normalize(point_sum, point_count),
            voxel=normalize(voxel_sum, voxel_count),
            consistency=normalize(consistency_sum, consistency_count),
            smooth=normalize(smooth_sum, smooth_count),
        )

    def loss_velocity(self, pred_vel, radar_points, radar_velocity, curr_ego2global,
                      prev_ego2global, timestamp, prev_timestamp, sampled_label,
                      prev_occ_label):
        if isinstance(pred_vel, (list, tuple)):
            if not pred_vel: raise ValueError('pred_vel is empty')
            pred_vel = pred_vel[-1]
        if pred_vel.ndim != 3 or pred_vel.shape[1:] != (3, 327680):
            raise ValueError(f'pred_vel must be [B,3,327680], got {tuple(pred_vel.shape)}')
        prediction = pred_vel.transpose(1, 2).contiguous(); batches = []
        for b in range(len(prediction)):
            previous_pose = self._item(prev_ego2global, b)
            previous_labels = self._item(prev_occ_label, b)
            previous_time = self._item(prev_timestamp, b)
            if previous_pose is None or previous_labels is None or previous_time is None:
                batches.append({name: prediction[b].sum() * 0. for name in self.lambdas}); continue
            dt = self._time(self._item(timestamp, b)) - self._time(previous_time)
            if not np.isfinite(dt):
                raise ValueError(f'invalid delta_t={dt}')
            if dt <= 0:
                # A non-positive interval cannot define a reliable displacement
                # target. Treat it like a missing previous frame while keeping
                # the zero connected to the velocity prediction graph.
                batches.append({name: prediction[b].sum() * 0. for name in self.lambdas})
                continue
            points = self._item(radar_points, b).to(prediction.device)
            doppler = self._item(radar_velocity, b).to(prediction.device)
            if points.ndim != 2 or points.shape[1] != 3 or doppler.shape != (len(points),):
                raise ValueError('radar inputs must be [R,3] and [R]')
            batches.append(self._batch_losses(
                prediction[b], self._item(sampled_label, b), previous_labels,
                points, doppler, self._pose(self._item(curr_ego2global, b)),
                self._pose(previous_pose), dt))
        self.last_components = {name: torch.stack([x[name] for x in batches]).mean()
                                for name in self.lambdas}
        return sum(self.lambdas[name] * self.last_components[name] for name in self.lambdas)
