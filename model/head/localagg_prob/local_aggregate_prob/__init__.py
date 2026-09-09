#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import re
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import _C


def _read_kernel_num_channels():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'config.h')
    try:
        with open(config_path, 'r') as f:
            for line in f:
                m = re.search(r'#define\s+NUM_CHANNELS\s+(\d+)', line)
                if m:
                    return int(m.group(1))
    except FileNotFoundError:
        pass
    return None


KERNEL_NUM_CHANNELS = _read_kernel_num_channels()


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opas,
        semantics,
        radii,
        cov3D,
        H, W, D
    ):
        args = (
            pts, points_int, means3D, means3D_int,
            opas, semantics, radii, cov3D,
            H, W, D
        )
        num_rendered, logits, bin_logits, density, probability, \
            geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args)

        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer, binningBuffer, imgBuffer,
            means3D, pts, points_int, cov3D,
            opas, semantics, logits,
            bin_logits, density, probability
        )

        return logits, bin_logits, density

    @staticmethod
    def backward(ctx, logits_grad, bin_logits_grad, density_grad):
        num_rendered = ctx.num_rendered
        H, W, D = ctx.H, ctx.W, ctx.D
        geomBuffer, binningBuffer, imgBuffer, \
            means3D, pts, points_int, cov3D, \
            opas, semantics, logits, \
            bin_logits, density, probability = ctx.saved_tensors

        args = (
            geomBuffer, binningBuffer, imgBuffer,
            H, W, D, num_rendered,
            means3D, pts, points_int, cov3D,
            opas, semantics, logits,
            bin_logits, density, probability,
            logits_grad, bin_logits_grad, density_grad)

        means3D_grad, opas_grad, semantics_grad, cov3D_grad = \
            _C.local_aggregate_backward(*args)

        return (
            None, None, means3D_grad, None,
            opas_grad, semantics_grad, None, cov3D_grad,
            None, None, None
        )


class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1):
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size
        self.radii_min = radii_min

    def forward(
        self,
        pts,
        means3D,
        opas,
        semantics,
        scales,
        cov3D):

        assert pts.shape[0] == 1
        pts = pts.squeeze(0)
        assert not pts.requires_grad
        means3D = means3D.squeeze(0)
        opas = opas.squeeze(0)
        semantics = semantics.squeeze(0)
        scales = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0)

        C_actual = semantics.shape[-1]
        K = KERNEL_NUM_CHANNELS

        if C_actual != 2:
            raise ValueError(
                f"RIGS localagg_prob requires exactly 2 non-empty semantic channels, got {C_actual}")
        if K != C_actual:
            raise RuntimeError(
                f"localagg_prob was compiled with NUM_CHANNELS={K}; clean and rebuild it for 2 channels")

        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int)
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        means3D_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int)
        assert means3D_int.min() >= 0 and means3D_int[:, 0].max() < self.H and means3D_int[:, 1].max() < self.W and means3D_int[:, 2].max() < self.D
        radii = torch.ceil(scales.max(dim=-1)[0] * self.scale_multiplier / self.grid_size).to(torch.int)
        radii = radii.clamp(min=self.radii_min)
        assert radii.min() >= 1
        cov3D = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        logits, bin_logits, density = _LocalAggregate.apply(
            pts, points_int, means3D, means3D_int,
            opas, semantics, radii, cov3D,
            self.H, self.W, self.D
        )

        if logits.shape[-1] != 2:
            raise RuntimeError(f"localagg_prob returned {logits.shape[-1]} channels; expected 2")

        return logits, bin_logits, density
