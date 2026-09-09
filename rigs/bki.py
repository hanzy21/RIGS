"""Strict two-class E2-BKI wrapper for release inference."""
from __future__ import annotations

import torch

from model.e2bki import E2BKIInference


class BKIProcessor:
    def __init__(self, config, device):
        config = dict(config)
        config["device"] = str(device)
        self.module = E2BKIInference(config).to(device)

    def reset(self):
        self.module.reset_alpha_history()

    def __call__(self, model_result, ego_to_global, sensor_position, frame_id):
        gaussian = model_result.get("gaussian")
        query = model_result.get("sampled_xyz")
        base = model_result.get("final_occ")
        if gaussian is None or query is None or base is None:
            raise KeyError("model result lacks gaussian, sampled_xyz, or final_occ")
        if gaussian.semantics.shape[-1] != 2:
            raise ValueError("BKI requires two non-empty Gaussian semantics")
        if ego_to_global.shape != (query.shape[0], 4, 4):
            raise ValueError("ego_to_global must be [B,4,4]")
        homogeneous = torch.cat((query, torch.ones_like(query[..., :1])), dim=-1)
        query_global = torch.einsum("bij,bnj->bni", ego_to_global, homogeneous)[..., :3]
        if sensor_position.ndim == 3:
            sensor_position = sensor_position[:, 0]
        output = self.module(
            gaussian, query, sensor_positions=sensor_position,
            query_points_global=query_global, frame_id=frame_id,
        )
        alpha = output["alpha_m"]
        if alpha.shape != query.shape[:-1] + (2,):
            raise ValueError(f"BKI alpha shape mismatch: {tuple(alpha.shape)}")
        semantic = output["semantic_map"].argmax(-1) + 1
        occupancy = torch.where(base > 0, semantic, torch.zeros_like(semantic))
        if occupancy.min() < 0 or occupancy.max() > 2:
            raise ValueError("BKI produced labels outside 0, 1, 2")
        return occupancy, alpha
