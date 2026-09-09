"""RIGS image/radar Gaussian occupancy segmentor."""
from __future__ import annotations

from mmseg.models import SEGMENTORS, build_backbone

from .base_segmentor import CustomBaseSegmentor


@SEGMENTORS.register_module()
class BEVSegmentor(CustomBaseSegmentor):
    def __init__(self, freeze_img_backbone=False, freeze_img_neck=False,
                 freeze_lifter=False, img_backbone_out_indices=(1, 2, 3),
                 extra_img_backbone=None, **kwargs):
        super().__init__(**kwargs)
        self.img_backbone_out_indices = tuple(img_backbone_out_indices)
        if freeze_img_backbone:
            self.img_backbone.requires_grad_(False)
        if freeze_img_neck:
            self.img_neck.requires_grad_(False)
        if freeze_lifter:
            self.lifter.requires_grad_(False)
            if hasattr(self.lifter, "random_anchors"):
                self.lifter.random_anchors.requires_grad = True
        if extra_img_backbone is not None:
            self.extra_img_backbone = build_backbone(extra_img_backbone)

    def extract_img_feat(self, imgs, **kwargs):
        if imgs.ndim != 5:
            raise ValueError(f"imgs must be [B,N,C,H,W], got {tuple(imgs.shape)}")
        batch, cameras = imgs.shape[:2]
        backbone_features = self.img_backbone(imgs.flatten(0, 1))
        if isinstance(backbone_features, dict):
            backbone_features = list(backbone_features.values())
        selected = [backbone_features[index] for index in self.img_backbone_out_indices]
        features = self.img_neck(selected)
        if isinstance(features, dict):
            features = features["fpn_out"]
        reshaped = []
        for feature in features:
            if feature.shape[0] != batch * cameras:
                raise ValueError("image feature batch/camera dimension mismatch")
            reshaped.append(feature.unflatten(0, (batch, cameras)))
        return {"ms_img_feats": reshaped}

    def forward_extra_img_backbone(self, imgs, **kwargs):
        if not hasattr(self, "extra_img_backbone"):
            raise RuntimeError("extra image backbone is not configured")
        batch, cameras = imgs.shape[:2]
        features = self.extra_img_backbone(imgs.flatten(0, 1))
        if isinstance(features, dict):
            features = list(features.values())
        return [feature.unflatten(0, (batch, cameras)) for feature in features]

    def forward(self, imgs=None, metas=None, points=None, extra_backbone=False,
                occ_only=False, rep_only=False, **kwargs):
        if imgs is None or metas is None:
            raise ValueError("imgs and metas are required")
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)
        results = {"imgs": imgs, "metas": metas, "points": points, **kwargs}
        results.update(self.extract_img_feat(**results))
        results.update(self.lifter(**results))

        inner = metas.get("metas", {}) if isinstance(metas.get("metas"), dict) else {}
        for key in (
            "radar_points", "radar_features", "radar_velocity",
            "curr_ego2global", "prev_ego2global", "timestamp",
            "prev_timestamp", "prev_occ_label",
        ):
            value = metas.get(key, inner.get(key))
            if key in ("radar_points", "radar_features", "curr_ego2global") and value is None:
                raise KeyError(f"required model input {key!r} is missing")
            results[key] = value

        encoded = self.encoder(**results)
        if rep_only:
            return encoded["representation"]
        results.update(encoded)
        output = self.head.forward_occ(**results) if occ_only and hasattr(self.head, "forward_occ") else self.head(**results)
        results.update(output)
        return results
