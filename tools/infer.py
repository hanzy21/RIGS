#!/usr/bin/env python3
"""Run sequential RIGS inference without loading occupancy ground truth."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
from mmengine import Config
from mmseg.models import build_segmentor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model/head/localagg_prob"))

import model  # noqa: E402,F401
from dataset import OPENOCC_DATASET  # noqa: E402
from misc.checkpoint_util import load_strict_checkpoint  # noqa: E402
from rigs.artifact import save_prediction  # noqa: E402
from rigs.bki import BKIProcessor  # noqa: E402
from rigs.contracts import GRID_SHAPE, PC_RANGE, VOXEL_SIZE  # noqa: E402
from rigs.manifest import read_manifest  # noqa: E402
from rigs.temporal import TemporalState  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def batch_to_device(sample, device):
    result = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            result[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, np.ndarray):
            result[key] = torch.from_numpy(value).unsqueeze(0).to(device)
        elif isinstance(value, (float, int, bool)):
            result[key] = torch.as_tensor([value], device=device)
        else:
            result[key] = value
    return result


def radar_support_mask(points) -> np.ndarray:
    """Return canonical-grid voxels that contain a radar return."""
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"radar points must be [R,3], got {points.shape}")
    lower = np.asarray(PC_RANGE[:3])
    indices = np.floor((points - lower) / VOXEL_SIZE).astype(np.int64)
    indices[:, 1] = GRID_SHAPE[1] - 1 - indices[:, 1]
    valid = ((indices >= 0) & (indices < np.asarray(GRID_SHAPE))).all(axis=1)
    mask = np.zeros(GRID_SHAPE, dtype=bool)
    if valid.any():
        mask[tuple(indices[valid].T)] = True
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/rigs_kradar.py")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RIGS inference requires an NVIDIA CUDA device")
    device = torch.device("cuda")
    cfg = Config.fromfile(args.config)
    dataset_config = dict(cfg.inference_dataset_config)
    dataset_config["data_root"] = str(args.data_root)
    dataset_config["imageset"] = str(args.index)
    dataset_config["pipeline"] = [
        item for item in dataset_config["pipeline"] if item["type"] != "LoadOccupancyKRadar"
    ]
    dataset_config["return_keys"] = [
        "img", "projection_mat", "image_wh", "cam_positions", "focal_positions",
    ]
    dataset = OPENOCC_DATASET.build(dataset_config)
    records = list(read_manifest(args.manifest))
    if len(records) != len(dataset):
        raise ValueError(f"manifest/index length mismatch: {len(records)} != {len(dataset)}")

    network = build_segmentor(cfg.model).to(device)
    network.init_weights()
    load_strict_checkpoint(network, args.checkpoint)
    network.eval()
    bki = BKIProcessor(cfg.e2bki_config, device)
    temporal = TemporalState()
    config_hash, checkpoint_hash = sha256(args.config), sha256(args.checkpoint)

    count = len(records) if args.limit is None else min(args.limit, len(records))
    for index in range(count):
        record = records[index]
        sample = batch_to_device(dataset[index], device)
        radar_mask = radar_support_mask(sample["radar_points"][0])
        images = sample.pop("img")
        temporal_inputs = temporal.prepare(
            sequence=record.sequence, frame_id=record.frame_id,
            timestamp=record.timestamp, previous_frame_id=record.previous_frame_id,
        )
        if temporal_inputs.pop("reset"):
            bki.reset()
        sample.update({key: value for key, value in temporal_inputs.items() if key != "previous_timestamp"})
        with torch.no_grad():
            result = network(imgs=images, metas=sample, return_loss=False)
            occupancy, alpha = bki(
                result, sample["curr_ego2global"], sample["cam_positions"], int(record.frame_id),
            )
        temporal.update(
            sequence=record.sequence, frame_id=record.frame_id, timestamp=record.timestamp,
            gaussian=result.get("refined_anchor"),
            gaussian_features=result.get("refined_rep_features", result.get("rep_features")),
        )
        velocity = result["pred_vel"][-1][0].transpose(0, 1).reshape(GRID_SHAPE + (3,))
        output = args.output_dir / f"{record.sequence}_{record.frame_id}.npz"
        save_prediction(
            output,
            occupancy=occupancy[0].reshape(GRID_SHAPE).cpu().numpy(),
            velocity=velocity.cpu().numpy(),
            alpha=alpha[0].reshape(GRID_SHAPE + (2,)).cpu().numpy(),
            metadata={
                "sequence": record.sequence, "frame_id": record.frame_id,
                "timestamp": record.timestamp, "config_sha256": config_hash,
                "checkpoint_sha256": checkpoint_hash,
            },
            gaussian={
                "means": result["gaussian"].means[0].cpu().numpy(),
                "scales": result["gaussian"].scales[0].cpu().numpy(),
                "rotations": result["gaussian"].rotations[0].cpu().numpy(),
                "opacities": result["gaussian"].opacities[0].cpu().numpy(),
                "semantics": result["gaussian"].semantics[0].cpu().numpy(),
                "velocities": result["gaussian"].velocities[0].cpu().numpy(),
            },
            radar_supported=radar_mask,
        )
        print(output)


if __name__ == "__main__":
    main()
