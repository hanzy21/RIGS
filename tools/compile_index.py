#!/usr/bin/env python3
"""Compile canonical JSONL into the generated MMEngine PKL consumed by KRadarDataset."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigs.manifest import read_manifest  # noqa: E402

CAM_KEYS = {"front": "CAM_FRONT", "left": "CAM_LEFT", "rear": "CAM_REAR", "right": "CAM_RIGHT"}


def read_flat_yaml(path):
    data = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        try:
            data[key.strip()] = float(value.strip())
        except ValueError:
            data[key.strip()] = value.strip()
    required = {
        "fx", "fy", "px", "py", "roll_ldr2cam", "pitch_ldr2cam", "yaw_ldr2cam",
        "x_ldr2cam", "y_ldr2cam", "z_ldr2cam",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"{path} lacks calibration keys: {sorted(missing)}")
    return data


def quaternion_wxyz(rotation):
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return [float(w), float(x), float(y), float(z)]


def camera_calibration(path):
    data = read_flat_yaml(path)
    intrinsic = np.array(
        [[data["fx"], 0, data["px"]], [0, data["fy"], data["py"]], [0, 0, 1]],
        dtype=np.float32,
    )
    lidar_to_camera = np.eye(4)
    lidar_to_camera[:3, :3] = Rotation.from_euler(
        "zyx",
        [data["yaw_ldr2cam"], data["pitch_ldr2cam"], data["roll_ldr2cam"]],
        degrees=True,
    ).as_matrix()
    lidar_to_camera[:3, 3] = [
        data["x_ldr2cam"], data["y_ldr2cam"], data["z_ldr2cam"],
    ]
    camera_to_lidar = np.linalg.inv(lidar_to_camera)
    return {
        "camera_intrinsic": intrinsic,
        "rotation": quaternion_wxyz(camera_to_lidar[:3, :3]),
        "translation": camera_to_lidar[:3, 3].astype(np.float32),
    }


def read_poses(path):
    poses = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank pose at {path}:{line_number}")
        values = np.fromstring(line, sep=" ")
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError(f"invalid 3x4 pose at {path}:{line_number}")
        matrix = values.reshape(3, 4)
        poses.append({
            "rotation": quaternion_wxyz(matrix[:, :3]),
            "translation": matrix[:, 3].astype(np.float32),
        })
    return poses


def radar_translation(sequence_dir):
    path = sequence_dir / "info_calib/calib_radar_lidar.txt"
    lines = path.read_text().strip().splitlines()
    if len(lines) != 2:
        raise ValueError(f"unexpected radar calibration format: {path}")
    values = lines[1].split(",")
    if len(values) != 3:
        raise ValueError(f"unexpected radar calibration row: {path}")
    return np.array([-float(values[1]), -float(values[2]), 0.0], dtype=np.float32)


def compile_records(records, data_root, resources_root):
    grouped = {}
    for record in records:
        grouped.setdefault(record.sequence, []).append(record)
    infos, metadata = {}, []
    for sequence, sequence_records in grouped.items():
        sequence_records.sort(key=lambda item: item.timestamp)
        poses = read_poses(resources_root / sequence_records[0].pose)
        first_lidar = int(sequence_records[0].sensor_indices["os2-64"])
        radar_t = radar_translation(data_root / str(sequence))
        scene = f"sequence-{sequence:02d}"
        scene_frames = []
        for local_index, record in enumerate(sequence_records):
            lidar = int(record.sensor_indices["os2-64"])
            pose_index = lidar - first_lidar
            if not 0 <= pose_index < len(poses):
                raise IndexError(f"no pose row for sequence {sequence}, lidar frame {lidar}")
            pose = poses[pose_index]
            frame = {
                "token": f"{scene}-{record.frame_id}",
                "previous_token": (
                    f"{scene}-{record.previous_frame_id}"
                    if record.previous_frame_id is not None else None
                ),
                "timestamp": record.timestamp,
                "occ_path": record.occupancy,
                "data": {
                    "LIDAR_TOP": {
                        "filename": record.frame_id,
                        "calib": {"rotation": [1, 0, 0, 0], "translation": np.zeros(3, np.float32)},
                        "pose": pose,
                    },
                    "TESSERACT": {
                        "filename": record.radar,
                        "calib": {"rotation": [1, 0, 0, 0], "translation": radar_t},
                        "pose": pose,
                    },
                },
            }
            for name, image in record.images.items():
                frame["data"][CAM_KEYS[name]] = {
                    "filename": image,
                    "calib": camera_calibration(resources_root / record.calibrations[name]),
                    "pose": pose,
                }
            scene_frames.append(frame)
            metadata.append((scene, local_index + 1))
        infos[scene] = scene_frames
    return {"infos": infos, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--resources-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compile_records(list(read_manifest(args.manifest)), args.data_root, args.resources_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        pickle.dump(payload, stream, protocol=4)
    print(f"{len(payload['metadata'])} frames -> {args.output}")


if __name__ == "__main__":
    main()
