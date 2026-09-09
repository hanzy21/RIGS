#!/usr/bin/env python3
"""Build canonical JSONL manifests directly from K-Radar metadata."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigs.manifest import FrameRecord, SPLIT_SEQUENCES, write_manifest  # noqa: E402

HEADER = re.compile(
    r"idx\(tesseract_os2-64_cam-front_os1-128_cam-lrr\)="
    r"(\d+)_(\d+)_(\d+)_(\d+)_(\d+),\s*timestamp=([0-9.]+)"
)
CAMERAS = {"front": "cam-front"}
CAMERA_CALIB = {"front": 1}
EXPECTED_COUNTS = {"train": 7588, "val": 227, "test": 1095, "weather": 2393}


def radar_path(sequence_dir: Path, index: int) -> Path:
    filename = f"EAsparse_{index:05d}.npz"
    candidates = [
        sequence_dir / "radar_tensor_8doppler_new" / filename,
        sequence_dir / "radar_tensor_8doppler_new" / "radar_tensor_8doppler_new" / filename,
    ]
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise ValueError(f"expected exactly one radar layout for {sequence_dir.name}/{filename}, got {present}")
    return present[0]


def records_for_sequence(data_root: Path, resources_root: Path, sequence: int, split: str):
    sequence_dir = data_root / str(sequence)
    label_files = sorted((sequence_dir / "info_label").glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"no info_label files in {sequence_dir}")
    previous = None
    records, excluded = [], []
    for label_file in label_files:
        first_line = label_file.open(encoding="utf-8").readline().strip()
        match = HEADER.search(first_line)
        if not match:
            raise ValueError(f"invalid label header: {label_file}: {first_line}")
        tesseract, lidar, cam_front, _os1, cam_lrr = map(int, match.groups()[:5])
        timestamp = float(match.group(6))
        images = {
            name: str((Path(str(sequence)) / folder / f"{folder}_{(cam_front if name == 'front' else cam_lrr):05d}.png").as_posix())
            for name, folder in CAMERAS.items()
        }
        occupancy = Path(str(sequence)) / "semantic_occupancy_gt" / f"occupancy_gt_with_semantic{lidar}.npy"
        radar_candidates = [
            sequence_dir / "radar_tensor_8doppler_new" / f"EAsparse_{tesseract:05d}.npz",
            sequence_dir / "radar_tensor_8doppler_new" / "radar_tensor_8doppler_new" / f"EAsparse_{tesseract:05d}.npz",
        ]
        radar_present = [path for path in radar_candidates if path.is_file()]
        # A dataset may expose the nested spelling as a symlink to the direct
        # directory.  Count distinct resolved files, not path aliases.  Two
        # genuinely different files still make the frame ambiguous.
        radar_targets = {}
        for path in radar_present:
            radar_targets.setdefault(path.resolve(), path)
        radar_unique = list(radar_targets.values())
        radar = (radar_unique[0] if radar_unique else radar_candidates[0]).relative_to(data_root)
        calibrations = {
            name: str((Path("cam_calib/calib_seq_v2") / f"seq_{sequence:02d}" / f"cam_{cam_id}.yml").as_posix())
            for name, cam_id in CAMERA_CALIB.items()
        }
        pose = Path("odometry/gt") / f"gt_{sequence:02d}.txt"
        reasons = []
        for path in list(images.values()) + [occupancy.as_posix()]:
            if not (data_root / path).is_file():
                reasons.append(f"missing:{path}")
        if len(radar_unique) != 1:
            reasons.append(
                f"radar_layout_count:{len(radar_unique)}:" +
                ",".join(path.relative_to(data_root).as_posix() for path in radar_unique)
            )
        for path in list(calibrations.values()) + [pose.as_posix()]:
            if not (resources_root / path).is_file():
                reasons.append(f"missing_resource:{path}")
        frame_id = f"{lidar:05d}"
        record = FrameRecord(
            sequence=sequence, frame_id=frame_id, split=split, timestamp=timestamp,
            images=images, radar=radar.as_posix(), occupancy=occupancy.as_posix(),
            calibrations=calibrations, pose=pose.as_posix(),
            sensor_indices={
                "tesseract": f"{tesseract:05d}", "os2-64": f"{lidar:05d}",
                "cam-front": f"{cam_front:05d}", "cam-lrr": f"{cam_lrr:05d}",
            },
            previous_frame_id=previous if not reasons else None,
            excluded_reason=";".join(reasons) if reasons else None,
        )
        if reasons:
            excluded.append(record)
            previous = None
        else:
            records.append(record)
            previous = frame_id
    return records, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--resources-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    all_excluded = []
    for split, sequences in SPLIT_SEQUENCES.items():
        records = []
        for sequence in sequences:
            complete, excluded = records_for_sequence(
                args.data_root, args.resources_root, sequence, split)
            records.extend(complete)
            all_excluded.extend(excluded)
        if len(records) != EXPECTED_COUNTS[split]:
            raise ValueError(
                f"{split} contains {len(records)} complete frames; expected {EXPECTED_COUNTS[split]}")
        output = args.output_dir / f"{split}.jsonl"
        write_manifest(records, output)
        print(f"{split}: {len(records)} -> {output}")
    write_manifest(all_excluded, args.output_dir / "excluded.jsonl")
    print(f"excluded: {len(all_excluded)} -> {args.output_dir / 'excluded.jsonl'}")


if __name__ == "__main__":
    main()
