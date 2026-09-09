"""Canonical JSONL manifest schema and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Optional

SPLIT_SEQUENCES = {
    "train": (9, 10, 11, 12, 14, 15, 16, 17, 18, 19),
    "val": (13,),
    "test": (5, 20),
    "weather": (27, 38, 42, 54),
}


@dataclass(frozen=True)
class FrameRecord:
    sequence: int
    frame_id: str
    split: str
    timestamp: float
    images: dict[str, str]
    radar: str
    occupancy: Optional[str]
    calibrations: dict[str, str]
    pose: str
    sensor_indices: dict[str, str]
    previous_frame_id: Optional[str] = None
    excluded_reason: Optional[str] = None

    def validate(self) -> None:
        if self.split not in SPLIT_SEQUENCES:
            raise ValueError(f"unknown split: {self.split}")
        if self.sequence not in SPLIT_SEQUENCES[self.split]:
            raise ValueError(f"sequence {self.sequence} is not assigned to {self.split}")
        if not self.frame_id:
            raise ValueError("frame_id is required")
        flat_paths = {"radar": self.radar, "pose": self.pose}
        flat_paths.update({f"image.{key}": value for key, value in self.images.items()})
        flat_paths.update({f"calibration.{key}": value for key, value in self.calibrations.items()})
        if self.occupancy is not None:
            flat_paths["occupancy"] = self.occupancy
        for key, value in flat_paths.items():
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{key} must be a data-root-relative POSIX path: {value}")


def write_manifest(records: Iterable[FrameRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            record.validate()
            key = (record.sequence, record.frame_id)
            if key in seen:
                raise ValueError(f"duplicate frame: {key}")
            seen.add(key)
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_manifest(path: Path) -> Iterator[FrameRecord]:
    seen = set()
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = FrameRecord(**json.loads(line))
            record.validate()
            key = (record.sequence, record.frame_id)
            if key in seen:
                raise ValueError(f"duplicate frame at line {line_number}: {key}")
            seen.add(key)
            yield record


def validate_files(records: Iterable[FrameRecord], data_root: Path,
                   resources_root: Optional[Path] = None) -> list[str]:
    root = Path(data_root).resolve()
    resource_root = Path(resources_root).resolve() if resources_root else root
    errors: list[str] = []
    previous_by_sequence: dict[int, str] = {}
    for record in records:
        paths = {"radar": record.radar, "pose": record.pose}
        paths.update({f"image.{key}": value for key, value in record.images.items()})
        paths.update({f"calibration.{key}": value for key, value in record.calibrations.items()})
        if record.occupancy is not None:
            paths["occupancy"] = record.occupancy
        for field, value in paths.items():
            base = resource_root if field == "pose" or field.startswith("calibration.") else root
            candidate = (base / value).resolve()
            if base not in candidate.parents or not candidate.is_file():
                errors.append(f"{record.sequence}/{record.frame_id}: missing {field}: {candidate}")
        expected_previous = previous_by_sequence.get(record.sequence)
        if record.previous_frame_id != expected_previous:
            errors.append(
                f"{record.sequence}/{record.frame_id}: previous_frame_id="
                f"{record.previous_frame_id!r}, expected {expected_previous!r}"
            )
        previous_by_sequence[record.sequence] = record.frame_id
    return errors
