"""Ground-truth-free temporal inference state management."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TemporalState:
    sequence: int | str | None = None
    frame_id: str | None = None
    timestamp: float | None = None
    gaussian: Any = None
    gaussian_features: Any = None

    def reset(self) -> None:
        self.sequence = self.frame_id = self.timestamp = None
        self.gaussian = self.gaussian_features = None

    def prepare(self, *, sequence, frame_id, timestamp, previous_frame_id):
        timestamp = float(timestamp)
        discontinuity = (
            self.sequence is None
            or sequence != self.sequence
            or previous_frame_id != self.frame_id
            or self.timestamp is None
            or timestamp <= self.timestamp
        )
        if discontinuity:
            self.reset()
        return {
            "reset": discontinuity,
            "prev_gaussian_representation": self.gaussian,
            "prev_gaussian_rep_features": self.gaussian_features,
            "previous_timestamp": self.timestamp,
        }

    def update(self, *, sequence, frame_id, timestamp, gaussian, gaussian_features=None):
        if gaussian is None:
            raise ValueError("model did not return Gaussian state")
        self.sequence = sequence
        self.frame_id = str(frame_id)
        self.timestamp = float(timestamp)
        self.gaussian = gaussian
        self.gaussian_features = gaussian_features
