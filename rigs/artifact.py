"""Versioned, self-describing inference artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import GRID_SHAPE, NUM_NONEMPTY_CLASSES, assert_occ_shape, assert_three_class


def save_prediction(path: Path, *, occupancy: np.ndarray, velocity: np.ndarray,
                    alpha: np.ndarray, metadata: dict[str, Any],
                    gaussian: dict[str, np.ndarray] | None = None,
                    radar_supported: np.ndarray | None = None) -> None:
    occupancy = np.asarray(occupancy, dtype=np.uint8)
    velocity = np.asarray(velocity, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    assert_occ_shape(occupancy)
    assert_three_class(occupancy)
    if velocity.shape != GRID_SHAPE + (3,):
        raise ValueError(f"velocity must be {GRID_SHAPE + (3,)}, got {velocity.shape}")
    if alpha.shape != GRID_SHAPE + (NUM_NONEMPTY_CLASSES,):
        raise ValueError(
            f"alpha must be {GRID_SHAPE + (NUM_NONEMPTY_CLASSES,)}, got {alpha.shape}"
        )
    if not np.isfinite(velocity).all() or not np.isfinite(alpha).all() or (alpha <= 0).any():
        raise ValueError("velocity must be finite and alpha must be finite and positive")
    required = {"sequence", "frame_id", "timestamp", "config_sha256", "checkpoint_sha256"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"missing metadata keys: {sorted(missing)}")
    payload = dict(metadata)
    payload.update({
        "schema_version": 1,
        "class_names": ["empty", "background", "foreground"],
        "coordinate_frame": "x_forward_y_left_z_up",
        "array_axes": "x_y_z",
        "uncertainty_formula": "2/(2+sum(alpha))",
    })
    strength = alpha.sum(axis=-1)
    uncertainty = 2.0 / (2.0 + strength)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        occupancy=occupancy,
        velocity=velocity,
        alpha=alpha,
        strength=strength.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
        metadata=np.asarray(json.dumps(payload, sort_keys=True)),
    )
    if gaussian is not None:
        required_gaussian = {"means", "scales", "rotations", "opacities", "semantics", "velocities"}
        missing_gaussian = required_gaussian - set(gaussian)
        if missing_gaussian:
            raise ValueError(f"missing Gaussian arrays: {sorted(missing_gaussian)}")
        arrays.update({f"gaussian_{key}": np.asarray(gaussian[key]) for key in required_gaussian})
    if radar_supported is not None:
        radar_supported = np.asarray(radar_supported, dtype=bool)
        if radar_supported.shape != GRID_SHAPE:
            raise ValueError(f"radar_supported must be {GRID_SHAPE}, got {radar_supported.shape}")
        arrays["radar_supported"] = radar_supported
    np.savez_compressed(path, **arrays)
