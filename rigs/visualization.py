"""Pure NumPy helpers for physically consistent release visualizations."""

from __future__ import annotations

import numpy as np

from .contracts import GRID_SHAPE, PC_RANGE, VOXEL_SIZE


def foreground_velocity_bev(occupancy: np.ndarray, velocity: np.ndarray):
    """Select the fastest foreground voxel in each vertical column.

    Columns without foreground receive a zero vector, so background or empty
    velocities cannot appear as learned object motion.
    """
    occupancy = np.asarray(occupancy)
    velocity = np.asarray(velocity)
    if occupancy.shape != GRID_SHAPE:
        raise ValueError(f"occupancy must have shape {GRID_SHAPE}, got {occupancy.shape}")
    if velocity.shape != GRID_SHAPE + (3,):
        raise ValueError(
            f"velocity must have shape {GRID_SHAPE + (3,)}, got {velocity.shape}"
        )
    foreground = occupancy == 2
    horizontal_speed = np.linalg.norm(velocity[..., :2], axis=-1)
    masked_speed = np.where(foreground, horizontal_speed, -np.inf)
    best_z = masked_speed.argmax(axis=2)
    gather = best_z[..., None]
    vx = np.take_along_axis(velocity[..., 0], gather, axis=2)[..., 0]
    vy = np.take_along_axis(velocity[..., 1], gather, axis=2)[..., 0]
    has_foreground = foreground.any(axis=2)
    return np.where(has_foreground, vx, 0.0), np.where(has_foreground, vy, 0.0)


def physical_bev_centres(step: int):
    """Return x/y sample coordinates following the canonical flipped-y grid."""
    if step <= 0:
        raise ValueError("step must be positive")
    x = PC_RANGE[0] + (np.arange(GRID_SHAPE[0]) + 0.5) * VOXEL_SIZE
    y = PC_RANGE[4] - (np.arange(GRID_SHAPE[1]) + 0.5) * VOXEL_SIZE
    return np.meshgrid(x[::step], y[::step], indexing="ij")
