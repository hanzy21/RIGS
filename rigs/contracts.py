"""Release-wide class, grid, and coordinate contracts."""

from __future__ import annotations

from typing import Any

import numpy as np

EMPTY = 0
BACKGROUND = 1
FOREGROUND = 2
CLASS_NAMES = ("empty", "background", "foreground")
NUM_CLASSES = 3
NUM_NONEMPTY_CLASSES = 2

# Physical frame: x forward, y left, z up. Array axes are [x, y, z].
PC_RANGE = (0.0, -25.6, -5.0, 51.2, 25.6, 3.0)
GRID_SHAPE = (128, 128, 20)
VOXEL_SIZE = 0.4
Y_FLIP_AT_LOAD = True

_KRADAR_TO_RIGS = np.asarray([0, 1, 2, 2, 2, 2, 2, 2, 2], dtype=np.uint8)


def map_kradar_labels(labels: Any):
    """Map native K-Radar labels 0..8 to the release's three classes."""
    if hasattr(labels, "detach"):
        import torch

        if labels.numel() and (labels.min() < 0 or labels.max() > 8):
            raise ValueError("K-Radar labels must be in [0, 8]")
        lut = torch.as_tensor(_KRADAR_TO_RIGS, device=labels.device)
        return lut[labels.long()].to(dtype=labels.dtype)

    array = np.asarray(labels)
    if array.size and (array.min() < 0 or array.max() > 8):
        raise ValueError("K-Radar labels must be in [0, 8]")
    return _KRADAR_TO_RIGS[array.astype(np.int64, copy=False)]


def assert_occ_shape(array: Any, *, name: str = "occupancy") -> None:
    if tuple(array.shape[-3:]) != GRID_SHAPE:
        raise ValueError(f"{name} must end in {GRID_SHAPE}, got {tuple(array.shape)}")


def assert_three_class(array: Any, *, name: str = "labels") -> None:
    if hasattr(array, "numel"):
        if array.numel() and (array.min() < 0 or array.max() >= NUM_CLASSES):
            raise ValueError(f"{name} must contain only 0, 1, 2")
    else:
        array = np.asarray(array)
        if array.size and (array.min() < 0 or array.max() >= NUM_CLASSES):
            raise ValueError(f"{name} must contain only 0, 1, 2")


def sparse_kradar_to_dense(sparse: Any) -> np.ndarray:
    """Convert K-Radar [x,y,z,label] rows to the canonical Y-flipped grid."""
    sparse = np.asarray(sparse)
    if sparse.ndim != 2 or sparse.shape[1] != 4:
        raise ValueError(f"sparse occupancy must be [N,4], got {sparse.shape}")
    coords = sparse[:, :3].astype(np.int64, copy=True)
    labels = sparse[:, 3]
    if len(coords) and ((coords < 0).any() or (coords >= np.asarray(GRID_SHAPE)).any()):
        raise ValueError("sparse occupancy contains out-of-grid coordinates")
    mapped = map_kradar_labels(labels)
    coords[:, 1] = GRID_SHAPE[1] - 1 - coords[:, 1]
    dense = np.zeros(GRID_SHAPE, dtype=np.uint8)
    dense[tuple(coords.T)] = mapped
    return dense


def load_occupancy(path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim == 2:
        return sparse_kradar_to_dense(array)
    assert_occ_shape(array)
    if array.shape != GRID_SHAPE:
        raise ValueError(f"dense occupancy must be {GRID_SHAPE}, got {array.shape}")
    return map_kradar_labels(array)


def canonical_grid(dtype=np.float32) -> np.ndarray:
    x = PC_RANGE[0] + (np.arange(GRID_SHAPE[0]) + 0.5) * VOXEL_SIZE
    y = PC_RANGE[1] + (np.arange(GRID_SHAPE[1] - 1, -1, -1) + 0.5) * VOXEL_SIZE
    z = PC_RANGE[2] + (np.arange(GRID_SHAPE[2]) + 0.5) * VOXEL_SIZE
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).astype(dtype)
