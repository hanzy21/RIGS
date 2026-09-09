#!/usr/bin/env python3
"""Render occupancy, BKI uncertainty, learned velocity and Gaussian primitives."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigs.contracts import PC_RANGE
from rigs.visualization import foreground_velocity_bev, physical_bev_centres


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--velocity-step", type=int, default=6)
    args = parser.parse_args()
    with np.load(args.artifact, allow_pickle=False) as item:
        occupancy = item["occupancy"]
        velocity = item["velocity"]
        uncertainty = item["uncertainty"]
        means = item["gaussian_means"] if "gaussian_means" in item else None
        semantics = item["gaussian_semantics"] if "gaussian_semantics" in item else None
    occupied = occupancy.max(axis=2)
    occupied_3d = occupancy > 0
    uncertainty_sum = np.where(occupied_3d, uncertainty, 0.0).sum(axis=2)
    uncertainty_count = occupied_3d.sum(axis=2)
    uncertainty_bev = np.full(uncertainty_sum.shape, np.nan, dtype=np.float32)
    np.divide(uncertainty_sum, uncertainty_count, out=uncertainty_bev,
              where=uncertainty_count > 0)
    vx, vy = foreground_velocity_bev(occupancy, velocity)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
    extent = (PC_RANGE[0], PC_RANGE[3], PC_RANGE[1], PC_RANGE[4])
    axes[0].imshow(occupied.T, origin="upper", extent=extent,
                   vmin=0, vmax=2, cmap="viridis")
    axes[0].set_title("Occupancy (0/1/2)")
    image = axes[1].imshow(uncertainty_bev.T, origin="upper", extent=extent,
                           vmin=0, vmax=1, cmap="magma")
    axes[1].set_title("E2-BKI uncertainty")
    fig.colorbar(image, ax=axes[1], fraction=.046)
    step = args.velocity_step
    x, y = physical_bev_centres(step)
    axes[2].imshow((occupied == 2).T, origin="upper", extent=extent,
                   cmap="Greys", alpha=.45)
    axes[2].quiver(x, y, vx[::step, ::step], vy[::step, ::step],
                   angles="xy", scale_units="xy", scale=1)
    axes[2].set_title("Learned velocity")
    if means is not None:
        classes = semantics.argmax(-1) + 1
        axes[3].scatter(means[:, 0], means[:, 1], c=classes, s=2, cmap="viridis", vmin=0, vmax=2)
        axes[3].set_xlim(0, 51.2); axes[3].set_ylim(-25.6, 25.6)
        axes[3].set_aspect("equal"); axes[3].set_title("Gaussian primitives")
    else:
        axes[3].text(.5, .5, "Gaussian arrays absent", ha="center", va="center")
    for axis in axes[:3]:
        axis.set_xlabel("x forward (m)"); axis.set_ylabel("y left (m)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
