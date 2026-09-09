#!/usr/bin/env python3
"""Generate the canonical 8D Doppler spectrum descriptor from arrDREA MAT files."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def descriptor(arr_drea: np.ndarray, points_per_range: int = 250):
    raw = np.asarray(arr_drea, dtype=np.float32)
    if raw.ndim != 4:
        raise ValueError(f"arrDREA must be [D,R,A,E], got {raw.shape}")
    data = np.log10(np.maximum(raw, 1e-10))
    d_dim, r_dim, a_dim, e_dim = data.shape
    k = min(points_per_range, a_dim * e_dim)
    selection = 0.7 * data.mean(0) + 0.3 * data.max(0)
    selected = np.argpartition(selection.reshape(r_dim, -1), -k, axis=1)[:, -k:]
    elevation = selected // e_dim
    azimuth = selected % e_dim
    ranges = np.repeat(np.arange(r_dim), k)
    elevation = elevation.reshape(-1)
    azimuth = azimuth.reshape(-1)
    spectra = data[:, ranges, elevation, azimuth]
    top = np.argpartition(spectra, -3, axis=0)[-3:]
    columns = np.arange(spectra.shape[1])
    values = spectra[top, columns]
    order = np.argsort(-values, axis=0)
    values = np.take_along_axis(values, order, axis=0)
    top = np.take_along_axis(top, order, axis=0)
    features = np.vstack((values, top, spectra.mean(0)[None], spectra.std(0)[None]))
    return ranges.astype(np.int16), elevation.astype(np.int16), azimuth.astype(np.int16), features.astype(np.float16)


def process_file(source: Path, destination: Path, points_per_range: int):
    payload = loadmat(source)
    if "arrDREA" not in payload:
        raise KeyError(f"{source} has no arrDREA")
    arrays = descriptor(payload["arrDREA"], points_per_range)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, range_ind=arrays[0], elevation_ind=arrays[1],
             azimuth_ind=arrays[2], power_val=arrays[3])
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequences", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--points-per-range", type=int, default=250)
    args = parser.parse_args()
    tasks = []
    for sequence in args.sequences:
        source_dir = args.data_root / str(sequence) / "radar_tesseract"
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        for source in sorted(source_dir.glob("*.mat")):
            index = source.stem.rsplit("_", 1)[-1]
            destination = args.output_root / str(sequence) / "radar_tensor_8doppler_new" / f"EAsparse_{index}.npz"
            tasks.append((source, destination, args.points_per_range))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda item: process_file(*item), tasks):
            print(result)


if __name__ == "__main__":
    main()
