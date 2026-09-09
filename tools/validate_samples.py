#!/usr/bin/env python3
"""Deep validation for canonical K-Radar samples without running the model."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigs.contracts import load_occupancy  # noqa: E402
from rigs.manifest import read_manifest  # noqa: E402


def check(record, data_root):
    errors = []
    try:
        for image in record.images.values():
            with Image.open(data_root / image) as handle:
                if handle.size != (2560, 720):
                    raise ValueError(f"unexpected image size {handle.size}")
                handle.verify()
    except Exception as error:
        errors.append(f"image:{error}")
    try:
        with np.load(data_root / record.radar, allow_pickle=False) as radar:
            required = {"range_ind", "elevation_ind", "azimuth_ind", "power_val"}
            if set(radar.files) != required:
                raise ValueError(f"keys={radar.files}")
            n = radar["range_ind"].shape[0]
            if radar["power_val"].shape != (8, n):
                raise ValueError(f"power_val={radar['power_val'].shape}, N={n}")
            if radar["elevation_ind"].shape != (n,) or radar["azimuth_ind"].shape != (n,):
                raise ValueError("coordinate array length mismatch")
            if not np.isfinite(radar["power_val"]).all():
                raise ValueError("non-finite power descriptor")
            bins = radar["power_val"][3:6]
            if ((bins < 0) | (bins >= 64)).any():
                raise ValueError("Doppler bin outside [0,63]")
    except Exception as error:
        errors.append(f"radar:{error}")
    try:
        if record.occupancy is None:
            raise ValueError("missing occupancy path")
        load_occupancy(data_root / record.occupancy)
    except Exception as error:
        errors.append(f"occupancy:{error}")
    return record, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    records = list(read_manifest(args.manifest))
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record, errors in pool.map(lambda item: check(item, args.data_root), records):
            if errors:
                failures.append((record.sequence, record.frame_id, errors))
    for failure in failures:
        print(failure)
    print(f"checked={len(records)} failures={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
