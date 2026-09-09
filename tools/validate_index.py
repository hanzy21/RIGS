#!/usr/bin/env python3
"""Validate a generated runtime PKL, including explicit temporal breaks."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    args = parser.parse_args()
    with args.index.open("rb") as stream:
        payload = pickle.load(stream)
    if set(payload) != {"infos", "metadata"}:
        raise ValueError("runtime index must contain exactly infos and metadata")
    metadata = payload["metadata"]
    total = breaks = 0
    expected_metadata = []
    for scene, frames in payload["infos"].items():
        previous_token = None
        for local_index, frame in enumerate(frames, 1):
            if "previous_token" not in frame:
                raise KeyError(f"{frame.get('token')} lacks previous_token")
            declared = frame["previous_token"]
            if declared is None:
                breaks += 1
            elif declared != previous_token:
                raise ValueError(
                    f"{frame.get('token')} declares {declared}, adjacent token is {previous_token}"
                )
            previous_token = frame.get("token")
            expected_metadata.append((scene, local_index))
            total += 1
    if metadata != expected_metadata:
        raise ValueError("metadata does not enumerate scene frames in canonical order")
    print(f"{total} frames, {len(payload['infos'])} scenes, {breaks} temporal starts/breaks")


if __name__ == "__main__":
    main()
