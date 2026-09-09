#!/usr/bin/env python3
"""Create a GT-free manifest for deployment/inference validation."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigs.manifest import read_manifest, write_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    records = list(read_manifest(args.input))
    if args.limit is not None:
        records = records[:args.limit]
    write_manifest([replace(record, occupancy=None) for record in records], args.output)
    print(f"{len(records)} GT-free records -> {args.output}")


if __name__ == "__main__":
    main()
