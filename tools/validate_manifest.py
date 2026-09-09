#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigs.manifest import read_manifest, validate_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--resources-root", type=Path)
    args = parser.parse_args()
    records = list(read_manifest(args.manifest))
    errors = validate_files(records, args.data_root, args.resources_root)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {len(records)} records")


if __name__ == "__main__":
    main()
