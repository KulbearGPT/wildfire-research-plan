#!/usr/bin/env python3
"""Hard-link the original and repaired years into the immutable experiment root."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


EXPECTED = {
    2016: 92,
    2017: 110,
    2018: 176,
    2019: 74,
    2020: 201,
    2021: 156,
    2022: 122,
    2023: 68,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--repaired-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original_root = args.original_root.resolve(strict=True)
    repaired_root = args.repaired_root.resolve(strict=True)
    target_root = args.target_root.resolve(strict=False)
    if target_root.exists():
        raise SystemExit(f"refusing existing target root: {target_root}")
    target_root.mkdir(parents=True)

    counts: dict[int, int] = {}
    for year, expected in EXPECTED.items():
        source_root = repaired_root if year in {2016, 2017, 2022, 2023} else original_root
        sources = tuple(sorted((source_root / str(year)).glob("*.hdf5")))
        if len(sources) != expected:
            raise SystemExit(f"unexpected source count for {year}: {len(sources)}")
        target_year = target_root / str(year)
        target_year.mkdir()
        for source in sources:
            os.link(source, target_year / source.name)
        counts[year] = len(sources)

    files = tuple(target_root.glob("*/*.hdf5"))
    payload = {
        "status": "pass",
        "dataset": "WSTS+ v1 active-fire-fixed",
        "source_record": "https://zenodo.org/records/17584629",
        "source_archive_bytes": 19920684684,
        "source_archive_md5": "42da7598cc33a170064e78d8027148c9",
        "original_root": str(original_root),
        "repaired_root": str(repaired_root),
        "hdf5_root": str(target_root),
        "hdf5_file_count": len(files),
        "hdf5_total_bytes": sum(path.stat().st_size for path in files),
        "years": {str(year): count for year, count in counts.items()},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
