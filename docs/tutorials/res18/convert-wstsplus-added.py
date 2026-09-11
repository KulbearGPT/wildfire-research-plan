#!/usr/bin/env python3
"""Convert the non-empty WSTS+ added-year GeoTIFF events to HDF5."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import rasterio
from tqdm import tqdm


YEARS = (2016, 2017, 2022, 2023)
EXPECTED = {2016: 92, 2017: 110, 2022: 122, 2023: 68}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def safe_lnglat(dataset: rasterio.io.DatasetReader) -> tuple[float, float]:
    """Return the raster center or the project's missing-location sentinel."""
    if dataset.crs is None:
        return (float("nan"), float("nan"))
    return dataset.lnglat()


def convert_event(event_dir: Path, target_path: Path, year: int) -> None:
    tiffs = tuple(sorted(event_dir.glob("*.tif"), key=lambda path: path.name))
    if not tiffs:
        return
    images: list[np.ndarray] = []
    lnglat: tuple[float, float] | None = None
    for path in tiffs:
        with rasterio.open(path, "r") as dataset:
            images.append(dataset.read())
            if lnglat is None:
                lnglat = safe_lnglat(dataset)
    data = np.stack(images, axis=0)
    data[:, 22] = np.nan_to_num(data[:, 22], nan=0.0)
    with h5py.File(target_path, "x") as handle:
        dataset = handle.create_dataset(
            "data", data=data, compression="lzf", shuffle=True
        )
        dataset.attrs["year"] = year
        dataset.attrs["fire_name"] = event_dir.name
        dataset.attrs["img_dates"] = [path.stem for path in tiffs]
        dataset.attrs["lnglat"] = lnglat


def convert_task(task: tuple[Path, Path, int]) -> None:
    convert_event(*task)


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve(strict=True)
    target_root = args.target_root.resolve(strict=False)
    if target_root.exists():
        raise SystemExit(f"refusing existing target root: {target_root}")
    target_root.mkdir(parents=True)

    counts: dict[int, int] = {}
    for year in YEARS:
        source_year = source_root / str(year)
        target_year = target_root / str(year)
        target_year.mkdir()
        events = tuple(sorted(
            (path for path in source_year.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ))
        tasks = [
            (event_dir, target_year / f"{event_dir.name}.hdf5", year)
            for event_dir in events
            if any(event_dir.glob("*.tif"))
        ]
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            tuple(tqdm(executor.map(convert_task, tasks), total=len(tasks), desc=str(year)))
        counts[year] = len(tuple(target_year.glob("*.hdf5")))

    if counts != EXPECTED:
        raise SystemExit(f"unexpected converted counts: {counts}")
    files = tuple(target_root.glob("*/*.hdf5"))
    payload = {
        "status": "pass",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "years": {str(year): count for year, count in counts.items()},
        "hdf5_file_count": len(files),
        "hdf5_total_bytes": sum(path.stat().st_size for path in files),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
