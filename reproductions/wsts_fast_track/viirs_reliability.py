"""Build acquisition-time VIIRS reliability fields for the fixed 2021 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np


CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
VNP14IMG_COLLECTION_ID = "C2734202914-LPCLOUD"
VALID_FIRE_MASK_CLASSES = (5, 8, 9)
FIXED_2021_GATE = (
    ("fire_24935885", "2021-02-01"),
    ("fire_25017416", "2021-03-27"),
    ("fire_25138935", "2021-05-06"),
    ("fire_25205936", "2021-06-01"),
    ("fire_25205363", "2021-06-07"),
    ("fire_25205362", "2021-06-17"),
    ("fire_25204859", "2021-06-20"),
    ("fire_25204877", "2021-06-30"),
    ("fire_25295099", "2021-07-06"),
    ("fire_25295877", "2021-07-08"),
    ("fire_25295908", "2021-07-09"),
    ("fire_25295073", "2021-07-11"),
    ("fire_25295927", "2021-07-16"),
    ("fire_25295856", "2021-07-22"),
    ("fire_25295966", "2021-07-27"),
    ("fire_25294984", "2021-07-31"),
    ("fire_25411029", "2021-08-04"),
    ("fire_25411057", "2021-08-07"),
    ("fire_25411896", "2021-08-10"),
    ("fire_25411825", "2021-08-13"),
    ("fire_25411013", "2021-08-22"),
    ("fire_25548182", "2021-09-01"),
    ("fire_25548269", "2021-09-13"),
    ("fire_25639315", "2021-10-21"),
)
FIXED_2021_MODEL_GATE = tuple(
    (
        event,
        (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat(),
    )
    for event, day in FIXED_2021_GATE
)
FIXED_2021_TARGET_GATE = tuple(
    (
        event,
        (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat(),
    )
    for event, day in FIXED_2021_MODEL_GATE
)
GATES = {
    "provenance": FIXED_2021_GATE,
    "model": FIXED_2021_MODEL_GATE,
    "target": FIXED_2021_TARGET_GATE,
}


@dataclass(frozen=True)
class CmrGranule:
    producer_id: str
    start: datetime
    end: datetime


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def select_causal_unique_granules(
    entries: Iterable[Mapping[str, Any]], prediction_time: datetime
) -> tuple[CmrGranule, ...]:
    """Normalize, deduplicate, and sort granules available before prediction."""
    unique: dict[str, CmrGranule] = {}
    for entry in entries:
        producer_id = str(entry["producer_granule_id"])
        if not producer_id.endswith(".nc"):
            producer_id = f"{producer_id}.nc"
        granule = CmrGranule(
            producer_id=producer_id,
            start=_parse_utc(str(entry["time_start"])),
            end=_parse_utc(str(entry["time_end"])),
        )
        if granule.end <= prediction_time:
            unique[producer_id] = granule
    return tuple(sorted(unique.values(), key=lambda granule: (granule.end, granule.producer_id)))


def update_latest_valid_minutes(
    latest_minutes: np.ndarray,
    flat_cells: np.ndarray,
    fire_mask_classes: np.ndarray,
    observation_minute: int,
) -> None:
    """Update cells observed as clear land or nominal/high-confidence fire."""
    if flat_cells.shape != fire_mask_classes.shape:
        raise ValueError("flat cells and fire-mask classes must have equal shape")
    reliable = np.isin(fire_mask_classes, VALID_FIRE_MASK_CLASSES)
    np.maximum.at(latest_minutes, flat_cells[reliable], observation_minute)


def finalize_reliability(
    latest_minutes: np.ndarray, active_fire_hhmm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge WSTS positives exactly and return finite reliability and age fields."""
    if latest_minutes.shape != active_fire_hhmm.shape:
        raise ValueError("latest minutes and WSTS active fire must have equal shape")
    merged = latest_minutes.astype(np.int16, copy=True)
    positive = np.isfinite(active_fire_hhmm) & (active_fire_hhmm > 0)
    hhmm = np.rint(active_fire_hhmm[positive]).astype(np.int16)
    hours, minutes = hhmm // 100, hhmm % 100
    if np.any(hours >= 24) or np.any(minutes >= 60):
        raise ValueError("WSTS active-fire values must be valid HHMM acquisition times")
    merged[positive] = np.maximum(merged[positive], hours * 60 + minutes)
    reliability = (merged >= 0).astype(np.uint8)
    age_hours = np.full(merged.shape, 24.0, dtype=np.float32)
    age_hours[reliability.astype(bool)] = (
        1440 - merged[reliability.astype(bool)]
    ) / 60.0
    return reliability, age_hours


def request_json_with_retries(
    request: Callable[[], Any],
    retry_errors: tuple[type[BaseException], ...],
    *,
    sleeper: Callable[[float], Any] = time.sleep,
    attempts: int = 3,
) -> Mapping[str, Any]:
    """Retry one idempotent metadata request after transient transport errors."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            response = request()
            response.raise_for_status()
            return response.json()
        except retry_errors:
            if attempt + 1 == attempts:
                raise
            sleeper(float(4**attempt))
    raise AssertionError("unreachable retry state")


def _cmr_entries(day: str, bounding_box: Sequence[float]) -> list[Mapping[str, Any]]:
    import requests

    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    parameters = {
            "collection_concept_id": VNP14IMG_COLLECTION_ID,
            "bounding_box": ",".join(str(value) for value in bounding_box),
            "temporal": (
                f"{start.isoformat().replace('+00:00', 'Z')},"
                f"{(start + timedelta(days=1)).isoformat().replace('+00:00', 'Z')}"
            ),
            "page_size": 100,
    }
    payload = request_json_with_retries(
        lambda: requests.get(
            CMR_URL,
            params=parameters,
            headers={"User-Agent": "wildfire-viirs-reliability/1"},
            timeout=60,
        ),
        (requests.ConnectionError, requests.Timeout),
    )
    return list(payload["feed"]["entry"])


def _laads_location(product: str, producer_id: str) -> tuple[str, str, str]:
    match = re.search(r"\.A(\d{4})(\d{3})\.", producer_id)
    if not match:
        raise ValueError(f"unrecognized VIIRS producer ID: {producer_id}")
    year, day_of_year = match.groups()
    url = (
        "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/"
        f"{product}/{year}/{day_of_year}/{producer_id}"
    )
    return year, day_of_year, url


def _cached_laads_download(
    product: str, producer_id: str, cache_root: Path, token: str
) -> tuple[Path, bool]:
    year, day_of_year, url = _laads_location(product, producer_id)
    target = cache_root / product / year / day_of_year / producer_id
    if target.is_file() and target.stat().st_size > 0:
        return target, True
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / f".{target.name}.{os.getpid()}.part"
    config = target.parent / f".{target.name}.{os.getpid()}.curl.conf"
    config.write_text(
        "\n".join(
            (
                f'url = "{url}"',
                "location",
                'cookie = "session"',
                f'header = "Authorization: Bearer {token}"',
                f'output = "{partial}"',
                "fail-with-body",
                "silent",
                "show-error",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    config.chmod(0o600)
    try:
        subprocess.run(["curl", "--config", str(config)], check=True)
        with h5py.File(partial, "r"):
            pass
        partial.replace(target)
    finally:
        config.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    return target, False


def _decode_attribute(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _project_fire_mask(
    latitude: np.ndarray,
    longitude: np.ndarray,
    fire_mask: np.ndarray,
    bounding_box_wgs84: Sequence[float],
    affine: Any,
    crs: Any,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    from rasterio.warp import transform as transform_coordinates

    west, south, east, north = bounding_box_wgs84
    geographic = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (longitude >= west - 0.01)
        & (longitude <= east + 0.01)
        & (latitude >= south - 0.01)
        & (latitude <= north + 0.01)
    )
    classes = fire_mask[geographic].astype(np.uint8, copy=False)
    projected_x, projected_y = transform_coordinates(
        "EPSG:4326",
        crs,
        longitude[geographic].astype(float).tolist(),
        latitude[geographic].astype(float).tolist(),
    )
    columns = np.floor((np.asarray(projected_x) - affine.c) / affine.a).astype(int)
    rows = np.floor((np.asarray(projected_y) - affine.f) / affine.e).astype(int)
    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    return rows[inside] * width + columns[inside], classes[inside]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_output(
    output_root: Path,
    event: str,
    day: str,
    reliability: np.ndarray,
    age_hours: np.ndarray,
) -> Path:
    event_root = output_root / event
    event_root.mkdir(parents=True, exist_ok=True)
    target = event_root / f"{day}.npz"
    if target.exists():
        raise FileExistsError(f"refusing existing reliability output: {target}")
    partial = target.with_suffix(".tmp.npz")
    np.savez_compressed(partial, reliability=reliability, age_hours=age_hours)
    partial.replace(target)
    return target


def _process_sample(
    archive: zipfile.ZipFile,
    event: str,
    day: str,
    output_root: Path,
    cache_root: Path,
    token: str,
) -> dict[str, Any]:
    from rasterio.io import MemoryFile
    from rasterio.warp import transform_bounds

    member = f"2021/{event}/{day}.tif"
    with MemoryFile(archive.read(member)) as memory_file, memory_file.open() as source:
        active_fire = source.read(23)
        affine = source.transform
        crs = source.crs
        height, width = source.height, source.width
        bounding_box = transform_bounds(crs, "EPSG:4326", *source.bounds, densify_pts=21)

    prediction_time = datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(days=1)
    granules = select_causal_unique_granules(
        _cmr_entries(day, bounding_box), prediction_time
    )
    if not granules:
        raise RuntimeError(f"CMR returned no causal VNP14IMG granules for {event} {day}")
    latest = np.full(height * width, -1, dtype=np.int16)
    sources: list[dict[str, Any]] = []
    cache_hits = 0

    for granule in granules:
        fire_path, fire_hit = _cached_laads_download(
            "VNP14IMG", granule.producer_id, cache_root, token
        )
        with h5py.File(fire_path, "r") as product:
            fire_mask = product["fire mask"][:]
            pointer = _decode_attribute(product.attrs["InputPointer"])
            match = re.search(r"(VNP03IMG\.A\d{7}\.\d{4}\.002\.\d+\.nc)", pointer)
            if not match:
                raise RuntimeError(f"no paired VNP03IMG in {granule.producer_id}")
            geo_id = match.group(1)
        geo_path, geo_hit = _cached_laads_download("VNP03IMG", geo_id, cache_root, token)
        with h5py.File(geo_path, "r") as geolocation:
            latitude = geolocation["geolocation_data/latitude"][:]
            longitude = geolocation["geolocation_data/longitude"][:]
        flat_cells, classes = _project_fire_mask(
            latitude,
            longitude,
            fire_mask,
            bounding_box,
            affine,
            crs,
            height,
            width,
        )
        observation_minute = granule.end.hour * 60 + granule.end.minute
        update_latest_valid_minutes(latest, flat_cells, classes, observation_minute)
        cache_hits += int(fire_hit) + int(geo_hit)
        sources.append(
            {
                "vnp14img": granule.producer_id,
                "vnp03img": geo_id,
                "start": granule.start.isoformat(),
                "end": granule.end.isoformat(),
                "projected_pixels": int(len(flat_cells)),
            }
        )

    reliability, age_hours = finalize_reliability(latest.reshape(height, width), active_fire)
    target = _write_output(output_root, event, day, reliability, age_hours)
    reliable = reliability.astype(bool)
    record = {
        "event": event,
        "day": day,
        "output": str(target.relative_to(output_root)),
        "sha256": _sha256(target),
        "shape": [height, width],
        "granules": len(granules),
        "cache_hits": cache_hits,
        "reliable_fraction": float(reliability.mean()),
        "mean_reliable_age_hours": float(age_hours[reliable].mean()) if reliable.any() else None,
        "wsts_positive_pixels": int((np.isfinite(active_fire) & (active_fire > 0)).sum()),
        "sources": sources,
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


def build_fixed_gate(
    archive_path: Path,
    output_root: Path,
    cache_root: Path,
    token_path: Path,
    *,
    gate: str = "provenance",
) -> dict[str, Any]:
    if gate not in GATES:
        raise ValueError(f"unknown reliability gate: {gate}")
    if output_root.exists():
        raise FileExistsError(f"refusing existing output root: {output_root}")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("Earthdata token file is empty")
    output_root.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        records = [
            _process_sample(archive, event, day, output_root, cache_root, token)
            for event, day in GATES[gate]
        ]
    summary = {
        "schema_version": 1,
        "purpose": f"fixed 24-sample 2021 VIIRS reliability {gate} gate",
        "gate": gate,
        "prediction_anchor": "next-calendar-day 00:00 UTC",
        "valid_fire_mask_classes": list(VALID_FIRE_MASK_CLASSES),
        "unknown_age_hours": 24.0,
        "samples": records,
        "summary": {
            "sample_count": len(records),
            "granule_count": sum(record["granules"] for record in records),
            "cache_hits": sum(record["cache_hits"] for record in records),
            "mean_reliable_fraction": sum(record["reliable_fraction"] for record in records)
            / len(records),
        },
    }
    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("summary", json.dumps(summary["summary"], sort_keys=True), flush=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--gate", choices=tuple(GATES), default="provenance")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    build_fixed_gate(
        arguments.archive,
        arguments.output_root,
        arguments.cache_root,
        arguments.token_file,
        gate=arguments.gate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
