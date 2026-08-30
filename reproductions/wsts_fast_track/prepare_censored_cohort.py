"""Select the fixed balanced 40-event target-QA training cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .censored_training import COHORT_PER_YEAR, COHORT_YEARS, select_balanced_candidates
from .contract import experiment_spec, validate_inventory
from .entrypoint import _install_runtime_contract, load_training_stats
from .environment_dro import resolve_dataset_index


def _day(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def _event_candidates(
    year: int,
    event: str,
    hdf5_path: Path,
    count: int,
    global_offset: int,
) -> list[dict[str, Any]]:
    """Return at most one deterministic positive and zero target per event."""

    order = sorted(
        range(count),
        key=lambda index: hashlib.sha256(f"{year}/{event}/{index}".encode()).hexdigest(),
    )
    found: dict[bool, dict[str, Any]] = {}
    with h5py.File(hdf5_path, "r") as source:
        data = source["data"]
        dates = data.attrs["img_dates"]
        for in_fire_index in order:
            target_index = in_fire_index + 1
            target = data[target_index, -1]
            positive_pixels = int(np.count_nonzero(np.isfinite(target) & (target > 0)))
            positive = positive_pixels > 0
            if positive not in found:
                found[positive] = {
                    "year": year,
                    "event": event,
                    "dataset_index": global_offset + in_fire_index,
                    "in_fire_index": in_fire_index,
                    "input_day": _day(dates[in_fire_index]),
                    "target_day": _day(dates[target_index]),
                    "target_positive_pixels": positive_pixels,
                    "hdf5": str(hdf5_path),
                }
            if len(found) == 2:
                break
    return list(found.values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    data_root = args.data_root.resolve()
    validate_inventory(data_root)
    stats = load_training_stats(args.stats_path)
    _install_runtime_contract(args.upstream_root.resolve(), "C00", stats)
    dataset_class = importlib.import_module("dataloader.FireSpreadDataset").FireSpreadDataset
    dataset_class.find_image_index_from_dataset_index = resolve_dataset_index
    spec = experiment_spec("C00")
    dataset = dataset_class(
        data_dir=str(data_root),
        included_fire_years=list(COHORT_YEARS),
        n_leading_observations=spec.n_leading_observations,
        n_leading_observations_test_adjustment=None,
        crop_side_length=128,
        load_from_hdf5=True,
        is_train=True,
        remove_duplicate_features=spec.remove_duplicate_features,
        features_to_keep=None,
        return_doy=False,
        stats_years=list(COHORT_YEARS),
        is_pad=False,
    )

    candidates: list[dict[str, Any]] = []
    global_offset = 0
    for year, fires in dataset.datapoints_per_fire.items():
        for event, count_value in fires.items():
            count = int(count_value)
            if count > 0:
                hdf5_path = Path(dataset.imgs_per_fire[year][event][0]).resolve()
                candidates.extend(
                    _event_candidates(year, event, hdf5_path, count, global_offset)
                )
            global_offset += count
    selected = select_balanced_candidates(
        candidates, years=COHORT_YEARS, per_year=COHORT_PER_YEAR
    )
    payload = {
        "schema_version": 1,
        "purpose": "fixed event-unique target-QA fine-tuning cohort",
        "years": list(COHORT_YEARS),
        "per_year": COHORT_PER_YEAR,
        "selection": "deterministic-hash; 4 positive and 4 zero targets per year",
        "samples": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(selected)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
