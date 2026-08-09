"""Run one real, seeded fold-2 training-loader batch without invoking test loading."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from control import verify_inventory, verify_upstream, write_json_atomic


EXPECTED_FOLD_2 = {
    "train": [2018, 2020],
    "validation": [2019],
    "test": [2021],
}
EXPECTED_COMMIT = "ed221d491fe2142a4b2e93462c2c0b7a1c7c31ad"


def _is_torch_tensor(value: object) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


def _shape(value: object) -> tuple[int, ...]:
    if _is_torch_tensor(value) or isinstance(value, np.ndarray):
        return tuple(int(dimension) for dimension in value.shape)
    return tuple(int(dimension) for dimension in np.asarray(value).shape)


def _all_finite(value: object) -> bool:
    if _is_torch_tensor(value):
        import torch

        return bool(torch.isfinite(value).all().item())
    return bool(np.isfinite(np.asarray(value)).all())


def _all_binary(value: object) -> bool:
    if _is_torch_tensor(value):
        import torch

        return bool(torch.logical_or(value == 0, value == 1).all().item())
    return bool(np.isin(np.asarray(value), (0, 1)).all())


def _array_equal(left: object, right: object) -> bool:
    if _is_torch_tensor(left) and _is_torch_tensor(right):
        import torch

        return bool(torch.equal(left, right))
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def inspect_batch(batch: object) -> dict[str, object]:
    """Validate an official loader batch and describe the model boundary."""
    if not isinstance(batch, (list, tuple)) or len(batch) != 2:
        raise ValueError("training batch must be a two-item input/target sequence")
    inputs = batch[0]
    targets = batch[1]
    input_shape = _shape(inputs)
    target_shape = _shape(targets)

    if len(input_shape) != 5:
        raise ValueError(f"loader inputs must be 5D [B,T,C,H,W], got {input_shape}")
    if len(target_shape) != 3:
        raise ValueError(f"loader targets must be 3D [B,H,W], got {target_shape}")
    batch_size, temporal_steps, channels, height, width = input_shape
    if temporal_steps != 1:
        raise ValueError(f"temporal dimension must be T=1, got {temporal_steps}")
    if channels != 40:
        raise ValueError(f"channel dimension must contain all 40 features, got {channels}")
    if (height, width) != (128, 128):
        raise ValueError(f"input spatial dimensions must be 128x128, got {height}x{width}")
    if target_shape != (batch_size, 128, 128):
        raise ValueError(
            "target spatial dimensions must match [B,128,128], "
            f"got {target_shape}"
        )
    if not _all_finite(inputs):
        raise ValueError("loader inputs must be finite")
    if not _all_finite(targets):
        raise ValueError("loader targets must be finite and binary")
    if not _all_binary(targets):
        raise ValueError("loader targets must be binary")

    input_day_active_fire = inputs[:, 0, -1, :, :]
    if _array_equal(input_day_active_fire, targets):
        raise ValueError("next-day target must be distinct from the input-day active-fire mask")

    return {
        "loader_input_shape": list(input_shape),
        "loader_target_shape": list(target_shape),
        "model_input_shape": [batch_size, temporal_steps * channels, height, width],
        "model_target_shape": [batch_size, 1, 128, 128],
        "temporal_steps": temporal_steps,
        "model_channels": temporal_steps * channels,
        "crop_side_length": height,
        "inputs_finite": True,
        "targets_binary": True,
        "next_day_target_distinct": True,
    }


def assert_fold_mapping(datamodule: object) -> dict[str, list[int]]:
    """Require the exact original-WSTS fold-2 year assignment."""
    split_fires = getattr(datamodule, "split_fires", None)
    if not callable(split_fires):
        raise ValueError("datamodule does not expose the official split_fires method")
    train, validation, test = split_fires(2, False)
    actual = {
        "train": list(train),
        "validation": list(validation),
        "test": list(test),
    }
    if actual != EXPECTED_FOLD_2:
        raise ValueError(f"official fold 2 mapping mismatch: expected {EXPECTED_FOLD_2}, got {actual}")
    return actual


def _load_lock(lock_path: Path) -> Mapping[str, Any]:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("upstream lock must contain a JSON object")
    return payload


def _run_real_smoke(
    upstream_root: Path, data_root: Path, output_dir: Path, num_workers: int
) -> dict[str, object]:
    lock_path = Path(__file__).resolve().parents[1] / "upstream.lock.json"
    lock = _load_lock(lock_path)
    expected_commit = lock["code"]["commit"]
    if expected_commit != EXPECTED_COMMIT:
        raise ValueError(f"unexpected code pin in lock manifest: {expected_commit}")
    actual_commit = verify_upstream(upstream_root, expected_commit)

    upstream_src = (upstream_root.resolve() / "src").resolve()
    if not upstream_src.is_dir():
        raise ValueError(f"pinned upstream src directory is missing: {upstream_src}")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream_src))

    import torch
    from dataloader.FireSpreadDataModule import FireSpreadDataModule

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    datamodule = FireSpreadDataModule(
        data_dir=str(data_root.resolve()),
        batch_size=64,
        n_leading_observations=1,
        n_leading_observations_test_adjustment=5,
        crop_side_length=128,
        load_from_hdf5=True,
        num_workers=num_workers,
        remove_duplicate_features=True,
        features_to_keep=None,
        return_doy=False,
        data_fold_id=2,
        additional_data=False,
    )
    fold_mapping = assert_fold_mapping(datamodule)
    selected_years = [
        *fold_mapping["train"],
        *fold_mapping["validation"],
        *fold_mapping["test"],
    ]
    inventory = verify_inventory(data_root, selected_years=selected_years)
    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))
    batch_report = inspect_batch(batch)

    report: dict[str, object] = {
        "status": "pass",
        "purpose": "fold-2 real-data loader smoke only; no training performed",
        "upstream_commit": actual_commit,
        "weights_revision": lock["weights"]["revision"],
        "seed": 0,
        "num_workers": num_workers,
        "inventory": inventory,
        "fold_mapping": fold_mapping,
        "batch_size": 64,
        "features": "All",
        **batch_report,
        "setup_stage": "fit",
        "test_metadata_constructed_in_setup_fit": datamodule.test_dataset is not None,
        "test_loader_called": False,
        "test_sample_loaded": False,
        "training_started": False,
    }
    write_json_atomic(output_dir / "smoke.json", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-workers", type=int, choices=(64, 8, 4, 0), default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        report = _run_real_smoke(
            arguments.upstream_root,
            arguments.data_root,
            arguments.output_dir.resolve(),
            arguments.num_workers,
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        traceback.print_exc()
        print(f"smoke failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
