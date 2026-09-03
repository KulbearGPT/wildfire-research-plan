"""Train the matched D2 standard/RNC continuations from corrected B3."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contract import experiment_spec, validate_inventory
from .corrected_baselines import install_corrected_baseline
from .entrypoint import TRAIN_YEARS, _install_runtime_contract, load_training_stats
from .evaluate_missingness import load_checkpoint_model
from .missingness import structured_block_mask
from .prototype import BLOCK_DROPOUT_PROBABILITY, FIRE_DROPOUT_PROBABILITY
from .reliability_normalized_conv import InputReliabilityNormalizedConv2d
from .train_predictive_consistency import validate_b3_record


PROCESSED_FEATURE_COUNT = 40
PROCESSED_ACTIVE_FIRE_VALUE = 38
PROCESSED_ACTIVE_FIRE_BINARY = 39
PROCESSED_DYNAMIC_NON_FIRE = tuple(range(12)) + (15,) + tuple(range(33, 38))
TRAINING_STEPS = 3_000
LEARNING_RATE = 1e-3


def apply_processed_reliability_corruption(
    x: torch.Tensor,
    *,
    fire_drop: bool,
    block_fraction: float,
    key_digest: str,
    active_fire_missing_value: float,
) -> torch.Tensor:
    """Apply aligned processed-space corruption and append invalidity."""

    if x.ndim != 4 or x.shape[1] != PROCESSED_FEATURE_COUNT:
        raise ValueError("processed C00 input must have shape (T, 40, H, W)")
    if block_fraction not in {0.0, 0.25, 0.5}:
        raise ValueError("block fraction must be 0, 0.25, or 0.5")
    result = x.clone()
    if fire_drop:
        result[:, PROCESSED_ACTIVE_FIRE_VALUE] = active_fire_missing_value
        result[:, PROCESSED_ACTIVE_FIRE_BINARY] = 0.0

    invalid = torch.zeros(x.shape[-2:], dtype=torch.bool, device=x.device)
    if block_fraction:
        numpy_mask = structured_block_mask(
            x.shape[-2], x.shape[-1], block_fraction, key_digest=key_digest
        )
        invalid = torch.as_tensor(numpy_mask, dtype=torch.bool, device=x.device)
        expanded = invalid[None, None]
        dynamic = result[:, PROCESSED_DYNAMIC_NON_FIRE]
        result[:, PROCESSED_DYNAMIC_NON_FIRE] = dynamic.masked_fill(expanded, 0.0)
        fire_value = result[:, PROCESSED_ACTIVE_FIRE_VALUE]
        result[:, PROCESSED_ACTIVE_FIRE_VALUE] = fire_value.masked_fill(
            invalid[None], active_fire_missing_value
        )
        fire_binary = result[:, PROCESSED_ACTIVE_FIRE_BINARY]
        result[:, PROCESSED_ACTIVE_FIRE_BINARY] = fire_binary.masked_fill(
            invalid[None], 0.0
        )
    invalid_channel = invalid.to(dtype=result.dtype)[None, None].expand(
        result.shape[0], 1, -1, -1
    )
    return torch.cat((result, invalid_channel), dim=1)


class ProcessedReliabilityDataset:
    """Add matched processed-space FireDrop/BlockDrop and an invalidity map."""

    def __init__(self, base_dataset: Any, active_fire_missing_value: float) -> None:
        if getattr(base_dataset, "is_train", None) is not True:
            raise ValueError("D2 requires an upstream training dataset")
        self.base = base_dataset
        self.active_fire_missing_value = active_fire_missing_value

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        x, target = self.base[index]
        fire_drop = bool(np.random.random() < FIRE_DROPOUT_PROBABILITY)
        block_drop = bool(np.random.random() < BLOCK_DROPOUT_PROBABILITY)
        block_fraction = 0.25 if np.random.random() < 0.5 else 0.5
        if not block_drop:
            block_fraction = 0.0
        return (
            apply_processed_reliability_corruption(
                x,
                fire_drop=fire_drop,
                block_fraction=block_fraction,
                key_digest=np.random.bytes(32).hex(),
                active_fire_missing_value=self.active_fire_missing_value,
            ),
            target,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3-record", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--variant", choices=("standard", "rnc"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    record_path = args.b3_record.resolve(strict=True)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("B3 completion record must be an object")
    checkpoint = Path(validate_b3_record(record)).resolve(strict=True)
    upstream = args.upstream_root.resolve()
    data_root = args.data_root.resolve()
    validate_inventory(data_root)
    stats = load_training_stats(args.stats_path)
    _install_runtime_contract(upstream, "C00", stats)
    install_corrected_baseline(upstream, "B0")

    dataset_class = importlib.import_module(
        "dataloader.FireSpreadDataset"
    ).FireSpreadDataset
    spec = experiment_spec("C00")
    base_dataset = dataset_class(
        data_dir=str(data_root),
        included_fire_years=list(TRAIN_YEARS),
        n_leading_observations=spec.n_leading_observations,
        n_leading_observations_test_adjustment=None,
        crop_side_length=128,
        load_from_hdf5=True,
        is_train=True,
        remove_duplicate_features=spec.remove_duplicate_features,
        features_to_keep=None,
        return_doy=False,
        stats_years=list(TRAIN_YEARS),
        is_pad=False,
    )
    means = np.asarray(base_dataset.means)
    stds = np.asarray(base_dataset.stds)
    active_fire_missing_value = float(
        (0.0 - means[0, 22, 0, 0]) / stds[0, 22, 0, 0]
    )
    dataset = ProcessedReliabilityDataset(
        base_dataset, active_fire_missing_value
    )

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but unavailable")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = load_checkpoint_model(
        checkpoint,
        experiment_id="C00",
        upstream_root=upstream,
        device=device,
    )
    if args.variant == "rnc":
        model.model.encoder.conv1 = InputReliabilityNormalizedConv2d(
            model.model.encoder.conv1
        ).to(device)
    model.train()
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    iterator = iter(loader)
    final_loss = float("nan")
    for step in range(1, TRAINING_STEPS + 1):
        try:
            x, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, target = next(iterator)
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        model_input = x if args.variant == "rnc" else x[:, :, :-1]
        logits = model(model_input).squeeze(1)
        loss = model.compute_loss(logits, target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % 100 == 0:
            print(
                f"D2_STEP={step} VARIANT={args.variant} LOSS={final_loss:.8f}",
                flush=True,
            )

    output = args.output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "candidate_id": "D2-RNC" if args.variant == "rnc" else "D2-STD",
        "matched_pair": "D2",
        "variant": args.variant,
        "base_b3_record": str(record_path),
        "base_b3_checkpoint": str(checkpoint),
        "experiment": "C00",
        "train_years": list(TRAIN_YEARS),
        "steps": TRAINING_STEPS,
        "seed": 0,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
        "block_dropout_probability": BLOCK_DROPOUT_PROBABILITY,
        "processed_space_matched_corruption": True,
        "final_loss": final_loss,
        "hyper_parameters": dict(model.hparams),
        "state_dict": model.state_dict(),
        "global_step": TRAINING_STEPS,
    }
    torch.save(payload, output)
    print(f"D2_CHECKPOINT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
