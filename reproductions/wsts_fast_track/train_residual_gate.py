"""Train only the 17-parameter P04 residual head on 2016--2020."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .contract import experiment_spec, validate_inventory
from .entrypoint import TRAIN_YEARS, _install_runtime_contract, load_training_stats
from .evaluate_missingness import load_checkpoint_model
from .evaluate_spatial_router import _checkpoint_from_record
from .prototype import FIRE_DROPOUT_PROBABILITY, PROTOTYPE_ID as P00_ID
from .residual_gate import (
    GATE_TRAINING_STEPS,
    PROTOTYPE_ID,
    FrozenSpatialResidualGate,
    install_training_processed_block_dropout,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p00-record", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    upstream = args.upstream_root.resolve()
    data_root = args.data_root.resolve()
    validate_inventory(data_root)
    checkpoint = _checkpoint_from_record(
        args.p00_record,
        prototype_id=P00_ID,
        training_policy={
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "training_only": True,
        },
    )
    stats = load_training_stats(args.stats_path)
    _install_runtime_contract(upstream, "C00", stats)
    install_training_processed_block_dropout(upstream)
    dataset_class = importlib.import_module(
        "dataloader.FireSpreadDataset"
    ).FireSpreadDataset
    spec = experiment_spec("C00")
    dataset = dataset_class(
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

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but unavailable")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(0),
    )
    default_model = load_checkpoint_model(
        checkpoint,
        experiment_id="C00",
        upstream_root=upstream,
        device=device,
    )
    gate = FrozenSpatialResidualGate(default_model).to(device)
    parameters = tuple(gate.trainable_parameters())
    if sum(parameter.numel() for parameter in parameters) != 17:
        raise ValueError("P04 must expose exactly 17 trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=1e-2)
    gate.train()
    iterator = iter(loader)
    final_loss = float("nan")
    for step in range(1, GATE_TRAINING_STEPS + 1):
        try:
            x, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, target = next(iterator)
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits = gate(x).squeeze(1)
        loss = gate.compute_loss(logits, target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % 100 == 0:
            print(f"P04_STEP={step} LOSS={final_loss:.8f}", flush=True)

    output = args.output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "prototype_id": PROTOTYPE_ID,
        "base_record": str(args.p00_record.resolve(strict=True)),
        "base_checkpoint": str(checkpoint),
        "train_years": list(TRAIN_YEARS),
        "steps": GATE_TRAINING_STEPS,
        "seed": 0,
        "block_fractions": [0.25, 0.5],
        "trainable_parameters": 17,
        "final_loss": final_loss,
        "residual_head": gate.residual_head.state_dict(),
    }
    if output.exists():
        raise FileExistsError(output)
    torch.save(payload, output)
    print(f"P04_CHECKPOINT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
