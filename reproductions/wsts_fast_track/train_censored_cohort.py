"""Run one of the two matched 3K target-QA cohort fine-tunes."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .censored_training import (
    COHORT_YEARS,
    LEARNING_RATE,
    PROTOTYPE_IDS,
    SEED,
    TRAINING_STEPS,
    CensoredCohortDataset,
    cohort_loss,
)
from .contract import experiment_spec, validate_inventory
from .entrypoint import _install_runtime_contract, load_training_stats
from .environment_dro import resolve_dataset_index
from .evaluate_missingness import load_checkpoint_model
from .evaluate_spatial_router import _checkpoint_from_record
from .prototype import FIRE_DROPOUT_PROBABILITY, PROTOTYPE_ID as P00_ID


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=tuple(PROTOTYPE_IDS), required=True)
    parser.add_argument("--p00-record", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--target-reliability-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    output = args.output_path.resolve()
    if output.exists():
        raise FileExistsError(output)

    p00_record = args.p00_record.resolve(strict=True)
    p00_checkpoint = _checkpoint_from_record(
        p00_record,
        prototype_id=P00_ID,
        training_policy={
            "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
            "training_only": True,
        },
    )
    upstream = args.upstream_root.resolve()
    data_root = args.data_root.resolve()
    validate_inventory(data_root)
    stats = load_training_stats(args.stats_path)
    _install_runtime_contract(upstream, "C00", stats)
    dataset_class = importlib.import_module("dataloader.FireSpreadDataset").FireSpreadDataset
    dataset_class.find_image_index_from_dataset_index = resolve_dataset_index
    spec = experiment_spec("C00")
    base = dataset_class(
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
    dataset = CensoredCohortDataset(
        base, args.cohort_manifest, args.target_reliability_root
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but unavailable")
    sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=True,
        num_samples=TRAINING_STEPS * args.batch_size,
        generator=torch.Generator().manual_seed(SEED),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = load_checkpoint_model(
        p00_checkpoint,
        experiment_id="C00",
        upstream_root=upstream,
        device=device,
    )
    model.train()
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    censored = args.objective == "censored"
    final_loss = float("nan")
    for step, (features, target, reliability) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        reliability = reliability.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features).squeeze(1)
        loss = cohort_loss(logits, target, reliability, censored=censored)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % 100 == 0:
            print(
                f"COHORT_STEP={step} OBJECTIVE={args.objective} LOSS={final_loss:.8f}",
                flush=True,
            )
        if step == TRAINING_STEPS:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "prototype_id": PROTOTYPE_IDS[args.objective],
        "objective": args.objective,
        "target_mask_policy": (
            "positive-or-reliably-observed" if censored else "all-pixels"
        ),
        "base_p00_record": str(p00_record),
        "base_p00_checkpoint": str(p00_checkpoint),
        "cohort_manifest": str(args.cohort_manifest.resolve(strict=True)),
        "target_reliability_root": str(args.target_reliability_root.resolve(strict=True)),
        "train_years": list(COHORT_YEARS),
        "cohort_samples": len(dataset),
        "steps": TRAINING_STEPS,
        "seed": SEED,
        "optimizer": "AdamW",
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "final_loss": final_loss,
        "model_state": model.state_dict(),
    }
    torch.save(payload, output)
    print(f"COHORT_CHECKPOINT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
