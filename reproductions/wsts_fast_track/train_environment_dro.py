"""Fine-tune P02 with matched year-corruption GroupDRO or ERM."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import sigmoid_focal_loss

from .contract import experiment_spec, validate_inventory
from .entrypoint import TRAIN_YEARS, _install_runtime_contract, load_training_stats
from .environment_dro import (
    CORRECTED_ALPHA_PROTOTYPE_ID,
    ERM_PROTOTYPE_ID,
    FIRE_ONLY_PROTOTYPE_ID,
    GROUP_COUNT,
    GROUP_DRO_STEP_SIZE,
    LEARNING_RATE,
    PROTOTYPE_ID,
    TRAINING_STEPS,
    balanced_year_sampling_weights,
    corrected_focal_alpha,
    erm_objective,
    group_dro_objective,
    install_training_environment_groups,
)
from .evaluate_missingness import load_checkpoint_model
from .evaluate_spatial_router import _checkpoint_from_record
from .prototype import (
    BLOCK_DROPOUT_PROBABILITY,
    BLOCK_PROTOTYPE_ID,
    FIRE_DROPOUT_PROBABILITY,
    PROTOTYPE_ID as P00_ID,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p02-record", type=Path)
    parser.add_argument("--p00-record", type=Path)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--objective", choices=("groupdro", "erm"), default="groupdro")
    parser.add_argument("--correct-focal-alpha", action="store_true")
    parser.add_argument("--fire-only", action="store_true")
    args = parser.parse_args(argv)
    if args.correct_focal_alpha and args.objective != "erm":
        raise ValueError("corrected focal alpha is only registered for matched ERM")
    if args.fire_only:
        if args.objective != "erm" or args.correct_focal_alpha:
            raise ValueError("FireDrop-only training requires legacy ERM")
        if args.p00_record is None or args.p02_record is not None:
            raise ValueError("FireDrop-only training requires only --p00-record")
    elif args.p02_record is None or args.p00_record is not None:
        raise ValueError("year-corruption training requires only --p02-record")

    upstream = args.upstream_root.resolve()
    data_root = args.data_root.resolve()
    validate_inventory(data_root)
    if args.fire_only:
        base_checkpoint = _checkpoint_from_record(
            args.p00_record,
            prototype_id=P00_ID,
            training_policy={
                "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
                "training_only": True,
            },
        )
    else:
        base_checkpoint = _checkpoint_from_record(
            args.p02_record,
            prototype_id=BLOCK_PROTOTYPE_ID,
            training_policy={
                "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
                "block_dropout_probability": BLOCK_DROPOUT_PROBABILITY,
                "block_fractions": [0.25, 0.5],
                "training_only": True,
            },
        )
    stats = load_training_stats(args.stats_path)
    _install_runtime_contract(upstream, "C00", stats)
    install_training_environment_groups(upstream, fire_only=args.fire_only)
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
    sampling_weights = balanced_year_sampling_weights(dataset)
    sampler = torch.utils.data.WeightedRandomSampler(
        sampling_weights,
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(0),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = load_checkpoint_model(
        base_checkpoint,
        experiment_id="C00",
        upstream_root=upstream,
        device=device,
    )
    model.train()
    focal_alpha = (
        corrected_focal_alpha(float(model.hparams.pos_class_weight))
        if args.correct_focal_alpha
        else 1.0 - float(model.hparams.pos_class_weight)
    )
    print(
        f"FOCAL_ALPHA_POLICY={'corrected-positive' if args.correct_focal_alpha else 'legacy-disabled'} "
        f"FOCAL_ALPHA={focal_alpha:.12f}",
        flush=True,
    )
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    log_weights = torch.zeros(GROUP_COUNT, device=device)
    iterator = iter(loader)
    final_loss = float("nan")
    last_group_losses: dict[int, float] = {}
    for step in range(1, TRAINING_STEPS + 1):
        try:
            x, target, group_ids = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, target, group_ids = next(iterator)
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        group_ids = group_ids.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        pixel_losses = sigmoid_focal_loss(
            logits,
            target.float(),
            alpha=focal_alpha,
            gamma=2.0,
            reduction="none",
        )
        per_sample_losses = pixel_losses.flatten(start_dim=1).mean(dim=1)
        if args.objective == "groupdro":
            loss, last_group_losses, weights = group_dro_objective(
                per_sample_losses,
                group_ids,
                log_weights,
                step_size=GROUP_DRO_STEP_SIZE,
            )
        else:
            loss = erm_objective(per_sample_losses)
            last_group_losses = {
                int(group): float(per_sample_losses[group_ids == group].mean().detach().cpu())
                for group in torch.unique(group_ids).tolist()
            }
            weights = torch.full(
                (GROUP_COUNT,), 1.0 / GROUP_COUNT, device=device
            )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % 100 == 0:
            print(
                f"{args.objective.upper()}_STEP={step} LOSS={final_loss:.8f} "
                f"MAX_GROUP={int(weights.argmax())} "
                f"WEIGHTS={','.join(f'{value:.6f}' for value in weights.cpu().tolist())}",
                flush=True,
            )

    output = args.output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "prototype_id": (
            FIRE_ONLY_PROTOTYPE_ID
            if args.fire_only
            else CORRECTED_ALPHA_PROTOTYPE_ID
            if args.correct_focal_alpha
            else PROTOTYPE_ID if args.objective == "groupdro" else ERM_PROTOTYPE_ID
        ),
        "objective": args.objective,
        "focal_alpha_policy": "corrected-positive" if args.correct_focal_alpha else "legacy-disabled",
        "focal_alpha": focal_alpha,
        "base_p00_record": (
            str(args.p00_record.resolve(strict=True)) if args.fire_only else None
        ),
        "base_p00_checkpoint": str(base_checkpoint) if args.fire_only else None,
        "base_p02_record": (
            str(args.p02_record.resolve(strict=True)) if not args.fire_only else None
        ),
        "base_p02_checkpoint": str(base_checkpoint) if not args.fire_only else None,
        "train_years": list(TRAIN_YEARS),
        "block_states": [0.0] if args.fire_only else [0.0, 0.25, 0.5],
        "corruption_policy": "fire-only" if args.fire_only else "year-block",
        "group_count": GROUP_COUNT,
        "steps": TRAINING_STEPS,
        "seed": 0,
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "group_dro_step_size": GROUP_DRO_STEP_SIZE if args.objective == "groupdro" else None,
        "active_fire_dropout_probability": FIRE_DROPOUT_PROBABILITY,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "final_loss": final_loss,
        "final_group_losses": last_group_losses,
        "group_weights": (
            torch.softmax(log_weights, dim=0).cpu().tolist()
            if args.objective == "groupdro"
            else None
        ),
        "model_state": model.state_dict(),
    }
    torch.save(payload, output)
    print(f"{args.objective.upper()}_CHECKPOINT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
