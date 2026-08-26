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
    BELIEF_PROTOTYPE_ID,
    BELIEF_SAMPLE_COUNT,
    BELIEF_TRAINING_STEPS,
    GATE_TRAINING_STEPS,
    LAST_BLOCK_PROTOTYPE_ID,
    PROTOTYPE_ID,
    SPATIAL_PROTOTYPE_ID,
    TEACHER_BELIEF_KL_WEIGHT,
    TEACHER_BELIEF_PROTOTYPE_ID,
    TEACHER_BELIEF_RECONSTRUCTION_WEIGHT,
    FrozenLastBlockRouter,
    FrozenSpatialResidualGate,
    FrozenStochasticBeliefResidual,
    FrozenTeacherPosteriorBelief,
    install_training_processed_block_dropout,
    install_training_teacher_belief_dropout,
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
    parser.add_argument(
        "--residual-kernel-size", type=int, choices=(1, 3), default=1
    )
    parser.add_argument("--adapt-last-block", action="store_true")
    parser.add_argument("--stochastic-belief", action="store_true")
    parser.add_argument("--teacher-belief", action="store_true")
    args = parser.parse_args(argv)
    if sum((args.adapt_last_block, args.stochastic_belief, args.teacher_belief)) > 1:
        raise ValueError("choose only one residual adaptation")

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
    if args.teacher_belief:
        install_training_teacher_belief_dropout(upstream)
    else:
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
    if args.teacher_belief:
        gate = FrozenTeacherPosteriorBelief(
            default_model, sample_count=BELIEF_SAMPLE_COUNT
        ).to(device)
        prototype_id = TEACHER_BELIEF_PROTOTYPE_ID
        gate_type = "teacher-posterior-belief"
        training_steps = BELIEF_TRAINING_STEPS
        learning_rate = 1e-3
    elif args.stochastic_belief:
        gate = FrozenStochasticBeliefResidual(
            default_model, sample_count=BELIEF_SAMPLE_COUNT
        ).to(device)
        prototype_id = BELIEF_PROTOTYPE_ID
        gate_type = "stochastic-belief"
        training_steps = BELIEF_TRAINING_STEPS
        learning_rate = 1e-3
    elif args.adapt_last_block:
        gate = FrozenLastBlockRouter(default_model).to(device)
        prototype_id = LAST_BLOCK_PROTOTYPE_ID
        gate_type = "last-decoder-block"
        training_steps = GATE_TRAINING_STEPS
        learning_rate = 1e-2
    else:
        gate = FrozenSpatialResidualGate(
            default_model, residual_kernel_size=args.residual_kernel_size
        ).to(device)
        prototype_id = (
            PROTOTYPE_ID
            if args.residual_kernel_size == 1
            else SPATIAL_PROTOTYPE_ID
        )
        gate_type = f"residual-{args.residual_kernel_size}x{args.residual_kernel_size}"
        training_steps = GATE_TRAINING_STEPS
        learning_rate = 1e-2
    parameters = tuple(gate.trainable_parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in parameters)
    expected_parameter_count = (
        14_465
        if args.teacher_belief
        else 4_945
        if args.stochastic_belief
        else (17 if args.residual_kernel_size == 1 else 145)
    )
    if not args.adapt_last_block and (
        trainable_parameter_count != expected_parameter_count
    ):
        raise ValueError("residual gate exposes an unexpected parameter count")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    gate.train()
    iterator = iter(loader)
    final_loss = float("nan")
    final_components: dict[str, float] | None = None
    for step in range(1, training_steps + 1):
        try:
            x, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, target = next(iterator)
        x = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        if args.teacher_belief:
            loss, components, _ = gate.training_objective(x, target)
            final_components = {
                name: float(value.detach().cpu())
                for name, value in components.items()
            }
        else:
            logits = gate(x).squeeze(1)
            loss = gate.compute_loss(logits, target)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % 100 == 0:
            if final_components is None:
                print(f"GATE_STEP={step} LOSS={final_loss:.8f}", flush=True)
            else:
                print(
                    "GATE_STEP="
                    f"{step} LOSS={final_loss:.8f} "
                    f"FORECAST={final_components['forecast']:.8f} "
                    f"KL={final_components['kl']:.8f} "
                    f"RECONSTRUCTION={final_components['reconstruction']:.8f}",
                    flush=True,
                )

    output = args.output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "prototype_id": prototype_id,
        "base_record": str(args.p00_record.resolve(strict=True)),
        "base_checkpoint": str(checkpoint),
        "train_years": list(TRAIN_YEARS),
        "steps": training_steps,
        "seed": 0,
        "block_fractions": [0.25, 0.5],
        "gate_type": gate_type,
        "residual_kernel_size": args.residual_kernel_size,
        "trainable_parameters": trainable_parameter_count,
        "learning_rate": learning_rate,
        "final_loss": final_loss,
    }
    if args.teacher_belief:
        payload["sample_count"] = gate.sample_count
        payload["kl_weight"] = TEACHER_BELIEF_KL_WEIGHT
        payload["reconstruction_weight"] = TEACHER_BELIEF_RECONSTRUCTION_WEIGHT
        payload["loss_components"] = final_components
        payload["posterior_head"] = gate.posterior_head.state_dict()
        payload["prior_head"] = gate.prior_head.state_dict()
        payload["reconstruction_head"] = gate.reconstruction_head.state_dict()
        payload["output_head"] = gate.output_head.state_dict()
    elif args.stochastic_belief:
        payload["sample_count"] = gate.sample_count
        payload["belief_head"] = gate.belief_head.state_dict()
        payload["output_head"] = gate.output_head.state_dict()
    elif args.adapt_last_block:
        payload["adapted_block"] = gate.adapted_block.state_dict()
        payload["adapted_head"] = gate.adapted_head.state_dict()
    else:
        payload["residual_head"] = gate.residual_head.state_dict()
    if output.exists():
        raise FileExistsError(output)
    torch.save(payload, output)
    print(f"GATE_CHECKPOINT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
