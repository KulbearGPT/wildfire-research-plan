"""Train one matched fire belief-state prototype on the frozen train split."""

from __future__ import annotations

import argparse
import importlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contract import validate_inventory
from .runtime import TRAIN_YEARS, _install_runtime_contract, load_training_stats
from .diagnostic_support import balanced_year_sampling_weights, resolve_dataset_index
from .evaluate_missingness import load_checkpoint_model
from .diagnostic_support import _checkpoint_from_record
from .latent_state_common import (
    TRAIN_SCENARIOS,
    BeliefStateTrainingDataset,
    masked_unweighted_focal,
    normalized_active_fire_zero,
    split_observations,
    trainable_state_dict,
)
from .diagnostic_support import FIRE_DROPOUT_PROBABILITY, PROTOTYPE_ID as P00_ID
from .reconstruction_baseline import ReconstructionFirstBeliefState
from .state_attention import TemporalAttentionBeliefState
from .state_filter import RecurrentBeliefFilter


PROTOTYPE_ID = "fire-belief-state-v1"
METHODS = ("filter", "attention", "reconstruction")
TRAINING_STEPS = 3_000
LEARNING_RATE = 1e-3
STATE_WEIGHT = 0.1
EFFECTIVE_BATCH_SIZE = 64
SEED = 0
TARGET_CARRIER_CHANNEL = 16


def build_model(
    method: str,
    default_model: torch.nn.Module,
    history: int,
) -> torch.nn.Module:
    """Construct exactly one registered belief-state model."""

    model_classes = {
        "filter": RecurrentBeliefFilter,
        "attention": TemporalAttentionBeliefState,
        "reconstruction": ReconstructionFirstBeliefState,
    }
    try:
        model_class = model_classes[method]
    except KeyError as error:
        raise ValueError(f"unknown belief-state method: {method}") from error
    return model_class(default_model, history)


def combine_losses(
    method: str,
    forecast_loss: torch.Tensor,
    state_loss: torch.Tensor,
) -> torch.Tensor:
    """Return the frozen task-oriented or reconstruction-only objective."""

    if method == "reconstruction":
        return state_loss
    if method in {"filter", "attention"}:
        return forecast_loss + STATE_WEIGHT * state_loss
    raise ValueError(f"unknown belief-state method: {method}")


def install_target_independent_augmentation(dataset_class: type) -> None:
    """Patch training geometry to transform targets without ranking crops by them."""

    original_augment = dataset_class.augment
    if getattr(original_augment, "_belief_target_independent", False):
        return

    def target_independent_augment(
        self: Any,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            x.ndim != 4
            or y.ndim != 2
            or x.shape[0] <= 0
            or x.shape[1] <= TARGET_CARRIER_CHANNEL
            or x.shape[-2:] != y.shape
        ):
            raise ValueError("raw training inputs and target have incompatible shapes")
        if TARGET_CARRIER_CHANNEL in getattr(
            self, "indices_of_degree_features", ()
        ):
            raise ValueError("target carrier channel cannot be angle-adjusted")

        carrier = torch.zeros_like(x[:1])
        carrier[0, TARGET_CARRIER_CHANNEL] = y.to(dtype=x.dtype)
        carried_x = torch.cat((x, carrier), dim=0)
        dummy_target = torch.zeros_like(y)
        transformed_x, transformed_dummy = original_augment(
            self, carried_x, dummy_target
        )
        if (
            transformed_x.ndim != 4
            or transformed_x.shape[:2] != carried_x.shape[:2]
            or transformed_x.shape[0] != x.shape[0] + 1
            or transformed_dummy.shape != transformed_x.shape[-2:]
        ):
            raise ValueError("upstream augmentation changed the carrier contract")
        transformed_target = transformed_x[-1, TARGET_CARRIER_CHANNEL].to(
            dtype=y.dtype
        )
        return transformed_x[:-1], transformed_target.clone()

    target_independent_augment._belief_target_independent = True  # type: ignore[attr-defined]
    dataset_class.augment = target_independent_augment


def build_training_loader(
    dataset: Any,
    sampler: torch.utils.data.Sampler,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> torch.utils.data.DataLoader:
    """Build a loader whose every emitted microbatch has the physical batch size."""

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--history", type=int, choices=(1, 5), required=True)
    parser.add_argument("--p00-record", type=Path, required=True)
    parser.add_argument("--artifact-map", type=Path, help="checksum-verified relocation map for historical artifact paths")
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"non-finite {name}")


def _next_batch(
    loader: torch.utils.data.DataLoader,
    iterator: object,
) -> tuple[object, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    try:
        batch = next(iterator)  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return iterator, batch


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0 or args.accumulation_steps <= 0:
        raise ValueError("batch size and accumulation steps must be positive")
    if args.num_workers < 0:
        raise ValueError("number of workers must be nonnegative")
    effective_batch_size = args.batch_size * args.accumulation_steps
    if effective_batch_size != EFFECTIVE_BATCH_SIZE:
        raise ValueError("effective batch size must equal 64")
    if re.fullmatch(r"[0-9a-f]{40}", args.git_commit) is None:
        raise ValueError("Git commit must be a full 40-character lowercase SHA")

    output = args.output_path.resolve()
    if output.exists():
        raise FileExistsError(output)
    p00_record = args.p00_record.resolve(strict=True)
    p00_checkpoint = _checkpoint_from_record(
        p00_record,
        prototype_id=P00_ID,
        artifact_map=args.artifact_map,
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
    dataset_class = importlib.import_module(
        "dataloader.FireSpreadDataset"
    ).FireSpreadDataset
    dataset_class.find_image_index_from_dataset_index = resolve_dataset_index
    install_target_independent_augmentation(dataset_class)
    base = dataset_class(
        data_dir=str(data_root),
        included_fire_years=list(TRAIN_YEARS),
        n_leading_observations=args.history,
        n_leading_observations_test_adjustment=None,
        crop_side_length=128,
        load_from_hdf5=True,
        is_train=True,
        remove_duplicate_features=False,
        features_to_keep=None,
        return_doy=False,
        stats_years=list(TRAIN_YEARS),
        is_pad=False,
    )
    dataset = BeliefStateTrainingDataset(base, normalized_active_fire_zero(base))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but unavailable")
    sampling_weights = balanced_year_sampling_weights(base)
    sampler = torch.utils.data.WeightedRandomSampler(
        sampling_weights,
        num_samples=len(base),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    loader = build_training_loader(
        dataset,
        sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    if args.method == "reconstruction":
        # Direction C never loads or calls P00 while optimizing its reconstructor.
        default_model: torch.nn.Module = torch.nn.Identity()
    else:
        default_model = load_checkpoint_model(
            p00_checkpoint,
            experiment_id="C00",
            upstream_root=upstream,
            device=device,
        )
    model = build_model(args.method, default_model, args.history).to(device)
    parameters = tuple(model.trainable_parameters())
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("belief-state model exposes invalid trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    model.train()

    iterator = iter(loader)
    final_loss = float("nan")
    final_forecast_loss: float | None = None
    final_state_loss = float("nan")
    for step in range(1, TRAINING_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        step_total = 0.0
        step_state = 0.0
        step_forecast = 0.0
        for _ in range(args.accumulation_steps):
            iterator, batch = _next_batch(loader, iterator)
            packed, clean_state, next_day_target, _scenario_index = batch
            packed = packed.to(device, non_blocking=True)
            clean_state = clean_state.to(device, non_blocking=True)
            features, reliability = split_observations(packed)
            if args.method == "reconstruction":
                state_logits = model.infer_state_logits(features, reliability)
                forecast_loss = state_logits.new_zeros(())
            else:
                next_day_target = next_day_target.to(
                    device, non_blocking=True
                ).long()
                forecast_logits, state_logits, _state = model.forward_state(packed)
                forecast_loss = model.compute_loss(
                    forecast_logits, next_day_target
                )
            _require_finite_tensor(state_logits, "state logits")
            state_loss = masked_unweighted_focal(
                state_logits,
                clean_state,
                ~reliability[:, -1].bool(),
            )
            loss = combine_losses(args.method, forecast_loss, state_loss)
            _require_finite_tensor(forecast_loss, "forecast loss")
            _require_finite_tensor(state_loss, "state loss")
            _require_finite_tensor(loss, "combined loss")
            (loss / args.accumulation_steps).backward()
            step_total += float(loss.detach().cpu())
            step_state += float(state_loss.detach().cpu())
            step_forecast += float(forecast_loss.detach().cpu())
        optimizer.step()
        final_loss = step_total / args.accumulation_steps
        final_state_loss = step_state / args.accumulation_steps
        final_forecast_loss = (
            None
            if args.method == "reconstruction"
            else step_forecast / args.accumulation_steps
        )
        if step == 1 or step % 100 == 0:
            forecast_text = (
                "null"
                if final_forecast_loss is None
                else f"{final_forecast_loss:.8f}"
            )
            print(
                f"BELIEF_STEP={step} LOSS={final_loss:.8f} "
                f"FORECAST={forecast_text} STATE={final_state_loss:.8f}",
                flush=True,
            )

    state = trainable_state_dict(model)
    if not state:
        raise ValueError("belief-state checkpoint has no trainable state")
    if any(key.startswith("default_model.") for key in state):
        raise ValueError("belief-state checkpoint unexpectedly contains P00 state")
    for key, value in state.items():
        _require_finite_tensor(value, f"checkpoint tensor {key}")

    payload = {
        "schema_version": 1,
        "status": "pass",
        "prototype_id": PROTOTYPE_ID,
        "method": args.method,
        "history": args.history,
        "p00_record": str(p00_record),
        "p00_checkpoint": str(p00_checkpoint),
        "git_commit": args.git_commit,
        "train_years": list(TRAIN_YEARS),
        "seed": SEED,
        "steps": TRAINING_STEPS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "physical_batch_size": args.batch_size,
        "effective_batch_size": effective_batch_size,
        "accumulation_steps": args.accumulation_steps,
        "corruptions": list(TRAIN_SCENARIOS),
        "state_weight": STATE_WEIGHT,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "final_loss": final_loss,
        "final_loss_components": {
            "forecast": final_forecast_loss,
            "state": final_state_loss,
        },
        "model_state": state,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        torch.save(payload, handle)
    print(f"BELIEF_CHECKPOINT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
